import os
import time
from typing import Optional
import pandas as pd

import torchview
from pytorch_lightning.utilities.model_summary import ModelSummary

try:
    import pynvml

    PYNVML_ENABLED = True
except Exception:
    PYNVML_ENABLED = False

import pytorch_lightning
import torch

import mlflow
from mlflow import MlflowException
from mlflow.entities import Run
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_ALREADY_EXISTS
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking.context import registry as context_registry
import logging
from mlflow.utils.file_utils import TempDir
from dotenv import load_dotenv
from copy import deepcopy

DEFAULT_AWAIT_MAX_SLEEP_SECONDS = 300
LOGGER = logging.getLogger('diagnome-mlflow-logger')
LOGGING_FORMAT = "[%(process)d - %(processName)s %(levelname)s] %(asctime)s ~~ %(relativeCreated)d [%(filename)s::%(" \
                 "lineno)d (%(funcName)s)]: %(message)s "
handler = logging.StreamHandler()
# handler2 = FileHandler(filename=os.path.join(ROOT, OUTPUT_DIR, LOG_DIR, PROGRAM_RUN_IDENTIFIER + '.log'))
handler.setFormatter(logging.Formatter(fmt=LOGGING_FORMAT))
# handler2.setFormatter(logging.Formatter(fmt=LOGGING_FORMAT))
LOGGER.addHandler(handler)
# LOGGER.addHandler(handler2)
LOGGER.setLevel(logging.DEBUG)


def are_state_dicts_equal(a, b):
    for key in a:
        if key not in b:
            return False
        if not torch.allclose(a[key], b[key], atol=1e-8):
            return False

    for key in b:
        if key not in a:
            return False

    return True


class BaseCallback:
    def __init__(self, logger):
        self.logging_enabled = True
        self.stage = 'default'

        self.pynvml_enabled = PYNVML_ENABLED
        if self.pynvml_enabled:
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError:
                self.pynvml_enabled = False
        self.logger = logger
        self.last_log_time = time.time()

    def disable(self):
        self.logging_enabled = False

    def enable(self):
        self.logging_enabled = True

    def set_stage(self, stage):
        self.stage = stage

    def perform_logging(self, epoch, logs):
        if not self.logging_enabled:
            return

        current_time = time.time()
        self.logger.log_metric(key=f'{self.stage}/system/epoch_time', value=current_time - self.last_log_time,
                               step=epoch)
        self.last_log_time = current_time

        for k, v in logs.items():
            self.logger.log_metric(key=f'{self.stage}/metric/{k}', value=v, step=epoch)

        if self.pynvml_enabled:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mi = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.logger.log_metric(key=f'{self.stage}/system/gpu/{i}/used_percentage', value=mi.used / mi.total,
                                       step=epoch)
                ur = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.logger.log_metric(key=f'{self.stage}/system/gpu/{i}/utilization/memory', value=ur.memory,
                                       step=epoch)
                self.logger.log_metric(key=f'{self.stage}/system/gpu/{i}/utilization/gpu', value=ur.gpu, step=epoch)


class LightningCallback(pytorch_lightning.Callback, BaseCallback):
    def __init__(self, logger):
        BaseCallback.__init__(self, logger)
        pytorch_lightning.Callback.__init__(self)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.perform_logging(trainer.current_epoch, trainer.logged_metrics)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.perform_logging(trainer.current_epoch, trainer.logged_metrics)


class EarlyStoppingPL(pytorch_lightning.callbacks.EarlyStopping):
    def __init__(
            self,
            monitor: str,
            min_delta: float = 0.0,
            patience: int = 3,
            verbose: bool = False,
            mode: str = "min",
            strict: bool = True,
            check_finite: bool = True,
            stopping_threshold: Optional[float] = None,
            divergence_threshold: Optional[float] = None,
            check_on_train_epoch_end: Optional[bool] = None,
            log_rank_zero_only: bool = False,
    ):
        super().__init__(monitor, min_delta, patience, verbose, mode, strict, check_finite, stopping_threshold,
                         divergence_threshold, check_on_train_epoch_end, log_rank_zero_only)
        self.stored_module_state_dict = None
        self.best_epoch = -1

    def perform_update_of_weights(self, trainer, pl_module):
        if self.wait_count == 0:  # new best
            self.best_epoch = trainer.current_epoch
            self.stored_module_state_dict = deepcopy(pl_module.state_dict())

    @property
    def best_state_dict(self):
        return self.stored_module_state_dict

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # super().on_train_epoch_end(trainer, pl_module)
        # self.perform_update_of_weights(trainer, pl_module)
        pass

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_end(trainer, pl_module)
        self.perform_update_of_weights(trainer, pl_module)


class MLFlowLogger:
    REQUIRED_ENV_VARS = (
        'MLFLOW_TRACKING_URI',
        'MLFLOW_TRACKING_USERNAME',
        'MLFLOW_TRACKING_PASSWORD',
        'MLFLOW_S3_ENDPOINT_URL',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
    )

    def __init__(self, experiment_name):
        load_dotenv()
        for required in self.REQUIRED_ENV_VARS:
            assert os.getenv(required) is not None, f"{required} must be set in a .env file, which should be " \
                                                    f"placed next to dgn_mlflow_logger/logger.py."

        LOGGER.debug("MLFLow logger loaded successfully.")

        self.client = mlflow.client.MlflowClient()
        exp = self.client.get_experiment_by_name(experiment_name)
        LOGGER.debug(f"found exp with name {experiment_name} = {exp}")
        self.experiment_id = exp.experiment_id if exp else self.client.create_experiment(name=experiment_name)
        LOGGER.debug(f"Experiment ID = {self.experiment_id}")
        self.current_run = None

    def start(self, tags=None, terminate_run: bool = False):
        """
        Starts a new MLFlow Run. Should be used with `with` syntax as `with instance.start()`.
        :param terminate_run: if the run should terminate (if not, can be continued)
        :type tags: dict
        :param tags: custom tags if needed
        :return: WithSyntaxRunWrapper instance wrapping the newly created MLFlow Run
        """
        self.current_run = self.client.create_run(self.experiment_id, tags=context_registry.resolve_tags(tags))
        return WithSyntaxRunWrapper(self.current_run, self, terminate_run)

    def continue_run(self, terminate_run: bool = False):
        """
        Continues a finished run. Should be used with `with` syntax as `with instance.continue_run()`.
        :param terminate_run: if the run should terminate (if not, can be continued)
        :return: WithSyntaxRunWrapper instance wrapping the restored MLFlow Run
        """
        return WithSyntaxRunWrapper(self.current_run, self, terminate_run)

    def get_keras_callback(self):
        raise ValueError("Not supported.")

    def get_lightning_callback(self):
        """
        Returns a logging callback for Pytorch Lightning.
        :return: LightningCallback instance
        """
        return LightningCallback(self)

    def log_params(self, dictionary):
        """
        Log multiple parameters at once.
        :type dictionary: dict
        :param dictionary: dictionary of logged parameters, see log_param docs for details.
        """
        for k, v in dictionary.items():
            self.log_param(k, v)

    def log_param(self, key, value):
        """
        Log a parameter.
        :type key: str
        :param key: Parameter name (string). This string may only contain alphanumerics, underscores
                    (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                    Supports keys up to length 250.
        :param value: Parameter value (string, but will be string-ified if not).
                      Supports value up to length 500.
        """
        if self.current_run is None:
            raise ValueError("No run exists. Start a run by calling the `start(...)` method first.")
        self.client.log_param(self.current_run.info.run_id, key, value)

    def log_metric(self, key, value, step):
        """
        Log a metric value. Probably unnecessary, used mainly by callbacks' functions.
        If you want to log metrics, consider creating a callback,
        for example using `get_keras_callback()`.

        :type value: float
        :type key: str
        :param key: Metric name (string). This string may only contain alphanumerics, underscores
                    (_), dashes (-), periods (.), spaces ( ), and slashes (/).
                    All backend stores will support keys up to length 250, but some may
                    support larger keys.
        :param value: Metric value (float). Note that some special values such
                      as +/- Infinity may be replaced by other values depending on the store. For
                      example, the SQLAlchemy store replaces +/- Inf with max / min float values.
                      All backend stores will support values up to length 5000, but some
                      may support larger values.
        :param step: Integer training step (iteration) at which was the metric calculated.
                     Defaults to 0.
        """
        if self.current_run is None:
            raise ValueError("No run exists. Start a run by calling the `start(...)` method first.")
        self.client.log_metric(self.current_run.info.run_id, key, value, step=step)

    def log_text(self, text, path):
        """
        Saves text to our artifact repository.
        :param text: any string
        :param path: resulting file path, for example path/to/file.txt
        """
        if self.current_run is None:
            raise ValueError("No run exists. Start a run by calling the `start(...)` method first.")
        self.client.log_text(self.current_run.info.run_id, text, path)

    def describe_torch_model(self, model: pytorch_lightning.LightningModule,
                             datamodule: pytorch_lightning.LightningDataModule,
                             path='description'):
        """
        Stores useful information about a torch model in our artifact repository.
        :param datamodule: datamodule for fetching input example
        :param model: target model
        :param path: resulting directory path, for example path/to/dir
        """
        if self.current_run is None:
            raise ValueError("No run exists. Start a run by calling the `start(...)` method first.")

        run_id = self.current_run.info.run_id
        example_input = self.fetch_example_input(datamodule)

        with TempDir() as tmp:
            torchview.draw_graph(model, input_data=example_input, expand_nested=True,
                                 directory=tmp.path(), depth=5, graph_name='model', save_graph=True)
            local_path = tmp.path('model.gv.png')
            self.client.log_artifact(run_id, local_path, path)

        self.log_text(str(ModelSummary(model, max_depth=-1)), os.path.join(path, "summary.txt"))

    def save_torch_model(self, model, path, registered_model_name=None, registered_model_tags=None,
                         datamodule_for_signature_inference=None):
        trainer = model.trainer
        if model.trainer is not None:
            model.trainer = None

        if datamodule_for_signature_inference is None:
            self.save_model(model, path, flavor=mlflow.pytorch, registered_model_name=registered_model_name,
                            registered_model_tags=registered_model_tags)
        else:
            try:
                signature = self.infer_signature_from_datamodule(datamodule_for_signature_inference, model)
            except Exception as e:
                print(f"Couldn't infer signature due to {e}.")
                signature = None
            self.save_model(model, path, flavor=mlflow.pytorch, registered_model_name=registered_model_name,
                            registered_model_tags=registered_model_tags,
                            signature=signature,
                            # input_example=example_input, # <-- very large in size, not useful
                            input_example=None,
                            )

        if trainer is not None:
            model.trainer = trainer

    def infer_signature_from_datamodule(self, datamodule_for_signature_inference, model):
        original_training_value = model.training
        model.train(False)
        example_input = self.fetch_example_input(datamodule_for_signature_inference)
        example_output = model(example_input)
        if isinstance(example_output, tuple):
            example_output = {f'output_{i}': example_output[i].detach().numpy() for i in range(len(example_output))}
        else:
            example_output = example_output.detach().numpy()
        example_input = example_input.detach().numpy()
        signature = mlflow.models.infer_signature(example_input, example_output)
        model.train(original_training_value)
        return signature

    def fetch_example_input(self, datamodule_for_signature_inference):
        example_input = next(iter(datamodule_for_signature_inference.train_dataloader()))
        if isinstance(example_input, (tuple, list)):
            example_input = example_input[0]
        return example_input

    def save_model(self, model, path, flavor, registered_model_name=None, registered_model_tags=None,
                   signature=None, input_example=None):
        """
        Saves a TF model to our artifact repository.

        :param input_example: example input for model (stored as JSON on mlflow server)
        :param signature: model signature created by mlflow from sample input/output
        :type registered_model_tags: dict
        :type registered_model_name: str
        :type path: str
        :param model: model instance (either Keras or Tensorflow module)
        :param path: resulting S3 path
        :param flavor: save_model module (mlflow.tensorflow for example)
        :param registered_model_name: Optional model name for registration. If you need the model to
        be registered, specify this. Versions will be automatically incremented.
        :param registered_model_tags: Optional tags for the registered model.
        :return: MLFlow Model metadata.
        """

        if self.current_run is None:
            raise ValueError("No run exists. Start a run by calling the `start(...)` method first.")

        run_id = self.current_run.info.run_id

        with TempDir() as tmp:
            local_path = tmp.path("model")
            mlflow_model = Model(run_id=run_id, artifact_path=path)

            # in tmpdir
            flavor.save_model(
                model,
                path=local_path,
                mlflow_model=mlflow_model,
                signature=signature,
                input_example=input_example,
                # pickle_module=pickle
            )

            # from tmpdir to our s3
            self.client.log_artifacts(
                run_id=run_id,
                local_dir=local_path,
                artifact_path=mlflow_model.artifact_path
            )

            try:
                # noinspection PyProtectedMember
                self.client._record_logged_model(run_id, mlflow_model)
            except MlflowException:
                LOGGER.error("Failed to record model to " + mlflow.get_artifact_uri())
            if registered_model_name is not None:
                self.__register_model(
                    mlflow_model,
                    registered_model_name,
                    registered_model_tags,
                    run_id
                )
        return mlflow_model.get_model_info()

    def set_tag_for_registered_model(self, name, version, key, value):
        version = str(version)
        if version in ('staging', 'production'):
            self.client.set_model_version_tag(name, None, key, value, stage=version)
        else:
            self.client.set_model_version_tag(name, version, key, value)

    def set_description_for_registered_model(self, name, version=None, text=""):
        if version is None:
            self.client.update_registered_model(name, text)
        else:
            version = str(version)
            self.client.update_model_version(name, version, text)

    def get_run_id_of_a_registered_model(self, model_name, version):
        version = str(version)
        if version in ('staging', 'production'):
            model_version = self.client.get_latest_versions(model_name, stages=[version])[0]
        else:
            model_version = self.client.get_model_version(model_name, version)

        return model_version.run_id

    def get_tag_of_a_registered_model(self, model_name, tag_name):
        try:
            latest_versions = self.client.get_latest_versions(model_name)
        except mlflow.exceptions.RestException:
            raise ValueError(f"Model {model_name} does not exist.")

        latest_version = max(int(v.version) for v in latest_versions)
        ret = []
        for version_number in range(1, latest_version + 1):
            try:
                mv = self.client.get_model_version(model_name, str(version_number))
            except mlflow.exceptions.RestException:
                print(f"Skipping {version_number} as it does not exist.")
                continue

            if tag_name in mv.tags:
                ret.append(mv.tags[tag_name])
            else:
                print(f"Skipping {version_number} as it does not have tag '{tag_name}'.")
                continue

        return ret

    def log_artifact_for_registered_model(self, name, version, local_path, remote_path):
        version_object = self.__get_version_object(name, version)

        self.client.log_artifact(version_object.run_id, local_path, remote_path)

    def log_artifacts(self, local_dir, remote_name=None, run_id=None):
        if run_id is None:
            if self.current_run is None:
                raise ValueError("No run exists. Start a run by calling the `start(...)` method first.")
            run_id = self.current_run.info.run_id

        if remote_name is None:
            remote_name = os.path.basename(local_dir)

        self.client.log_artifacts(run_id, local_dir, remote_name)

    def remove_artifacts_for_registered_model(self, name, version, remote_path_root):
        version_object = self.__get_version_object(name, version)
        run_obj = self.client.get_run(version_object.run_id)
        repo = get_artifact_repository(run_obj.info.artifact_uri)
        repo.delete_artifacts(remote_path_root)

    def get_experiment_dataframe(self, experiment_name):
        exp = self.client.get_experiment_by_name(experiment_name)
        exp_id = exp.experiment_id

        list_of_runs = self.client.search_runs([exp_id], max_results=10000)
        return pd.DataFrame.from_dict([r.data.metrics | r.data.params | r.data.tags for r in list_of_runs])

    def unregister_model(self, model_name):
        """
        Removes all version of a given model.
        :param model_name: model name
        """
        self.client.delete_registered_model(model_name)

    @staticmethod
    def load_torch_model(uri):
        """
        Loads a PL model from our S3 repository.
        :type uri: str
        :param uri: Model path.
        :return: Loaded model instance.
        """
        return mlflow.pytorch.load_model(uri)

    @staticmethod
    def load_keras_model(uri):
        """
        Loads a TF model from our S3 repository.
        :type uri: str
        :param uri: Model path.
        :return: Loaded model instance.
        """
        raise ValueError("Not supported.")

    def terminate_current_run(self, status):
        """
        Terminates the current run. Should not be used manually.
        :param status: final run status
        """
        self.client.set_terminated(self.current_run.info.run_id, status)
        self.current_run = None

    def __get_version_object(self, name, version):
        version = str(version)
        version_object = self.client.get_model_version(name, version)
        if version_object is None:
            raise ValueError(f"No such model {name} and version {version}.")
        return version_object

    def __register_model(self, mlflow_model, registered_model_name, registered_model_tags, run_id):
        model_uri = "runs:/{}/{}".format(run_id, mlflow_model.artifact_path)
        LOGGER.debug(f"Model URI: {model_uri}")
        await_registration_for = DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        try:
            create_model_response = self.client.create_registered_model(registered_model_name)
            LOGGER.debug("Successfully registered model '%s'." % create_model_response.name)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                LOGGER.error(
                    "Registered model '%s' already exists. Creating a new version of this model..."
                    % registered_model_name
                )
            else:
                raise e
        source = RunsArtifactRepository.get_underlying_uri(model_uri)
        (run_id, _) = RunsArtifactRepository.parse_runs_uri(model_uri)
        create_version_response = self.client.create_model_version(
            registered_model_name,
            source,
            run_id,
            tags=registered_model_tags,
            await_creation_for=await_registration_for
        )
        LOGGER.debug(
            "Created version '{version}' of model '{model_name}'.".format(
                version=create_version_response.version, model_name=create_version_response.name
            )
        )
        return create_version_response


class WithSyntaxRunWrapper(Run):
    def __init__(self, run, logger: MLFlowLogger, terminate_run: bool):
        Run.__init__(self, run.info, run.data)
        self.logger = logger
        self.terminate_run = terminate_run

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            if self.terminate_run:
                self.logger.terminate_current_run('FINISHED')
        else:
            import traceback
            self.logger.log_text("\n".join(traceback.format_exception(exc_type, exc_val, exc_tb)),
                                 "traceback.txt")
            self.logger.terminate_current_run('FAILED')

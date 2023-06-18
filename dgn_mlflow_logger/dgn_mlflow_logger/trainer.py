import os.path
from datetime import timedelta

import pytorch_lightning
from mlflow.utils.file_utils import TempDir
from pytorch_lightning import Trainer
from dgn_mlflow_logger.logger import EarlyStoppingPL, MLFlowLogger


class ArtifactBuilder:
    def __init__(self):
        self.model: pytorch_lightning.LightningModule = None
        self.datamodule: pytorch_lightning.LightningDataModule = None

    @property
    def save_dir(self):
        """
        Where the artifacts will be stored locally.
        :return:
        """
        raise NotImplementedError

    def build(self, tmp):
        """
        Creates the artifacts in save_dir.

        :param tmp: Temporary folder.
        :return:
        """
        raise NotImplementedError

    def load_model_and_datamodule(self, model: pytorch_lightning.LightningModule,
                                  datamodule: pytorch_lightning.LightningDataModule):
        self.model = model
        self.datamodule = datamodule


class DGNTrainer:
    def __init__(self,
                 experiment_name,

                 max_epochs,
                 max_time: timedelta = None,

                 accelerator: str = 'cpu',
                 devices: int = 1,

                 es_monitor: str = 'val_loss',
                 es_mode: str = 'min',
                 es_patience: int = 5,
                 ):
        """
        Initializes the Diagnome trainer.
        Early stopping only supports validation metrics or losses (on purpose).

        :param experiment_name: name of the experiment (mlflow)
        :param max_epochs: how many epochs to train at most
        :param max_time: how much time to train at most
        :param accelerator: 'cpu' or 'gpu' or other if necessary, see pl.Trainer
        :param devices: number of devices to train on
        :param es_monitor: monitor for early stopping, default 'val_loss'
        :param es_mode: monitor mode for early stopping, default 'min' -> should minimize monitor, otherwise use 'max'
        :param es_patience: how many epochs to give monitor chance to improve, default 5
        """

        self.mlogger = MLFlowLogger(experiment_name)
        self.__logging_callback = self.mlogger.get_lightning_callback()

        self.__trainer = None
        self.__accelerator = accelerator
        self.__devices = devices
        self.__max_epochs = max_epochs
        self.__max_time = max_time

        self.__es_monitor = es_monitor
        self.__es_mode = es_mode
        self.__es_patience = es_patience

        self.__rebuild_trainer_and_es()

    def __rebuild_trainer_and_es(self):
        self.__early_stopping = EarlyStoppingPL(
            self.__es_monitor, mode=self.__es_mode, patience=self.__es_patience, verbose=False,
            check_on_train_epoch_end=False)

        self.__trainer = Trainer(
            max_epochs=self.__max_epochs,
            max_time=self.__max_time,
            accelerator=self.__accelerator,
            devices=self.__devices,
            callbacks=[
                self.__logging_callback,
                self.__early_stopping,
            ],
            enable_checkpointing=False,
            logger=False
        )

    def fit_and_validate(self,
                         model,
                         datamodule,

                         parameters: dict = None,
                         run_tags: dict = None,
                         should_save_model: bool = True,
                         should_describe_model: bool = True,
                         should_evaluate_model: bool = False,
                         registered_model_name: str = None,
                         registered_model_tags: dict = None,
                         artifact_builders: list[ArtifactBuilder] = None,

                         terminate_run: bool = True,
                         continues_run: bool = False,
                         logger_stage: str = None,
                         ):
        """
        Main method of the trainer, fits the datamodule, validates the model and stores related artifacts
        using our MLFlow tracking server.

        :param model: LightningModule to use
        :param datamodule: LightningDataModule to use
        :param parameters: Model parameters
        :param run_tags: tags for this run
        :param should_save_model: whether to save model weights or not
        :param should_describe_model: whether to save model architecture diagram or not
        :param should_evaluate_model: whether to evaluate the model on test data or not
        :param registered_model_name: how to register model
        :param registered_model_tags: what tags to include for the registered version
        :param artifact_builders: auxiliary builders for additional artifacts
        :param terminate_run: if the run should terminate (if not, can be continued)
        :param continues_run: if the run should not be created and instead continued from previous
        :param logger_stage: stage for the mlflow logger ({stage}/metric/val_loss etc.)
        """

        if continues_run:
            run = self.mlogger.continue_run(terminate_run)
            self.__rebuild_trainer_and_es()
        else:
            run = self.mlogger.start(run_tags, terminate_run)

        with run:
            if parameters is not None:
                self.mlogger.log_params(parameters)

            if logger_stage is not None:
                self.__logging_callback.set_stage(logger_stage)

            self.__trainer.fit(model, datamodule=datamodule)

            # Actually, it works better without disabling, since MLFlow aggregates metrics
            # into the last logged value anyway... 
            # self.__logging_callback.disable()

            model.load_state_dict(self.__early_stopping.best_state_dict)
            results = self.__trainer.validate(model, datamodule=datamodule)

            if should_evaluate_model:
                print("Evaluation not supported yet, only validation.")
                # results |= self.__trainer.test(model, datamodule=datamodule)

            if logger_stage is None:
                tags = {
                    'best_step': self.__early_stopping.best_epoch,
                    'last_step': self.__trainer.current_epoch,
                    'target_metric': self.__early_stopping.monitor,
                }
            else:
                tags = {
                    f'{logger_stage}_best_step': self.__early_stopping.best_epoch,
                    f'{logger_stage}_last_step': self.__trainer.current_epoch,
                    f'{logger_stage}_target_metric': self.__early_stopping.monitor,
                }

            for r in results:
                tags |= r

            if registered_model_tags is not None:
                tags |= registered_model_tags

            if should_save_model:
                self.mlogger.save_torch_model(model,
                                              'saved_model',
                                              registered_model_name=registered_model_name,
                                              registered_model_tags=tags,
                                              datamodule_for_signature_inference=datamodule
                                              )

            if should_describe_model:
                self.mlogger.describe_torch_model(model, datamodule)

            if artifact_builders is not None:
                for builder in artifact_builders:
                    builder.load_model_and_datamodule(model, datamodule)
                    with TempDir() as tmp:
                        builder.build(tmp)
                        self.mlogger.log_artifacts(tmp.path(builder.save_dir))

            print(f"Run stored at {os.getenv('MLFLOW_TRACKING_URI')}/#/experiments/"
                  f"{self.mlogger.experiment_id}/runs/{self.mlogger.current_run.info.run_id}")

    def validate_registered(self,
                            model_name,
                            version,
                            datamodule,
                            context: str,
                            should_validate: bool = True,
                            artifact_builders: list[ArtifactBuilder] = None):
        """
        Validates (dev dataset) a model registered in MLFlow using the supplied datamodule and/or
        stores related artifacts using our MLFlow tracking server.

        :param model_name: Name of the registered model
        :param version: Target version of the registered model
        :param datamodule: LightningDataModule to use
        :param context: Why is this model being evaluated post-train? Short description.
        :param should_validate: Whether to actually validate the model or just build artifacts.
        :param artifact_builders: auxiliary builders for additional artifacts
        """

        # only torch is supported, yeah, whatchu gonna do about it, huh? HUH?
        model = self.mlogger.load_torch_model(f"models:/{model_name}/{version}")
        self.__logging_callback.disable()

        if should_validate:
            results = self.__trainer.validate(model, datamodule=datamodule)
            for loader in results:
                for k, v in loader.items():
                    self.mlogger.set_tag_for_registered_model(model_name, version, f"{context}_{k}", v)

        run_id = self.mlogger.get_run_id_of_a_registered_model(model_name, version)

        if artifact_builders is not None:
            for builder in artifact_builders:
                builder.load_model_and_datamodule(model, datamodule)
                with TempDir() as tmp:
                    builder.build(tmp)
                    self.mlogger.log_artifacts(tmp.path(builder.save_dir), f"{context}_{builder.save_dir}", run_id)

    def validate_local(self,
                       model,
                       datamodule,
                       context: str,
                       continue_run: bool = True,
                       terminate_run: bool = True,
                       artifact_builders: list[ArtifactBuilder] = None):
        """
        Validates (dev dataset) a model instance using the supplied datamodule and/or
        stores related artifacts using our MLFlow tracking server.

        :param model: LightningModule to validate
        :param datamodule: LightningDataModule to use
        :param context: Why is this model being evaluated post-train? Short description.
        :param continue_run: if the run should continue (if not, will be started)
        :param terminate_run: if the run should terminate (if not, can be continued)
        :param artifact_builders: auxiliary builders for additional artifacts
        """

        if continue_run:
            run = self.mlogger.continue_run
        else:
            run = self.mlogger.start

        with run(terminate_run=terminate_run):
            self.__logging_callback.enable()
            self.__logging_callback.set_stage(f'{context}')
            self.__trainer.validate(model, datamodule=datamodule)

            if artifact_builders is not None:
                for builder in artifact_builders:
                    builder.load_model_and_datamodule(model, datamodule)
                    with TempDir() as tmp:
                        builder.build(tmp)
                        self.mlogger.log_artifacts(tmp.path(builder.save_dir), f"{context}_{builder.save_dir}",
                                                   self.mlogger.current_run.info.run_id)

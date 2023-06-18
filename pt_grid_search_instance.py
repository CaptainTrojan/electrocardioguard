import argparse
import os
import warnings

from dgn_mlflow_logger import DGNTrainer

from h5_pt_dataloader import HDF5ECGDataset, ECGDataModule
from models.pt_cdil_cnn import CDIL
from models.pt_discriminator_head import DiscriminatorHead
from models.pt_embedings_model import ResNet
from models.pt_long_conv import LongConvModel
from models.pt_metric_learning_module import MetricLearning
from models.pt_pair_module import PairLoss
from models.pt_s4 import S4Model
from utils.pt_frequency_interpolation import str2bool

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    dataset_options = ['ikem', 'c15p', 'ptbxl']

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help="What to call the experiment.", required=True)
    parser.add_argument('-dsr', '--dataset_root', type=str, help="Absolute path to the datasets folder or relative to "
                                                                 "this script.", required=True)
    parser.add_argument('-d', '--dataset', choices=dataset_options, help="Which data to train on.", required=True)
    parser.add_argument('-save', '--save_model', type=str2bool, help="Whether to save the model or not.", default=False)

    # first gridsearch
    parser.add_argument('-m', '--model', choices=['1d-rn', 'cdil', 's4', 'long-conv'],
                        help="Which model to use.", required=True)
    parser.add_argument('-p', '--metric', choices=['triplet', 'circle'], required=True, help='What pretraining to use.')
    parser.add_argument('-e', '--embedding_size', type=int, choices=[128, 256, 384, 512], required=True,
                        help='Embedding size.')
    parser.add_argument('-norm', '--normalize', type=str2bool, help="Whether to normalize the leads or not.")
    parser.add_argument('-bwr', '--baseline_wander_removal',
                        type=str2bool, help="Whether to remove baseline wander from the leads or not.")
    parser.add_argument('-hfw', '--high_frequency_removal',
                        type=str2bool, help="Whether to remove high frequency noise from the leads or not.")
    parser.add_argument('-f', '--freeze_embeddings', type=str2bool, required=True, help='Whether to freeze embeddings')

    # second gridsearch
    parser.add_argument('-dh', '--discriminator_hidden_size', type=int, choices=[0, 16, 64], required=True,
                        help="Discriminator hidden size (0 if single layer).")
    parser.add_argument('-dl2', '--discriminator_l2_approach', type=str, choices=['none', 'merge', 'full'],
                        help="How to handle L2 distance in discriminator.")
    parser.add_argument('-dl1', '--discriminator_l1_approach', type=str, choices=['none', 'merge', 'full'],
                        help="How to handle L1 distance in discriminator.")
    parser.add_argument('-dcos', '--discriminator_cos_approach', type=str, choices=['none', 'merge', 'full'],
                        help="How to handle cosine distance in discriminator.")

    # other
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-tss', '--train_sample_size', type=int, default=500)
    parser.add_argument('-vss', '--val_sample_size', type=int, default=200)
    parser.add_argument('-me', '--max_epochs', type=int, default=1)
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-acc', '--accel', type=str, default='cpu')

    args = parser.parse_args()

    ROOT = args.dataset_root
    params = {k: v for k, v in args.__dict__.items()}
    params['es_patience'] = 15

    trainer = DGNTrainer(args.name,
                         max_epochs=params['max_epochs'],
                         accelerator=params['accel'],
                         es_patience=params['es_patience'])

    # Create model

    if args.model == '1d-rn':
        embedding = ResNet(args.normalize, False, args.baseline_wander_removal, args.high_frequency_removal,
                           args.embedding_size)
    elif args.model == 'cdil':
        embedding = CDIL(args.normalize, args.baseline_wander_removal, args.high_frequency_removal,
                         embedding_size=args.embedding_size)
    elif args.model == 's4':
        embedding = S4Model(normalize=args.normalize, remove_baseline=args.baseline_wander_removal,
                            remove_hf_noise=args.high_frequency_removal, d_output=args.embedding_size,
                            d_input=12)
    elif args.model == 'long-conv':
        embedding = LongConvModel(d_input=12, normalize=args.normalize, remove_baseline=args.baseline_wander_removal,
                                  remove_hf_noise=args.high_frequency_removal, d_output=args.embedding_size)
    else:
        raise ValueError(f"Unknown value of {args.model=}")

    # First phase (train embeddings)

    should_tranpose = {
        '1d-rn': True,
        'cdil': True,
        's4': False,
        'long-conv': False,
    }

    first_phase = MetricLearning(embedding, variant=args.metric, should_transpose=should_tranpose[args.model])
    first_phase_dm = ECGDataModule(os.path.join(ROOT, args.dataset), params['batch_size'],
                                   mode=HDF5ECGDataset.Mode.MODE_TRIPLETS,
                                   sample_size=params['train_sample_size'],
                                   num_workers=params['num_workers'])

    trainer.fit_and_validate(first_phase, first_phase_dm, parameters=params,
                             should_save_model=False,
                             should_describe_model=False,
                             logger_stage='embedding',

                             terminate_run=False)

    # Second phase (train pair discriminator)

    embedding = first_phase.embedding_model
    discriminator = DiscriminatorHead(args.embedding_size, args.discriminator_hidden_size,
                                      args.discriminator_l2_approach,
                                      args.discriminator_l1_approach,
                                      args.discriminator_cos_approach,
                                      )
    second_phase = PairLoss(embedding, discriminator, args.freeze_embeddings,
                            should_transpose=should_tranpose[args.model])
    second_phase_dm = ECGDataModule(os.path.join(ROOT, args.dataset), params['batch_size'],
                                    mode=HDF5ECGDataset.Mode.MODE_PAIRS,
                                    sample_size=params['train_sample_size'],
                                    num_workers=params['num_workers'])
    trainer.fit_and_validate(second_phase, second_phase_dm,
                             should_save_model=args.save_model,
                             should_describe_model=False,
                             logger_stage='pairs',

                             continues_run=True,
                             terminate_run=False)

    # Validate result on all datasets

    for should_terminate, ds in zip(
            [False if i < len(dataset_options) - 1 else True for i in range(len(dataset_options))],
            dataset_options,
    ):
        tf = 0.7 if ds == args.dataset else 0.0
        df = 0.1 if ds == args.dataset else 0.5
        ef = 0.2 if ds == args.dataset else 0.5

        dm = ECGDataModule(os.path.join(ROOT, ds), params['batch_size'], HDF5ECGDataset.Mode.MODE_PAIRS,
                           sample_size=params['val_sample_size'],
                           num_workers=params['num_workers'],
                           train_fraction=tf, dev_fraction=df, test_fraction=ef)

        trainer.validate_local(second_phase, dm, context=f'{ds}_20k', terminate_run=should_terminate)

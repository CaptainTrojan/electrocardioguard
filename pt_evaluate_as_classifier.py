import argparse
from time import sleep
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.pt_classifier import Classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument('-m', '--model_path', type=str,
                        help="Path to the saved embedder + discriminator model. Exclusive with -e and -d.")
    parser.add_argument('-e', '--embedder_path', type=str,
                        help="Path to the saved embedder model. Exclusive with -m, requires -d.",
                        default="best_model/best_embedding")
    parser.add_argument('-d', '--discriminator_path', type=str,
                        help="Path to the saved discriminator model. Exclusive with -m, requires -e.",
                        default="best_model/best_discriminator")

    # Dataset settings
    parser.add_argument('-ds', '--dataset', type=str, help="Dataset name.", choices=['c15p', 'ikem', 'ptb', 'ptbxl'],
                        required=True)
    parser.add_argument('-set', type=str, choices=['dev', 'test', 'whole'], help="Which set to evaluate on.",
                        required=True)
    parser.add_argument('-sm', '--sample_mode', type=str, choices=['random', 'fill'],
                        help="Whether to sample entirely random ECGs or minimize the amount of patients with only "
                             "one sample.", default='fill')

    # Gallery-probe test
    parser.add_argument('-gp', action='store_true', help="Should perform the gallery-probe matching test.")
    parser.add_argument('-gps', '--gallery_probe_samples', type=int,
                        help="How many patients to use for the gallery-probe evaluation.", default=2127)

    # Overseer simulation test
    parser.add_argument('-os', action='store_true', help="Should perform the overseer simulation test.")
    parser.add_argument('-omr', '--overseer_mistake_rates', nargs="+", type=float, default=[1.0, 0.5, 0.05, 0.02, 0.01],
                        help="How many mistakes does an overseer make.")
    parser.add_argument('-ip', '--init_patients', type=int, help="How many patients to initialize the database with.",
                        default=10000)
    parser.add_argument('-es', '--eval_samples', type=int, help="How many samples to classify into the database.",
                        default=1000)
    parser.add_argument('-dt', '--decision_threshold', type=float,
                        help="Discriminator decision threshold (same/different). Should be calculated on the dev set "
                             "after training.", default=0.5)
    parser.add_argument('-lt', '--likelihood_threshold', type=float, help="Likelihood threshold (correct/mistake).",
                        default=0.5)
    parser.add_argument('-a', '--approach', type=str, choices=['a1', 'a2', 'a3', 'a4'],
                        help="Approach of selecting the patient (see paper for details).", default='a1')

    # Other settings
    parser.add_argument('-r', '--repeats', type=int, default=1)
    parser.add_argument('-seed', type=int, default=9713)

    args = parser.parse_args()

    if args.model_path is None:
        assert args.embedder_path is not None, parser.print_help()
        assert args.discriminator_path is not None, parser.print_help()

        embedder = torch.load(args.embedder_path)
        discriminator = torch.load(args.discriminator_path)
    else:
        assert args.embedder_path is None, parser.print_help()
        assert args.discriminator_path is None, parser.print_help()

        model = torch.load(args.model_path)
        embedder = model.__embedding
        discriminator = model.__discriminator

    c = Classifier(embedder, discriminator, args.approach)
    torch.manual_seed(7)

    created_files = []

    if args.gp:
        gallery_probe_result = c.gallery_and_probe(args.dataset, args.set, args.gallery_probe_samples,
                                                   args.seed, args.repeats)

        df = pd.DataFrame.from_records(gallery_probe_result)
        df.reset_index(inplace=True)

        fn = f"gp_{args.dataset}_{args.set}_{args.seed}.csv"
        df.describe().to_csv(fn)

        created_files.append(("Gallery-probe results", fn))
        # print(df)

    if args.os:
        evaluation_result = c.evaluate(args.dataset, args.set, args.sample_mode, args.init_patients, args.eval_samples,
                                       args.overseer_mistake_rates, args.decision_threshold, args.likelihood_threshold,
                                       args.seed, args.repeats)

        df = pd.DataFrame.from_records(evaluation_result)
        df.reset_index(inplace=True)
        fn1 = f"os_{args.dataset}_{args.set}_{args.approach}_{args.seed}.csv"
        df.to_csv(fn1)
        res = df.groupby('overseer_mistakes')[['OM_P@R95', 'OM_P@R95_threshold']].describe()
        fn2 = f"os_{args.dataset}_{args.set}_{args.approach}_{args.seed}_pre_at_rec_95.csv"
        res.to_csv(fn2)

        created_files.append(("Overseer simulation, all runs", fn1))
        created_files.append(("Overseer simulation, p@r95", fn1))

    if not args.gp and not args.os:
        print("You have not selected any evaluation tasks for the model. Try adding -gp or -os to the argument list.")
    else:
        for desc, fn in created_files:
            print(f"{desc} --> '{fn}'")
        print("done")

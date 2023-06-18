import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from dgn_mlflow_logger.trainer import ArtifactBuilder
from torchmetrics.functional import confusion_matrix
from tqdm import tqdm


class BinaryClassifierVisualization(ArtifactBuilder):
    def __init__(self, mode):
        """
        :param mode: One of 'val' or 'test' depending on whether to use val_dataloader() or test_dataloader().
        """
        super().__init__()
        self.mode = mode

    @property
    def save_dir(self):
        return 'validation_curves' if self.mode == 'dev' else 'evaluation_curves'

    def build(self, tmp):
        local_dir = tmp.path(self.save_dir)
        os.mkdir(local_dir)

        thresholds = 2001
        # roc = BinaryROC(thresholds=thresholds)
        # auroc = BinaryAUROC(thresholds=thresholds)
        # prc = BinaryPrecisionRecallCurve(thresholds=thresholds)

        train_value = self.model.training
        self.model.train(False)
        negative_population_test_values = []
        positive_population_test_values = []
        all_predictions = []
        all_targets = []

        dataloader = self.datamodule.val_dataloader()

        for test_batch in tqdm(dataloader, desc='Loading data'):
            x, y = test_batch
            x = x.to(self.model.device)
            y_pred = self.model(x).cpu().detach()

            y = y.to(int)
            # roc(y_pred, y)
            # auroc(y_pred, y)
            # prc(y_pred, y)

            for true_answer, predicted_probability in zip(y, y_pred):
                all_predictions.append(predicted_probability)
                all_targets.append(true_answer)

                if true_answer == 1:
                    positive_population_test_values.append(predicted_probability.numpy())
                else:
                    negative_population_test_values.append(predicted_probability.numpy())

        self.model.train(train_value)

        all_predictions = torch.stack(all_predictions)
        all_targets = torch.stack(all_targets)

        _tp = []
        _fp = []
        _fn = []
        _tn = []
        T = []
        for t in tqdm(torch.linspace(0.0, 1.0, thresholds), desc='Calculating confusion matrices'):
            (tn, fp), (fn, tp) = confusion_matrix(all_predictions, all_targets, task='binary', threshold=float(t))
            total = tp + fp + fn + tn

            T.append(t)
            _tp.append(tp / total)
            _fp.append(fp / total)
            _fn.append(fn / total)
            _tn.append(tn / total)

        tp = torch.as_tensor(_tp)
        fp = torch.as_tensor(_fp)
        fn = torch.as_tensor(_fn)
        tn = torch.as_tensor(_tn)

        fpr = (fp / (fp + tn))
        tpr = (tp / (tp + fn))
        rec = (tp / (tp + fn))
        pre = (tp / (tp + fp))
        acc = ((tp + tn) / (tp + fp + fn + tn))

        f1 = torch.nan_to_num(2 * pre * rec / (pre + rec))
        f1_best_threshold_index = torch.argmax(f1)
        acc_best_threshold_index = torch.argmax(acc)
        f1_best_threshold = T[f1_best_threshold_index]
        acc_best_threshold = T[acc_best_threshold_index]
        figure_size = (8, 8)
        bins = np.linspace(0, 1, 51)

        # Create a figure and axes
        fig, ax = plt.subplots()
        fig.set_figheight(figure_size[1] - 3)
        fig.set_figwidth(figure_size[0] + 3)

        # Create the histograms for each population
        positive_heights = ax.hist(positive_population_test_values, bins=bins, alpha=0.5, label='Positive population',
                                   color='#96C281')
        negative_heights = ax.hist(negative_population_test_values, bins=bins, alpha=0.5, label='Negative population',
                                   color='#D78484')
        max_y = max([*positive_heights[0], *negative_heights[0]])

        ax.plot([f1_best_threshold, f1_best_threshold], [0, max_y], 'k--',
                label=f'Optimal F1 threshold {f1_best_threshold:.2f} -> {f1[f1_best_threshold_index]:.3f}')

        ax.plot([acc_best_threshold, acc_best_threshold], [0, max_y], 'k--',
                label=f'Optimal ACC threshold {acc_best_threshold:.2f} -> {acc[acc_best_threshold_index]:.3f}')

        ax.set_xticks(torch.linspace(0, 1, 21), rotation=45)
        ax.tick_params(axis='x', labelrotation=45)

        # Add a legend and title
        ax.legend()
        ax.set_title('Population histograms')
        plt.legend()
        plt.savefig(os.path.join(local_dir, 'populations.png'))
        plt.clf()

        plt.figure(figsize=figure_size)
        plt.xticks(torch.linspace(0, 1, 21), rotation=45)
        plt.yticks(torch.linspace(0, 1, 21))
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().tick_params(axis='x', labelrotation=45)
        plt.grid(alpha=0.3)
        plt.plot(fpr, tpr, label=f'ROC curve')
        plt.scatter(
            x=[fpr[thresholds - f1_best_threshold_index - 1]],
            y=[tpr[thresholds - f1_best_threshold_index - 1]],
            s=60, marker='x', c='red',
            label=f'Optimal F1 threshold {f1_best_threshold:.3f} -> {f1[f1_best_threshold_index]:.2f}')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title("Receiver operating characteristic curve")
        plt.legend()
        plt.savefig(os.path.join(local_dir, 'roc.png'))
        plt.clf()

        plt.figure(figsize=figure_size)
        plt.xticks(torch.linspace(0, 1, 21), rotation=45)
        plt.yticks(torch.linspace(0, 1, 21))
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().tick_params(axis='x', labelrotation=45)
        plt.grid(alpha=0.3)
        plt.plot(rec, pre, label=f'Precision/Recall')
        plt.scatter(x=[rec[f1_best_threshold_index]], y=[pre[f1_best_threshold_index]],
                    s=60, marker='x', c='red',
                    label=f'Optimal F1 threshold {f1_best_threshold:.3f} -> {f1[f1_best_threshold_index]:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("Precision-recall curve")
        plt.legend()
        plt.savefig(os.path.join(local_dir, 'precall.png'))
        plt.clf()

        plt.figure(figsize=figure_size)
        plt.xticks(torch.linspace(0, 1, 21), rotation=45)
        plt.yticks(torch.linspace(0, 1, 21))
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().tick_params(axis='x', labelrotation=45)
        plt.grid(alpha=0.3)
        plt.plot(T, f1, label='F1')
        plt.plot(T, pre, label='Precision')
        plt.plot(T, rec, label='Recall')
        plt.plot([f1_best_threshold, f1_best_threshold], [0, 1], 'k--',
                 label=f'Optimal F1 threshold {f1_best_threshold:.4f} -> {f1[f1_best_threshold_index]:.4f}')
        plt.scatter(x=[f1_best_threshold], y=[f1[f1_best_threshold_index]],
                    s=60, marker='x', c='red',
                    label=f'Threshold intersection with the F1 curve')
        plt.xlabel('Decision threshold t (answer = 1 if output > t else 0)')
        plt.ylabel('F1 score')
        plt.title("F1 threshold curve")
        plt.legend()
        plt.savefig(os.path.join(local_dir, 'f1.png'))
        plt.clf()

        plt.figure(figsize=figure_size)
        plt.xticks(torch.linspace(0, 1, 21), rotation=45)
        plt.yticks(torch.linspace(0, 1, 21))
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().tick_params(axis='x', labelrotation=45)
        plt.grid(alpha=0.3)
        plt.plot(T, f1, label='F1', alpha=0.5)
        plt.plot(T, acc, label='ACC', alpha=0.5)
        plt.plot(T, tp * 2, label='TP', alpha=0.5)
        # plt.plot(T, fp, label='FP', alpha=0.5)
        plt.plot(T, tn * 2, label='TN', alpha=0.5)
        # plt.plot(T, fn, label='FN', alpha=0.5)
        plt.plot([f1_best_threshold, f1_best_threshold], [0, 1], '--',
                 label=f'Optimal F1 threshold {f1_best_threshold:.4f} -> {f1[f1_best_threshold_index]:.4f}')
        plt.plot([acc_best_threshold, acc_best_threshold], [0, 1], '--',
                 label=f'Optimal ACC threshold {acc_best_threshold:.4f} -> {acc[acc_best_threshold_index]:.4f}')

        plt.xlabel('Decision threshold t (answer = 1 if output > t else 0)')
        plt.ylabel('Value')
        plt.title("Confusion matrix plot")
        plt.legend()
        plt.savefig(os.path.join(local_dir, 'tp_tn.png'))
        plt.clf()

        with open(os.path.join(local_dir, 'optimal_f1_threshold.txt'), 'w') as f:
            f.write(f"{f1_best_threshold}")

        with open(os.path.join(local_dir, 'optimal_acc_threshold.txt'), 'w') as f:
            f.write(f"{acc_best_threshold}")

        with open(os.path.join(local_dir, 'confusion_matrix_acc.txt'), 'w') as f:
            f.write(
                f"---------------\n"
                f"|{tn[acc_best_threshold_index] * 2:.4f}|{fp[acc_best_threshold_index] * 2:.4f}|\n"
                f"|{fn[acc_best_threshold_index] * 2:.4f}|{tp[acc_best_threshold_index] * 2:.4f}|\n"
                f"---------------\n"
            )

import argparse
import os

import torch
from dgn_mlflow_logger.trainer import ArtifactBuilder
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm


class FrequencyInterpolationVisualization(ArtifactBuilder):
    def __init__(self, mode, base_frequency, start=100, end=1100, count=51):
        """
        :param mode: One of 'val' or 'test' depending on whether to use val_dataloader() or test_dataloader().
        """
        super().__init__()
        self.mode = mode
        self.start = start
        self.end = end
        self.count = count
        self.base_frequency = base_frequency

    @property
    def save_dir(self):
        return 'validation_sample_freq_analysis' if self.mode == 'val' else 'evaluation_sample_freq_analysis'

    def interpolate(self, x, new_frequency):
        if len(x.shape) == 4:
            time_dimension = 2
            resulting_size = int(x.shape[time_dimension] * new_frequency / self.base_frequency)

            ekg_array = [v.squeeze(1) for v in x.split(1, dim=1)]
            interpolated = []
            for ekg in ekg_array:
                interpolated.append(
                    torch.nn.functional.interpolate(
                        ekg.transpose(1, 2),  # switch time and lead dimensions to comply with F.interpolate
                        size=resulting_size, align_corners=True, mode='linear'
                    ).transpose(1, 2)  # and switch them back
                )
            ret = torch.stack(interpolated, dim=1)

        else:
            time_dimension = 1
            resulting_size = int(x.shape[time_dimension] * new_frequency / self.base_frequency)

            ret = torch.nn.functional.interpolate(
                x.transpose(1, 2),  # switch time and lead dimensions to comply with F.interpolate
                size=resulting_size, align_corners=True, mode='linear'
            ).transpose(1, 2)  # and switch them back

        # padding needed
        if resulting_size < x.shape[time_dimension]:
            left_pad = (x.shape[time_dimension] - resulting_size) // 2
            right_pad = (x.shape[time_dimension] - resulting_size) // 2 + \
                        (x.shape[time_dimension] - resulting_size) % 2
            ret = torch.nn.functional.pad(ret, (0, 0, left_pad, right_pad))

        # trimming needed
        else:
            left_trim = (resulting_size - x.shape[time_dimension]) // 2
            ret = ret[..., left_trim:left_trim + x.shape[time_dimension], :]

        return ret

    def build(self, tmp):
        local_dir = tmp.path(self.save_dir)
        os.mkdir(local_dir)

        thresholds = 1999
        frequencies_to_sample = torch.linspace(self.start, self.end, self.count)
        metrics = {f: BinaryAUROC(thresholds=thresholds) for f in frequencies_to_sample}

        train_value = self.model.training
        self.model.train(False)

        dataloader = self.datamodule.val_dataloader()

        for test_batch in tqdm(dataloader, desc='Building curves', position=0):
            x, y = test_batch
            x = x.to(self.model.device)

            for frequency, metric in metrics.items():
                interpolated = self.interpolate(x, frequency)
                y_pred = self.model(interpolated).cpu().detach()
                y = y.to(int)
                metric(y_pred, y)

        plt.figure(figsize=(12, 5))
        plt.xticks(torch.linspace(self.start, self.end, min(self.count, 21)), rotation=45)
        plt.yticks(torch.linspace(0, 1, 21))
        plt.grid(alpha=0.3)
        plt.plot(metrics.keys(), [m.compute() for m in metrics.values()], label='AUROC')
        plt.plot([self.base_frequency, self.base_frequency], [0, 1], 'k--', label='Original frequency.')
        plt.xlabel('Simulated frequency')
        plt.ylabel(f'{self.mode} AUROC')
        plt.title("Influence of frequency interpolation on result")
        plt.legend()
        plt.savefig(os.path.join(local_dir, 'interfreq.png'))
        plt.clf()

        self.model.train(train_value)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError("Wrong boolean type.")

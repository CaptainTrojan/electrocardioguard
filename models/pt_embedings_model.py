# Created by David at 01.03.2023
# Project name main.py
# Created by David at 25.02.2023
# Project name main.py
import torch
import torch.nn as nn

from models.pt_preprocess_module import Preprocessing


class ResidualUnit(nn.Module):
    def __init__(self, n_samples_out, n_filters_out, n_samples_in, n_filters_in, kernel_initializer='he_normal',
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='gelu'):
        super(ResidualUnit, self).__init__()
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        self._gen_layer(n_samples_in, n_filters_in)

    def _skip_connection(self, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        layers = []

        if downsample > 1:
            layers.append(nn.MaxPool1d(downsample, padding=int((downsample - 1) / 2)))
        elif downsample == 1:
            pass
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            layers.append(
                nn.Conv1d(in_channels=n_filters_in, out_channels=self.n_filters_out, kernel_size=1, padding='same'))
        return nn.Sequential(*layers)

    def _gen_layer(self, n_samples_in, n_filters_in):
        downsample = n_samples_in // self.n_samples_out

        self.skip_layer = self._skip_connection(downsample, n_filters_in)
        self.layer_1 = nn.Sequential(nn.Conv1d(n_filters_in, self.n_filters_out, self.kernel_size,
                                               padding='same'),
                                     # TODO not too sure
                                     nn.BatchNorm1d(self.n_filters_out),
                                     nn.GELU(),
                                     nn.Dropout1d(p=self.dropout_rate)
                                     )
        self.layer_2 = nn.Sequential(
            nn.Conv1d(self.n_filters_out, self.n_filters_out, self.kernel_size, stride=downsample),
        )
        self.layer_3 = nn.Sequential(
            nn.BatchNorm1d(self.n_filters_out),
            nn.GELU(),
            nn.Dropout1d(p=self.dropout_rate)
        )

    def forward(self, x):
        z = self.layer_1(x[0])
        z = self.layer_2(z)
        y = self.skip_layer(x[1])
        y = y + z
        x = self.layer_3(y)

        return [x, y]


class ResNet(nn.Module):
    def __init__(self, normalize=False, propagate_normalization=False, remove_baseline=False, remove_hf_noise=False,
                 embedding_size=256):
        super().__init__()
        self.preprocessing = Preprocessing(normalize, remove_baseline, remove_hf_noise, result_only=True)

        if propagate_normalization and not normalize:
            raise ValueError("Propagation is only meaningful when normalizing.")
        self.propagate_normalization = propagate_normalization

        kernel_size = 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 12, kernel_size=kernel_size, stride=1, padding="same"),
            nn.BatchNorm1d(12),
            nn.GELU())

        self.layer1 = ResidualUnit(1024, 128, n_filters_in=12, n_samples_in=4096, kernel_size=kernel_size)
        self.layer2 = ResidualUnit(256, 196, n_filters_in=128, n_samples_in=1024, kernel_size=kernel_size)
        self.layer3 = ResidualUnit(64, 256, n_filters_in=196, n_samples_in=256, kernel_size=kernel_size)
        self.layer4 = ResidualUnit(16, 320, n_filters_in=256, n_samples_in=64, kernel_size=kernel_size)
        self.flattening = nn.Flatten(start_dim=1, end_dim=2)
        self.dense = nn.Sequential(torch.nn.Linear(5120 + (12 * 2 if propagate_normalization else 0), 512),
                                   nn.GELU(),
                                   torch.nn.Linear(512, embedding_size),
                                   nn.GELU(),
                                   torch.nn.Linear(embedding_size, embedding_size))

    def forward(self, x):
        x = self.preprocessing(x)

        x = self.conv1(x)
        x, y = self.layer1([x, x])
        x, y = self.layer2([x, y])
        x, y = self.layer3([x, y])
        x, _ = self.layer4([x, y])
        x = self.flattening(x)

        if self.propagate_normalization:
            raise NotImplementedError("Unsupported.")
            # x = torch.concat([x, mean.squeeze(), std.squeeze()], dim=1)

        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = ResNet(normalize=True, propagate_normalization=True)
    X = model.forward(torch.normal(4, 15, size=(8, 12, 4096)))
    print(X.shape)
    print("done")

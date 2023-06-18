import matplotlib.pyplot as plt
from torch import nn
import torch
import ptwt
from copy import deepcopy


class Preprocessing(nn.Module):
    def __init__(self, normalize, remove_baseline, remove_hf_noise, result_only=True,
                 should_pre_transpose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = []
        self.result_only = result_only
        self.should_pre_transpose = should_pre_transpose

        if remove_baseline:
            self.transforms.append(self.f_remove_baseline)

        if remove_hf_noise:
            self.transforms.append(self.f_remove_hf_noise)

        if normalize:
            self.transforms.append(self.f_normalize)

    @staticmethod
    def threshold_fn(x, threshold):
        x[x < threshold] = 0
        return x

        # N = torch.as_tensor(x.clone().detach().shape[-1:], dtype=torch.float32).view(-1)
        #
        # # Compute the soft threshold
        # lambda_value = threshold * torch.sqrt(torch.tensor(2.0 * torch.log(N)))
        # soft_threshold = lambda_value / N
        #
        # # Apply the soft threshold to the input tensor
        # soft_thresholded = torch.sign(x) * torch.max(torch.abs(x) - soft_threshold, torch.tensor(0.0))
        #
        # return soft_thresholded

    @classmethod
    def f_remove_hf_noise(cls, signal):
        level = 4
        wavelet = 'db4'
        initial_signal = signal.reshape(-1, signal.shape[-1])

        coeffs = ptwt.wavedec(initial_signal, wavelet, level=level)

        # Estimate noise standard deviation using MAD-based method
        sigma = torch.median(torch.abs(coeffs[-level])) / 0.6745

        # Apply soft thresholding to coefficients
        coeffs[1:] = [cls.threshold_fn(c, sigma) for c in coeffs[1:]]

        # Reconstruct denoised signal using inverse wavelet transform
        denoised_signal = ptwt.waverec(coeffs, wavelet)

        return denoised_signal.reshape(*signal.shape)

    @classmethod
    def calculate_baseline(cls, signal):
        initial_signal = signal.detach().clone().reshape(-1, signal.shape[-1])

        ssd_shape = (initial_signal.shape[0], )

        generations = [{
            'signal': initial_signal,
            'mask': torch.ones(initial_signal.shape[0], dtype=torch.bool, device=signal.device)
        }]

        current_iter = 0

        while True:
            sig = generations[-1]['signal']
            lp, hp = ptwt.wavedec(sig, 'db4', level=1)
            new_ssd = torch.zeros(ssd_shape, device=signal.device)
            new_ssd[generations[-1]['mask']] = torch.sum(hp ** 2, dim=-1)
            generations[-1]['ssd'] = new_ssd

            if len(generations) >= 3:
                newly_stopped = torch.logical_and(
                    torch.gt(
                        generations[-3]['ssd'][generations[-1]['mask']],
                        generations[-2]['ssd'][generations[-1]['mask']],
                    ),
                    torch.lt(
                        generations[-2]['ssd'][generations[-1]['mask']],
                        generations[-1]['ssd'][generations[-1]['mask']],
                    ),
                )

                if torch.all(newly_stopped) or lp.shape[-1] < 8 or current_iter > 7:
                    break

                new_sig = lp[~newly_stopped]
                new_mask = torch.clone(generations[-1]['mask'])
                new_mask[generations[-1]['mask']] = ~newly_stopped

                generations.append({
                    'signal': new_sig,
                    'mask': new_mask
                })
            else:
                generations.append({
                    'signal': lp,
                    'mask': generations[-1]['mask']
                })

            current_iter += 1

        # for i in range(len(generations)):
        #     g = generations[i]
        #     plt.scatter([i for v in range(len(g['ssd']))], [v for v in g['ssd']], c=[v for v in range(len(g['ssd']))])
        #
        # plt.show()
        # exit()

        for i in range(len(generations) - 2, -1, -1):
            lp = generations[i + 1]['signal']
            sig = generations[i]['signal']

            recovered = ptwt.waverec([lp, torch.zeros_like(lp)], 'db4')
            if recovered.shape[-1] == sig.shape[-1] + 1:
                recovered = recovered[..., :-1]

            mask_diff = generations[i + 1]['mask'][generations[i]['mask']]  # massive tricks omg (or maskive?)
            sig[mask_diff] = recovered
        baseline = generations[0]['signal'].reshape(*signal.shape)
        return baseline

    @classmethod
    def f_remove_baseline(cls, x):
        bl = cls.calculate_baseline(x)

        return x - bl

    @classmethod
    def f_normalize(cls, x):
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        x = torch.nan_to_num((x - mean) / std)
        return x, std, mean

    def forward(self, x):
        aux = []

        if self.should_pre_transpose:
            x = x.transpose(-1, -2)

        # plt.plot(x[0, 7, :2000], linewidth=0.5, color='red')
        # plt.savefig("pp_1.pdf")
        # plt.clf()
        # i = 2
        for t in self.transforms:
            output = t(x)
            if isinstance(output, tuple):
                x, *other = output
                aux += other
            else:
                x = output

            # plt.plot(x[0, 7, :2000], linewidth=0.5, color='red')
            # plt.savefig(f"pp_{i}.pdf")
            # plt.clf()
            # i += 1

        if self.should_pre_transpose:
            x = x.transpose(-1, -2)

        if self.result_only:
            return x
        else:
            return x, *aux

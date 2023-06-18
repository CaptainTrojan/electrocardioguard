from pytorch_lightning import LightningModule
from torch import nn
import torch


class DiscriminatorHead(LightningModule):
    def __init__(self, embedding_size, hidden_size,
                 l2_approach,
                 l1_approach,
                 cos_approach,
                 ):
        super().__init__()

        input_size = 0
        self.l2_approach = l2_approach
        self.l1_approach = l1_approach
        self.cos_approach = cos_approach

        for a in (l2_approach, l1_approach, cos_approach):
            if a == 'none':
                pass
            elif a == 'merge':
                input_size += 1
            elif a == 'full':
                input_size += embedding_size
            else:
                raise ValueError(f"Unknown distance approach {a}.")

        if hidden_size == 0:
            self.sequence = nn.Sequential(
                nn.Dropout(),
                nn.Linear(input_size, 1)
            )
        else:
            self.sequence = nn.Sequential(
                nn.Dropout(),
                nn.Linear(input_size, hidden_size),
                nn.Dropout(),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

        self.logit_to_probability = nn.Sigmoid()

    def forward(self, embedded_a, embedded_b):
        euclidean_distance_members = torch.square(embedded_b - embedded_a)
        manhattan_distance_members = torch.abs(embedded_b - embedded_a)
        cosine_distance_members = \
            embedded_b * embedded_a / (torch.sqrt(torch.sum(torch.square(embedded_a))) * torch.sqrt(torch.sum(torch.square(embedded_a))))

        to_merge = []
        for members, approach in (
                (euclidean_distance_members, self.l2_approach),
                (manhattan_distance_members, self.l1_approach),
                (cosine_distance_members, self.cos_approach),
        ):
            if approach == 'none':
                pass
            elif approach == 'merge':
                to_merge.append(torch.sum(members, dim=1).reshape(-1, 1))
            else:
                to_merge.append(members)

        if len(to_merge) == 0:
            to_merge = [torch.empty(embedded_b.shape[0], 0, device=embedded_a.device)]

        flat = torch.concat(to_merge, dim=1)
        return self.logit_to_probability(self.sequence(flat)).squeeze()


if __name__ == '__main__':
    for var1 in ('none', 'merge', 'full'):
        for var2 in ('none', 'merge', 'full'):
            for var3 in ('none', 'merge', 'full'):
                for l in (0, 32, 64):
                    d = DiscriminatorHead(10, l, var1, var2, var3)
                    a = torch.rand(8, 10)
                    b = torch.rand(8, 10)
                    print(d(a, b).size())

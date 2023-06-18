import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, anchor, positive, negative):
        ap_dist = F.pairwise_distance(anchor, positive, p=2)
        an_dist = F.pairwise_distance(anchor, negative, p=2)
        sp = F.relu(ap_dist - self.m + 1)
        sn = F.relu(self.m - an_dist + 1)
        loss = torch.mean(sp + self.gamma*sn)
        return loss


class MetricLearning(pl.LightningModule):
    def __init__(self, embedding_model, variant='triplet', should_transpose=True, margin=1.0, gamma=0.1):
        super(MetricLearning, self).__init__()
        self.embedding_model = embedding_model
        self.should_transpose = should_transpose
        if variant == 'triplet':
            self.loss = nn.TripletMarginLoss(margin=margin, p=2)
        elif variant == 'circle':
            self.loss = CircleLoss(m=margin, gamma=gamma)
        else:
            raise ValueError(f"Unknown variant '{variant}'.")

    def forward(self, x):
        if self.should_transpose:
            x = x.transpose(2, 3)
        embedded_a = self.embedding_model(x[:, 0])
        embedded_p = self.embedding_model(x[:, 1])
        embedded_n = self.embedding_model(x[:, 2])

        return embedded_a, embedded_p, embedded_n

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        embedded_a, embedded_p, embedded_n = self.forward(x)
        loss = self.loss(embedded_a, embedded_p, embedded_n)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        embedded_a, embedded_p, embedded_n = self.forward(x)
        loss = self.loss(embedded_a, embedded_p, embedded_n)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

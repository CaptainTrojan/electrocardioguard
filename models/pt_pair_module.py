import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, AveragePrecision


class PairLoss(pl.LightningModule):
    def __init__(self, embedding_model: nn.Module, discriminator_model: nn.Module, freeze_embeddings=False,
                 should_transpose=True):
        super().__init__()
        self.embedding = embedding_model
        self.discriminator = discriminator_model
        self.should_transpose = should_transpose

        if freeze_embeddings:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.loss = nn.BCELoss()

        self.train_auc = BinaryAUROC()
        self.train_ap = AveragePrecision('binary', thresholds=1999)

        self.val_auc = BinaryAUROC()
        self.val_ap = AveragePrecision('binary', thresholds=1999)

    def forward(self, x):
        if self.should_transpose:
            x = x.transpose(2, 3)
        embedded_a = self.embedding(x[:, 0])
        embedded_b = self.embedding(x[:, 1])

        return self.discriminator(embedded_a, embedded_b)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        probability = self.forward(x)
        loss = self.loss(probability, y)

        self.train_auc(probability, y)
        self.train_ap(probability, y.to(int))

        self.log('train_loss', loss)
        self.log('train_auc', self.train_auc)
        self.log('train_ap', self.train_ap)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        probability = self.forward(x)
        loss = self.loss(probability, y)

        self.val_auc(probability, y)
        self.val_ap(probability, y.to(int))

        self.log('val_loss', loss)
        self.log('val_auc', self.val_auc)
        self.log('val_ap', self.val_ap)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

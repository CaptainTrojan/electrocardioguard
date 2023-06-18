import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            s: float,
            m: float,
            easy_margin: bool,
            ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_tensor: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input_tensor), F.normalize(self.weight))
        # Enable 16 bit precision  TODO Michal: how does cosine.to(torch.float32) enable 16bit precision?
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # one_hot = torch.zeros(cosine.size(), device=device)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = label
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * label + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArchNetTraining(pl.LightningModule):
    def __init__(self, embedding_model, in_features: int, out_features: int, s: float, margin: float, easy_margin: bool,
                 ls_eps: float):
        super(ArchNetTraining, self).__init__()
        self.resnet = embedding_model
        self.arc = ArcMarginProduct(
            in_features=in_features,
            out_features=out_features,
            s=s,
            m=margin,
            easy_margin=easy_margin,
            ls_eps=ls_eps,
        )
        self.loss_fn = F.cross_entropy
        self.cos_loss = torch.nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, x):
        x = x.transpose(1, 2)
        fow_pass = self.resnet(x)
        return fow_pass

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self.forward(x)
        outputs = self.arc(x, y)
        loss = self.loss_fn(outputs, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        ekg_s = val_batch[0]
        to_embed_0 = ekg_s[:, 0, :, :]
        to_embed_1 = ekg_s[:, 1, :, :]

        fow_pass_0 = self.forward(to_embed_0)
        fow_pass_1 = self.forward(to_embed_1)
        y = val_batch[1]
        y[y == 0] = -1
        val_loss = self.cos_loss(fow_pass_0, fow_pass_1, y)
        self.log('val_loss', val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

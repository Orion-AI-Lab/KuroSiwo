import torch
from torch import nn
import torch.nn.functional as F

from .dice import DiceLoss

class BCEandDiceLoss(nn.Module):
    def __init__(self, weights=None, ignore_index=None, use_softmax=False):
        super(BCEandDiceLoss, self).__init__()

        if ignore_index is not None:
            self.bce = nn.CrossEntropyLoss(weight=torch.Tensor(weights), ignore_index=ignore_index)
            self.dice = DiceLoss(ignore_index=ignore_index, use_softmax=use_softmax)
        else:
            self.bce = nn.CrossEntropyLoss(weight=torch.Tensor(weights))
            self.dice = DiceLoss(use_softmax=use_softmax)

    def forward(self, preds, lbl):
        dice_loss = self.dice(preds, lbl)
        bce_loss = self.bce(preds, lbl)

        loss = dice_loss + bce_loss

        return loss
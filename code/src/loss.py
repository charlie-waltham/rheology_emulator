import torch
import torch.nn as nn
import torch.nn.functional as F


class ShrinkageLoss(nn.Module):
    """
    Implements Shrinkage Loss as seen in
    https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.pdf
    """

    def __init__(self, a=10, c=0.2, reduction="mean"):
        super(ShrinkageLoss, self).__init__()
        self.a = a
        self.c = c
        self.reduction = reduction

    def forward(self, x, y):
        mae = F.l1_loss(x, y, reduction="none")
        loss = (mae**2) / (1 + torch.exp(self.a * (self.c - mae)))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

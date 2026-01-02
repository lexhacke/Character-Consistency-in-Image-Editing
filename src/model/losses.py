import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossFromLogits(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLossFromLogits, self).__init__()
        self.gamma = gamma
        self.eps = 1e-4
        self.alpha = alpha

    def forward(self, y, yhat):
        """
        Expects y, yhat of shape B, 1, H, W
        """
        B, _, H, W = y.shape
        P = F.sigmoid(yhat)
        focal_weight = y * (1 - self.alpha) * (1 - P)**self.gamma + (1 - y) * self.alpha * P**self.gamma
        bce = F.binary_cross_entropy_with_logits(yhat, y, reduction='none')
        return (bce * focal_weight).mean()

class DiceLossFromLogits(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLossFromLogits, self).__init__()
        self.smooth = smooth

    def forward(self, y, yhat):
        """
        Expects y, yhat of shape B, 1, H, W
        """
        B, _, H, W = y.shape
        P = F.sigmoid(yhat)
        yP = (y * P).sum(dim=(1,2,3))
        y, P = y.sum(dim=(1,2,3)), P.sum(dim=(1,2,3))
        loss = 1 - (2 * yP + self.smooth) / (y + P + self.smooth)
        return loss.mean()
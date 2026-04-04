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
        focal_weight = y * self.alpha * (1 - P)**self.gamma + (1 - y) * (1 - self.alpha) * P**self.gamma
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

if __name__ == "__main__":
    B, C, H, W = 4, 1, 64, 64

    y = torch.zeros(B, C, H, W)
    num_ones = int(0.1 * B * C * H * W)
    indices = torch.randperm(B * C * H * W)[:num_ones]
    y.view(-1)[indices] = 1
    yhat = torch.randn(B, C, H, W)
    loss = FocalLossFromLogits(gamma=2)(y, yhat) + DiceLossFromLogits()(y, yhat)
    print(f"\nCalculated Loss: {loss.item()}")
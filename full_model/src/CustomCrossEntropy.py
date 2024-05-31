import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        y_pred = F.log_softmax(y_pred, dim=1)
        loss = -torch.sum(y_true * y_pred, dim=1)
        return loss.mean()

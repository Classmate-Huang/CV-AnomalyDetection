import torch
import torch.nn as nn

class EntropyLoss(nn.Module):
    ''' Entropy Loss '''
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        ''' x [N,M] : attention weight '''
        loss = x * torch.log(x + self.eps)
        loss = -1.0 * torch.sum(loss, dim=1)
        return loss.mean()


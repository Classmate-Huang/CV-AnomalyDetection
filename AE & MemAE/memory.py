import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


def hard_shrink_relu(x, lambd=0, epsilon=1e-12):
    ''' Hard Shrinking '''
    return (F.relu(x-lambd) * x) / (torch.abs(x-lambd) + epsilon)


class MemoryModule(nn.Module):
    ''' Memory Module '''
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        # attention
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))   # [M, C]
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        ''' init memory elements : Very Important !! '''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        ''' x [B,C,H,W] : latent code Z'''
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).flatten(end_dim=2) # Fea : [NxC]  N=BxHxW
        # calculate attention weight
        att_weight = F.linear(x, self.weight)   # Fea*Mem^T : [NxC] x [CxM] = [N, M]
        att_weight = F.softmax(att_weight, dim=1)   # [N, M]

        if self.shrink_thres > 0:
            # hard shrink
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # re-normalize
            att_weight = F.normalize(att_weight, p=1, dim=1)    # [N, M]
        
        # generate code z'
        mem_T = self.weight.permute(1, 0)
        output = F.linear(att_weight, mem_T) # Fea*Mem^T^T : [N, M] x [M, C] = [N, C]
        output = output.view(B,H,W,C).permute(0,3,1,2)  # [N,C,H,W]

        return att_weight, output


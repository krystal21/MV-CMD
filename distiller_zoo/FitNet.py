from __future__ import print_function

import torch.nn as nn


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""

    def __init__(self, dim_in=2048, dim_out=128, predictor=False, raw=False):
        super(HintLoss, self).__init__()
        self.embed_s = Reg(dim_in, dim_out)
        self.embed_t = Reg(dim_in, dim_out)
        self.crit = nn.MSELoss()
        self.raw = raw

    def forward(self, f_s, f_t):
        if not self.raw:
            f_s = self.embed_s(f_s)
        loss = self.crit(f_s, f_t)
        return loss


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=2048, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class Reg(nn.Module):
    """Linear regressor"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Reg, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.linear(x)
        return x
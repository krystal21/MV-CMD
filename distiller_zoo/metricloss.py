from __future__ import print_function

import torch.nn as nn


class MetricLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""

    def __init__(self):
        super(MetricLoss, self).__init__()
        self.embed_s = Embed(2048, 128)
        self.embed_t = Embed(2048, 128)
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        loss = self.crit(f_s, f_t)
        return loss


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

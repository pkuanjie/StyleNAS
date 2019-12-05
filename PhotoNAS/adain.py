import torch
import torch.nn
import numpy as np

def transform(cF, sF, alpha):
    eps = 1e-6
    n, c, h, w = cF.size(0), cF.size(1), cF.size(2), cF.size(3)
    cF = cF.view(n, c, h * w)
    sF = sF.view(n, c, h * w)
    cF_mean = cF.mean(-1).unsqueeze(-1)
    sF_mean = sF.mean(-1).unsqueeze(-1)
    cF_std = cF.std(-1).unsqueeze(-1) + eps
    sF_std = sF.std(-1).unsqueeze(-1)
    csF = (cF - cF_mean) / cF_std * sF_std + sF_mean
    csF = csF.view(n, c, h, w)
    return csF


if __name__ == '__main__':
    cF = torch.rand((2, 4, 10, 10))
    sF = torch.rand((2, 4, 10, 10))
    alpha = 1.0
    csF = transform(cF, sF, alpha)
    print(csF.size())

import torch.nn as nn
import torch
from einops import rearrange
import json
import numpy as np
import math as mt
from scipy.stats import entropy

def cal_nMSE(pred, target):  # normalized MSE
    pred = rearrange(pred, "b ... -> b (...)")
    target = rearrange(target, "b ... -> b (...)")
    err_norm = torch.norm(pred - target, p=2, dim=1)**2
    tar_norm = torch.norm(target, p=2, dim=1)**2
    err_nrmse = (err_norm / tar_norm)
    return err_nrmse.detach().cpu().numpy()

def cal_RMSE(pred, target):  # RMSE
    pred = rearrange(pred, "b h w c t -> b c (h w) t")
    target = rearrange(target, "b h w c t -> b c (h w) t")
    err_mean = torch.sqrt(torch.mean((pred - target) ** 2, dim=2))
    err_RMSE = torch.mean(err_mean, dim=0)
    return err_RMSE.detach().cpu().numpy()

def cal_fRMSE(pred, target):  # RMSE in Fourier space
    pred = rearrange(pred, "b h w c t -> b c h w t")
    target = rearrange(target, "b h w c t -> b c h w t")
    idxs = target.size()
    nb, nc, nx, ny, nt = idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]

    pred_F = torch.fft.fftn(pred, dim=[2, 3])  # (B, C, Nx, Ny, T)
    target_F = torch.fft.fftn(target, dim=[2, 3])  # (B, C, Nx, Ny, T)

    _err_F = torch.abs(pred_F - target_F) ** 2
    err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(pred.device)
    for i in range(nx // 2):
        for j in range(ny // 2):
            it = mt.floor(mt.sqrt(i ** 2 + j ** 2))
            if it > min(nx // 2, ny // 2) - 1:
                continue
            err_F[:, :, it] += _err_F[:, :, i, j]
    _err_F = torch.sqrt(err_F) / (nx * ny)
    _err_F = torch.mean(torch.mean(_err_F, dim=2), dim=[1, -1])
    return _err_F.detach().cpu().numpy()

def cal_spectral_entropy(target):
    target_F = np.fft.rfftn(target, axes=[0, 1, 2])
    target_psd = np.abs(target_F) ** 2
    target_psd /= np.sum(target_psd)
    se = -np.sum(target_psd * np.log2(target_psd + 1e-10))
    return se

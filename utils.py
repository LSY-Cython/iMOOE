import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import json
import os
import operator
from functools import reduce
import math
import xarray as xr


def fix_seed(seed):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_train_loss(error, file_name):
    plt.plot(error)
    plt.savefig(f"output/{file_name}_train_loss.png")
    plt.clf()
    train_error = {}
    for i in range(len(error)):
        train_error[f"epoch{i}"] = error[i]
    with open(f"output/{file_name}_train_loss.json", "w") as f:
        json.dump(train_error, f, indent=4)


def plot_state_data(true_data, pred_data, t, channel, t_fraction, plt_cfg, ablate_idx, fig_name, is_naive=False):
    if not isinstance(true_data, np.ndarray):
        true_data = true_data.detach().cpu().numpy()
    if not isinstance(pred_data, np.ndarray):
        pred_data = pred_data.detach().cpu().numpy()
    if not isinstance(t, np.ndarray):
        t = t[0].detach().cpu().numpy()
    t_idx = list(range(0, len(t), t_fraction))

    # Plot data at t=t_idx, use imshow for 2D data
    plt.figure(figsize=(2*len(t_idx), 4))
    for i in range(len(t_idx)):
        plt.subplot(2, len(t_idx), i+1)
        plt.title(f"$t={t[t_idx[i]]}s$")
        plt.imshow(
            true_data[0, :, :, channel, t_idx[i]].transpose(),
            aspect="auto",
            origin="lower",
            extent=[
                plt_cfg["x_left"],
                plt_cfg["x_right"],
                plt_cfg["y_bottom"],
                plt_cfg["y_top"],
            ],
        )
        plt.xlabel("$x-true$")
        plt.ylabel("$y-true$")

        plt.subplot(2, len(t_idx), i+1+len(t_idx))
        plt.imshow(
            pred_data[0, :, :, channel, t_idx[i]].transpose(),
            aspect="auto",
            origin="lower",
            extent=[
                plt_cfg["x_left"],
                plt_cfg["x_right"],
                plt_cfg["y_bottom"],
                plt_cfg["y_top"],
            ],
        )
        plt.xlabel("$x-pred$")
        plt.ylabel("$y-pred$")

    fig_dir = plt_cfg["fig_dir"]
    if is_naive:
        fig_dir = fig_dir.replace("MoE", ablate_idx)
    else:
        fig_dir += f"_{ablate_idx}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{fig_name}.png")
    plt.close()
    plt.clf()


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


def plot_freq_distr(state_img):  # RMSE in Fourier space
    nx, ny = state_img.shape
    state_F = np.abs(np.fft.fftn(state_img, axes=[0, 1])) ** 2 / (nx * ny)  # (Nx, Ny)

    amp = np.zeros((min(nx // 2, ny // 2), ))
    for i in range(nx // 2):
        for j in range(ny // 2):
            it = math.floor(math.sqrt(i ** 2 + j ** 2))
            if it > min(nx // 2, ny // 2) - 1:
                continue
            amp[it] += state_F[i, j]
    return amp
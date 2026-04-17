import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torchdiffeq import odeint
import pysindy as ps
from einops import rearrange
from models.framework import DyMoE


class Forecaster(nn.Module):
    def __init__(self, config, exp_net, int_method, int_options, int_step_scale, device):
        super(Forecaster, self).__init__()
        self.int_method = int_method
        self.int_step_scale = int_step_scale
        self.device = device
        self.int_ = odeint
        self.exp_cfg = config["expert"]
        self.env_dim = config["data"]["env_dim"]
        self.init_step = config["data"]["init_step"]
        self.t_train = config["data"]["Nt"]
        self.dt = config["data"]["dt"]
        self.dx = config["data"]["dx"]
        self.dy = config["data"]["dy"]
        self.diff_order = config["differentiation"]["diff_order"]  # the highest order of spatial derivatives
        self.diff_method = config["differentiation"]["method"]
        self.num_var = config["data"]["num_var"]

        self.hist_channels = self.init_step * self.num_var
        if self.diff_order == 1:
            self.diff_channels = self.num_var * 2
        elif self.diff_order == 2:
            self.diff_channels = self.num_var * 3
        elif self.diff_order == [1, 2]:
            self.diff_channels = self.num_var * 5
        else:
            raise NotImplementedError

        self.exp_net = exp_net
        self.moe_model = DyMoE(self.exp_cfg, exp_net, self.env_dim, self.hist_channels, self.diff_channels, self.device)

        if int_options == "{}":
            self.int_options = {}
        else:
            int_step_size = self.dt * self.int_step_scale  # smaller step size leads to better integration accuracy
            self.int_options = dict(step_size=int_step_size)

    def differentiation(self, x, delta, order, axis):
        if not isinstance(x, np.ndarray):
            x = x.detach().cpu().numpy()
        # two-point central differencing
        if self.diff_method == "finite_difference":
            diff_sol = ps.FiniteDifference(d=order, axis=axis, drop_endpoints=False)._differentiate(x, delta)
        # Fourier derivative
        elif self.diff_method == "spectral":
            diff_sol = ps.SpectralDerivative(d=order, axis=axis)._differentiate(x, delta)
        else:
            raise NotImplementedError

        # Spatial scaling
        if order == 1:
            diff_scale = delta
        elif order == 2:
            diff_scale = delta ** 2
        else:
            raise NotImplementedError
        diff_sol_norm = torch.from_numpy(diff_sol) * diff_scale

        return diff_sol_norm.to(self.device)

    def cal_spatial_derivative(self, x):  # (B, Nx, Ny, C)
        # first-order
        ux = self.differentiation(x, delta=self.dx, order=1, axis=1)
        uy = self.differentiation(x, delta=self.dy, order=1, axis=2)
        # second-order
        uxx = self.differentiation(x, delta=self.dx, order=2, axis=1)
        uyy = self.differentiation(x, delta=self.dy, order=2, axis=2)
        uxy = self.differentiation(ux, delta=self.dy, order=1, axis=2)

        if self.diff_order == 1:
            return torch.cat([ux, uy], dim=3)  # (B, Nx, Ny, C*2)
        elif self.diff_order == 2:
            return torch.cat([uxx, uyy, uxy], dim=3)  # (B, Nx, Ny, C*3)
        elif self.diff_order == [1, 2]:
            return torch.cat([ux, uy, uxx, uyy, uxy], dim=3)  # (B, Nx, Ny, C*5)
        else:
            raise NotImplementedError

    def forward(self, x, c, t, mode, mask=None):  # x: (B, Nx, Ny, C, T)
        if len(t.shape) > 1:
            t = t[0]  # t must be one-dimensional in odeint

        def derivative_func(t, y):  # (B, Nx, Ny, C)
            y_diff = self.cal_spatial_derivative(y)  # (B, Nx, Ny, C_diff)
            y_hist = rearrange(input_t, 'b h w c t -> b h w (c t)')  # (B, Nx, Ny, C*init_step)
            output = self.moe_model(y_hist, y_diff, c, t)
            return output["preds"]  # (B, Nx, Ny, C)

        preds_out, targets_out = [], []

        input_t = x[..., :self.init_step]  # (B, Nx, Ny, C, init_step)
        for t_eval in range(self.init_step, self.t_train):
            x0 = input_t[..., -1]  # (B, Nx, Ny, C)
            res_t = odeint(derivative_func, y0=x0, t=t[t_eval-1:t_eval+1],
                           method=self.int_method, options=self.int_options)  # (2, B, Nx, Ny, C)
            pred_t = res_t[-1].unsqueeze(-1)  # (B, Nx, Ny, C, 1)

            ######
            if mask is not None:
                pred_t = pred_t * mask
            ######

            target_t = x[..., t_eval].unsqueeze(-1)  # (B, Nx, Ny, C, 1)

            input_t = torch.cat((input_t[..., 1:], pred_t), dim=-1)  # (B, Nx, Ny, C, init_step)
            preds_out.append(pred_t)
            targets_out.append(target_t)

        if mode == "train":
            preds_out = rearrange(torch.cat(preds_out, dim=-1), 'b h w c t -> (b t) h w c')  # (B*(Nt-init_step), Nx, Ny, C)
            targets_out = rearrange(torch.cat(targets_out, dim=-1), 'b h w c t -> (b t) h w c')  # (B*(Nt-init_step), Nx, Ny, C)
        else:
            preds_out = torch.cat(preds_out, dim=-1)  # (B, Nx, Ny, C, Nt-init_step)
            targets_out = torch.cat(targets_out, dim=-1)  # (B, Nx, Ny, C, Nt-init_step)
        masks = self.moe_model.exp_masks  # (C_diff, num_expert)
        eval_steps = np.tile(t[self.init_step:].detach().cpu().numpy(), x.shape[0])  # (B*(Nt-init_step), )
        return {"preds": preds_out, "targets": targets_out, "masks": masks, "eval_steps": eval_steps}

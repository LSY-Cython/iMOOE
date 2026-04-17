import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from models.networks.fno import FNO2d, FNOLayer2d, MLP
from models.networks.oformer.oformer import OFormerUniform2d


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)  # restrict the straight-through gradient to [-1, 1]


"""
Wrapping nn.Parameter() into nn.Module(), then it can be registered by nn.ModuleList()
"""
class ParameterWrapper(nn.Module):
    def __init__(self, init_param):
        super(ParameterWrapper, self).__init__()
        self.param = nn.Parameter(init_param)

"""
Characterizing the invariant relations between exogenous parameters and individual sub-operator experts.
"""
class CorrNet(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super(CorrNet, self).__init__()
        self.corr_layer = nn.Conv2d(in_channels, hid_channels, 1)

    def forward(self, x):
        x = self.corr_layer(x)  # (B, hid_channels, Nx, Ny)
        return x

class CorrNetLin(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(CorrNetLin, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, hid_channels, 1)
        self.mlp2 = nn.Conv2d(hid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

"""
Fusing expert outputs and physical parameters to represent their (non-)linear relationships.
"""
class FusionNet(nn.Module):
    def __init__(self, hid_channels, out_channels, n_layers, modes1, modes2):
        super(FusionNet, self).__init__()
        self.fusion_layers = nn.Sequential()
        for _ in range(n_layers):
            self.fusion_layers.append(FNOLayer2d(
                modes1=modes1,
                modes2=modes2,
                width=hid_channels,
                spatial_size=64,
                weight_init=1,
                act="sin"
            ))

        self.head_layer = MLP(hid_channels, out_channels, hid_channels)

    def forward(self, x):  # (B, num_expert*corr_dim, Nx, Ny)
        x = self.fusion_layers(x)  # (B, fuse_dim, Nx, Ny)
        x = self.head_layer(x)  # (B, C, Nx, Ny)
        return x


class ExpertNet(nn.Module):
    def __init__(self, exp_config, exp_net, hist_channels, diff_channels, env_dim, corr_dim, device):
        super(ExpertNet, self).__init__()

        self.exp_net = exp_net
        self.hist_channels = hist_channels
        self.diff_channels = diff_channels
        self.env_dim = env_dim
        self.device = device
        self.in_channels = exp_config["in_channels"]
        self.is_select = exp_config["is_select"]
        self.select_type = exp_config["select_type"]
        self.init_type = exp_config["init_type"]
        self.is_op_lin = exp_config["is_op_lin"]
        self.corr_dim = corr_dim

        # initialize the differentiable element selection module
        if self.init_type == "ones":
            init_param = torch.ones((self.diff_channels, ))
        elif self.init_type == "rand":  # ~[0, 1] uniform distribution
            init_param = torch.rand((self.diff_channels, )) - 0.5
        elif self.init_type == "randn":  # ~N(0, 1)
            init_param = torch.randn((self.diff_channels, ))
        else:
            raise NotImplementedError
        self.W = ParameterWrapper(init_param).to(self.device)

        # Instantiate expert network
        if self.is_select:
            self.comb_channels = self.hist_channels + self.diff_channels
        else:
            self.comb_channels = self.hist_channels

        if self.exp_net == "FNO2d":
            exp_fno2d_config = exp_config["FNO2d"]
            exp_out_channels = exp_fno2d_config["exp_out_channels"]
            self.expert = FNO2d(in_channels=self.comb_channels, out_channels=exp_out_channels,
                                modes1=exp_fno2d_config["modes1"], modes2=exp_fno2d_config["modes2"],
                                width=exp_fno2d_config["width"], spatial_size=exp_fno2d_config["spatial_size"],
                                n_layers=exp_fno2d_config["n_layers"], act=exp_fno2d_config["act_type"],
                                padding=exp_fno2d_config["padding"], weight_init=exp_fno2d_config["weight_init"],
                                x_span=exp_config["x_span"], y_span=exp_config["y_span"])
        elif self.exp_net == "OFormerUniform2d":
            exp_ofu2d_config = exp_config["OFormerUniform2d"]
            exp_out_channels = exp_ofu2d_config["exp_out_channels"]
            self.expert = OFormerUniform2d(in_channels=self.comb_channels, out_channels=exp_out_channels,
                                           latent_channels=exp_ofu2d_config["latent_channels"],
                                           encoder_emb_dim=exp_ofu2d_config["encoder_emb_dim"],
                                           encoder_heads=exp_ofu2d_config["encoder_heads"],
                                           encoder_depth=exp_ofu2d_config["encoder_depth"],
                                           x_span=exp_config["x_span"], y_span=exp_config["y_span"])
        else:
            raise NotImplementedError

        # Instantiate correlation network
        corr_in_channels = exp_out_channels + env_dim
        if self.is_op_lin:
            self.corr_net = CorrNetLin(in_channels=corr_in_channels, hid_channels=self.corr_dim,
                                       out_channels=exp_out_channels)
        else:
            self.corr_net = CorrNet(in_channels=corr_in_channels, hid_channels=self.corr_dim)

    def forward(self, x_hist, x_diff, c, t):  # x: (B, Nx, Ny, C_hist/C_diff), c: (B, env_dim)
        # masking must be operated in invoking function
        if self.select_type == "hard":
            mask = STEFunction.apply(self.W.param)
        elif self.select_type == "soft":
            mask = F.softmax(self.W.param)
        else:
            raise NotImplementedError
        if len(mask.shape) != len(x_diff.shape):
            for _ in range(len(x_diff.shape)-1):  # (1, 1, 1, diff_channels)
                mask = mask.unsqueeze(0)

        if self.is_select:
            input_masked = torch.mul(x_diff, mask)  # (B, Nx, Ny, diff_channels)
            input_combed = torch.cat((input_masked, x_hist), dim=-1)  # (B, Nx, Ny, comb_channels)
            exp_pred = self.expert(input_combed.permute(0, 3, 1, 2))  # (B, exp_out_channels, Nx, Ny)
        else:
            exp_pred = self.expert(x_hist.permute(0, 3, 1, 2))  # (B, exp_out_channels, Nx, Ny)

        env_vec = torch.repeat_interleave(c, repeats=x_diff.shape[0]//c.shape[0], dim=0)  # (B, env_dim)
        env_vec = env_vec.unsqueeze(-1).unsqueeze(-1)  # (B, env_dim, 1, 1)
        env_vec = env_vec.repeat(1, 1, x_diff.shape[1], x_diff.shape[2])  # (B, env_dim, Nx, Ny)
        corr_input = torch.concatenate([exp_pred, env_vec], dim=1)  # (B, exp_out_channels+env_dim, Nx, Ny)

        exp_pred = self.corr_net(corr_input)  # (B, corr_dim, Nx, Ny)

        exp_out = {"pred": exp_pred, "mask": mask.reshape(self.diff_channels, )}
        return exp_out


class DyMoE(nn.Module):
    def __init__(self, exp_cfg, exp_net, env_dim, hist_channels, diff_channels, device):
        super(DyMoE, self).__init__()

        self.exp_net = exp_net
        self.env_dim = env_dim
        self.hist_channels = hist_channels
        self.diff_channels = diff_channels
        self.device = device
        self.out_channels = exp_cfg["out_channels"]

        self.is_op_lin = exp_cfg["is_op_lin"]
        self.fuse_cfg = exp_cfg["fusion"]
        self.corr_dim = self.fuse_cfg["corr_dim"]

        # instantiate expert network
        # using channel-wise experts to model multi-channel coupled physical states
        self.expert_set = nn.ModuleList()  # can't register nn.Parameters() to learnable network params
        self.num_expert = exp_cfg["num_expert"]
        for _ in range(self.num_expert):
            self.expert_set.append(ExpertNet(exp_cfg, self.exp_net, self.hist_channels, self.diff_channels,
                                             self.env_dim, self.corr_dim, self.device))

        # instantiate fusion network
        self.fuse_dim = self.num_expert * self.corr_dim
        self.fuse_net = FusionNet(hid_channels=self.fuse_dim, out_channels=self.out_channels,
                                  n_layers=self.fuse_cfg["n_layers"],
                                  modes1=self.fuse_cfg["modes1"], modes2=self.fuse_cfg["modes2"])

    def forward(self, x_hist, x_diff, c, t):
        exp_preds, exp_masks = [], []
        for expert in self.expert_set:
            exp_out = expert(x_hist, x_diff, c, t)
            exp_preds.append(exp_out["pred"])
            exp_masks.append(exp_out["mask"])

        self.exp_masks = torch.stack(exp_masks, dim=-1)  # (C_diff, num_experts)

        if self.is_op_lin:
            exp_preds = torch.stack(exp_preds, dim=-1)  # (B, exp_out_channels, Nx, Ny, num_experts)
            preds_out = torch.sum(exp_preds, dim=-1).permute(0, 2, 3, 1)  # (B, Nx, Ny, out_channels)
        else:
            exp_preds = torch.cat(exp_preds, dim=1)  # (B, num_expert*corr_dim, Nx, Ny)
            preds_out = self.fuse_net(exp_preds).permute(0, 2, 3, 1)  # (B, Nx, Ny, out_channels)

        return {"preds": preds_out, "masks": self.exp_masks}

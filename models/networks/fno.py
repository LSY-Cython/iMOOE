import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, spatial_size, weight_init=1):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        """
        weight_init: [1: "naive", 2: "vp", 3: "half-naive", 4: "zero"]: vp (variance preserving) or naive as per Li et al.
        """
        if weight_init == 1:  # naive
            self.scale = (1 / (in_channels * out_channels))
        elif weight_init == 2:  # vp
            self.scale = math.sqrt((1 / (in_channels)) * (spatial_size**2) / (4 * modes1 * modes2 + 4))
        elif weight_init == 3:  # half-naive
            self.scale = 1 / in_channels
        elif weight_init == 4:  # zero
            self.scale = 1e-8

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # first element-wise multiplication: ixy*ixy->ixy, then summation along the i-axis -> xy
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)  # (B, width, S, S//2+1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNOLayer2d(nn.Module):
    def __init__(self, modes1, modes2, width, spatial_size, weight_init, act):
        super(FNOLayer2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.spatial_size = spatial_size
        self.weight_init = weight_init

        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, self.spatial_size, self.weight_init)
        self.mlp = MLP(self.width, self.width, self.width)
        self.w = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.act = torch.sin if act == "sin" else F.gelu

    # refer to Li.el
    def forward(self, x):
        x1 = self.norm(self.conv(self.norm(x)))
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x = x1 + x2
        x = self.act(x)
        return x

    # # refer to LSM
    # def forward(self, x):
    #     x1 = self.conv(x)
    #     x2 = self.w(x)
    #     x = x1 + x2
    #     x = self.act(x)
    #     return x

class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width, spatial_size, n_layers,
                 act="sin", padding=0, weight_init=1, x_span=1, y_span=1):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.padding = padding  # pad the domain if input is non-periodic
        self.weight_init = weight_init
        self.act = act
        self.x_span = x_span
        self.y_span = y_span

        self.p = nn.Linear(2 + self.in_channels, self.width)  # input channel is: (a(x, y), x, y)
        self.q = MLP(self.width, self.out_channels, self.width*4)

        self.fno2d_layers = nn.Sequential()
        for _ in range(self.n_layers):
            self.fno2d_layers.append(FNOLayer2d(
                modes1=self.modes1,
                modes2=self.modes2,
                width=self.width,
                spatial_size=self.spatial_size,
                weight_init=self.weight_init,
                act=self.act
            ))

    def forward(self, x):  # (B, C_comb, Nx, Ny)
        x = x.permute(0, 2, 3, 1)  # (B, Nx, Ny, in_channels)
        grid = self.get_grid(x.shape, x.device)  # (B, Nx, Ny, 2)
        x = torch.cat((x, grid), dim=-1)  # (B, Nx, Ny, in_channels+2)

        x = self.p(x)  # (B, Nx, Ny, width)
        x = x.permute(0, 3, 1, 2)  # (B, width, Nx, Ny)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])  # pad the domain if input is non-periodic

        x = self.fno2d_layers(x)

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]  # pad the domain if input is non-periodic
        x = self.q(x)  # (B, out_channels, Nx, Ny)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.x_span, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.y_span, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
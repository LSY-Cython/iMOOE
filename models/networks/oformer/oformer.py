import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
from .encoder_module import SpatialTemporalEncoder2D
from .decoder_module import PointWiseDecoder2DSimple


class OFormerUniform2d(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, encoder_emb_dim,
                 encoder_heads, encoder_depth, x_span, y_span):
        super(OFormerUniform2d, self).__init__()
        self.x_span = x_span
        self.y_span = y_span

        self.encoder = SpatialTemporalEncoder2D(in_channels+2, encoder_emb_dim,
                                                latent_channels, encoder_heads, encoder_depth)
        self.decoder = PointWiseDecoder2DSimple(latent_channels, out_channels)

    def forward(self, x):  # (B, C_comb, Nx, Ny)
        in_seq = rearrange(x, 'b c h w -> b (h w) c')  # (B, N, C_comb)
        input_pos = prop_pos = self.get_grid(x.shape, x.device)  # [B, N, 2]
        in_seq = torch.cat((in_seq, input_pos), dim=-1)  # [B, N, C_comb+2]

        z = self.encoder(in_seq, input_pos)  # [B, N, latent_channels]
        x_out = self.decoder(z, prop_pos, input_pos)  # [B, N, C]
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=x.shape[2], w=x.shape[3])

        return x_out  # (B, C, Nx, Ny)

    def get_grid(self, shape, device):
        batchsize, Nx, Ny = shape[0], shape[2], shape[3]
        x0, y0 = np.meshgrid(np.linspace(0, self.x_span, Nx), np.linspace(0, self.y_span, Ny))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, Nx, Ny]
        grid = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').unsqueeze(0).float()  # [1, N, 2]
        pos = repeat(grid, '() n c -> b n c', b=batchsize)  # [B, N, 2]

        return pos.to(device)
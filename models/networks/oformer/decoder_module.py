import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from .attention_module import CrossLinearAttention, FeedForward


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping,
    code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks.
    Given an input of size [batches, n, num_input_channels],
    returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                      heads=heads, dim_head=dim_head, dropout=dropout,
                                                      relative_emb=relative_emb,
                                                      scale=scale,
                                                      relative_emb_dim=relative_emb_dim,
                                                      min_freq=min_freq,
                                                      init_method='orthogonal',
                                                      cat_pos=cat_pos,
                                                      pos_dim=relative_emb_dim)
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x


class PointWiseDecoder2DSimple(nn.Module):
    def __init__(self,
                 latent_channels,
                 out_channels,
                 scale=8
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n, c]
                propagate_pos,  # [b, n, 2]
                input_pos=None,
                ):
        x = self.coordinate_projection.forward(propagate_pos)  # [b, n, latent_channels]
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)  # [b, n, latent_channels]
        z = self.decode(torch.cat((z, propagate_pos), dim=-1))  # [b, n, out_channels]

        return z  # [b, n, c]

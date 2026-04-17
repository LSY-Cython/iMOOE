import torch.nn as nn
from .attention_module import LinearAttention, FeedForward, ReLUFeedForward


class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        for d in range(depth):
            if scale[d] != -1 or cat_pos:
                attn_module = LinearAttention(dim, attn_type,
                                               heads=heads, dim_head=dim_head, dropout=dropout,
                                               relative_emb=True, scale=scale[d],
                                               relative_emb_dim=relative_emb_dim,
                                               min_freq=min_freq,
                                               init_method=attention_init,
                                               init_gain=init_gain
                                               )
            else:
                attn_module = LinearAttention(dim, attn_type,
                                              heads=heads, dim_head=dim_head, dropout=dropout,
                                              cat_pos=True,
                                              pos_dim=relative_emb_dim,
                                              relative_emb=False,
                                              init_method=attention_init,
                                              init_gain=init_gain
                                              )
            if not use_ln:
                self.layers.append(
                    nn.ModuleList([
                                    attn_module,
                                    FeedForward(dim, mlp_dim, dropout=dropout)
                                    if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                    ]),
                    )
            else:
                self.layers.append(
                    nn.ModuleList([
                        nn.LayerNorm(dim),
                        attn_module,
                        nn.LayerNorm(dim),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                    ]),
                )

    def forward(self, x, pos_embedding):  # x in [b n c], pos_embedding in [b n 2]
        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding) + x
                x = ln2(x)
                x = ffn(x) + x
        return x


class SpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,
                 in_emb_dim,               # embedding dim of token
                 out_seq_emb_dim,          # embedding dim of encoded sequence
                 heads,
                 depth,                    # depth of transformer
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32, 16, 8, 8] +
                                                                             [1] * (depth - 4),
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32] + [16]*(depth-2) + [1],
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, t(*c)+2]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x
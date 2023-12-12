from torch import nn
from local_attention import LocalAttention
from zeta.nn import (
    FeedForward,
    RMSNorm,
)


# Transformer Blocks
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        window_size: int = 512,
        causal: bool = True,
        look_backward: int = 1,
        look_forward: int = 0,
        dropout: float = 0.1,
        shared_qk: bool = True,
        exact_window_size: bool = False,
        heads: int = None,
        dim_head: int = None,
        ff_mult=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                LocalAttention(
                    dim=dim,
                    window_size=window_size,
                    causal=causal,
                    look_backward=look_backward,
                    look_forward=look_forward,
                    dropout=dropout,
                    shared_qk=shared_qk,
                ),
            )

            self.ffn_layers.append(
                FeedForward(dim=dim, dim_out=dim, mult=ff_mult, dropout=dropout),
            )

    def forward(self, x):
        for attn, ffn in zip(self.layers, self.ffn_layers):
            x = ffn(x) + x
            x = attn(x, x, x) + x
            x = ffn(x) + x
        return x


# classes


class TeraGPT(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = TransformerBlock(dim, depth, heads, dim_head, ff_mult)

        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)

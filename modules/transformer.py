"""
Define and implement transformer block.
"""

import torch.nn as nn
import torch

from modules.multihead_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    """
    Transfromer Block

    Interesting discussion on layernorm: https://arxiv.org/pdf/2002.04745.pdf
    We use Pre-LN transformer here, which applies layernorms before multi-head attention and MLP.
    """
    def __init__(
        self, n_embed: int, n_heads: int, window_size: int, dropout: float, use_mask: bool = True
    ):
        super().__init__()
        self.attention_module = MultiHeadAttention(
            n_embed, n_heads, window_size, dropout, use_mask
        )
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.LayerNorm(n_embed),
            nn.Linear(n_embed, n_embed),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(n_embed, n_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fwd function for transformer."""
        yln = self.layer_norm1(x)
        y = self.attention_module({'q':yln, 'k':yln, 'v':yln,}) + x
        y = y + self.mlp(y)

        return y

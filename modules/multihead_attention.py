"""
Attention module and aux modules, this includes self-attention and cross-attention

The input of this modeul can be configured using a dict pointing to q, k or v
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class MultiHeadAttentionConfig:
    """Data class defining configuration for attention module."""
    n_embed: int
    n_heads: int
    window_size: int
    dropout: float
    use_mask: bool


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module

    Assume the input size is (B, T, C), where B is batch size, T is the sequence length, and 
    C is model size (embedding size)

    """
    def __init__(self, n_embed, n_heads, window_size, dropout, use_mask: bool = True):
        """
        n_embed is the total embedding space size, and share by n_heads heads
        """
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.use_mask = use_mask
        """
        based on nn.Linear doc, the inputs and outputs are (*, Hin), (*, Hout),
        """
        self.q_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.k_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.v_proj = nn.Linear(n_embed, n_embed, bias=False)
        if use_mask:
            # This defines a lower-triangular matrix buffer to mask the attention for decoding.
            # Augment the dimension to ()
            self.register_buffer("mask", torch.tril(torch.ones(window_size, window_size)))
        self.output_proj = nn.Linear(n_embed, n_embed)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward operations
        x is a dictionary with keys "q", "k", "v", each corresponds to a value tensor shape 
        (B, T, C)
        """
        x_in_q, x_in_k, x_in_v = x['q'],  x['k'],  x['v']
        # TODO: assertion same size
        B, T, C = x_in_q.shape
        # split into q, k, v
        q, k, v = self.q_proj(x_in_q), self.k_proj(x_in_k), self.v_proj(x_in_v)
        # following lines: split q,k,v as multiple heads (B, T, n_heads, embed_per_head),
        # and then transpose to (B, n_heads, T, embed_per_head)
        q = q.view(B, T, self.n_heads, -1).transpose(1,2)
        k = k.view(B, T, self.n_heads, -1).transpose(1,2)
        v = v.view(B, T, self.n_heads, -1).transpose(1,2)

        # The @ operator works on the last two dimensions
        attention = (q @ k.transpose(-1, -2)) * (1.0/math.sqrt(k.shape[-1])) # (B, n_heads, T, T)
        if self.use_mask:
            # self.mask is a buffer that is registered as a lower triangular matrix with dimension
            # (1,1,window_size, window_size), here clip with sequence length T, and the dimension is 
            # broadcastable with attention of size (B, n_heads, T, T)
            attention = attention.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        # softmax across the last dimention, prepare for applying on v
        attention = self.attention_dropout(F.softmax(attention, dim = -1))
        # (B, n_heads, T, T) @ (B, n_heads, T, embed_per_head) -> (B, n_heads, T, embed_per_head)
        # softmax(attention) @ v means soft query on v space, the last dimension of softmax(attention)
        # adds up to 1.       
        # What does contiguous() do? It creates a memory contiguous copy 
        # Don't worry if forget this, when contiguous is needed, there will be an error asking for 
        # contiguous memory and you will know to add it.
        y = (attention @ v).transpose(1,2).contiguous().view(B,T,C)
        # Where to place dropout? Can be placed on fully-connected layers - they tend to have lots of 
        # parameters and thus are prone to introduce overfitting
        y = self.output_dropout(self.output_proj(y))

        return y
    
    @classmethod
    def from_config(cls, config: MultiHeadAttentionConfig):
        """Factory method for creating MultiHeadSelfAttention from config"""
        return cls(
            config.n_embed,
            config.n_heads,
            config.window_size,
            config.dropout,
            config.use_mask
        )
"""
A super simple implementation of "GPT".

Some parts referenced https://github.com/karpathy/minGPT
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
from modules.transformer import TransformerBlock

class SimpleGpt(nn.Module):
    """Simple GPT"""
    def __init__(
        self,
        n_vocab: int,
        n_embed: int,
        n_head: int,
        window_size: int,
        n_layer: int,
        dropout: float,
        device: str,    
    ):
        super().__init__()
        self.token_embed_proj = nn.Embedding(n_vocab, n_embed)
        self.pos_embed_proj = nn.Embedding(window_size, n_embed)
        self.transformers = nn.Sequential(
            *[TransformerBlock(
                n_embed, n_head, window_size, dropout, use_mask=True
            ) for _ in range(n_layer)]
        )
        self.output_layer = nn.Linear(n_embed, n_vocab)
        self.device = device
        self.window_size = window_size
        self.to(device)

    def forward(self, sequence: torch.Tensor, targets: torch.Tensor = None):
        B, T = sequence.shape
        token_embedding = self.token_embed_proj(sequence)
        # extract the embedding at each position from 0 to T-1
        pos_embedding = self.pos_embed_proj(torch.arange(T, device = self. device))
        x = token_embedding + pos_embedding
        x = self.transformers(x)
        logits = self.output_layer(x)

        if targets is None:
            return logits, None
        else:
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        
    def generate(self, seq: torch.Tensor, max_new_tokens: int, temperature: float):
        """Generate sequence starting with start_seq, until reaching max_new_tokens"""
        for _ in range(max_new_tokens):
            seq_in_window = seq[:,-self.window_size:] # (B, T, C)
            logits, _ = self(seq_in_window)
            logits = logits[:, -1, :]
            # temperature is used to tune the randomness. The higher temperature,
            # the more randomness in generated sequence. Generation is purely random
            # when temperature is 1.0.
            probs = F.softmax(logits*(1.0-temperature), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat((seq, next_token), dim=1)
        return seq
    
    @torch.no_grad()
    def estimate_loss(self, pred: torch.Tensor, target: torch.Tensor, eval_iters: int) -> torch.Tensor:
        self.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = self(pred, target)
            losses[k] = loss.item()
        mean_loss = losses.mean()
        self.train()
        return mean_loss

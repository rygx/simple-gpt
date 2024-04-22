"""
Train/eval batcher based on super simple char tokenizer

Some parts referenced https://github.com/karpathy/minGPT
"""

import torch
from batcher.tokenizer_batcher import TokenizerBatcher

class CharTokenizerBatcher(TokenizerBatcher):
    """Batcher using simple char tokenizer. """
    def __init__(
        self,
        filename: str,
        batch_size: int,
        window_size: int,
        split_ratio: float,
    ):
        with open(filename, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.batch_size = batch_size
        self.window_size = window_size
        self.vocab = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: "".join([self.itos[i] for i in l])

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(split_ratio * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split_mode: str, device=None):
        """
        Get a batch from the dataset.
        """
        return self._get_batch_helper(self.train_data, self.val_data, split_mode, self.window_size, self.batch_size, device)

    def vocab_size(self):
        """Returns the size of the dataset."""
        return len(self.vocab)

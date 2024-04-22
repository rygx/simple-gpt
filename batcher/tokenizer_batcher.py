"""
Interface for tokenizer bathcer
"""

import torch

class TokenizerBatcher:
    """Interface for tokenizer batcher behavior."""
    def __init__(self):
        pass

    def get_batch(self,split_mode: str):
        """
        Gets a batch.

        Args:
        split_mode - train/eval mode to retrieve batch. If "train" then batch train data,
            otherwise batch eval data.
        """
        raise NotImplementedError("Please extend this class to define get_batch function.")
    
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        raise NotImplementedError("Please extend this class to define vocab_size function.")
    
    def _get_batch_helper(
            self,
            train_data: torch.Tensor,
            val_data: torch.Tensor,
            split_mode: str,
            window_size: int,
            batch_size: int,
            device = None,
            ):
        data = train_data if split_mode == 'train' else val_data
        ix = torch.randint(len(data) - window_size, (batch_size,))
        x = torch.stack([data[i:i+window_size] for i in ix])
        y = torch.stack([data[i+1:i+window_size+1] for i in ix])
        if device is not None:
            x, y = x.to(device), y.to(device)
        return x, y
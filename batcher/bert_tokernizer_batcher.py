"""
Train/eval batcher using BERT tokenizer
"""

import torch
from transformers import AutoTokenizer
from batcher.tokenizer_batcher import TokenizerBatcher

class BertTokenizerBatcher(TokenizerBatcher):
    def __init__(
            self, 
            filename: str,
            batch_size: int,
            window_size: int,
            split_ratio: float,
        ):        
        with open(filename, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.batch_size = batch_size
        self.window_size = window_size
        encoded_text = self.tokenizer(self.text)['input_ids']
        self.unique_id = sorted(list(set(encoded_text)))
        self.token_id_to_idx = { self.unique_id[idx]:idx for idx in range(len(self.unique_id))}

        self.vocab_sz = len(self.unique_id)

        self.encode = lambda s: [self.token_id_to_idx[tk] for tk in self.tokenizer(s)['input_ids']]
        self.decode = lambda l: self.tokenizer.decode([self.unique_id[v] for v in l])

        self.data = torch.tensor([self.token_id_to_idx[et] for et in encoded_text], dtype=torch.long)
        n = int(split_ratio*len(self.data)) # first 90% will be train, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split_mode: str, device=None):
        """
        Get a batch from the dataset.
        """
        return self._get_batch_helper(self.train_data, self.val_data, split_mode, self.window_size, self.batch_size, device)
    
    def vocab_size(self):
        """Returns the size of the dataset."""
        return self.vocab_sz
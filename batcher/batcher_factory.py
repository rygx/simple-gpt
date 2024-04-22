"""
Makes batcher
"""

from batcher.char_tokenizer_batcher import CharTokenizerBatcher
from batcher.bert_tokernizer_batcher import BertTokenizerBatcher

def create_bacther(filename, batch_size, window_size, split_ratio, batcher_type: str = "bert"):
    """Creates tokenizer batcher based on bather type and parameters."""
    if batcher_type == "bert":
        return BertTokenizerBatcher(filename, batch_size, window_size, split_ratio)
    else:
        return CharTokenizerBatcher(filename, batch_size, window_size, split_ratio)
    
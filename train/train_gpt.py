"""
Trains GPT.

You can download the tiny-shakespeare text sample using command
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O input1.dat
"""

import torch
from util.model_management import save_model
from batcher.batcher_factory import create_bacther
from batcher.tokenizer_batcher import TokenizerBatcher
from modules.gpt import SimpleGpt


def estimate_gpt_loss(
    model: SimpleGpt, batcher: TokenizerBatcher, eval_iters: int, split: str
) -> torch.Tensor:
    """
    Estimate gpt model loss.

    Args:
    eval_iters - iterations to estimate the mean loss
    split - "train" or "val"
    """
    X, Y = batcher.get_batch(split)
    return model.estimate_loss(X.to(model.device), Y.to(model.device), eval_iters)


if __name__ == "__main__":    
    # TODO: rygx - add argparser
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16  # how many independent sequences will we process in parallel?
    window_size = 100  # what is the maximum context length for predictions?
    max_iters = 7500
    eval_interval = 100
    learning_rate = 5e-4
    eval_iters = 200
    n_embed = 128
    n_heads = 16
    n_layer = 4
    dropout = 0.1
    batcher_mode = "simple"
    filename = "input1.dat"
    batcher_split = 0.9
    model_save_dir = "models"

    batcher = create_bacther(
        filename, batch_size, window_size, batcher_split, batcher_type=batcher_mode
    )
    n_vocab = batcher.vocab_size()
    print(f"{n_vocab=}")
    print(f"{len(batcher.text)=}")

    model = SimpleGpt(
        n_vocab,
        n_embed,
        n_heads,
        window_size,
        n_layer,
        dropout,
        device,
    )
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = estimate_gpt_loss(model, batcher, eval_iters, "train")
            val_loss = estimate_gpt_loss(model, batcher, eval_iters, "val")
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        # sample a batch of data
        xb, yb = batcher.get_batch("train")

        # evaluate the loss
        logits, loss = model(xb.to(device), yb.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    saved_model_uuid = save_model(
        model.state_dict(),
        model_save_dir,
        locals(),
        n_vocab,
        n_embed,
        n_heads,
        window_size,
        dropout,
        n_layer,
        batch_size,
        batcher_split,
        filename,
        batcher_mode,
    )
    print(f"{saved_model_uuid=}")

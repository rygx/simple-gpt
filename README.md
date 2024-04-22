# A PyTorch implementation of generative pre-trained transformers (GPT)

This is a personal exercise to build, train and use GPT, inspired by [minGPT](https://github.com/karpathy/minGPT).

## Installation
```
git clone https://github.com/rygx/simple-gpt.git && cd simple-gpt
python -m venv .venv
pip install pip-tools
./update_deps.sh
```

## Usage
### Train
Use [train/train_gpt.py](train/train_gpt.py). After training, model state dictionary and hyper parameters will be stored in `models` directory.

### Generate
Use [train/generate.py](train/generate.py).

## TODOs

# A PyTorch implementation of generative pre-trained transformers (GPT)

This is a personal exercise to build, train and use GPT, inspired by [minGPT](https://github.com/karpathy/minGPT).

## Prerequisite
- [direnv](https://direnv.net/)
- Python 3.9+

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
Use [train/generate.py](train/generate.py). Under [models/](models/) directory there is already a coarsely pre-trained model with ID `9cdb42ed-0b16-4a3a-88e2-fffa61fa4f50`. Generate texts using this model with the following command:
```
python train/generate.py --dir "models" -u "9cdb42ed-0b16-4a3a-88e2-fffa61fa4f50" --prompt "QUEEN: "
```
Generation length (in number of tokens) and temperature can also be tuned using `--length/-l` and `--temp/-t` options.

## TODOs/plans
- Easier setup (e.g., using setuptool)
- Unit tests :P
- Implement own tokenizer

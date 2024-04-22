"""
Load model and generate

Example:
python train/generate.py --dir "models"  --prompt "KING:" -u "{UUID}" -t 0.1 -l 100
"""

import argparse
import torch
from util.model_management import load_model
from batcher.batcher_factory import create_bacther
from modules.gpt import SimpleGpt

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "Generate using SimpleGPT")
 
    parser.add_argument("-d", "--dir", type = str, 
                        help = "Directory of model and parameter storage.")
     
     
    parser.add_argument("-u", "--uuid", type = str, 
                        help = "UUID of model and parameter.")
    
    parser.add_argument("-p", "--prompt", type = str, 
                        help = "Starting prompt")
     
    parser.add_argument("-t", "--temp", type = float, default= 0.1,
                        help = "Generation temperature, should be <=1.0")
     
    parser.add_argument("-l", "--length", type = int, default= 500,
                        help = "Generation length")
     
    args = parser.parse_args()

    assert args.length > 0, f"Illegal generation length: {args.length}. Length should be > 0."
    assert args.temp <=1.0, f"Illegal temperature: {args.temp}. Temperature should be <=1.0."

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = load_model(args.dir,args.uuid,locals())
    model = SimpleGpt(
        n_vocab, n_embed, n_heads, window_size, n_layer, dropout ,device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    
    batcher = create_bacther(filename, batch_size, window_size, batcher_split, batcher_mode)
    print()
    # generate from the model
    context = torch.LongTensor(batcher.encode(f" {args.prompt} ")[1:-1]).unsqueeze(0).to(device)
    print(batcher.decode(model.generate(context, max_new_tokens=args.length, temperature=args.temp)[0].tolist()))
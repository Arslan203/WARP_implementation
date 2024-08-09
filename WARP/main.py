import json
import argparse
from train import train
from pathlib import Path
import os.path as osp
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequence classification model with LoRA")
    parser.add_argument('--model_name', type=str, help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--I', type=int)
    parser.add_argument('--M', type=int)
    parser.add_argument('--T', type=int)
    parser.add_argument('--nu', type=int)
    parser.add_argument('--mu', type=int)
    parser.add_argument('--lambda', type=int)
    return parser.parse_args()

def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    args = parse_args()

    config_path = osp.abspath(osp.join(__file__, osp.pardir, 'config.json'))
    
    # Update the Config with command-line arguments
    config = load_config(config_path)['WARP_args']
    for name, val in args._get_kwargs():
        if val is not None:
            if name == 'learning_rate':
                config['optimizers_args']['learning_rate'] = args.learning_rate
            else:
                config[name] = args.__getattr__(name)

    # check reward_path
    if Path(config['reward_model']).is_dir():
        config['reward_model'] = osp.abspath(config['reward_model'])

    config['save_path'] = osp.join(os.getcwd(), config['save_path'])  
    # Run training
    train(config)

if __name__ == "__main__":
    main()

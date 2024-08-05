import json
import argparse
from train import train

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequence classification model with LoRA")
    parser.add_argument('--model_name', type=str, help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--I', type=int)
    parser.add_argument('--M', type=int)
    parser.add_argument('--T', type=int)
    return parser.parse_args()

def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    args = parse_args()
    
    # Update the Config with command-line arguments
    config = load_config('config.json')['WARP_args']
    for name, val in args._get_kwargs():
        if val is not None:
            if name == 'learning_rate':
                config['optimizers_args']['learning_rate'] = args.learning_rate
            else:
                config[name] = args.__getattr__(name)

    # Run training
    train(config)

if __name__ == "__main__":
    main()

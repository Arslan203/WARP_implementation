import json
import argparse
from train import train

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequence classification model with LoRA")
    parser.add_argument('--model_name', type=str, default="distilbert/distilbert-base-cased", help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    return parser.parse_args()

def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    args = parse_args()
    
    # Update the Config with command-line arguments
    config = load_config('config.json')['reward_trainer_args']
    config['model_name'] = args.model_name
    config['optimizers_args']['learning_rate'] = args.learning_rate
    config['train_batch_size'] = args.train_batch_size
    config['num_epochs'] = args.epochs

    # Run training
    train(config)

if __name__ == "__main__":
    main()

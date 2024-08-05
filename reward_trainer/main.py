import json
import argparse
from train import train

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequence classification model with LoRA")
    parser.add_argument('--model_name', type=str, default="distilbert/distilbert-base-cased", help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size for evaluate')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_per_epoch', type=int, default=1000, help='Number of batchs in epoch')
    parser.add_argument('--epoch_to_save', type=int, default=3, help='Number of epochs per saving')
    parser.add_argument('--batch_per_eval', type=int, default=100, help='Number of batchs per evaluating')
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
    config['eval_batch_size'] = args.train_batch_size
    config['batch_per_epoch'] = args.batch_per_epoch
    config['batch_per_eval'] = args.batch_per_eval
    config['epoch_to_save'] = args.epoch_to_save
    config['num_epochs'] = args.num_epochs
    # Run training
    train(config)

if __name__ == "__main__":
    main()

import json
import argparse
from train import train

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequence classification model with LoRA")
    parser.add_argument('--model_name', type=str, help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('--train_batch_size', type=int, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, help='Batch size for evaluate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_per_epoch', type=int, help='Number of batchs in epoch')
    parser.add_argument('--epoch_to_save', type=int, help='Number of epochs per saving')
    parser.add_argument('--batch_per_eval', type=int, help='Number of batchs per evaluating')
    return parser.parse_args()

def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    args = parse_args()
    
    # Update the Config with command-line arguments
    config = load_config('config.json')['reward_trainer_args']
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

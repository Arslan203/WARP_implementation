
import argparse
from config import Config
from train import train

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequence classification model with LoRA")
    parser.add_argument('--model_name', type=str, default="distilbert/distilbert-base-cased", help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update the Config with command-line arguments
    config = Config()
    config.model_name = args.model_name
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.epochs = args.epochs

    # Run training
    train()

if __name__ == "__main__":
    main()

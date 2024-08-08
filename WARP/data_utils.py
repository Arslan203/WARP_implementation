import torch
from datasets import load_dataset
import numpy as np

def load_imdb_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    # Remove 'unsupervised' split if it exists
    if 'unsupervised' in dataset:
        del dataset['unsupervised']
    if 'test' in dataset:
        del dataset['test']
    return dataset

class IMDbPrompts(torch.utils.data.Dataset):
  def __init__(self, encodings: torch.Tensor, truncate_range=(5, 15)):
    self.encodings = encodings
    self.trunc_range = truncate_range

  def __getitem__(self, idx):
    prompt_len = torch.randint(low=self.trunc_range[0], high=self.trunc_range[1] + 1, size=(1, )).item()
    return self.encodings[idx][:prompt_len]

  def __len__(self):
    return len(self.encodings)


def collate_fn_WARP(batch, tokenizer):
  batch = [{'input_ids': b} for b in batch]
  ids = tokenizer.pad(
              batch,
              padding=True,
              return_tensors='pt')
  return ids['input_ids']
import torch
from datasets import load_dataset
import numpy as np

def load_imdb_dataset(dataset_name):
  dataset = load_dataset(dataset_name)
  # Remove 'unsupervised' split if it exists
  if 'unsupervised' in dataset:
      del dataset['unsupervised']
  return dataset

class IMDbDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels = labels
      self.preprocess_label()


  def preprocess_label(self):
      labels = np.array(self.labels)
      self.chose_mask = np.where(labels == 1)[0]
      self.reject_mask = np.where(labels == 0)[0]
      # self.indices = list(product(self.chose_mask, self.reject_mask))


  def __getitem__(self, idx):
    ch_ind, rej_ind = idx // self.reject_mask.shape[0], idx % self.reject_mask.shape[0]
    ch_ind, rej_ind = self.chose_mask[ch_ind], self.reject_mask[rej_ind]
    # item = {"input_ids_chosen": self.encodings['input_ids'][ch_ind],
    #         "attention_mask_chosen": self.encodings['attention_mask'][ch_ind],
    #         "input_ids_rejected": self.encodings['input_ids'][rej_ind],
    #         "attention_mask_rejected": self.encodings['attention_mask'][rej_ind],
    #         }
    item = {'chosen': self.encodings[ch_ind],
            'rejected': self.encodings[rej_ind]}
    return item

  def __len__(self):
      return self.reject_mask.shape[0] * self.chose_mask.shape[0]

class EVALIMDbDataset(IMDbDataset):
  def __init__(self, texts, labels):
    super().__init__(texts, labels)
    self.perm = torch.randperm(self.chose_mask.shape[0])

  def shuffle(self):
    self.perm = torch.randperm(self.chose_mask.shape[0])

  def __getitem__(self, idx):
    ch_ind, rej_ind = idx, self.perm[idx]
    ch_ind, rej_ind = self.chose_mask[ch_ind], self.reject_mask[rej_ind]
    # item = {"input_ids_chosen": self.encodings['input_ids'][ch_ind],
    #         "attention_mask_chosen": self.encodings['attention_mask'][ch_ind],
    #         "input_ids_rejected": self.encodings['input_ids'][rej_ind],
    #         "attention_mask_rejected": self.encodings['attention_mask'][rej_ind],
    #         }
    item = {'chosen': self.encodings[ch_ind],
            'rejected': self.encodings[rej_ind],
            }
    # item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return self.chose_mask.shape[0]



def collate_fn_RT(data, tokenizer):
  texts = {'chosen': [], 'rejected': []}
  for x in data:
    for k in x.keys():
        texts[k].append(x[k])
  chosen = tokenizer(texts['chosen'], truncation=True, padding=True, return_tensors='pt')
  rejected = tokenizer(texts['rejected'], truncation=True, padding=True, return_tensors='pt')
  batch = {"input_ids_chosen": chosen["input_ids"], "attention_mask_chosen": chosen['attention_mask'],
          "input_ids_rejected": rejected["input_ids"], "attention_mask_rejected": rejected['attention_mask'],
          'margin': 0,
  }
  return batch

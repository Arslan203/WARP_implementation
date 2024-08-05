from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from functools import partial
import os

def load_model_and_tokenizer(model_name):
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', add_pad_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    generation_config = GenerationConfig(
        max_new_tokens=53,
        min_new_tokens=53,
        temperature=0.9,
        top_p=1.0,
        top_k=0.0,
        do_sample=True
    )
    return model, tokenizer, generation_config


def get_optimizer_and_scheduler(model, args):
    def linear_scheduler(current_step, num_training_steps):
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps)))

    optimizer = torch.optim.AdamW(model.parameters(), args['optimizers_args']['learning_rate'], weight_decay=args['optimizers_args']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(linear_scheduler, num_training_steps=args['num_epochs'] * args['batch_per_epoch']))
    return optimizer, scheduler

class RewardBenchmark:
  def __init__(self):
    self.all_chosen = []
    self.all_rejected = []

  def add_batch(self, chosen, rejected):
    self.all_chosen.append(chosen)
    self.all_rejected.append(rejected)

  def compute(self):
    self.all_chosen = torch.cat(self.all_chosen, 0).squeeze()
    self.all_rejected = torch.cat(self.all_rejected, 0).squeeze()
    labels = torch.cat((torch.zeros_like(self.all_rejected), torch.ones_like(self.all_chosen)), 0)
    labels = labels[torch.argsort(torch.cat((self.all_rejected, self.all_chosen), 0))]


    counter, score = 0, 0
    for mask in labels:
      if mask == 1:
        score += counter
      else:
        counter += 1
    score /= self.all_chosen.shape[0] * self.all_rejected.shape[0]


    # del self.all_chosen
    # del self.all_rejected
    return score
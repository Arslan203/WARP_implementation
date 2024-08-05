from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig
import torch
from functools import partial

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.enable_input_require_grads()
    return model, tokenizer

def configure_lora(model, lora_args):
    lora_config = LoraConfig(
        task_type='SEQ_CLS',
        **lora_args,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def get_optimizer_and_scheduler(model, args):
    def linear_scheduler(current_step, num_training_steps):
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps)))

    optimizer = torch.optim.AdamW(model.parameters(), args['optimizer_args']['learning_rate'], weight_decay=args['optimizer_args']['weight_decay'])
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
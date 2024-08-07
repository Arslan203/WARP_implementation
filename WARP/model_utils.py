from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, GenerationConfig
import torch
from functools import partial
import os

def load_model_and_tokenizer(model_name, generation_config_args):
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', add_pad_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    generation_config = GenerationConfig(
        **generation_config_args
    )
    return model, tokenizer, generation_config


def get_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.Adam(model.parameters(), args['optimizers_args']['learning_rate'], weight_decay=args['optimizers_args']['weight_decay'])

    return optimizer, None

class RM(torch.nn.Module):
  def __init__(self, model, tokenizer_from, tokenizer_to, device='cuda'):
    super(RM, self).__init__()
    self.model = model
    self.tokenizer_from = tokenizer_from
    self.tokenizer_to = tokenizer_to
    self.device = device

    if isinstance(self.model, str):
       name = self.model
       self.model = AutoModelForSequenceClassification.from_pretrained(name)
       self.tokenizer_to = AutoTokenizer.from_pretrained(name)

    self.model.eval()

  def forward(self, query_completion):
    text = self.tokenizer_from.batch_decode(
        query_completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    tokens = self.tokenizer_to(text, return_tensors='pt', padding=True, truncation=True).to(self.device)

    return self.model(**tokens).logits.mean(dim=-1)

def generate(model, queries, generation_config, pad_token_id=0):
  context_len = queries.shape[1]
  attention_mask = queries.ne(pad_token_id)
#   position_ids = attention_mask.cumsum(1) - attention_mask.long()
  res = model.generate(
      input_ids=queries,
      attention_mask=attention_mask,
#       position_ids=position_ids,
      return_dict_in_generate=True,
      output_scores=True,
      generation_config=generation_config,
      )
  return torch.cat((queries, res.sequences[:, context_len:]), 1), torch.stack(res.scores, 1)

def forward(model, query_completion, pad_token_id=0):
  attention_mask = query_completion.ne(pad_token_id)
  # i dont think we need pos_ids
  # position_ids = attention_mask.cumsum(1) - attention_mask.long()
  return model(
        input_ids=query_completion,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )

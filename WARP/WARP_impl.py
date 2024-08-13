from tqdm import tqdm
import gc
from accelerate import Accelerator
from collections import defaultdict
import torch
import torch.nn.functional as F
from copy import deepcopy
from ema_pytorch import EMA
from model_utils import forward, generate, get_optimizer_and_scheduler
import numpy as np


def gather_logprobs(logits, index):
  all_logprobs = F.log_softmax(logits, dim=-1)
  logprobs = torch.gather(all_logprobs, 2, index.unsqueeze(-1)).squeeze(-1)
  return logprobs

def inner_loop(sft_model, reward_model, prompt_dataset, T, **kwargs):

  generation_config = kwargs.get('generation_config', None)
  beta = kwargs.get('beta', 0.1)
  verbose = kwargs.get('verbose', True)
  grad_acc_steps = kwargs.get('gradient_accumulation_steps', 2)
  mu = kwargs.get('mu', 0.01)
  mu = 1 - mu
  pad_token_id = reward_model.tokenizer_from.pad_token_id

  if generation_config is None:
    generation_config = sft_model.generation_config
    generation_config.max_new_tokens = 100
#     generation_config.temperature = 0.9

  model = sft_model
  state_dict_innate = deepcopy(model.state_dict())
  ema_model = EMA(model, beta=mu,
                  update_after_step=-1, update_every=grad_acc_steps)
  ema_model.eval()

  optimizer, _ = get_optimizer_and_scheduler(model, kwargs.get('args', None))

  pbar = tqdm(total=T, unit='text', disable=not verbose)

  accelerator = Accelerator(gradient_accumulation_steps=grad_acc_steps)

  prompt_dataset, model, ema_model, reward_model, optimizer = accelerator.prepare(prompt_dataset, model, ema_model, reward_model, optimizer)

  data_iterable = iter(prompt_dataset)

  logs = {'reward': [], 'kl': []}

  for t in range(1, T * grad_acc_steps + 1):
    with accelerator.accumulate(model):
      model.eval()

      x = next(data_iterable)
      pbar.set_description('generation')
      with torch.no_grad():
        query_completion, logits_non_grad = generate(model, x, generation_config, pad_token_id)
      completion = query_completion[:, x.shape[1]:]
      logprobs_non_grad = gather_logprobs(logits_non_grad, completion)

      pbar.set_description('ema_forward')
      with torch.no_grad():
        logits_ref = forward(ema_model, query_completion).logits
        logprobs_ref = gather_logprobs(logits_ref[:, x.shape[1] - 1 : -1, :], completion)

        reward = reward_model(query_completion)

      model.train()
      pbar.set_description('model_forward')
      logits = forward(model, query_completion).logits
      logprobs = gather_logprobs(logits[:, x.shape[1] - 1 : -1, :], completion)

      kl_loss = (logprobs_non_grad - logprobs_ref).sum(-1)

      loss = - torch.mean(logprobs.sum(-1) * (reward - beta * kl_loss))

      logs['reward'].append(reward.mean(-1).item())
      logs['kl'].append(kl_loss.mean(-1).item())

      accelerator.backward(loss)

      optimizer.step()

      optimizer.zero_grad()

      ema_model.update()

      if t % grad_acc_steps == 0:
        pbar.update(1)

  for name, metric in logs.items():
    logs[name] = torch.tensor(metric)

  vector = deepcopy(model.state_dict())

  model.load_state_dict(state_dict_innate)

  del ema_model

  gc.collect()

  return {'vector': vector} | logs

def compute_vectors(model, reward_model, dataset, M=2, T=100, **kwargs):
    res = defaultdict(list)
    for m in range(M):
      ret_dict = inner_loop(sft_model=model, reward_model=reward_model, prompt_dataset=dataset, T=T, **kwargs)

      for key in ret_dict.keys(): # gathering logs
        res[key].append(ret_dict[key])

    for key, val in res.items(): # merging res dict
      if key != 'vector':
        res[key] = torch.stack(val, 0)
    return res

def SLERP(w_init, w_1, w_2, lamb, eps=1e-6):
  for name in w_1.keys():
     w_1[name].data.add_(w_init[name].data, alpha=-1)
     w_2[name].data.add_(w_init[name].data, alpha=-1)
  
  w1_flatten, w2_flatten = torch.cat([w1.flatten() for w1 in w_1.values()], 0), torch.cat([w2.flatten() for w2 in w_2.values()], 0)
  angle = torch.acos(F.cosine_similarity(w1_flatten, w2_flatten, 0)) + eps
  coef_1 = (torch.sin((1 - lamb) * angle) / torch.sin(angle)).item()
  coef_2 = (torch.sin(lamb * angle) / torch.sin(angle)).item()
  print(f'angle_between_task_vectors:{180 * angle.item() / np.pi:.1f} degrees')
  for name, par in w_init.items():
    yield name, par.data.detach().clone().mul_(1 - coef_1 - coef_2).add_(w_1[name].data, alpha=coef_1).add_(w_2[name].data, alpha=coef_2)


def WARP_method(model, reward_model, dataset, I, M=2, nu=0.5, lamb=0.5, **kwargs): # for now supports only M=2
    device = kwargs.get('device', 'cuda')
    model_sft = deepcopy(model).to('cpu')
    model.to(device)
    res = defaultdict(list)
    for i in range(I):
        vector_logs = compute_vectors(model, reward_model, dataset, **kwargs)
        vectors = vector_logs.pop('vector')
        model_st_cp = deepcopy(model.state_dict())
        if i == I - 1:
            model_sft.to(device)
            model_st = dict(model_sft.state_dict())
        else:
            model_st = dict(model.state_dict())
        for name, slerp_par in SLERP(model_st_cp, *vectors, lamb):
            model_st[name].data.mul_(1 - nu).add_(slerp_par, alpha=nu)

        for key, val in vector_logs.items():
          res[key].append(val)

        gc.collect()

    for key, val in res.items(): # merging res dict
      res[key] = torch.stack(val, 0)

    return {'model': model_sft} | res

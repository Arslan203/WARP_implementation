import numpy as np
import torch
import torch.nn.functional as F
from model_utils import *
from data_utils import load_imdb_dataset, IMDbDataset, EVALIMDbDataset, collate_fn_RT
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from functools import partial

def train(args):
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    set_seed(args['seed'])

    dataset = load_imdb_dataset(args['dataset_name'])
    model, tokenizer = load_model_and_tokenizer(args['model_name'])
    model = configure_lora(model, args['lora_args'])

    # prepare dataset
    train_dataset = IMDbDataset(dataset['train']['text'], dataset['train']['label'])
    eval_dataset = EVALIMDbDataset(dataset['test']['text'], dataset['test']['label'])

    # Create DataLoaders
    collate_fn_ = partial(collate_fn_RT, tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['train_batch_size'], shuffle=True,
                               num_workers=args['num_workers'], collate_fn=collate_fn_)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args['eval_batch_size'], shuffle=False,
                              num_workers=args['num_workers'], collate_fn=collate_fn_)

    # Set up optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    logging = args['log_with'] != 'none'
    if logging:
      accelerator = Accelerator(log_with=args['log_with'])
      accelerator.init_trackers('reward_trainer')
    else:
      accelerator = Accelerator()
    train_loader, eval_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, eval_loader, model, optimizer, scheduler
    )

    # Training loop
    for epoch in range(1, args['num_epochs'] + 1):
        model.train()
        losses = []
        for i, batch in tqdm(enumerate(train_loader), disable=not args['verbose'], desc='train_epoch'):
            optimizer.zero_grad()
            chosen = model(input_ids=batch['input_ids_chosen'], attention_mask=batch['attention_mask_chosen']).logits
            rejected = model(input_ids=batch['input_ids_rejected'], attention_mask=batch['attention_mask_rejected']).logits
            loss = -F.logsigmoid(chosen - rejected - batch['margin']).mean()
            losses.append(loss.item())
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            if i == args['batch_per_epoch']:
                break

        if epoch % args['epoch_to_save'] == 0:
          if args['HF_repository'] == 'none':
            model.save_pretrained(args['save_path'] + f'/{epoch}_epoch')
          else:
            model.push_to_hub(args['HF_repository'], commit_message=f'train {epoch} epoch')


        accelerator.log({'loss/train': sum(losses) / len(losses)})

        model.eval()
        losses = []
        metric_reward = RewardBenchmark()
        for i, batch in tqdm(enumerate(eval_loader), disable=not args['verbose'], desc='eval_epoch'):
            with torch.no_grad():
                chosen = model(input_ids=batch['input_ids_chosen'], attention_mask=batch['attention_mask_chosen']).logits
                rejected = model(input_ids=batch['input_ids_rejected'], attention_mask=batch['attention_mask_rejected']).logits
                loss = -F.logsigmoid(chosen - rejected - batch['margin']).mean()
                losses.append(loss.item())

            metric_reward.add_batch(chosen=chosen.clone().mean(dim=-1),
                                    rejected=rejected.clone().mean(dim=-1),
                                    )

            if i == args['batch_per_eval']:
                break
        metric_score = metric_reward.compute()
        reward_chosen = metric_reward.all_chosen.mean()
        reward_rejected = metric_reward.all_rejected.mean()


        accelerator.log({'reward/metric': metric_score,
                        'reward/chosen': reward_chosen,
                        'reward/rejected': reward_rejected,
                        'loss/eval': sum(losses) / len(losses)})

    accelerator.end_training()

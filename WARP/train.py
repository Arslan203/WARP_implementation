import torch
import torch.nn.functional as F
import numpy as np
from model_utils import load_model_and_tokenizer, RM
from data_utils import load_imdb_dataset, IMDbPrompts, collate_fn_WARP
from functools import partial
from WARP_impl import WARP_method
from accelerate.utils import set_seed
from transformers.utils import logging
import transformers


def train(args):
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    set_seed(args['seed'])

    verbosity_level = {
        'FATAL': transformers.logging.FATAL,
        'ERROR': transformers.logging.ERROR,
        'WARNING': transformers.logging.WARN,
        'WARN': transformers.logging.WARN,
        'INFO': transformers.logging.INFO,
        'DEBUG': transformers.logging.DEBUG,
    }
    logging.set_verbosity(verbosity_level[args['logging']])
    logging.disable_progress_bar()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_imdb_dataset(args['dataset_name'])
    model, tokenizer, generation_config = load_model_and_tokenizer(args['model_name'], args['generation_config_args'])

    # prepare dataset
    train_dataset = IMDbPrompts(tokenizer(dataset['train']['text'], truncation=True, max_length=args['truncate_range'][1])['input_ids'], args['truncate_range'])

    # Create DataLoader
    collate_fn_ = partial(collate_fn_WARP, tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                               num_workers=args['num_workers'], collate_fn=collate_fn_)

    reward_model = RM(args['reward_model'], tokenizer, tokenizer_to=None, device=device)

    results = WARP_method(model, reward_model, train_loader, args['I'], M=args['M'], nu=args['nu'], lamb=args['lambda'],
                gradient_accumulation_steps=args['gradient_accumulation_steps'], generation_config=generation_config,
                device=device, mu=args['mu'], args=args, beta=args['beta'], verbose=args['verbose'])
    
    results['model'].save_pretrained(args['save_path'])

    torch.cuda.empty_cache()

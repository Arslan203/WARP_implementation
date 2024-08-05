import torch
import torch.nn.functional as F
from model_utils import load_model_and_tokenizer, RM
from data_utils import load_imdb_dataset, IMDbPrompts, collate_fn_WARP
from functools import partial
from WARP_impl import WARP_method


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_imdb_dataset(args['dataset_name'])
    model, tokenizer, generation_config = load_model_and_tokenizer(args['model_name'])

    # prepare dataset
    train_dataset = IMDbPrompts(tokenizer(dataset['train']['text'], truncation=True, max_length=args['truncate_range'][1])['input_ids'], args['truncate_range'])

    # Create DataLoader
    collate_fn_ = partial(collate_fn_WARP, tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['train_batch_size'], shuffle=True,
                               num_workers=args['num_workers'], collate_fn=collate_fn_)

    reward_model = RM(args['reward_model'], tokenizer, tokenizer_to=None, device=device)

    results = WARP_method(model, reward_model, train_loader, args['I'], args['nu'], args['lambda'],
                gradient_accumulation_steps=args['gradient_accumulation_steps'], generation_config=generation_config, device=device)
    
    results['model'].save_pretrained(args['save_path'])

import json
import os
from collections import defaultdict
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import sys

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
 
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    save_path = os.path.join(args.save_path, f'{name}.pt' )
    torch.save(model, save_path)#save_path

def save_extra_model(args, model, name=''):
    name = save_load_name(args, name)
    save_path = os.path.join(args.save_path, f'state_dict_{name}.pt' )
    torch.save(model, save_path)#save_path
    
def load_model(args, name=''):
    name = save_load_name(args, name)
    save_path = os.path.join(args.save_path, f'{name}.pt' )
    model = torch.load(save_path)#f'/pre_trained_models/{name}.pt'
    return model

def adjust_learning_rate(optimizer, epoch, args):# 
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.n_epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups: # 
        param_group['lr'] = lr

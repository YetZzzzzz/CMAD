from __future__ import absolute_import, division, print_function

import sys
import argparse
import random
import pickle
import numpy as np
from typing import *
import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, get_cosine_schedule_with_warmup
from transformers.optimization import AdamW
from itertools import chain
import os
import argparse
from student_model import StudentModel
from teacher_model import TeacherModel
from transformers import BertConfig
from transformers import BertTokenizer, XLNetTokenizer, get_cosine_schedule_with_warmup
from train_student_model_mosei import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = torch.device("cuda:0")
import warnings
warnings.filterwarnings("ignore")
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected. Recieved {0}".format(s)
        )


def seed(s):
    if isinstance(s, int):
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError(
                "Seed must be between 0 and 2**32 - 1. Received {0}".format(s)
            )
    elif s == "random":
        return random.randint(0, 9999)
    else:
        raise argparse.ArgumentTypeError(
            "Integer value is expected. Recieved {0}".format(s)
        )


def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



def convert_to_features(args, examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example# 
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        if args.model == "bert-base-uncased":
            prepare_input = prepare_bert_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            args, tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_bert_input(args, tokens, visual, acoustic, tokenizer):# include the text or not 
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, args.ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, args.VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, args.ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, args.VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids




def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained('./BERT_EN/')
    
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(
                model
            )
        )


def get_appropriate_dataset(data):

    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(args, data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(f"./datasets/{args.dataset}.pkl", "rb") as handle:# 
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


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


def prep_for_training(num_train_optimization_steps: int):

    if args.model == "bert-base-uncased":
        model = StudentModel.from_pretrained(
            './BERT_EN/', num_labels=1, args = args,
        )

    total_para = 0
    for param in model.parameters():
        total_para += np.prod(param.size())
    print('total parameter for the model: ', total_para)
    
    if args.load:
        model.load_state_dict(torch.load(args.model_path))

    # model.to(DEVICE)

    return model
    
def adjust_learning_rate(optimizer, epoch, args):# 
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups: # 
        param_group['lr'] = lr

parser = argparse.ArgumentParser(description='StudentModel')
parser.add_argument('-f', default='', type=str)

# teacher model settings 
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosei")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--dropout_prob", type=float, default=0.3)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased"],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=2e-5)# 2E-5 
parser.add_argument("--gradient_accumulation_step", type=int, default=1) # don't need this 
parser.add_argument("--d_l", type=int, default=96)# 80
parser.add_argument("--seed", type=int, default=5576)

parser.add_argument("--attn_dropout", type=float, default=0.5) #attn_dropout
parser.add_argument("--num_heads", type=int, default=16)#5 
parser.add_argument("--relu_dropout", type=float, default=0.3)
parser.add_argument("--res_dropout", type=float, default=0.3)
parser.add_argument("--embed_dropout", type=float, default=0.2)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay') # 0.01
parser.add_argument('--schedule', default=[180, 200], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')# needs to adjust based on n_epochs []
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--load", type=int, default=0)
parser.add_argument("--test", type=int, default=0)   ####test or not
parser.add_argument("--model_path", type=str, default='bert_tm.pth')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
# new 
parser.add_argument('--p_attn_dropout', type=float, default=0.0,
                    help='feedforward layer dropout in perceiver')
parser.add_argument('--p_ff_dropout', type=float, default=0.0,
                    help='feedforward layer dropout in perceiver')
parser.add_argument('--te_layers', type=int, default=2,
                    help='layers of the transformer encoders')
parser.add_argument('--depth', type=int, default=2,
                    help='layers of the perceiver')
parser.add_argument('--num_latents', type=int, default=5,
                    help='number of learnable latents')
parser.add_argument('--latent_dim', type=int, default=96,
                    help='hidden_dimensions of the learnable units')
parser.add_argument('--save_path', type=str, default='./CMAD_sentiment/Teacher_Model',
                    help='path for storing the dataset')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip value (default: 0.8)')

# hyperparameters for student settings
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--model_root', type=str, default='./CMAD_sentiment/Teacher_Model',
                    help='directory of the saved models')
parser.add_argument('--log_root', type=str, default='./CMAD_sentiment/Student_Model/output/logs',
                    help='directory of the logs')
parser.add_argument('--lr_decrease', type=str, default='cos', help='the methods of learning rate decay  ')
parser.add_argument('--lr_warmup', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--p', default=[0, 0, 0], help='para for modality dropout')
parser.add_argument('--temperature', type=float, default=2.0, help='temperature for dkd')
parser.add_argument('--alpha', type=float, default=1, help='weights for dkd')#0.9
parser.add_argument('--beta', type=float, default=1, help='weights for dkd')#0.1
parser.add_argument('--gamma', type=float, default=1, help='weights of aux loss')#0.4
parser.add_argument('--delta', type=float, default=0.1, help='weights of C2FD loss')#0.4
parser.add_argument('--begin_epoch', type=int, default=20)
parser.add_argument('--weights', type=float, default=20.0, help='weights of mar weights in begin epoch')#0.1
parser.add_argument('--version', type=int, default=0)
parser.add_argument('--total_epoch', type=int, default=20, help='warmup epochs')
parser.add_argument('--tau', type=float, default=0.2, help='tau in the first module')

args = parser.parse_args()
torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
args = parser.parse_args()

if args.dataset == 'mosei':
    args.TEXT_DIM = 768
    args.ACOUSTIC_DIM = 74
    args.VISUAL_DIM = 35

else:
    print('wrong dataset')



args.name = str(args.dataset) + '_version_' + str(args.version) + '_WMSE_' + str(args.delta) +  '_MAR_' + str(args.gamma) + '_DKD_' + str(args.alpha) + '_' + str(args.beta)
args.output_dim = 1
set_random_seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def train_main(args):
    print(args)
    args.log_name = args.name + '.csv'
    args.model_name = args.name
    (
        train_loader,
        dev_loader,
        test_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    
    # Note that the structures of the student and teacher models are exactly the same
    teacher_config = BertConfig.from_json_file("./BERT_EN/config.json")
    teacher_config.num_labels = 1
    teacher_model = TeacherModel(teacher_config, args)
    
    # initialize the network and freeze the teacher network
    if args.dataset=='mosei':
        teacher_model.load_state_dict(torch.load(os.path.join(args.model_root, 'state_dict_mosei.pt')))
    else:
        print('Wrong dataset')

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    student_model = prep_for_training(
        num_train_optimization_steps)
    teacher_model.to(DEVICE)
    student_model.to(DEVICE)


    nets = {'snet': student_model, 'tnet': teacher_model}

    train_model(net_dict=nets, train_loader=train_loader, test_loader=test_loader, args=args)#optimizer=optimizer, 



if __name__ == '__main__':
    start_time = time.time()
    train_main(args)
    end_time = time.time()
    print('Cost time of 100 epochs: %s ms' %((end_time - start_time) * 1000))
from __future__ import absolute_import, division, print_function

import argparse
import os
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
from teacher_model import TeacherModel
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, get_cosine_schedule_with_warmup
from transformers.optimization import AdamW

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = torch.device("cuda:0")

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
        acoustic[acoustic == -np.inf] = 0
        visual[visual == -np.inf] = 0
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)
        if len(tokens) > args.text_length - 2:
            tokens = tokens[: args.text_length - 2]
        if len(acoustic) > args.acoustic_length - 2:
            acoustic = acoustic[: args.acoustic_length - 2]
        if len(visual) > args.visual_length - 2:
            visual = visual[: args.visual_length - 2]

        if args.model == "bert-base-uncased":
            prepare_input = prepare_bert_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            args, tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.acoustic_length
        assert visual.shape[0] == args.visual_length

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

    pad_length = args.text_length - len(input_ids)

    # pad_length_text = args.max_seq_length - len(input_ids)
    pad_length_audio = args.acoustic_length - len(acoustic)
    pad_length_video = args.visual_length - len(visual)

    acoustic_padding = np.zeros((pad_length_audio, args.ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length_video, args.VISUAL_DIM))
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
        model = TeacherModel.from_pretrained(
            './BERT_EN/', num_labels=1, args = args,
        )

    total_para = 0
    for param in model.parameters():
        total_para += np.prod(param.size())
    print('total parameter for the model: ', total_para)
    
    if args.load:
        model.load_state_dict(torch.load(args.model_path))

    model.to(DEVICE)

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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosei")
parser.add_argument("--max_seq_length", type=int, default=60)
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=60)
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
parser.add_argument('--schedule', default=[80, 100], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')# needs to adjust based on n_epochs []
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--load", type=int, default=0)
parser.add_argument("--test", type=int, default=0)   ####test or not
parser.add_argument("--model_path", type=str, default='bert_tm.pth')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument("--alignment", type=str,
                    choices=["align", "unalign"], default="align")

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


args = parser.parse_args()
if args.dataset == 'mosei':
    args.TEXT_DIM = 768
    args.ACOUSTIC_DIM = 74
    args.VISUAL_DIM = 35

else:
    print('wrong dataset')

if args.alignment == 'align':
    args.text_length = args.max_seq_length
    args.visual_length = args.max_seq_length
    args.acoustic_length = args.max_seq_length
else:
    args.text_length = args.max_seq_length
    args.visual_length = 500
    args.acoustic_length = 375



def train_epoch(model: nn.Module, train_dataloader: DataLoader, epoch=None):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # 
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    adjust_learning_rate(optimizer, epoch, args)  
 
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    embeddings = []
    mm_labels = []

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs, h = model(
            input_ids,
            visual,
            acoustic,
            label_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
        )

        logits = outputs

        loss_fct = L1Loss()
        loss_all = loss_fct(logits.view(-1), label_ids.view(-1))
        
        optimizer.zero_grad()
        loss_all.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
        embeddings.append(h.detach().cpu().numpy())
        mm_labels.append(label_ids.detach().cpu().numpy())


        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        tr_loss += loss_all.item()
        nb_tr_steps += 1
    embeddings = np.concatenate(embeddings)
    mm_labels = np.concatenate(mm_labels)
    return tr_loss / nb_tr_steps, embeddings, mm_labels


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_steps = 0
    embeddings = []
    mm_labels = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs,h = model.test(
                input_ids,
                 visual,
                 acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

            logits = outputs

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            embeddings.append(h.detach().cpu().numpy())
            mm_labels.append(label_ids.detach().cpu().numpy())

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1
    embeddings = np.concatenate(embeddings)
    mm_labels = np.concatenate(mm_labels)
    return dev_loss / nb_dev_steps, embeddings, mm_labels


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs, h = model.test(
                input_ids,
                 visual,
                 acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

            logits = outputs

            logits = logits.detach().cpu().numpy()
            
            label_ids = label_ids.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)
     
    preds = np.array(preds)
    labels = np.array(labels)

    return preds, labels


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    test_preds, test_truth = test_epoch(model, test_dataloader)
    mae = np.mean(np.absolute(test_preds - test_truth)) 
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    
    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or use_zero])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    binary_truth_o = (test_truth[non_zeros] > 0) # 
    binary_preds_o = (test_preds[non_zeros] > 0) # 
    acc2_non_zero = accuracy_score(binary_truth_o, binary_preds_o)
    f_score_non_zero = f1_score(binary_truth_o, binary_preds_o,  average='weighted')
    

    binary_truth = (test_truth >= 0) # 
    binary_preds = (test_preds >= 0) # 
    acc2 = accuracy_score(binary_truth, binary_preds) # 
    f_score = f1_score(binary_truth, binary_preds, average='weighted')
    f_score_bias = f1_score((test_preds > 0), (test_truth >= 0), average='weighted')

    return mae, corr, mult_a7, mult_a5, acc2_non_zero, f_score_non_zero, acc2, f_score, test_preds, test_truth


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader
):
    valid_losses = []
    test_accuracies = []
    f1_scores = []
    best_loss = 10
    best_mae=0
    best_corr=0
    best_acc7=0
    best_acc5=0
    best_acc2_non_zero=0
    best_f_score_non_zero=0
    best_acc2=0
    best_f_score=0
    
    for epoch_i in range(int(args.n_epochs)):
        train_loss, train_mm_embeddings, train_labels = train_epoch(model, train_dataloader, epoch_i)
        valid_loss, valid_mm_embeddings, valid_labels = eval_epoch(model, validation_dataloader)
        test_mae, test_corr, test_acc7, test_acc5, test_acc2_non_zero, test_f_score_non_zero, test_acc2, test_f_score, test_preds, test_labels = test_score_model(
            model, test_data_loader
        )

        print(
            "epoch:{}, train_loss:{:.4f}, valid_loss:{:.4f}, test_acc2:{:.4f}".format(
                epoch_i, train_loss, valid_loss, test_acc2
            )
        )
 

        print(
            "current mae:{:.4f}, current corr:{:.4f}, acc7:{:.4f}, acc5:{:.4f},acc2_non_zero:{:.4f}, f_score_non_zero:{:.4f}, acc2:{:.4f}, f_score:{:.4f}".format(
                test_mae, test_corr, test_acc7, test_acc5, test_acc2_non_zero, test_f_score_non_zero, test_acc2, test_f_score
            )
        )
        print(f"Saved model at ./CMAD_sentiment/Teacher_Model/mosei.pkl!")# modify for rename
        torch.save(model.state_dict(),f'./CMAD_sentiment/Teacher_Model/mosei.pkl')

        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc2)
        f1_scores.append(test_f_score_non_zero)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_mae = test_mae
            best_corr = test_corr
            best_acc7 = test_acc7
            best_acc5 = test_acc5
            best_acc2_non_zero = test_acc2_non_zero
            best_f_score_non_zero = test_f_score_non_zero
            best_acc2 = test_acc2
            best_f_score = test_f_score
            
            
            
        print(
            "best mae:{:.4f}, current corr:{:.4f}, acc7:{:.4f}, acc5:{:.4f},acc2_non_zero:{:.4f}, f_score_non_zero:{:.4f}, acc2:{:.4f}, f_score:{:.4f}".format(
            best_mae, best_corr, best_acc7, best_acc5, best_acc2_non_zero, best_f_score_non_zero, best_acc2, best_f_score
            )
        )


def main():
    set_random_seed(args.seed)
    start_time = time.time()

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model = prep_for_training(
        num_train_optimization_steps)#

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader
    )
    end_time = time.time()
    print('Cost time of 100 epochs: %s ms' %((end_time - start_time) * 1000))


if __name__ == "__main__":
    main()



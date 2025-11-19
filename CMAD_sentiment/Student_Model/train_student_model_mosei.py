import sys
sys.path.append('..')

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import csv
import time
import datetime
import os
import torch.nn as nn
import torch.nn.functional as F
from loss import *
from eval_metrics import eval_results
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from teacher_model import TeacherModel
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, get_cosine_schedule_with_warmup
from transformers.optimization import AdamW
from utils import adjust_learning_rate

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = torch.device("cuda:0")
import warnings
warnings.filterwarnings("ignore")



def train_model(net_dict, train_loader, dev_loader, test_loader, args):
    '''
    :param model:
    :param optimizer:
    :param train_loader:
    :param dev_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)

    student_model = net_dict['snet']
    teacher_model = net_dict['tnet']

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # 
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # Training initialization
    epoch_num = args.n_epochs

    epoch = 0
    F1_best = 0

    # in the first several epochs, the weights of each modality is the same
    # default_weights = torch.ones(3).to(DEVICE)#, requires_grad=True
    # tensor combinations
    combinations = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]]).to(DEVICE)
    epoch_stu_ps = []
    epoch_tea_gt_loss = []
    epoch_stu_gt_loss = []


    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            student_model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retraining")
    # Train
    
    while epoch < epoch_num:
        train_loss_all = 0.0
        epoch_spe_loss = 0.0
        epoch_auxi_loss = 0.0
        epoch_wei_mse_loss = 0.0

        student_model.p = [0, 0, 0]
        start = datetime.datetime.now()
        
        if epoch == 0:
            default_mar_weights = torch.ones(7).to(DEVICE)
        elif epoch < args.begin_epoch:
            pt_stu_ps = torch.cat(epoch_stu_ps, dim=0)
            pt_tea_gt_loss = torch.cat(epoch_tea_gt_loss, dim=0)
            pt_stu_gt_loss = torch.cat(epoch_stu_gt_loss, dim=0)
            pt_mar_weights = combination_importance(pt_stu_gt_loss, pt_tea_gt_loss, pt_stu_ps, combinations, args)
            pt_mar_weights = pt_mar_weights * args.weights
            default_mar_weights = torch.ones(7).to(DEVICE)
        elif epoch == args.begin_epoch:
            epoch_stu_ps = torch.cat(epoch_stu_ps, dim=0)
            epoch_tea_gt_loss = torch.cat(epoch_tea_gt_loss, dim=0)
            epoch_stu_gt_loss = torch.cat(epoch_stu_gt_loss, dim=0)
            default_mar_weights = combination_importance(epoch_stu_gt_loss, epoch_tea_gt_loss, epoch_stu_ps, combinations, args)
            default_mar_weights = default_mar_weights * args.weights
            epoch_stu_ps = []
            epoch_tea_gt_loss = []
            epoch_stu_gt_loss = []
        else:
            default_mar_weights = default_mar_weights
        
        # for the first epoch, use default loss
        for step, batch in enumerate(tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            data_read_time = (datetime.datetime.now() - start)
            optimizer.zero_grad()
            # extract the features and the logits
            tea_logits, tea_hidden  = teacher_model(input_ids,
            visual,
            acoustic,
            label_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,)
            
            stu_logits, stu_hidden, ps = student_model(input_ids,
            visual,
            acoustic,
            label_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,)
            
            # weighted_mse loss
            wei_mse_loss, simis, mses = compute_weighted_mse_loss(stu_hidden, tea_hidden, args.tau)


            # AUX LOSS
            task_loss_fc = nn.L1Loss(reduction='none')
            stu_logits = stu_logits.view(-1)
            tea_logits = tea_logits.view(-1)
            loss_auxi = regression_loss(stu_logits, tea_logits, args.temperature)# [64,64]
            label_ids = label_ids.view(-1)
            loss_task = task_loss_fc(stu_logits, label_ids)#, args.temperature) 
            batch_loss_tea_gt = task_loss_fc(tea_logits, label_ids)
            batch_loss_std_gt = task_loss_fc(stu_logits, label_ids)
                
            # TASK-SPECIFIC LOSS    
            loss_task = batch_loss_std_gt

            epoch_stu_ps.append(ps)
            epoch_tea_gt_loss.append(batch_loss_tea_gt)
            epoch_stu_gt_loss.append(batch_loss_std_gt)
            
            mask_new = (ps.unsqueeze(1) == combinations).all(dim=2).float()
            weighted_auxi_loss = torch.einsum('i,ij,j->i', loss_auxi, mask_new, default_mar_weights)
            mar_auxi_loss = weighted_auxi_loss.mean()
            weighted_spe_loss = torch.einsum('i,ij,j->i', loss_task, mask_new, default_mar_weights)
            mar_spe_loss = weighted_spe_loss.mean()
            loss_all = args.delta * wei_mse_loss + mar_spe_loss + args.gamma * mar_auxi_loss

            train_loss_all += loss_all.item()
            epoch_spe_loss += mar_spe_loss.item()
            epoch_auxi_loss += mar_auxi_loss.item()
            epoch_wei_mse_loss += wei_mse_loss.item()
            adjust_learning_rate(optimizer, epoch, args) 
            loss_all.backward()
            optimizer.step()

        student_model.p = [1, 1, 1]
        print("-" * 50 + 'DEV--[1,1,1]')
        acc_all, f1_all = evulate(model=student_model, loader=dev_loader, args=args)
        student_model.p = [1, 0, 0]
        print("-" * 50 + 'DEV--[1,0,0]')
        acc_1, f1_1 = evulate(model=student_model, loader=dev_loader, args=args)
        student_model.p = [0, 1, 0]
        print("-" * 50 + 'DEV--[0,1,0]')
        acc_2, f1_2 = evulate(model=student_model, loader=dev_loader, args=args)
        student_model.p = [0, 0, 1]
        print("-" * 50 + 'DEV--[0,0,1]')
        acc_3, f1_3 = evulate(model=student_model, loader=dev_loader, args=args)
        student_model.p = [1, 1, 0]
        print("-" * 50 + 'DEV--[1,1,0]')
        acc_4, f1_4 = evulate(model=student_model, loader=dev_loader, args=args)
        student_model.p = [1, 0, 1]
        print("-" * 50 + 'DEV--[1,0,1]')
        acc_5, f1_5 = evulate(model=student_model, loader=dev_loader, args=args)
        student_model.p = [0, 1, 1]
        print("-" * 50 + 'DEV--[0,1,1]')
        acc_6, f1_6 = evulate(model=student_model, loader=dev_loader, args=args)
        print('DEV--avg_acc', (acc_all + acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6) / 7.0)
        print('DEV--avg_f1', (f1_all + f1_1 + f1_2 + f1_3 + f1_4 + f1_5 + f1_6) / 7.0)

        F1_test = f1_all
        print("Epoch {}, spe_loss={:.5f}, auxi_loss={:.5f},  wei_mse_loss={:.5f}".format(epoch, epoch_spe_loss / len(train_loader), epoch_auxi_loss / len(train_loader),
              epoch_wei_mse_loss / len(train_loader)))

        print("Epoch {}, loss={:.5f}, F1_test={:.5f},  F1_best={:.5f}".format(epoch, train_loss_all / len(train_loader), F1_test, F1_best))
        

        if F1_test > F1_best:
            F1_best = F1_test
            # save_path = os.path.join(args.model_root, args.dataset + '_f1_best_' + '.pth')
            # torch.save(student_model.state_dict(), save_path)

            student_model.p = [1, 1, 1]
            print("-" * 50 + 'TEST--[1,1,1]')
            acc_all, f1_all = evulate(model=student_model, loader=test_loader, args=args)
            student_model.p = [1, 0, 0]
            print("-" * 50 + 'TEST--[1,0,0]')
            acc_1, f1_1 = evulate(model=student_model, loader=test_loader, args=args)
            student_model.p = [0, 1, 0]
            print("-" * 50 + 'TEST--[0,1,0]')
            acc_2, f1_2 = evulate(model=student_model, loader=test_loader, args=args)
            student_model.p = [0, 0, 1]
            print("-" * 50 + 'TEST--[0,0,1]')
            acc_3, f1_3 = evulate(model=student_model, loader=test_loader, args=args)
            student_model.p = [1, 1, 0]
            print("-" * 50 + 'TEST--[1,1,0]')
            acc_4, f1_4 = evulate(model=student_model, loader=test_loader, args=args)
            student_model.p = [1, 0, 1]
            print("-" * 50 + 'TEST--[1,0,1]')
            acc_5, f1_5 = evulate(model=student_model, loader=test_loader, args=args)
            student_model.p = [0, 1, 1]
            print("-" * 50 + 'TEST--[0,1,1]')
            acc_6, f1_6 = evulate(model=student_model, loader=test_loader, args=args)
            print('TEST--avg_acc', (acc_all + acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6) / 7.0)
            print('TEST--avg_f1', (f1_all + f1_1 + f1_2 + f1_3 + f1_4 + f1_5 + f1_6) / 7.0)
            
        print("Epoch {}, loss={:.5f}, F1_test={:.5f},  F1_best={:.5f}".format(epoch, train_loss_all / len(train_loader), F1_test, F1_best))

   
        train_state = {
            "Epoch": epoch,
            "model_state": student_model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "args": args
            }

        epoch = epoch + 1


def evulate(model, loader, args):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :return F1 score, accuracy, float
    """
    mode_saved = model.training#false
    model.train(False)
    model.to(DEVICE)
    results = []
    truths = []
    for step, batch in enumerate(loader):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        with torch.no_grad():
            outputs = model(input_ids,
            visual,
            acoustic,
            label_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,)
            
            if isinstance(outputs, tuple):
                preds = outputs[-3]
            
            preds = preds.view(-1)
            label_ids = label_ids.view(-1)
            results.append(preds)
            truths.append(label_ids)
    results = torch.cat(results)
    truths = torch.cat(truths)
    acc, f1 = eval_results(results, truths, args)
    return acc, f1

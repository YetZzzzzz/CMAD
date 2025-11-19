import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
# DEVICE = torch.device("cuda:2")
import sys
sys.path.append('..')


def modality_drop(modal1, modal2, modal3, p, args):
    # The code is adapted from MMANet: https://github.com/shicaiwei123/MMANet-CVPR2023/blob/main/classification/lib/model_arch.py
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    index_list = [x for x in range(7)]
    # for training with uncertain conditions
    # save ps for uncertain modalities reweighting
    if p == [0, 0, 0]:
        p = []
        prob = np.array((2 / 14, 3 / 14, 3 / 14, 1 / 14, 1 / 14, 3 / 14, 1 / 14))
        # prob = np.array((1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7))# 2/14, 3/14, 3/14, 1/14, 1/14, 3/14, 1/14,1 / 28, 8 / 28, 8 / 28, 1 / 28, 1 / 28, 8 / 28, 1 / 28
        for i in range(modal1.shape[0]):# bsz
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])# store each missing conditions
            
        p = np.array(p)
        p = torch.from_numpy(p)# adds the missing conditions into the list
        p1 = torch.unsqueeze(p, 2)
        p1 = torch.unsqueeze(p1, 3)
        # p1 = torch.unsqueeze(p1, 4)
    # for testing with fixed conditions
    else:
        p = p
        # print(p)
        p = [p * modal1.shape[0]]
        # print(p)
        p = np.array(p).reshape(modal1.shape[0], 3)# for 3 modalities
        p = torch.from_numpy(p)
        p1 = torch.unsqueeze(p, 2)
        p1 = torch.unsqueeze(p1, 3)
        # p1 = torch.unsqueeze(p1, 4)

        # print(p[:, 0], p[:, 1], p[:, 2])
    p1 = p1.float().to(modal1.device)#.to(DEVICE)
    p = p.float().to(modal1.device)#.to(DEVICE)

    modal1 = modal1 * p1[:, 0].to(modal1.device)
    modal2 = modal2 * p1[:, 1].to(modal2.device)
    modal3 = modal3 * p1[:, 2].to(modal3.device)

    return modal1, modal2, modal3, p
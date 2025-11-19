import torch
import torch.nn as nn
import torch.nn.functional as F

def bi_kl_divergence(p,q, reduction='none'):
    # distance between two distributions
    kl_pq = F.kl_div(F.log_softmax(p), F.softmax(q), reduction=reduction).sum(dim=-1)
    kl_qp = F.kl_div(F.log_softmax(q), F.softmax(p), reduction=reduction).sum(dim=-1)
    bidirectional_kl = (kl_pq + kl_qp) / 2
    return bidirectional_kl

def compute_similarity(x, y):
    """compute the cosine similarity between two tensors"""
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    similarity = torch.matmul(x_norm, y_norm.transpose(0, 1))#F.cosine_similarity(stu_repres.unsqueeze(1), tea_repres.unsqueeze(0), dim=2)
    return similarity


def compute_weighted_mse_loss(stu_repres, tea_repres, tau=0.1):
    # 1. compute the mse loss
    mse_loss = F.mse_loss(stu_repres, tea_repres, reduction='none').mean(dim=1)
    
    # 2. similarity matrix
    sim_stu_tea = compute_similarity(stu_repres, tea_repres)
    sim_tea_tea = compute_similarity(tea_repres, tea_repres)
    
    # 3. compute the distance between the matrixes
    sim_diff = torch.abs(sim_tea_tea - sim_stu_tea)
    diagonal_sum = torch.diagonal(sim_diff).sum()
    total_sum = sim_diff.sum()
    loss_total = diagonal_sum / total_sum

    # 5. bi_kl_divergence
    bi_kl_loss = bi_kl_divergence(sim_stu_tea/tau , sim_tea_tea/tau)
    
    # 6. combine together
    loss_alls = (bi_kl_loss * mse_loss).mean() + loss_total
    
    return loss_alls, bi_kl_loss.mean()+loss_total, mse_loss.mean()

def compute_correlation_loss(stu_repres, tea_repres, tau=0.1):
    mse_loss = F.mse_loss(stu_repres, tea_repres, reduction='none').mean(dim=1)
     # 2. similarity matrix
    sim_stu_stu = compute_similarity(stu_repres, stu_repres)
    sim_tea_tea = compute_similarity(tea_repres, tea_repres)
    bi_kl_loss = bi_kl_divergence(sim_stu_stu/tau , sim_tea_tea/tau)
    # # 3. compute the distance between the matrixes
    # sim_diff = torch.abs(sim_tea_tea - sim_stu_tea)
    # diagonal_sum = torch.diagonal(sim_diff).sum()
    loss_alls = bi_kl_loss.mean()
    return loss_alls, bi_kl_loss.mean(), mse_loss.mean()

def Print_modality(stu_gt_dis, tea_gt_dis, ps, combinations, args):
    with torch.no_grad():
        distance_tensor = stu_gt_dis
        mask = (ps.unsqueeze(1) == combinations).all(dim=2).float()
        if args.dataset == 'iemocap':
            mask = mask.repeat_interleave(4, dim=0)
            loss_sum = torch.mm(distance_tensor.unsqueeze(1).t(), mask).squeeze()
        else:
            loss_sum = torch.einsum('i,ij->j', distance_tensor, mask)
            # loss_sum = torch.mm(distance_tensor.t(), mask).squeeze()
        count = mask.sum(dim=0)
        # compute the avg loss
        avg_loss = torch.where(count > 0, loss_sum / count, torch.zeros_like(loss_sum))
        
        norm_avg = avg_loss / avg_loss.max()

        importance_scores = norm_avg ** 2
        # importance_scores = F.softmax(norm_avg, dim=0)
        # print('softmax_scores', importance_scores)
        # for extraction the combination loss
        # use the following code , if the new tensor is tensor_new
        # mask_new = (tensor_new.unsqueeze(1) == tensor0).all(dim=2).float()
        # weighted_loss = torch.mm(tensornew_loss.t(), mask_new).squeeze() * importance_scores
        # weighted_loss_sum = weighted_loss.sum()
    return avg_loss


def combination_importance(stu_gt_dis, tea_gt_dis, ps, combinations, args):
    # This module computes the weights for each modality combination
    # for 3 modalities, the weights for 7 combinations are computated based on the average performances in all samples
    # input: stu_gt_dis:the distance between teacher loss and ground truth
    #        tea_gt_dis: the distance between student loss and ground truth
    #        ps: the combination of each sample
    # output: weights of 7 combinations in certain order, the weights of the former epoch
    # dynamically update the weights
    # initial the 7 combinations
    # tensor = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0,1,1], [0,0,1], [1,0,0]], device=stu_gt_dis.device)
    # tensorim = torch.zeros(7, device=stu_gt_dis.device, requires_grad=True)
    # compute the distance between two distances, if the student is better than teacher, then keeps by setting distance as 0, otherwise set as distance
    with torch.no_grad():
        distance_tensor = torch.relu(stu_gt_dis - tea_gt_dis)
        # create the mask to identify the combination each sample belongs to
        mask = (ps.unsqueeze(1) == combinations).all(dim=2).float()
        if args.dataset == 'iemocap':
            mask = mask.repeat_interleave(4, dim=0)
        # compute sample count and the whole loss of each combination
            loss_sum = torch.mm(distance_tensor.unsqueeze(1).t(), mask).squeeze()
        else:
            loss_sum = torch.einsum('i,ij->j', distance_tensor, mask)
            # loss_sum = torch.mm(distance_tensor.t(), mask).squeeze()
        count = mask.sum(dim=0)
        # compute the avg loss
        avg_loss = torch.where(count > 0, loss_sum / count, torch.zeros_like(loss_sum))
        norm_avg = avg_loss / avg_loss.max()
        importance_scores = norm_avg ** 2
    return importance_scores

def combination_importance_wt(stu_gt_dis, tea_gt_dis, ps, combinations, args):
    # This module computes the weights for each modality combination
    # for 3 modalities, the weights for 7 combinations are computated based on the average performances in all samples
    # input: stu_gt_dis:the distance between teacher loss and ground truth
    #        tea_gt_dis: the distance between student loss and ground truth
    #        ps: the combination of each sample
    # output: weights of 7 combinations in certain order, the weights of the former epoch
    # dynamically update the weights
    # initial the 7 combinations
    # tensor = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0,1,1], [0,0,1], [1,0,0]], device=stu_gt_dis.device)
    # tensorim = torch.zeros(7, device=stu_gt_dis.device, requires_grad=True)
    # compute the distance between two distances, if the student is better than teacher, then keeps by setting distance as 0, otherwise set as distance
    with torch.no_grad():
        # distance_tensor = torch.relu(stu_gt_dis - tea_gt_dis)
        
        distance_tensor = stu_gt_dis
        # print('distance_tensor_shape', distance_tensor.shape)
        # create the mask to identify the combination each sample belongs to
        mask = (ps.unsqueeze(1) == combinations).all(dim=2).float()
        if args.dataset == 'iemocap':
            mask = mask.repeat_interleave(4, dim=0)
            # print(mask.shape)
        # compute sample count and the whole loss of each combination
            loss_sum = torch.mm(distance_tensor.unsqueeze(1).t(), mask).squeeze()
        else:
            loss_sum = torch.einsum('i,ij->j', distance_tensor, mask)
            # loss_sum = torch.mm(distance_tensor.t(), mask).squeeze()
        count = mask.sum(dim=0)
        # compute the avg loss
        avg_loss = torch.where(count > 0, loss_sum / count, torch.zeros_like(loss_sum))
        # use softmax to normalize the average loss of the 7 combinations, the final results are the weights of each combination 
        # the larger the number, the larger the distance, the harder the samples or the combination
        norm_avg = avg_loss / avg_loss.max()
        # print('norm_avg', norm_avg)
        # importance_scores = (avg_loss / avg_loss.mean()) ** 2 *10

        importance_scores = norm_avg ** 2
        # importance_scores = F.softmax(norm_avg, dim=0)
        # print('softmax_scores', importance_scores)
        # for extraction the combination loss
        # use the following code , if the new tensor is tensor_new
        # mask_new = (tensor_new.unsqueeze(1) == tensor0).all(dim=2).float()
        # weighted_loss = torch.mm(tensornew_loss.t(), mask_new).squeeze() * importance_scores
        # weighted_loss_sum = weighted_loss.sum()
    return importance_scores


def cross_correlation_distillation(stu_repres, tea_repres, temperature):
    # compute the corresponding distance between representations
    mse_loss = F.mse_loss(stu_repres, tea_repres)
    # compute the similarity matrix
    similarity_matrix = F.cosine_similarity(stu_repres.unsqueeze(1), tea_repres.unsqueeze(0), dim=2)
    diagonal_similarities = similarity_matrix.diag()
    off_diagonal_similarities = similarity_matrix - torch.eye(batchsize)
    # compute the exp similarity of the diagonal and off-diagonal elements
    diagonal_exp_similarities = torch.exp(diagonal_similarities / temperature)
    off_diagonal_exp_similarities = torch.exp(off_diagonal_similarities / temperature)
    # compute the loss of the diagonal
    diagonal_loss = -torch.log(diagonal_exp_similarities / (diagonal_exp_similarities + off_diagonal_exp_similarities.sum(dim=1)))
    contrastive_loss = diagonal_loss.mean()
    total_loss = mse_loss + contrastive_loss
    return total_loss


def regression_loss(logits_student, logits_teacher, temperature):#, alpha, beta, temperature):
    # for uniform 
    # regression task needs not target, alpha and beta
    pred_student = logits_student / temperature
    pred_teacher = logits_teacher / temperature

    # mse loss
    distill_loss = F.mse_loss(pred_student, pred_teacher, reduction = 'none') * (temperature**2)
    return distill_loss


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    # """Decoupled Knowledge Distillation(CVPR 2022)
    # adapted from https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    # compute tckd_loss for each sample
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none')
        * (temperature**2)
    )
    tckd_loss = tckd_loss.sum(dim=1)  # Sum over the classes to get per-sample loss
    # tckd_loss = (
    #     F.kl_div(log_pred_student, pred_teacher, size_average=False, reduction='none')
    #     * (temperature**2)
    #     / target.shape[0]
    # )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none')
        * (temperature**2)
    )
    nckd_loss = nckd_loss.sum(dim=1)
    # nckd_loss = (
    #     F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False, reduction='none')
    #     * (temperature**2)
    #     / target.shape[0]
    # )
    # Combine the losses for each sample
    per_sample_loss = alpha * tckd_loss + beta * nckd_loss
    return per_sample_loss



def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt



    
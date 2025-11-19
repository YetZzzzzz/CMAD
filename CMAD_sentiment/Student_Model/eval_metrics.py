import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def eval_results(results, truths, args):
    if args.dataset == 'iemocap':
        acc, f1 = eval_iemocap_avg(results, truths)
    elif args.dataset == 'mosi':
        acc, f1 = eval_mosi_binary(results, truths)
    else:
        acc, f1 = eval_mosei_senti_binary(results, truths)
    return acc, f1

def eval_iemocap_avg(results, truths):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    total_f1 = 0.0
    total_acc = 0.0
    num_classes = 4
    test_preds = results.view(-1, 4,2).cpu().detach().numpy()
    test_truth = truths.view(-1, 4).cpu().detach().numpy()
    
    for emo_ind in range(4):
        test_preds_i = np.argmax(test_preds[:,emo_ind], axis=1)
        test_truth_i = test_truth[:,emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        total_f1 += f1
        total_acc += acc
    avg_f1 = total_f1 / num_classes
    avg_acc = total_acc / num_classes
    print("\n Average Metrics:")
    print("  - Average F1 Score: ", avg_f1)
    print("  - Average Accuracy: ", avg_acc)
    return avg_acc, avg_f1
    

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)

# here changes to mosei results into the positive/negative ones for a fair comparison
def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("F1 score: ", f_score)
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))

    print("-" * 50)


def eval_mosei_senti_binary(preds, truths):
    preds = preds.view(-1).cpu().detach().numpy()
    truths = truths.view(-1).cpu().detach().numpy()
    mae = np.mean(np.absolute(preds - truths))   # Average L1 distance between preds and truths
    corr = np.corrcoef(preds, truths)[0][1]
    non_zeros = np.array(
        [i for i, e in enumerate(truths) if e != 0])

    binary_truth_o = (truths[non_zeros] > 0) # 
    binary_preds_o = (preds[non_zeros] > 0) # 
    acc2_non_zero = accuracy_score(binary_truth_o, binary_preds_o)
    f_score_non_zero = f1_score(binary_truth_o, binary_preds_o,  average='weighted')
    print("-" * 50)
    print('Accuracy of negative/positive: ', acc2_non_zero)
    print('F1 score of negative/positive: ', f_score_non_zero)
    print("-" * 50)
    return acc2_non_zero, f_score_non_zero

def eval_mosi_binary(results, truths):
    return eval_mosei_senti_binary(results, truths)
    
def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4,2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()
        
        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
            test_truth_i = test_truth[:,emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()
        
        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds,axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)




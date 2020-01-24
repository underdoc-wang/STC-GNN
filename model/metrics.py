import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, fbeta_score, \
     roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, \
     log_loss, mean_absolute_error, recall_score



# global variables
H, W = (20, 10)
C_lst = ['Violation', 'Misdemeanor', 'Felony', 'EMS', 'Rescue', 'Fire']
beta = 2          # for F-beta score: beta stands for weight of recall(FN) over precision(FP)



def eval_metrics(out_dir, y_true, y_pred_proba, ha):
    '''
    # global model
    # evaluate on macro/micro-F1/F2, recall, AUC/AP, BCE, MAE

    :param out_dir:
    :param y_true: ground truth (t+1) - (T, 20, 10, 6)
    :param y_pred: prediction (t+1) - (T, 20, 10, 6)
    :param ha: regional occurence rate of each category - threshold for y_pred
    :return:
    '''
    assert y_true.shape == y_pred_proba.shape, f"Prediction's dimension doesn't match ground truth \n" \
                                               f"truth: {y_true.shape}, pred: {y_pred_proba.shape}"

    with open(out_dir + '/eval_metrics.txt', 'a') as wf:
        wf.write(' '.join(['*'*10, 'model evaluation started at', time.ctime(), '*'*10]) + '\n')
        print(' '.join(['*'*10, 'model evaluation started at', time.ctime(), '*'*10]))

    # loop timestamp - proba to binary label
    y_pred = []
    for t in range(y_pred_proba.shape[0]):
        y_pred.append(np.where(y_pred_proba[t,:,:,:] >= ha, 1, 0))
    y_pred = np.array(y_pred)

    tp_lst, fn_lst, fp_lst = [], [], []     # to calculate Macro-F1/F2
    f1_lst = []          # to calculate Micro-F1
    f2_lst = []          # to calculate Micro-F2

    # loop category
    for c in range(y_true.shape[-1]):
        y_true_c = y_true[:,:,:,c].flatten()
        y_pred_c = y_pred[:,:,:,c].flatten()
        y_pred_proba_c = y_pred_proba[:,:,:,c].flatten()

        # single category evaluation
        print(f"{C_lst[c]} \n \
                             F1-score: {round(f1_score(y_true_c, y_pred_c), 5)} \n \
                             F2-score: {round(fbeta_score(y_true_c, y_pred_c, 2), 5)} \n \
                             AUC score: {round(roc_auc_score(y_true_c, y_pred_proba_c), 5)} \n \
                             AP score: {round(average_precision_score(y_true_c, y_pred_proba_c), 5)} \n")
        # check recall vs. precision
        f1_report = classification_report(y_true_c, y_pred_c, labels=np.unique(y_pred_c))
        print(f1_report)

        # confusion matrix
        print('confusion matrix: \n', confusion_matrix(y_true_c, y_pred_c), '\n')

        TN, FP, FN, TP = confusion_matrix(y_true_c, y_pred_c).ravel()
        tp_lst.append(TP)
        fn_lst.append(FN)
        fp_lst.append(FP)
        f1_lst.append(2 * TP / (2 * TP + FN + FP))
        f2_lst.append((1 + beta ** 2) * TP / ((1 + beta ** 2) * TP + (beta ** 2) * FN + FP))
    # F1
    macro_f1 = 2 * sum(tp_lst) / (2 * sum(tp_lst) + sum(fn_lst) + sum(fp_lst))
    micro_f1 = sum(f1_lst) / len(f1_lst)
    # F2
    macro_f2 = (1 + beta ** 2) * sum(tp_lst) / ((1 + beta ** 2) * sum(tp_lst) + (beta ** 2) * sum(fn_lst) + sum(fp_lst))
    micro_f2 = sum(f2_lst) / len(f2_lst)

    macro_f1, micro_f1, macro_f2, micro_f2 = round(macro_f1, 5), round(micro_f1, 5), round(macro_f2, 5), round(micro_f2, 5)

    # flatten arrays for global metrics
    y_true, y_pred_proba, y_pred = y_true.flatten(), y_pred_proba.flatten(), y_pred.flatten()

    # overall recall
    RECALL = round(recall_score(y_true, y_pred), 5)

    # overall AUC / AP
    AUC = round(roc_auc_score(y_true, y_pred_proba), 5)
    AP = round(average_precision_score(y_true, y_pred_proba), 5)

    # Cross-Entropy
    CE = round(log_loss(y_true=y_true, y_pred=y_pred_proba), 5)
    # Mean Absolute Error
    MAE = round(mean_absolute_error(y_true=y_true, y_pred=y_pred_proba), 5)

    # output
    with open(out_dir + '/eval_metrics.txt', 'a') as wf:
        print(f' Macro-F1: {macro_f1}, Micro-F1: {micro_f1} \n Macro-F2: {macro_f2}, Micro-F2: {micro_f2} \n'
              f' Overall recall: {RECALL} \n'
              f' Overall AUC score: {AUC}, AP score: {AP} \n'
              f' Binary Cross-Entropy: {CE}, MAE: {MAE} \n')
        wf.write(f' Macro-F1: {macro_f1}, Micro-F1: {micro_f1} \n Macro-F2: {macro_f2}, Micro-F2: {micro_f2} \n'
                 f' Overall recall: {RECALL} \n'
                 f' Overall AUC score: {AUC}, AP score: {AP} \n'
                 f' Binary Cross-Entropy: {CE}, MAE: {MAE} \n')
        wf.write(' '.join(['*'*10, 'model evaluation ended at', time.ctime(), '*'*10]) + '\n \n')
        print(' '.join(['*'*10, 'model evaluation ended at', time.ctime(), '*'*10]))

    return None


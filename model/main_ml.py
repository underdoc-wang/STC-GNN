import os
import time
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, fbeta_score, \
     roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, \
     log_loss, mean_absolute_error, recall_score



def load_data(in_dir):
    data = np.load(in_dir)
    print('Loaded Emergency NYC data.. \n   Shape:', data.shape)

    # historical avg. - regional occurence rate
    ha = np.mean(data, axis=0)

    return data, ha


def split_data(data, test_ratio):
    test_len = int(data.shape[0] * test_ratio)
    trainSet = data[:-test_len, :,:,:]
    testSet = data[-test_len:, :,:,:]

    print('Train set shape:', trainSet.shape)
    print('Test set shape:', testSet.shape)

    return trainSet, testSet


def run_model(model_name, model_dir, trainSet, testSet, timestep):
    print(f'Running model {model_name}.. \n   Observing last {timestep} steps to predict next one step.')
    if model_name == 'VAR':
        from model.baseline.VAR import run_VAR
        y_true, y_pred = run_VAR(model_dir, trainSet, testSet, timestep)

    elif model_name == 'LR':
        from model.baseline.LR import run_LR
        y_true, y_pred = run_LR(model_dir, trainSet, testSet, timestep)

    elif model_name == 'LASSO':
        # TODO: LASSO - Cai 1/31
        pass
    elif model_name == 'SVM':
        # TODO: SVM - Cai 1/31
        pass
    else:
        raise Exception('Unknown model')

    return y_true, y_pred


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



# global
C_lst = ['Violation', 'Misdemeanor', 'Felony', 'EMS', 'Rescue', 'Fire']
beta = 2          # for F-beta score: beta stands for weight of recall(FN) over precision(FP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ML models for emergency prediction')
    parser.add_argument('-in', '--in_dir', type=str, help='Input directory', default='../data')
    parser.add_argument('-model', '--model_name', type=str, help='Choose prediction model',
                        choices=['LR', 'VAR', 'LASSO', 'SVM'], default='VAR')
    parser.add_argument('-t', '--delta_t', type=int, default=4, help='Time interval in hour(s)')
    parser.add_argument('-l', '--seq_len', type=int, default=6, help='Sequence length of observation steps')

    args = parser.parse_args()

    # input dir
    in_dir = os.path.join(args.in_dir, f'{args.delta_t}h', 'EmergNYC_bi_20x10.npy')
    # output dir
    out_dir = os.path.join('./baseline', args.model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load data
    data, ha = load_data(in_dir)

    # split train/test data
    trainSet, testSet = split_data(data, test_ratio=61/365)     # test on last 2 months

    # run model
    y_true, y_pred = run_model(args.model_name, out_dir, trainSet, testSet, args.seq_len)

    # evaluate prediction performance
    eval_metrics(out_dir, y_true, y_pred, ha)

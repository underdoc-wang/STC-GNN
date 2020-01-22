import os
import time
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, fbeta_score, \
     roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, \
     log_loss, mean_absolute_error, recall_score

from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler


def load_data(in_dir):
    data = np.load(in_dir)
    print('Loaded Emergency NYC data.. \n   Shape:', data.shape)

    # historical avg. - regional occurence rate
    ha = np.mean(data, axis=0)

    return data, ha


def getXSYS(data, len_test, len_seq):
    X_seq, Y_seq = [], []
    for i in range(data.shape[0] - len_seq):
        X_seq.append(data[i:i + len_seq])
        Y_seq.append(data[i + len_seq])

    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)
    print('X_sequence shape:', X_seq.shape,
          'Y_sequence shape:', Y_seq.shape)

    # get train/test sets
    print('Constructing train/test sets...')
    trainX, testX = X_seq[:-len_test], X_seq[-len_test:]
    trainY, testY = Y_seq[:-len_test], Y_seq[-len_test:]

    print('Train set shape:', trainX.shape, trainY.shape)
    print('Test set shape:', testX.shape, testY.shape)

    return (trainX, trainY), (testX, testY)


def get_model(model_name, timestep, n_lstm_layers, n_hidden_units):
    print(f'Building model {model_name}.. \n   Observing last {timestep} steps to predict next one step.')
    if model_name == 'GRU':
        from baseline.GRU import get_GRU
        model = get_GRU(timestep, n_lstm_layers, n_hidden_units)

    elif model_name == 'ConvLSTM':
        from baseline.ConvLSTM import get_ConvLSTM
        model = get_ConvLSTM(timestep, n_lstm_layers, n_hidden_units)

    else:
        raise Exception('Unknown model')

    # compile
    model.compile(loss = binary_crossentropy, optimizer = 'adam')
    model.summary()

    return model


def train_model(args, out_dir, model, trainSet, HA):
    # unpack
    lr, batchs, epochs, val_split = args.lr, args.batch_size, args.max_epoch, args.val_ratio
    trainX, trainY = trainSet

    # callbacks
    csv_logger = CSVLogger(out_dir + '/training.log')
    check_pointer = ModelCheckpoint(out_dir + '/model_temp.h5', verbose=1, save_best_only=True)
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    learning_rater = LearningRateScheduler(lambda epoch: lr)

    with open(out_dir + '/eval_metrics.txt', 'a') as wf:
        print(' '.join(['*' * 10, 'model training started at', time.ctime(), '*' * 10]))
        wf.write(' '.join(['*' * 10, 'model training started at', time.ctime(), '*' * 10]) + '\n')
        wf.write(f'Hyper-paramters: lr {lr}    batch_size {batchs}    epochs {epochs} \n')
        print(f'Hyper-paramters: lr {lr}    batch_size {batchs}    epochs {epochs}')

    # fit model
    history = model.fit(trainX, trainY, batch_size=batchs, epochs=epochs,
                        verbose=1, validation_split=val_split, shuffle=False,
                        callbacks=[csv_logger, check_pointer, early_stopper, learning_rater])
    # predict/evaluate
    predY = model.predict(trainX)
    eval_metrics(out_dir, trainY, predY, HA)

    model.save_weights(out_dir + '/trained_weights.h5', overwrite=True)

    return None


def test_model(out_dir, model, testSet, HA):
    # unpack
    testX, testY = testSet

    # load model weight
    model.load_weights(out_dir + '/trained_weights.h5')

    with open(out_dir + '/eval_metrics.txt', 'a') as wf:
        wf.write(' '.join(['*' * 10, 'model testing started at', time.ctime(), '*' * 10]) + '\n')
        print(' '.join(['*' * 10, 'model testing started at', time.ctime(), '*' * 10]))

    # predict/evaluate
    predY = model.predict(testX)
    eval_metrics(out_dir, testY, predY, HA)

    return None


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


def combo_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# global
C_lst = ['Violation', 'Misdemeanor', 'Felony', 'EMS', 'Rescue', 'Fire']
beta = 2          # for F-beta score: beta stands for weight of recall(FN) over precision(FP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DL models for emergency prediction')
    parser.add_argument('-in', '--in_dir', type=str, help='Input directory', default='../data')
    parser.add_argument('-model', '--model_name', type=str, help='Choose prediction model',
                        choices=['GRU', 'ConvLSTM'], default='GRU')
    parser.add_argument('-t', '--delta_t', type=int, default=4, help='Time interval in hour(s)')
    parser.add_argument('-l', '--seq_len', type=int, default=6, help='Sequence length of observation steps')
    parser.add_argument('--GPU', type=str, help='Specify which GPU to run with (-1 for run on CPU)', default='-1')
    parser.add_argument('-test_len', '--days_test', type=int, help='Specify how many days for test', default=61)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
    parser.add_argument('-batch', '--batch_size', type=int, default=64)
    parser.add_argument('-epoch', '--max_epoch', type=int, default=100)
    parser.add_argument('-split', '--val_ratio', type=float, help='Validate ratio', default=0.2)
    parser.add_argument('-unit', '--lstm_hidden_units', type=int, help='#Hidden units for LSTM', default=32)
    parser.add_argument('-layer', '--lstm_n_layers', type=int, help='#LSTM Layers', default=2)

    args = parser.parse_args()

    # GPU usage
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    if args.GPU == '-1':
        gpu_config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        gpu_config = tf.ConfigProto()
        #gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        gpu_config.gpu_options.allow_growth = True
        gpu_config.gpu_options.visible_device_list = args.GPU
    set_session(tf.Session(config=gpu_config))

    # input dir
    in_dir = os.path.join(args.in_dir, f'{args.delta_t}h', 'EmergNYC_bi_20x10.npy')
    # output dir
    out_dir = os.path.join('./baseline', args.model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load data
    data, ha = load_data(in_dir)

    # split train/test data
    len_test = int(args.days_test * 24 / args.delta_t)
    trainSet, testSet = getXSYS(data, len_test, args.seq_len)

    # get model
    model = get_model(args.model_name, args.seq_len, args.lstm_n_layers, args.lstm_hidden_units)

    # train/test model
    train_model(args, out_dir, model, trainSet, ha)
    test_model(out_dir, model, testSet, ha)
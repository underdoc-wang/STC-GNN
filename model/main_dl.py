import os
import time
import argparse
import numpy as np
import pandas as pd
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
from model.metrics import eval_metrics, H, W, C_lst


def load_data(in_dir):
    data = np.load(in_dir)
    print('Loaded Emergency NYC data.. \n   Shape:', data.shape)

    # historical avg. - regional occurence rate
    ha = np.mean(data, axis=0)

    return data, ha


def getXSYS(data, timestep):
    XS, YS = [], []
    for i in range(data.shape[0] - timestep):
        x = data[i : i + timestep, :, :, :]
        y = data[i + timestep, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)

    return XS, YS


def split_data(data, dates, delta_t, len_seq):
    assert len(dates) == 4, 'Invalid dates input.. Please input a sequence with four items.' \
                            ' Input example: -date 20150101 20150531 20150601 20150630'
    train_start, train_end, test_start, test_end = dates     # unpack dates
    day_timestep = int(24/delta_t)     # number of timesteps per day

    dates_range = pd.date_range('20150101', '20151231').strftime('%Y%m%d').tolist()     # date range covers entire dataset
    assert len(dates_range) == int(data.shape[0]/day_timestep)

    # train set
    start_index, end_index = dates_range.index(train_start), dates_range.index(train_end)
    trainSet = data[start_index*day_timestep:(end_index+1)*day_timestep]
    print(f'Train set {train_start}-{train_end}: {trainSet.shape}')
    # test set
    start_index, end_index = dates_range.index(test_start), dates_range.index(test_end)
    testSet = data[start_index*day_timestep:(end_index+1)*day_timestep]
    print(f'Test set {test_start}-{test_end}: {testSet.shape}')

    # get X/Y features
    trainX, trainY = getXSYS(trainSet, len_seq)
    testX, testY = getXSYS(testSet, len_seq)
    print('Train set shape: X/Y', trainX.shape, trainY.shape)
    print('Test set shape: X/Y', testX.shape, testY.shape)

    return (trainX, trainY), (testX, testY)


def get_model(model_name, timestep, n_layers, n_hidden_units, region_size, n_channel):
    print(f'Building model {model_name}.. \n   Observing last {timestep} steps to predict next one step.')
    if model_name == 'GRU':
        from baseline.GRU import get_GRU
        model = get_GRU(timestep, n_layers, n_hidden_units)

    elif model_name == 'ConvLSTM':
        from baseline.ConvLSTM import get_ConvLSTM
        model = get_ConvLSTM(timestep, n_layers, n_hidden_units)

    elif model_name == 'MiST':
        from baseline.MiST import get_MiST
        model = get_MiST(timestep, n_layers, n_hidden_units, region_size, n_channel)

    elif model_name == 'Hetero-ConvLSTM':
        from baseline.Hetero_ConvLSTM import get_Hetero_ConvLSTM
        model = get_Hetero_ConvLSTM(timestep, n_layers, n_hidden_units)

    else:
        raise Exception('Unknown model')

    # compile
    model.compile(loss = binary_crossentropy, optimizer = 'adam')
    model.summary()

    return model


def train_model(args, out_dir, model, trainSet):
    # unpack
    lr, batchs, epochs, val_split = args.learn_rate, args.batch_size, args.max_epoch, args.val_ratio
    trainX, trainY = trainSet

    # callbacks
    csv_logger = CSVLogger(out_dir + '/training.log')
    check_pointer = ModelCheckpoint(out_dir + '/model_temp.h5', verbose=1, save_best_only=True)
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    learning_rater = LearningRateScheduler(lambda epoch: lr)

    with open(out_dir + '/eval_metrics.txt', 'a') as wf:
        print(' '.join(['*' * 10, 'model training started at', time.ctime(), '*' * 10]))
        wf.write(' '.join(['*' * 10, 'model training started at', time.ctime(), '*' * 10]) + '\n')
        wf.write(f'Training configs: {args}\n')
        print(f'Training configs: {args}')

    # fit model
    history = model.fit(trainX, trainY, batch_size=batchs, epochs=epochs,
                        verbose=1, validation_split=val_split, shuffle=False,
                        callbacks=[csv_logger, check_pointer, early_stopper, learning_rater])
    model.save_weights(out_dir + '/trained_weights.h5', overwrite=True)

    # predict
    predY = model.predict(trainX)
    # check out for local model
    if len(trainY.shape) != 4:   # 2
        trainY = trainY.reshape((-1, H, W, len(C_lst)))
        predY = predY.reshape((-1, H, W, len(C_lst)))

    return trainY, predY


def test_model(out_dir, model, testSet):
    # unpack
    testX, testY = testSet

    # load model weight
    model.load_weights(out_dir + '/trained_weights.h5')

    with open(out_dir + '/eval_metrics.txt', 'a') as wf:
        wf.write(' '.join(['*' * 10, 'model testing started at', time.ctime(), '*' * 10]) + '\n')
        print(' '.join(['*' * 10, 'model testing started at', time.ctime(), '*' * 10]))

    # predict
    predY = model.predict(testX)
    # check out for local model
    if len(testY.shape) != 4:   # 2
        testY = testY.reshape((-1, H, W, len(C_lst)))
        predY = predY.reshape((-1, H, W, len(C_lst)))

    return testY, predY



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DL models for emergency prediction')
    parser.add_argument('--GPU', type=str, help='Specify which GPU to run with (-1 for run on CPU)', default='-1')
    parser.add_argument('-in', '--in_dir', type=str, help='Input directory', default='../data')
    parser.add_argument('-model', '--model_name', type=str, help='Choose prediction model',
                        choices=['GRU', 'ConvLSTM', 'MiST', 'Hetero-ConvLSTM'], default='Hetero-ConvLSTM')
    parser.add_argument('-t', '--delta_t', type=int, default=4, help='Time interval in hour(s)')
    parser.add_argument('-l', '--seq_len', type=int, default=6, help='Sequence length of observation steps')
    parser.add_argument('-date', '--dates', type=str, nargs='+',
                        help='Start/end dates of train/test sets. Test follows train.'
                             ' Example: -date 20150101 20150531 20150601 20150630',
                        default=['20150101', '20150531', '20150601', '20150630'])
    parser.add_argument('-epoch', '--max_epoch', type=int, default=100)
    parser.add_argument('-batch', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3)
    parser.add_argument('-split', '--val_ratio', type=float, help='Train-to-validation ratio', default=0.2)
    parser.add_argument('-unit', '--n_hidden_units', type=int,
                        help='#Hidden units for LSTM/ConvLSTM/Embedding/Attention', default=32)
    parser.add_argument('-layer', '--n_layers', type=int,
                        help='#Layers for LSTM/ConvLSTM/MLP', default=3)
    parser.add_argument('-rsize', '--region_size', type=int, help='Local region size for MiST', default=3)

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

    # output dir
    out_dir = os.path.join('./baseline', args.model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load data
    in_dir = os.path.join(args.in_dir, f'{args.delta_t}h', 'EmergNYC_bi_20x10.npy')
    data, ha = load_data(in_dir)
    if args.model_name == 'MiST':
        data_local = np.load(os.path.join(args.in_dir, f'{args.delta_t}h',
                                          f'EmergNYC_bi_20x10_{args.region_size}x{args.region_size}region.npy'))
        print('Loaded Emergency NYC data local.. \n   Shape: ', data_local.shape)
    elif args.model_name == 'Hetero-ConvLSTM':
        data_t_vari = np.load(os.path.join(args.in_dir, f'{args.delta_t}h', 'moblty', 'pcount-in_out.npy'))
        data_t_invari = np.load(os.path.join(args.in_dir, f'{args.delta_t}h', 'Hetero_invar_feat.npy'))
        print('Loaded time-variant features: ', data_t_vari.shape, '\n',
              '      time-invariant + spatial_graph features: ', data_t_invari.shape)

    # split train/test data
    #len_test = int(args.days_test * 24 / args.delta_t)
    if args.model_name not in ['MiST', 'Hetero-ConvLSTM']:
        trainSet, testSet = split_data(data, args.dates, args.delta_t, args.seq_len)
    elif args.model_name == 'MiST':
        from model.baseline.MiST import split_data_MiST
        trainSet, testSet = split_data_MiST(data_local, args.dates, args.delta_t, args.seq_len,
                                            args.region_size, len(C_lst))
    elif args.model_name == 'Hetero-ConvLSTM':
        from model.baseline.Hetero_ConvLSTM import split_data_Hetero
        trainSet, testSet = split_data_Hetero(data, data_t_vari, data_t_invari, args.dates, args.delta_t, args.seq_len)

    if args.model_name != 'Hetero-ConvLSTM':
        # get model
        model = get_model(args.model_name, args.seq_len, args.n_layers, args.n_hidden_units, args.region_size, len(C_lst))

        # train & evaluate
        trainY, predY = train_model(args, out_dir, model, trainSet)
        eval_metrics(out_dir, trainY, predY, ha)
        # test & evaluate
        testY, predY = test_model(out_dir, model, testSet)
        eval_metrics(out_dir, testY, predY, ha)

    else:
        assert len(trainSet) == len(testSet) == len(C_lst)

        train_y_true, train_y_pred = [], []
        test_y_true, test_y_pred = [], []

        for c in range(len(C_lst)):
            # get Hetero-ConvLSTM: category-separate train/test
            print(f'Processing category {C_lst[c]}..')
            locals()[f'model_{C_lst[c]}'] = get_model(args.model_name, args.seq_len, args.n_layers,
                                                      args.n_hidden_units, args.region_size, len(C_lst))
            trainY, predY = train_model(args, out_dir, locals()[f'model_{C_lst[c]}'], trainSet[c])
            train_y_true.append(trainY)
            train_y_pred.append(predY)
            testY, predY = test_model(out_dir, locals()[f'model_{C_lst[c]}'], testSet[c])
            test_y_true.append(testY)
            test_y_pred.append(predY)

        # evaluate together
        train_y_true = np.squeeze(np.array(train_y_true).transpose((1, 2, 3, 4, 0)))
        train_y_pred = np.squeeze(np.array(train_y_pred).transpose((1, 2, 3, 4, 0)))
        eval_metrics(out_dir, train_y_true, train_y_pred, ha)
        test_y_true = np.squeeze(np.array(test_y_true).transpose((1, 2, 3, 4, 0)))
        test_y_pred = np.squeeze(np.array(test_y_pred).transpose((1, 2, 3, 4, 0)))
        eval_metrics(out_dir, test_y_true, test_y_pred, ha)

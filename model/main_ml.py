import os
import argparse
import numpy as np
import pandas as pd
from model.metrics import eval_metrics



def load_data(in_dir):
    data = np.load(in_dir)
    print('Loaded Emergency NYC data.. \n   Shape:', data.shape)

    # historical avg. - regional occurence rate
    ha = np.mean(data, axis=0)

    return data, ha


def split_data(data, dates, delta_t):
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
        raise Exception('Unknown model name..')

    return y_true, y_pred



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ML models for emergency prediction')
    parser.add_argument('-in', '--in_dir', type=str, help='Input directory', default='../data')
    parser.add_argument('-model', '--model_name', type=str, help='Choose prediction model',
                        choices=['LR', 'VAR', 'LASSO', 'SVM'], default='LR')
    parser.add_argument('-t', '--delta_t', type=int, default=4, help='Time interval in hour(s)')
    parser.add_argument('-l', '--seq_len', type=int, default=6, help='Sequence length of observation steps')
    parser.add_argument('-date', '--dates', type=str, nargs='+',
                        help='Start/end dates of train/test sets. Test follows train.'
                             ' Example: -date 20150101 20150531 20150601 20150630',
                        default=['20150101', '20150531', '20150601', '20150630'])

    args = parser.parse_args()

    # input dir
    in_dir = os.path.join(args.in_dir, f'{args.delta_t}h', 'EmergNYC_bi_20x10.npy')
    # output dir
    out_dir = os.path.join('./baseline', args.model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load data
    data, ha = load_data(in_dir)

    # split train/test sets
    trainSet, testSet = split_data(data, args.dates, args.delta_t)

    # run model
    y_true, y_pred = run_model(args.model_name, out_dir, trainSet, testSet, args.seq_len)

    # evaluate prediction performance
    eval_metrics(out_dir, y_true, y_pred, ha)

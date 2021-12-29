import os
import time
import argparse
import Data_Container, Model_Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multi-incident co-prediction.')

    # command line arguments
    parser.add_argument('-device', '--device', type=str, help='Specify device usage', default='cuda:0',
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('-in', '--input_dir', type=str, default='../data')
    parser.add_argument('-out', '--output_dir', type=str, default='./output')
    parser.add_argument('-city', '--city', type=str, help='Specify city', choices=['SF'])
    parser.add_argument('-model', '--model', type=str, help='Specify model', default='STC-GNN', choices=['STC-GNN'])
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation sequence', default=9)
    parser.add_argument('-pred', '--pred_len', type=int, help='Length of prediction sequence', default=3)
    parser.add_argument('-split', '--split_ratio', type=int, nargs='+',
                        help='Relative data split ratio in train : validate : test. Example: -split 6 1 1',
                        default=[6, 1, 1])
    parser.add_argument('-batch', '--batch_size', type=int, default=32)
    parser.add_argument('-hidden', '--hidden_dim', type=int, default=16)
    parser.add_argument('-K', '--cheby_order', type=int, default=2)
    parser.add_argument('-nn', '--nn_layers', type=int, default=2)
    parser.add_argument('-lr', '--learn_rate', type=float, default=2e-3)
    parser.add_argument('-dr', '--decay_rate', type=float, default=1e-4)    # weight decay: L2 regularization
    parser.add_argument('-epoch', '--num_epochs', type=int, default=100)
    parser.add_argument('-test', '--test_only', type=int, default=0, choices=[0, 1])    # 1 for test only

    params = parser.parse_args().__dict__       # save in dict

    if params['city'] == 'NYC':
        params['C_inc'] = ['Crime', 'EMS', 'Rescue', 'Traffic_Accident',
                           'Noise', 'Illegal_Parking', 'Blocked_Driveway', 'Malfunctioning']
        params['C'] = len(params['C_inc'])
        params['H'], params['W'] = (20, 15)
        params['time_slice'] = 6
        start, end = (2013, 7), (2016, 6)       # 36 months: test on last 4.5 months
    elif params['city'] == 'CHI':
        params['C_inc'] = ['Crime', 'Traffic_Accident', 'Blocked_Driveway', 'Malfunctioning']
        params['C'] = len(params['C_inc'])
        params['H'], params['W'] = (10, 24)
        params['time_slice'] = 4
        start, end = (2018, 7), (2020, 6)       # 24 months: test on last 3 months
    elif params['city'] == 'SF':
        params['C_inc'] = ['Crime', 'EMS', 'Rescue', 'Illegal_Parking', 'Malfunctioning']
        params['C'] = len(params['C_inc'])
        params['H'], params['W'] = (10, 10)
        params['time_slice'] = 4
        start, end = (2018, 5), (2020, 8)       # 28 months: test on last 3.5 months
    else:
        raise ValueError('Invalid input city.')

    # paths
    data_dir = os.path.join(params['input_dir'], f'{params["city"]}-incidents-{params["time_slice"]}h.npz')
    params['output_dir'] = os.path.join(params['output_dir'], params['city'])
    os.makedirs(params['output_dir'], exist_ok=True)

    # load data
    print('\n', time.ctime())
    print(f'    Loading {params["city"]} data: {start}~{end}, '
          f'at {params["time_slice"]}h time slice, on a {params["H"]}x{params["W"]} grid map.')
    data_input = Data_Container.DataInput(data_dir=data_dir)
    data = data_input.load_data()
    print('    Incident data shape:', data['inc'].shape, '\n')

    # get data loader
    data_generator = Data_Container.DataGenerator(obs_len=params['obs_len'],
                                                  pred_len=params['pred_len'],
                                                  data_split_ratio=params['split_ratio'])
    data_loader = data_generator.get_data_loader(params=params, data=data)

    # get model
    trainer = Model_Trainer.ModelTrainer(params=params, data=data)
    if not params['test_only']:
        trainer.train(data_loader=data_loader,
                      modes=['train', 'validate'])
    trainer.test(data_loader=data_loader,
                 modes=['test'])

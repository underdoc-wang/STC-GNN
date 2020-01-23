import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def comb_moblty_color(in_path, year, month):
    volm_lst, flow_lst = [], []
    valid_count = 0
    for m in range(1, month+1):
        volm_lst_c, flow_lst_c = [], []
        for c in ['green', 'yellow']:
            with np.load(os.path.join(in_path, f'{c}{year}-{str(m).zfill(2)}-data.npz')) as data:
                vdata, fdata, trips, errors = data['vdata'], data['fdata'], data['trips'], data['errors']
            volm_lst_c.append(vdata)
            flow_lst_c.append(fdata)
            # check
            print(c, m, 'invalid:', errors[0], 'unparsable:', errors[1])
            print(c, m, '#people starts/ends within range:', int(trips[0, 0, 1]), '%total', round(trips[0, 0, 1] / np.sum(trips[:, :, 1]), 5))
            print('Max of flow:', np.amax(fdata))
            valid_count += trips[0, 0, 1]

        volm_lst.append(sum(volm_lst_c))
        flow_lst.append(sum(flow_lst_c))

    print('Total trips starting/ending within range: ', int(valid_count))

    stdn_volm = np.concatenate(volm_lst, axis = 0)
    print(year, 'Volume data shape:', stdn_volm.shape)
    stdn_flow = np.concatenate(flow_lst, axis = 1)
    print(year, 'Flow data shape:', stdn_flow.shape)

    return stdn_volm, stdn_flow


def get_moblty_feat(out_path, stdn_volm, stdn_flow, channel, strategy):
    # TODO: try flow from last frame?
    '''
    # check flows - current vs. last frame
    print(stdn_flow[0].max(), stdn_flow[1].max())
    plt.hist(stdn_flow[0].flatten())
    plt.title('current')
    plt.show()

    plt.hist(stdn_flow[1].flatten())
    plt.title('last')
    plt.show()
    '''

    # channel: passenger/trip count
    if channel == 'pcount':
        if strategy == 'in_out':
            moblty = stdn_volm[:, :, :, :, 0]     # strat 1
        elif strategy == 'inflow':
            moblty = stdn_flow[0, :, :, :, :, :, 0].transpose((0, 3, 4, 1, 2))     # strat 2
            moblty = moblty.reshape((moblty.shape[0], moblty.shape[1], moblty.shape[2], -1))
        elif strategy == 'outflow':
            moblty = stdn_flow[0, :, :, :, :, :, 0]     # strat 3
            moblty = moblty.reshape((moblty.shape[0], moblty.shape[1], moblty.shape[2], -1))
        elif strategy == 'spars_mx':
            curr_out = stdn_flow[0, :, :, :, :, :, 0]       # strat 4
            curr_out = curr_out.reshape((curr_out.shape[0], curr_out.shape[1], curr_out.shape[2], -1))
            curr_in = stdn_flow[0, :, :, :, :, :, 0].transpose((0, 3, 4, 1, 2))
            curr_in = curr_in.reshape((curr_in.shape[0], curr_in.shape[1], curr_in.shape[2], -1))
            # sparse matrix - concatenate outflow & inflow TKDE'19
            moblty = np.concatenate([curr_out, curr_in], axis=-1)

    elif channel == 'trip':
        if strategy == 'in_out':
            moblty = stdn_volm[:, :, :, :, 1]     # strat 5
        elif strategy == 'inflow':
            moblty = stdn_flow[0, :, :, :, :, :, 1].transpose((0, 3, 4, 1, 2))     # strat 6
            moblty = moblty.reshape((moblty.shape[0], moblty.shape[1], moblty.shape[2], -1))
        elif strategy == 'outflow':
            moblty = stdn_flow[0, :, :, :, :, :, 1]     # strat 7
            moblty = moblty.reshape((moblty.shape[0], moblty.shape[1], moblty.shape[2], -1))
        elif strategy == 'spars_mx':
            curr_out = stdn_flow[0, :, :, :, :, :, 1]       # strat 8
            curr_out = curr_out.reshape((curr_out.shape[0], curr_out.shape[1], curr_out.shape[2], -1))
            curr_in = stdn_flow[0, :, :, :, :, :, 1].transpose((0, 3, 4, 1, 2))
            curr_in = curr_in.reshape((curr_in.shape[0], curr_in.shape[1], curr_in.shape[2], -1))
            moblty = np.concatenate([curr_out, curr_in], axis=-1)

    print('Strategy: ', channel, strategy, '\n',
          'Mobility shape: ', moblty.shape)
    # save
    np.save(os.path.join(out_path, f'{channel}-{strategy}.npy'), moblty)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Dynamic mobility feature generation')
    parser.add_argument('-in', '--in_dir', type=str, help='Input directory',
                        default='../../NYCDatasetProcessing')
    parser.add_argument('-out', '--out_dir', type=str, help='Output directory',
                        default='../data')
    parser.add_argument('-y', '--year', type=int, default=2015, help='Aim year')
    parser.add_argument('-m', '--month_span', type=int, default=12, help='How many months')
    parser.add_argument('-t', '--t_interval', type=int, default=4, help='Time interval in hour(s)')
    parser.add_argument('-channel', '--trip_pcount', type=str, help='Use trip or passenger count',
                        choices=['trip', 'pcount'], default='pcount')
    parser.add_argument('-feat', '--feat_strat', type=str, help='Use which strategy to construct feature',
                        choices=['in_out', 'inflow', 'outflow', 'spars_mx'], default='in_out')

    args = parser.parse_args()

    in_path = os.path.join(args.in_dir, f'mobility{str(args.year)[-2:]}', f'{args.t_interval}h')
    print('Processing:', in_path)

    # mobility of colors combine on a monthly base
    volm_data, flow_data = comb_moblty_color(in_path, args.year, args.month_span)

    # output dir
    out_path = os.path.join(args.out_dir, f'{args.t_interval}h', 'moblty')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # generate mobility feature
    get_moblty_feat(out_path, volm_data, flow_data, channel=args.trip_pcount, strategy=args.feat_strat)

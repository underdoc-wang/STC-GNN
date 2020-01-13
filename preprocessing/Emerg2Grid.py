import os
import argparse
import time
import numpy as np
import pandas as pd
from math import floor
import matplotlib.pyplot as plt


# 900 x 900 m - spatial granularity
H = 20
W = 10

# top as origin
# top->left as axis x
# top->right as axis y
top_lon, top_lat = (-73.94115647069, 40.854811344)
left_lon, left_lat = (-74.03774052394, 40.7102999775)
right_lon, right_lat = (-73.8460741444, 40.818024909)
#
bottom_lon, bottom_lat = (-73.94281094938, 40.67359246767)


# values for prebaked function
origin_array = np.array([top_lon, top_lat])   # Origin

x_vector = np.array([left_lon, left_lat]) - origin_array
y_vector = np.array([right_lon, right_lat]) - origin_array
axis_base = np.array([x_vector, y_vector])

inv_basis = np.linalg.inv(axis_base)   # for coordinate transformation
#  [[-2.12718769  8.35641724]
#   [-5.49816675 -5.58500461]]
print('Inversion base: \n', inv_basis)


def pgps_to_xy(lon, lat):
    # original function from NYCDatasetProcessing repo
    # https://github.com/znwang918/NYCDatasetProcessing
    ''' gps_to_xy, using prebaked values to increase performance.'''
    x = (lon, lat) - origin_array
    c = np.matmul(x, inv_basis)
    #print(lon, lat, c[0], c[1], floor(c[0]*H), floor(c[1]*W))
    return c[0], c[1]


def load_data(file_dir, type, year, month, interval):
    print(time.ctime(), f'Loading {type} data at {interval}h interval...')
    raw = pd.read_csv(file_dir)

    raw.dropna(subset=['Latitude', 'Longitude', 'Timestamp'], inplace=True)
    raw.drop_duplicates(inplace=True)
    raw['T'] = raw.apply(lambda L: pd.to_datetime(L['Timestamp'], format='%Y-%m-%d %H:%M:%S'), axis=1)

    # time range
    deltaT = pd.Timedelta(hours=interval)
    if 1 <= month < 12:
        rangeT = pd.date_range(start=f'1/1/{year}', end=f'{1+month}/1/{year}', freq=deltaT)
        print(f'{year}/1~{month} total frames: {len(rangeT) - 1} at time interval {interval} hour(s)')
    elif month == 12:
        rangeT = pd.date_range(start=f'1/1/{year}', end=f'1/1/{year+1}', freq=deltaT)
        print(f'{year}/1~{year+1}/1 total frames: {len(rangeT) - 1} at time interval {interval} hour(s)')
    else:
        raise Exception('Invalid month input.')

    # category list
    if type == 'crime':
        category_lst = ['VIOLATION', 'MISDEMEANOR', 'FELONY']
    elif type == 'fire':
        category_lst = ['RESCUE', 'FIRE']
    elif type == 'ems':
        category_lst = ['EMS']

    data = []
    for t in range(len(rangeT) - 1):
        frame_t = raw[(raw['T'] >= rangeT[t]) & (raw['T'] < rangeT[t + 1])]
        maps_t = []
        for category in category_lst:
            category_t = frame_t[frame_t['Category'] == category]
            category_map_t = np.zeros((H, W))
            for i, row in category_t.iterrows():
                lon, lat = row['Longitude'], row['Latitude']
                x, y = pgps_to_xy(lon, lat)     # calculate using origin array & inverse bias
                inside = (0<=x<=1) and (0<=y<=1)     # T/F whether point within
                if inside:
                    grid_x = floor(x * H)
                    grid_y = floor(y * W)
                    category_map_t[grid_x, grid_y] += 1
            #print(category, np.sum(category_map_t))
            category_map_t1 = np.expand_dims(category_map_t, axis=-1)
            maps_t.append(category_map_t1)
        maps_t_concat = np.concatenate(maps_t, axis=-1)
        #print(rangeT[t], maps_t_concat.shape)
        data.append(maps_t_concat)

    data_np = np.array(data)
    print(data_np.shape)

    return data_np


def check_data(data, data_01):
    # check any pixel is zero-out
    zero_count = 0
    for i in range(H):
        for j in range(W):
            if not np.any(data_01[:,i,j,:]):     # all zero
                print('zero target grid:', i, j)
                zero_count += 1
    print(f'#Empty target pixel: {zero_count}')

    # statistics
    cat_lst = ['violation', 'misdemeanor', 'felony', 'ems', 'rescue', 'fire']
    for c in range(data_01.shape[-1]):
        # numerical
        c_sum = np.sum(data[:, :, :, c])
        # binary
        neg, pos = np.bincount(data_01[:, :, :, c].flatten())
        total = neg + pos # - data_01.shape[0]*zero_count      # minus zero-out pixels
        print(f'Occurence% {cat_lst[c]}: \n    Total numerical count: {c_sum} \n    Total pixel: {total}\n    Positive pixel: {pos} ({round(pos/total, 4) * 100}% of total)')

    return None


def plot_histo(data, data_01, args):
    plt.hist(data.flatten())
    plt.show()

    plt.hist(data_01.flatten())
    #counts, bins = np.histogram(data_01)
    #print(counts, bins)
    #plt.hist(bins[:-1], bins, weights=counts)
    #plt.xticks([0, 1])
    plt.title(f'Binary Distribution of Occurence@{args.t_interval}h Interval')
    #plt.show()
    plt.savefig(os.path.join(args.out_dir, str(args.t_interval)+'h', 'binary_distribution.png'))

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NYC emergency preprocessing')
    parser.add_argument('-in', '--aim_dir', type=str, help='Aim directory', default='./from_raw/raw')
    parser.add_argument('-out', '--out_dir', type=str, help='Output directory', default='../data')
    parser.add_argument('-type', '--emerg_type', type=str, choices=['all', 'crime', 'ems', 'fire'],
                        default='all', help='Emergency type')
    parser.add_argument('-y', '--year', type=int, default=2015, help='Aim year')
    parser.add_argument('-m', '--month_span', type=int, default=12, help='How many months')
    parser.add_argument('-t', '--t_interval', type=int, default=4, help='Time interval in hour(s)')
    #parser.add_argument('--print_steps', type=int, default=24, help='Length of print sequence')
    #parser.add_argument('--print_category', type=int, default=5, help='Which channel to print')

    args = parser.parse_args()

    if args.emerg_type != 'all':
        file_dir = os.path.join(args.aim_dir, f'{args.emerg_type}{str(args.year)[-2:]}.csv')
        data = load_data(file_dir, args.emerg_type, args.year, args.month_span, args.t_interval)
    else:
        emerg_lst = []
        for type in ['crime', 'ems', 'fire']:
            file_dir = os.path.join(args.aim_dir, f'{type}{str(args.year)[-2:]}.csv')
            type_data = load_data(file_dir, type, args.year, args.month_span, args.t_interval)
            emerg_lst.append(type_data)
        data = np.concatenate(emerg_lst, axis=-1)

    print('Emergency NYC shape: ', data.shape)
    data_int = data.astype(int)
    # total counts for all categories
    for c in range(data_int.shape[-1]):
        print(np.sum(data_int[:,:,:,c]))

    # clip to 0/1
    data_01 = np.clip(data_int, 0, 1)
    np.save(os.path.join(args.out_dir, str(args.t_interval)+'h', f'EmergNYC_bi_{H}x{W}.npy'), data_01)

    check_data(data_int, data_01)
    plot_histo(data_int, data_01, args)


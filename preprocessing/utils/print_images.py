import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def probalistic_cross_k_function(y_true, y_pred, minkov_degree=1, max_distance=10):
    y_true_list = np.stack(np.where(y_true == 1)).T
    y_true_kdtree = KDTree(y_true_list)

    y_pred_list = np.stack(np.where(y_pred >= 0)).T
    y_pred_kdtree = KDTree(y_pred_list)

    dense = np.mean(y_pred)

    k_value_list = []
    random_list = []

    for i in range(max_distance + 1):
        total_result = y_true_kdtree.sparse_distance_matrix(y_pred_kdtree, p=minkov_degree, max_distance=i)
        y_list = []
        y_random_list = []
        for j in range(len(y_true_list)):
            neighbor_obj = total_result.getrow(j).indices
            final_list = y_pred_list[list(neighbor_obj)].T
            y_list.append(sum(y_pred[final_list[0], final_list[1]]))
            y_random_list.append(len(neighbor_obj) * dense)
        k_value_list.append(np.mean(y_list) / dense)
        random_list.append(np.mean(y_random_list) / dense)

    return k_value_list, random_list


def print_cross_k_onetime(y_true, y_pred_lst, print_time):
    for c in range(len(cat_lst)):
        for n in range(len(y_pred_lst)):
            y_true_t_c = y_true[print_time, :, :, c]
            y_pred_t_c = y_pred_lst[n][print_time, :, :, c]

            df = pd.DataFrame(probalistic_cross_k_function(y_true_t_c, y_pred_t_c, minkov_degree=1, max_distance=6))
            if n == 0:
                linestyle = '-'
                marker = 'D'
            else:
                linestyle = '-.'
                marker = ''
            plt.plot(df.columns, df.iloc[0], label=f'{model_lst[n]} kf', linestyle=linestyle, color=col_lst[n], marker=marker, markersize=7)

        plt.plot(df.columns, df.iloc[1], label='Random kf', linestyle='--', marker='x', markersize=8, color='black')
        plt.title(f'{print_time}-{cat_lst[c]}')
        plt.legend()
        plt.show()

    return None


def print_overlay_image(emergency_data, volume_data, time):
    fig = plt.figure()
    plt.xticks(np.arange(10, step=2))
    plt.yticks(np.arange(20, step=2))
    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    #plt.title(f'2015-01-01 {str(4*t).zfill(2)}:00')

    #for cat in range(1): #range(emergency_data.shape[-1]):
    #plt.imshow(emergency_data[6, :, :, 0], alpha=0.7, extent=(-0.5, 9.5, 20.5, -0.5), cmap='rainbow', interpolation='kaiser')
    # plot ending volume (inflow); passanger count
    plt.imshow(volume_data[5, :, :, 1], alpha=0.7, extent=(-0.5, 9.5, 20.5, -0.5), cmap='inferno', interpolation='sinc')
    plt.show()

    return None


def print_emerg_images(emerg_data, t, type):
    if type == 'binary':
        col_lst = ['viridis', 'CMRmap', 'gnuplot2', 'inferno', 'cubehelix', 'afmhot']  # 'cividis'
    elif type == 'risk':
        col_lst = ['coolwarm'] * len(cat_lst)
    else:
        raise Exception('Unkown type..')

    for c, color in zip(range(emerg_data.shape[-1]), col_lst):
        fig = plt.figure()
        plt.xticks(np.arange(10, step=2))
        plt.yticks(np.arange(20, step=2))
        plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        plt.imshow(emerg_data[t, :, :, c], extent=(-0.5, 9.5, 20.5, -0.5), cmap=color, interpolation='kaiser')
        #plt.show()
        plt.savefig(f'./for_figure/{t_intvl}h/{t}/{cat_lst[c]}-{type}.png')

    return None


# Hyper
t_intvl = 4
H, W = (20, 10)

# prediction color scheme
# mid-product: 'rainbow'
# pred_proba: 'coolwarm'

# emergency color scheme
cat_lst = ['Violation', 'Misdemeanor', 'Felony', 'EMS', 'Rescue', 'Fire']
model_lst = ['CitySiren', 'MiST', 'Hetero-ConvLSTM', 'ConvLSTM', 'GRU', 'VAR']
col_lst = ['red', 'goldenrod', 'lime', 'aqua', 'blue', 'fuchsia']

# (t-1): 5   t: 6   (t+1): 7
#print_time = 6


if __name__ == '__main__':
    '''
    # load data
    emerg_data = np.load(f'../../data/{t_intvl}h/EmergNYC_bi_{H}x{W}.npy')
    volm_data = np.load(f'../../data/{t_intvl}h/moblty/pcount-in_out.npy')
    print('Loaded emergency data: ', emerg_data.shape)
    print('Loaded mobility data:', volm_data.shape)
    '''
    # input print_time
    print('Enter print_time:')
    t = int(input())
    # check output directory
    if not os.path.exists(f'./for_figure/{t_intvl}h/{t}'):
        os.makedirs(f'./for_figure/{t_intvl}h/{t}')
        print(f'Made an output directory for {t}.')

    # print images
    #print_emerg_images(emerg_data, print_time)
    #print_overlay_image(emerg_data, volm_data, print_time)

    # print_cross_f
    y_true = np.load('./for_figure/y_true_jun.npy')

    y_pred_0 = np.load('./for_figure/y_pred_citysiren_jun.npy')
    y_pred_1 = np.load('./for_figure/y_pred_mist_jun.npy')
    y_pred_2 = np.load('./for_figure/y_pred_hetero_jun.npy')
    y_pred_3 = np.load('./for_figure/y_pred_convlstm_jun.npy')
    y_pred_4 = np.load('./for_figure/y_pred_gru_jun.npy')
    y_pred_5 = np.load('./for_figure/y_pred_var_jun.npy')
    y_pred_lst = [y_pred_0, y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5]


    #print_cross_k_onetime(y_true, y_pred_lst, t)
    #print_emerg_images(y_true, t, 'binary')
    print_emerg_images(y_pred_0, t, 'risk')

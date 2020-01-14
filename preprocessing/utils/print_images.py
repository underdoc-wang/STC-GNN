import os
import numpy as np
import matplotlib.pyplot as plt


def print_overlay_images(emergency_data, volume_data, steps):
    for t in range(steps):
        fig = plt.figure()
        plt.xticks(np.arange(10, step=2))
        plt.yticks(np.arange(20, step=2))
        plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.title(f'2016-01-01 {str(4*t).zfill(2)}:00')

        for cat in range(emergency_data.shape[-1]):
            plt.imshow(emergency_data[t, :, :, cat], extent=(-0.5, 9.5, 20.5, -0.5), cmap='rainbow', interpolation='kaiser')
        # plot ending volume (inflow); passanger count
        plt.imshow(volume_data[t, :, :, 1, 0], alpha=0.7, extent=(-0.5, 9.5, 20.5, -0.5), cmap='inferno')
        plt.show()

    return None


def print_emerg_images(emerg_data, t):
    # emergency color scheme
    cat_lst = ['violation', 'misdemeanor', 'felony', 'EMS', 'rescue', 'fire']
    col_lst = ['viridis', 'CMRmap', 'gnuplot2', 'inferno', 'cubehelix', 'afmhot']   # 'cividis'

    for cat, color in zip(range(emerg_data.shape[-1]), col_lst):
        fig = plt.figure()
        plt.xticks(np.arange(10, step=2))
        plt.yticks(np.arange(20, step=2))
        plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        plt.imshow(emerg_data[t, :, :, cat], extent=(-0.5, 9.5, 20.5, -0.5), cmap=color, interpolation='kaiser')
        plt.show()
        #plt.savefig(f'../data/{t_intvl}h/images/{cat_lst[cat]}-t{t}.png')

    return None


t_intvl = 2
H = 20
W = 10

print_emerg_time = 5
print_overlay_steps = 1


if __name__ == '__main__':
    # load emergency & mobility
    emerg_data = np.load(f'../../data/{t_intvl}h/EmergNYC_bi_{H}x{W}.npy')
    #volm_data = np.load('../Data/VolumNYC.npy')
    print('Emergency data shape:', emerg_data.shape)
    #print('Mobility data shape:', volm_data.shape)

    # check output directory
    if not os.path.exists(f'../../data/{t_intvl}h/images'):
        os.makedirs(f'../data/{t_intvl}h/images')
    print_emerg_images(emerg_data, print_emerg_time)
    #print_overlay_images(emerg_data, volm_data, print_overlay_steps)
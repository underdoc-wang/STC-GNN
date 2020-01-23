import numpy as np


def build_region_image(data, size):
    T, X, Y, C = data.shape
    padding = size // 2
    target_loc = size // 2
    
    pad_data = np.pad(data, ((0,0), (padding, padding), (padding, padding), (0,0)), 'constant')
    print('pad data shape:', pad_data.shape)
    
    region_image = np.zeros((T, X*Y, size, size, C))
    
    for t in range(T):
        if t % 1000 == 0:
            print(f'processing {round(t/T, 4)*100} data')
        for x in range(X):
            for y in range(Y):
                region_image[t, x*Y+y] = pad_data[t, x:x+size, y:y+size]
    
    print(region_image.shape)
    print('finish building region images.')
    
    return region_image


# global
delta_t = 4       # time interval
region_size = 3   # local size

in_dir = f'../../../data/{delta_t}h/EmergNYC_bi_20x10.npy'
out_dir = f'../../../data/{delta_t}h/EmergNYC_bi_20x10_{region_size}x{region_size}region.npy'


if __name__ == '__main__':
    data = np.load(in_dir)
    print('data shape:', data.shape)
    
    local_images = build_region_image(data, region_size)
    np.save(out_dir, local_images)
    
    
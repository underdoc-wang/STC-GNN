import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DataInput(object):
    def __init__(self, data_dir:str):
        self.data_dir = data_dir

    def load_data(self):
        npz_data = np.load(self.data_dir)
        print('Dataset contents:', list(npz_data.keys()))

        dataset = dict()
        dataset['inc'] = npz_data['incident']
        dataset['mask'] = [tuple(a) for a in npz_data['mask']]      # list of all-zero coordinates: masked from evaluation
        dataset['HA'] = npz_data['threshold']
        dataset['s_adj'] = npz_data['s_adj']
        dataset['c_cor'] = npz_data['c_cor']

        return dataset

    def minmax_normalize(self, x:np.array):     # normalize to [0, 1]
        self._max, self._min = x.max(), x.min()
        print('min:', self._min, 'max:', self._max)
        x = (x - self._min) / (self._max - self._min)
        return x

    def minmax_denormalize(self, x:np.array):
        x = (self._max - self._min) * x + self._min
        return x

    def std_normalize(self, x:np.array):        # normalize to N(0, 1)
        self._mean, self._std = x.mean(), x.std()
        print('mean:', round(self._mean, 4), 'std:', round(self._std, 4))
        x = (x - self._mean)/self._std
        return x

    def std_denormalize(self, x:np.array):
        x = x * self._std + self._mean
        return x


class IncDataset(Dataset):
    def __init__(self, inputs:dict, output:torch.Tensor, mode:str, mode_len:dict):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_seq'][item], self.output[item]

    def prepare_xy(self, inputs:dict, output:torch.Tensor):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:       # test
            start_idx = self.mode_len['train']+self.mode_len['validate']

        x = dict()
        x['x_seq'] = inputs['x_seq'][start_idx : (start_idx + self.mode_len[self.mode])]
        y = output[start_idx : start_idx + self.mode_len[self.mode]]
        return x, y


class DataGenerator(object):
    def __init__(self, obs_len:int, pred_len:int, data_split_ratio:tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len:int):
        mode_len = dict()
        mode_len['validate'] = int(self.data_split_ratio[1]/sum(self.data_split_ratio) * data_len)
        mode_len['test'] = int(self.data_split_ratio[2]/sum(self.data_split_ratio) * data_len)
        mode_len['train'] = data_len - mode_len['validate'] - mode_len['test']

        return mode_len

    def get_data_loader(self, params:dict, data:dict):
        incident = data['inc'].reshape(data['inc'].shape[0], params['H']*params['W'], params['C'])

        feat_dict = dict()
        # incident
        x_seq, y_seq = self.get_feats(incident)
        if params['device'].startswith('cuda'):
            feat_dict['x_seq'] = torch.from_numpy(np.asarray(x_seq)).float().to(params['device'])
            y_seq = torch.from_numpy(np.asarray(y_seq)).float().to(params['device'])
        else:
            feat_dict['x_seq'] = torch.from_numpy(np.asarray(x_seq)).float()
            y_seq = torch.from_numpy(np.asarray(y_seq)).float()

        mode_len = self.split2len(data_len=y_seq.shape[0])
        data_loader = dict()        # data_loader for [train, validate, test]
        for mode in ['train', 'validate', 'test']:
            dataset = IncDataset(inputs=feat_dict, output=y_seq,
                                 mode=mode, mode_len=mode_len)
            data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False)
        # data loading default: single-processing | for multi-processing: num_workers=pos_int or pin_memory=True (GPU)

        return data_loader

    def get_feats(self, data:np.array):
        x, y = [], []
        for i in range(self.obs_len, data.shape[0]-self.pred_len):
            x.append(data[i-self.obs_len : i])
            y.append(data[i : i+self.pred_len])
        return x, y


import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from src.utils.util import config

from datetime import datetime
import numpy as np
import arrow
import torch
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data


class HazeData(data.Dataset):
    def __init__(self, graph,
                       hist_len=24,
                       pred_len=24,
                       dataset_num=2015,
                       flag='train',
                       norm=None):
        self.flag = flag
        self.norm = norm
        if flag == 'train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'test_default':
            start_time_str = 'test_start_ood'
            end_time_str = 'test_end_ood'
        elif flag == 'test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')

        self.start_time = self._get_time(config['dataset'][dataset_num][start_time_str])
        self.end_time = self._get_time(config['dataset'][dataset_num][end_time_str])
        self.data_start = self._get_time(config['dataset']['data_start'])
        self.data_end = self._get_time(config['dataset']['data_end'])
        
        self.knowair_fp = os.path.join('data', 'KnowAir.npy')

        self.graph = graph

        self._load_npy()
        self._gen_time_arr()
        self._process_time()
        self._process_feature()
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        self._calc_mean_std()
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()
        print(flag+' done!')

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std

    def _add_time_dim(self, seq_len):

        def _add_t(arr, seq_len):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)
        self.feature = _add_t(self.feature, seq_len)
        self.time_arr = _add_t(self.time_arr, seq_len)
        self.day = _add_t(self.day, seq_len)
        self.week = _add_t(self.week, seq_len)

    def _calc_mean_std(self):
        if self.flag == 'train':
            print('Self norm!')
            self.feature_mean = self.feature.mean(axis=(0,1))
            self.feature_std = self.feature.std(axis=(0,1))
            self.wind_mean = self.feature_mean[-2:]
            self.wind_std = self.feature_std[-2:]
            self.pm25_mean = self.pm25.mean()
            self.pm25_std = self.pm25.std()
        else:
            print('Training set norm!')
            self.feature_mean, self.feature_std, self.wind_mean, self.wind_std, self.pm25_mean, self.pm25_std = self.norm
        # print(self.pm25_mean)
        # print(self.pm25_std)
        # print(self.wind_mean.shape)
        # print(self.wind_std.shape)

    def return_mean_std(self):
        return [self.feature_mean, self.feature_std, self.wind_mean, self.wind_std, self.pm25_mean, self.pm25_std]

    def _process_feature(self):
        metero_var = config['data']['metero_var']
        metero_use = config['experiments']['metero_use']
        metero_idx = [metero_var.index(var) for var in metero_use]
        self.feature = self.feature[:,:,metero_idx]

        u = self.feature[:, :, -2] * units.meter / units.second
        v = self.feature[:, :, -1] * units.meter / units.second
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
        direc = mpcalc.wind_direction(u, v)._magnitude

        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            h_arr.append(i.hour)
            w_arr.append(i.isoweekday())
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)

        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direc[:, :, None]
                                       ], axis=-1)

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]
        self.day = self.day[start_idx: end_idx + 1]
        self.week = self.week[start_idx: end_idx + 1]

    def _gen_time_arr(self):
        self.time_arrow = []
        self.time_arr = []
        self.day = []
        self.week = []
        i = 1.
        j = 4*8. 
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+3), 3):
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].timestamp)
            self.day.append((i-1)/8.)
            self.week.append(((j//8) - 1)/7.)
            i += 1
            j += 1
            if i == (8+1): i = 1.
            if j == (7*8)+1: j = 1.
        self.time_arr = np.stack(self.time_arr, axis=-1)
        # print(np.stack(self.time_arrow, axis=-1).dayofweek)
        # sys.exit()
        self.day = np.repeat(np.stack(self.day, axis=-1).reshape(-1, 1, 1), self.graph.node_num, axis=-2)
        self.week = np.repeat(np.stack(self.week, axis=-1).reshape(-1, 1, 1), self.graph.node_num, axis=-2)
        
    def _load_npy(self):
        self.knowair = np.load(self.knowair_fp)
        self.feature = self.knowair[:,:,:-1]
        self.pm25 = self.knowair[:,:,-1:]

    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp - t0.timestamp) / (60 * 60 * 3))

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index], self.time_arr[index], self.day[index], self.week[index]


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(graph, hist_len, pred_len, dataset_num, batch_size):
    dataloader = {}
    scaler = {}
    objective = {}
    for cat in ['train', 'val', 'test']:
        objective[cat] = HazeData(graph, 
                                  hist_len, 
                                  pred_len, 
                                  dataset_num, 
                                  flag=cat, 
                                  norm=None if cat == 'train' else objective['train'].return_mean_std())
        dataloader[cat + '_loader'] = torch.utils.data.DataLoader(objective[cat], 
                                                                  batch_size=batch_size, 
                                                                  shuffle=True if cat == 'train' else False, 
                                                                  drop_last=True)
        
        scaler[cat] = StandardScaler(mean=objective[cat].pm25_mean, 
                                     std=objective[cat].pm25_std)
    return dataloader, scaler

if __name__ == '__main__':
    from graph import Graph
    graph = Graph()
    train_data = HazeData(graph, flag='Train')
    val_data = HazeData(graph, flag='Val')
    test_data = HazeData(graph, flag='Test')

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))

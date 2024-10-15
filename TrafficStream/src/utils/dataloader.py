import os
import pickle
import torch
import numpy as np
import random
import threading
import multiprocessing as mp


class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, years, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
                
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + 
                    str(self.idx.shape[0]) + 
                    ', Batch num: ' + 
                    str(self.num_batch) + 
                    ', in year: ' +
                    str(years))

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon
    
    # shuffle data order in one batch
    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx
    
    # shuffle batch order in one epoch
    def shuffle_batch(self):
        blocks = [self.idx[i:i+self.bs] for i in range(0, self.size, self.bs)]
        random.shuffle(blocks)
        idx = np.concatenate(blocks)
        if self.size % self.bs != 0:
            print('Warning: In shuffle_batch, the batch size does not evenly divide the array size.')

    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        r = 0.01
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]
                
                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) if array_size == 1 else len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, \
                                              args=(x, y, idx_ind, start_index, end_index))

                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(data_path, args, logger):
    years = int(args.years)
    check_years = int(args.checkyears)
    ptr = np.load(os.path.join(data_path, str(years), 'his.npz'))
    logger.info('Data shape: ' + str(ptr['data'].shape))

    if args.tood:
        ptr_ood = np.load(os.path.join(data_path, str(check_years), 'his.npz'))
        logger.info('Data shape in '+ args.checkyears + ':' + str(ptr_ood['data'].shape))

    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, 
                                   str(check_years) if (cat == 'test' and args.tood) else str(years), 
                                   'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(ptr_ood['data'][..., :args.input_dim] if (cat == 'test' and args.tood) else ptr['data'][..., :args.input_dim], 
                                                 idx, 
                                                 args.seq_len, 
                                                 args.horizon, 
                                                 args.bs, 
                                                 logger,
                                                 check_years if (cat == 'test' and args.tood) else years,
                                                 )
    scaler = [StandardScaler(mean=ptr['mean'], std=ptr['std']),
              StandardScaler(mean=ptr['mean'], std=ptr['std']) if args.tood else None]
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         '2011': [base_dir, base_dir+'/2011/2011_adj.npy', 655],
         '2012': [base_dir, base_dir+'/2012/2012_adj.npy', 715],
         '2013': [base_dir, base_dir+'/2013/2013_adj.npy', 786],
         '2014': [base_dir, base_dir+'/2014/2014_adj.npy', 822],
         '2015': [base_dir, base_dir+'/2015/2015_adj.npy', 834],
         '2016': [base_dir, base_dir+'/2016/2016_adj.npy', 850],
         '2017': [base_dir, base_dir+'/2017/2017_adj.npy', 871],
        }
    assert dataset in d.keys()
    return d[dataset]

        
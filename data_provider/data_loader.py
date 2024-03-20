import os

import numpy as np
import pandas as pd
import torch

# from sklearn.preprocessing import StandardScaler
from utils.tools import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class Dataset_METRLA(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, scale=True, time_to_feature=1):
        super(Dataset_METRLA, self).__init__()
        if size == None:
            self.seq_len = 12
            self.label_len = 1
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.time_to_feature = time_to_feature

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        data_file_path =os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_hdf(data_file_path)  # [time_len, node_num]
        print(df_raw.shape)
        print(df_raw.head())
        print(df_raw.columns)
        print(df_raw.index.values)
        dates = df_raw.index.values.astype('datetime64')
        print(dates)
        num_train = round(len(df_raw) * 0.7)
        num_test = round(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = df_raw[border1s[0]:border2s[0]].values
            self.scaler = StandardScaler(mean=train_data.mean(), std=train_data.std())
            print('mean:', train_data.mean())
            print('std:', train_data.std())
            # 对数据进行标准化
            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values
        print('data shape:', data.shape)

        if self.time_to_feature == 0:
            num_samples, num_nodes = df_raw.shape
            time_ind = (df_raw.index.values - df_raw.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            print('type(df.index.values):', type(df_raw.index.values))
            print('df.index.values', df_raw.index.values)
            print('df.index.values.astype("datetime64[D]")', df_raw.index.values.astype("datetime64[D]"))
            print('time_ind', time_ind[:100])
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            feature_list = [np.expand_dims(data, axis=-1), time_in_day]
            processed_data = np.concatenate(feature_list, axis=-1)
        elif self.time_to_feature == 1:
            feature_list = [np.expand_dims(data, axis=-1)]
            l, n = data.shape
            # 对时间进行编码，返回是一个编码后的矩阵，每一行对应一个时间，列为编码后的特征
            stamp = time_features(pd.to_datetime(df_raw.index.values), freq='T')
            # 进行转置，每一行对应一个特征，列为对应的时间
            stamp = stamp.transpose(1, 0)
            # print(stamp[:10])
            # print(stamp.shape)
            print("type stamp:", type(stamp))
            stamp_tiled = np.tile(stamp, [n, 1, 1]).transpose((1, 0, 2))
            feature_list.append(stamp_tiled)
            processed_data = np.concatenate(feature_list, axis=-1)
        else:
            processed_data = data

        time_stamp = {'date': dates}
        df_stamp = pd.DataFrame(time_stamp)
        df_stamp['year'] = df_stamp.date.apply(lambda row: row.year, 1)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        data_stamp = df_stamp.drop(columns=['date']).values
        print(data_stamp[:10])

        self.data_x = processed_data[border1:border2]
        self.data_y = processed_data[border1:border2]
        self.data_stamp = data_stamp
        print(self.data_x[:10, 0])
        print(self.data_x.shape)
        print(type(self.data_x))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len - self.label_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

if __name__ == '__main__':
    root_path = '../datasets/raw_data/'
    data_path = 'METR-LA/METR-LA.h5'
    data = Dataset_METRLA(root_path=root_path, data_path=data_path, time_to_feature=True)
    seq_x, seq_y, seq_x_mark, seq_y_mark = data.__getitem__(0)
    print(seq_x.shape)
    print(type(seq_x[0, 0, 0]))
    print(type(seq_x[0, 0, :]))
    print(type(seq_x[0, :, :]))
    print(type(seq_x))
    print(seq_x_mark)
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from basicts.metrics import mape
from utils import metrics as m

print(os.getcwd())
folder_path = './STEP_METR-LA_sl24_pl12_el2_dl2_eh8_dh8_de256_ee4_2024_02_20_16_43_0/'
metrics_path = 'metrics.npy'
pred_path = 'pred.npy'
true_path = 'true.npy'
x_trues_path = 'histroy_trues.npy'
x_marks_path = 'x_marks.npy'
y_marks_path = 'y_marks.npy'

metrics = np.load(folder_path + metrics_path)
pred = np.load(folder_path + pred_path)
true = np.load(folder_path + true_path)
x_true = np.load(folder_path + x_trues_path)
# x_marks = np.load(folder_path + x_marks_path)
# y_marks = np.load(folder_path + y_marks_path)
#
# x_marks = x_marks.astype(dtype=str)[:, :, 1:]
# y_marks = y_marks.astype(dtype=str)[:, 1:, 1:]

# N, L, _ = x_marks.shape
# x_date = []
# for i in range(N):
#     dates = []
#     for j in range(L):
#         d = x_marks[i, j, 0] + '.' + x_marks[i, j, 1] + '.' + x_marks[i, j, 2] + '.' + x_marks[i, j, 3]
#         dates.append(d)
#     x_date.append(dates)
#
# N, L, _ = y_marks.shape
# y_date = []
# for i in range(N):
#     dates = []
#     for j in range(L):
#         d = y_marks[i, j, 0] + '.' + y_marks[i, j, 1] + '.' + y_marks[i, j, 2] + '.' + y_marks[i, j, 3]
#         dates.append(d)
#     y_date.append(dates)
#
# x_date, y_date = np.array(x_date), np.array(y_date)

# d = x_marks[0, 22, 0] + x_marks[0, 22, 1] + x_marks[0, 22, 2] + x_marks[0, 22, 3]
# print(d)
# print(type(d))
#
# print('x_marks:', x_marks)
# print(x_marks.shape)
# print('y_marks:', y_marks)
# print(y_marks.shape)


# def toDate(mark):
#     print(mark)
#     date = mark[0] + '.' + mark[1] + '.' + mark[2] + '.' + mark[3]
#     print(date)
#     return date
#
#
# x_marks = np.apply_along_axis(toDate, axis=-1, arr=x_marks)
# y_marks = np.apply_along_axis(toDate, axis=-1, arr=y_marks)
# date = np.concatenate((x_date, y_date), axis=1)


print(metrics)

print('pred:', pred)  # [len, pred_len, n_nodes]
print(pred.shape)
print('true:', true)  # [len, pred_len, n_nodes]
print(true.shape)
print('x_true:', x_true)
print(x_true.shape)

Mape = m.MAPE(pred, true)
print('Mape:', Mape)
# print('x_date:', x_date[0, :])
# print(x_marks.shape)
# print('y_date:', y_date[0, :])
# print(y_marks.shape)
# print('date:', date[0, :])
# print(date.shape)

# print(x_marks[0, 0])



for i in range(x_true.shape[0]):
    pos = i
    GroundTruth = np.concatenate((x_true[pos, :, 0], true[pos, :, 0]), axis=0)
    Prediction = np.concatenate((x_true[pos, :, 0], pred[pos, :, 0]), axis=0)

    plt.figure()
    plt.plot(GroundTruth, label='GroundTruth', linewidth=2)
    plt.plot(Prediction, label='Prediction', linewidth=2)

    # 创建刻度的位置
    # x_ticks = date[100, :]
    x_ticks = np.arange(24)
    tick_positions = np.arange(len(x_ticks))
    font_properties = FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
    plt.xticks(tick_positions, x_ticks, rotation=45)
    plt.legend()
    plt.show()

# root_path = '../datasets/raw_data/'
# data_path = 'METR-LA/METR-LA.h5'
# data_file_path =os.path.join(root_path, data_path)
# df_raw = pd.read_hdf(data_file_path)  # [time_len, node_num]
# print(df_raw.head())
# date = df_raw.index.values
# data_sensor0 = df_raw.values[:, 0]
#
# plt.plot(date[:24], data_sensor0[:24])
# plt.grid(True)  # 添加网格线，可选
# plt.show()

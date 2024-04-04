import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from basicts.metrics import mape
from utils import metrics as m

print(os.getcwd())
folder_path = './TAformer_seqlen288/'
metrics_path = 'metrics.npy'
pred_path = 'pred.npy'
true_path = 'true.npy'
x_trues_path = 'x_trues.npy'
x_marks_path = 'x_marks.npy'
y_marks_path = 'y_marks.npy'
maes_path = 'maes.npy'
rmses_path = 'rmses.npy'
mapes_path = 'mapes.npy'

metrics = np.load(folder_path + metrics_path)
pred = np.load(folder_path + pred_path)
true = np.load(folder_path + true_path)
x_true = np.load(folder_path + x_trues_path)
x_marks = np.load(folder_path + x_marks_path)
y_marks = np.load(folder_path + y_marks_path)
maes = np.load(folder_path + maes_path)
rmses = np.load(folder_path + rmses_path)
mapes = np.load(folder_path + mapes_path)

x_marks = x_marks.astype(dtype=str)[:, :, 1:]
y_marks = y_marks.astype(dtype=str)[:, :, 1:]

N, L, _ = x_marks.shape
x_date = []
for i in range(N):
    dates = []
    for j in range(L):
        d = x_marks[i, j, 0] + '.' + x_marks[i, j, 1] + '.' + x_marks[i, j, 2] + '.' + x_marks[i, j, 3]
        dates.append(d)
    x_date.append(dates)

N, L, _ = y_marks.shape
y_date = []
for i in range(N):
    dates = []
    for j in range(L):
        d = y_marks[i, j, 0] + '.' + y_marks[i, j, 1] + '.' + y_marks[i, j, 2] + '.' + y_marks[i, j, 3]
        dates.append(d)
    y_date.append(dates)

x_date, y_date = np.array(x_date), np.array(y_date)

d = x_marks[0, 22, 0] + x_marks[0, 22, 1] + x_marks[0, 22, 2] + x_marks[0, 22, 3]
print(type(d))


def toDate(mark):
    date = mark[0] + '.' + mark[1] + '.' + mark[2] + '.' + mark[3]
    return date


x_marks = np.apply_along_axis(toDate, axis=-1, arr=x_marks)
y_marks = np.apply_along_axis(toDate, axis=-1, arr=y_marks)
date = np.concatenate((x_date, y_date), axis=1)

print(metrics)


def showTimeSeries(x_true, true, pred, time_index):
    GroundTruth = np.concatenate((x_true, true), axis=0)
    Prediction = np.concatenate((x_true, pred), axis=0)

    plt.figure()
    plt.plot(Prediction, label='Prediction', linewidth=2)
    plt.plot(GroundTruth, label='GroundTruth', linewidth=2)

    # 创建刻度的位置
    x_ticks = date[time_index, :]
    # x_ticks = np.arange(24)
    tick_positions = np.arange(len(x_ticks))
    font_properties = FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
    plt.xticks(tick_positions[::12], x_ticks[::12], rotation=45)
    plt.ylim(-2, 75)
    plt.text(150.05, 7.5, 'MAE:{0}'.format(maes[i]), ha='left', va='center', size=10, weight='bold')
    plt.text(150.05, 4.5, 'RMSE:{0}'.format(rmses[i]), ha='left', va='center', size=10, weight='bold')
    plt.text(150.05, 1.5, 'MAPE:{0}'.format(mapes[i]), ha='left', va='center', size=10, weight='bold')
    plt.legend(loc='lower left')
    plt.show()


for i in range(x_true.shape[0]):
    if maes[i] < 4:
        continue
    for j in range(207):
        true_x = x_true[i, :, j]
        true_y = true[i, :, j]
        pred_y = pred[i, :, j]
        showTimeSeries(true_x, true_y, pred_y, i)

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

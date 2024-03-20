import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

maes = np.load('maes.npy')
mapes = np.load('mapes.npy')
metrics = np.load('metrics.npy')
mses = np.load('mses.npy')
mspes = np.load('mspes.npy')
preds = np.load('pred.npy')
rmses = np.load('rmses.npy')
trues = np.load('true.npy')
x_marks = np.load('x_marks.npy')
x_trues = np.load('x_trues.npy')
y_marks = np.load('y_marks.npy')

x_marks = x_marks.astype(dtype=str)[:, :, 1:]
y_marks = y_marks.astype(dtype=str)[:, 1:, 1:]

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

# def toDate(mark):
#     # print(mark)
#     date = mark[0] + '.' + mark[1] + '.' + mark[2] + '.' + mark[3]
#     # print(date)
#     return date
#
#
# x_marks = np.apply_along_axis(toDate, axis=-1, arr=x_marks)
# y_marks = np.apply_along_axis(toDate, axis=-1, arr=y_marks)
date = np.concatenate((x_date, y_date), axis=1)

for i in range(x_trues.shape[0]):
    pos = i
    GroundTruth = np.concatenate((x_trues[pos, :, 0], trues[pos, :, 0]), axis=0)
    Prediction = np.concatenate((x_trues[pos, :, 0], preds[pos, :, 0]), axis=0)

    print(preds[pos, :, 0])
    print('#################################################################')

    plt.figure()
    plt.plot(Prediction, label='Prediction', linewidth=2)
    plt.plot(GroundTruth, label='GroundTruth', linewidth=2)

    # 创建刻度的位置
    x_ticks = date[i, :][::10]
    # x_ticks = np.arange(24)
    tick_positions = np.arange(0, len(date[i, :]), 10)
    font_properties = FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
    plt.xticks(tick_positions, x_ticks, rotation=45)
    plt.ylim(45, 75)
    plt.text(-0.05, 63.5, 'MAE={0:.4f}'.format(maes[i]), ha='left', va='center', size=14, weight='bold')
    plt.text(-0.05, 62.5, 'MSE={0:.4f}'.format(mses[i]), ha='left', va='center', size=14, weight='bold')
    plt.text(-0.05, 61.5, 'MAPE={0:.4f}'.format(mapes[i]), ha='left', va='center', size=14, weight='bold')
    plt.text(-0.05, 61, 'MSPE={0:.4f}'.format(mspes[i]), ha='left', va='center', size=14, weight='bold')
    plt.text(-0.05, 60.5, 'RMSE={0:.4f}'.format(rmses[i]), ha='left', va='center', size=14, weight='bold')
    plt.legend()
    plt.show()

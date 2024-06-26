import pickle

import scipy.sparse as sp
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric, masked_mae
from utils.tools import EarlyStopping, adjust_learning_rate, visual, StandardScaler, DataLoader
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_GWNET(Exp_Basic):
    def __init__(self, args):
        super(Exp_GWNET, self).__init__(args)

    def _build_model(self):
        # 读取邻接矩阵
        adj_path = r'/kaggle/input/traffic-datasets/datasets/PEMS-BAY/adj_PEMS-BAY.pkl'
        with open(adj_path, 'rb') as f:
            pickle_data = pickle.load(f, encoding="latin1")
        adj_mx = pickle_data[2]
        adj = [self.asym_adj(adj_mx), self.asym_adj(np.transpose(adj_mx))]
        num_nodes = len(pickle_data[0])
        supports = [torch.tensor(i).to(self.device) for i in adj]
        # .float(): 将模型的参数和张量转换为浮点数类型
        model = self.model_dict[self.args.model].Model(device=self.device, supports=supports, num_nodes=num_nodes).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            # nn.DataParallel: 这是 PyTorch 中的一个模块，用于在多个 GPU 上并行地运行模型。
            # 它将输入模型封装在一个新的 DataParallel 模型中。
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        """
        data = {}
        dataset_dir = r'D:\博士阶段科研\文献阅读与汇报\时空数据挖掘\KDD\STEP-github\datasets\METR-LA'
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        print('mean:', data['x_train'][..., 0].mean())
        print('std:', data['x_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], 32)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], 32)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], 1)
        data['scaler'] = scaler
        """
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.0001)
        return model_optim

    def _select_criterion(self):
        criterion = masked_mae
        return criterion

    def asym_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        return d_mat.dot(adj).astype(np.float32).todense()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss, maes, mses, rmses, mapes, mspes = [], [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_x = torch.Tensor(batch_x).to(self.device)
                # batch_y = torch.Tensor(batch_y).to(self.device)

                outputs = self.model(batch_x)

                outputs = outputs.squeeze(-1)
                y = batch_y[:, self.args.label_len:, :, 0]
                if vali_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes = outputs.shape
                    outputs = vali_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                                 pred_len, n_nodes)
                    y = vali_data.inverse_transform(y.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                     pred_len, n_nodes)
                # predict = vali_data.inverse_transform(outputs)
                loss = criterion(outputs, y, 0.0)
                total_loss.append(loss.item())

                outputs = outputs.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                mae, mse, rmse, mape, mspe = metric(outputs, y)
                maes.append(mae)
                mses.append(mse)
                rmses.append(rmse)
                mapes.append(mape)
                mspes.append(mspe)

        total_loss, maes, mses, rmses, mapes, mspes = np.average(total_loss), np.average(maes), \
                                                      np.average(mses), np.average(rmses), \
                                                      np.average(mapes), np.average(mspes)
        self.model.train()
        return total_loss, maes, mses, rmses, mapes, mspes

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # dataloader = self._get_data(flag='train')
        # scaler = dataloader['scaler']

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        path = '/kaggle/working/'  # 使用kaggle跑代码的路径

        time_now = time.time()

        train_steps = len(train_data)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("Model = %s" % str(self.model))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # tensorboard_path = os.path.join('./runs/{}/'.format(setting))
        # if not os.path.exists(tensorboard_path):
        #     os.makedirs(tensorboard_path)
        tensorboard_path = '/kaggle/working/'  # 使用kaggle跑实验时的路径
        writer = SummaryWriter(log_dir=tensorboard_path)

        step = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # dataloader['train_loader'].shuffle()
            train_pbar = tqdm(train_loader, position=0, leave=True)  # 可视化训练的过程

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_pbar):
                iter_count += 1
                step += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_x = torch.Tensor(batch_x).to(self.device)
                # batch_y = torch.Tensor(batch_y).to(self.device)

                outputs = self.model(batch_x)
                outputs = outputs.squeeze(-1)
                y = batch_y[:, self.args.label_len:, :, 0]
                if train_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes = outputs.shape
                    outputs = train_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                                 pred_len, n_nodes)
                    y = train_data.inverse_transform(y.reshape(-1, n_nodes)).reshape(batch_size,
                                                                                     pred_len, n_nodes)
                # outputs = scaler.inverse_transform(outputs)
                loss = criterion(outputs, y, 0.0)
                train_loss.append(loss.item())

                train_pbar.set_description(f'Epoch [{epoch + 1}/{self.args.train_epochs}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})

                if (i + 1) % 100 == 0:
                    outputs = outputs.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                    mae, mse, rmse, mape, mspe = metric(outputs, y)
                    # print('pred:', outputs)
                    # print(outputs.shape)
                    # print('true:', y)
                    # print(y.shape)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | mae: {3:.7f} | mse: {4:.7f} | rmse: {5:.7f} | mape: {6:.7f} | mspe: {7:.7f}".
                          format(i + 1, epoch + 1, loss.item(), mae, mse, rmse, mape, mspe))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if self.clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali mae: {4:.7f} Vali mse: {5:.7f} "
                  "Vali rmse: {6:.7f} Vali mape: {7:.7f} Vali mspe: {8:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            writer.add_scalar(scalar_value=train_loss, global_step=step, tag='Loss/train')
            writer.add_scalar(scalar_value=vali_loss, global_step=step, tag='Loss/valid')
            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            # path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            path = '/kaggle/input/gwnet-models/checkpoint.pth'
            self.model.load_state_dict(torch.load(path))

        preds = []
        trues = []
        x_trues = []
        x_marks = []
        y_marks = []
        maes, mses, rmses, mapes, mspes = [], [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                outputs = outputs.squeeze(-1)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y[:, self.args.label_len:, :, 0].detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    batch_size, pred_len, n_nodes = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size, pred_len,
                                                                                                n_nodes)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, n_nodes)).reshape(batch_size, pred_len,
                                                                                                n_nodes)
                mae, mse, rmse, mape, mspe = metric(outputs, batch_y)
                maes.append(mae), mses.append(mse), rmses.append(rmse), mapes.append(mape), mspes.append(mspe)
                print("\tmae: {0:.7f} | mse: {1:.7f} | rmse: {2:.7f} | mape: {3:.7f} | mspe: {4:.7f}".format(mae, mse, rmse, mape, mspe))
                # print('pred:', outputs)
                # print('true:', batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                inputs = batch_x[:, :, :, 0].detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    batch_size, seq_len, n_nodes = inputs.shape
                    inputs = test_data.inverse_transform(inputs.reshape(-1, n_nodes)).reshape(batch_size, seq_len,
                                                                                              n_nodes)
                x_trues.append(inputs)
                batch_x_mark = batch_x_mark.detach().cpu().numpy()
                batch_y_mark = batch_y_mark.detach().cpu().numpy()
                x_marks.append(batch_x_mark)
                y_marks.append(batch_y_mark)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        x_trues = np.array(x_trues)
        x_marks = np.array(x_marks)
        y_marks = np.array(y_marks)
        maes = np.array(maes)
        mses = np.array(mses)
        rmses = np.array(rmses)
        mspes = np.array(mspes)
        mapes = np.array(mapes)
        print('test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape, mses.shape, rmses.shape, mspes.shape, mapes.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        x_trues = x_trues.reshape(-1, x_trues.shape[-2], x_trues.shape[-1])
        x_marks = x_marks.reshape(-1, x_marks.shape[-2], x_marks.shape[-1])
        y_marks = y_marks.reshape(-1, y_marks.shape[-2], y_marks.shape[-1])
        print('test shape:', preds.shape, trues.shape, x_trues.shape, x_marks.shape, y_marks.shape, maes.shape, mses.shape, rmses.shape, mspes.shape, mapes.shape)

        # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        folder_path = '/kaggle/working/'  # 使用kaggle跑代码时的输出路径

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x_trues.npy', x_trues)
        np.save(folder_path + 'x_marks.npy', x_marks)
        np.save(folder_path + 'y_marks.npy', y_marks)
        np.save(folder_path + 'maes.npy', maes)
        np.save(folder_path + 'mses.npy', mses)
        np.save(folder_path + 'rmses.npy', rmses)
        np.save(folder_path + 'mapes.npy', mapes)
        np.save(folder_path + 'mspes.npy', mspes)

        return


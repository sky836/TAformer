import argparse
import math
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from basicts.utils import load_pkl
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_STEP(Exp_Basic):
    def __init__(self, args, cfg):
        super(Exp_STEP, self).__init__(args, cfg)
        self.dataset_name = cfg["DATASET_NAME"]
        self.dataset_type = cfg["DATASET_TYPE"]
        self.scaler = load_pkl(
            "./{0}/scaler_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"],
                                                    cfg["DATASET_OUTPUT_LEN"]))

    def _build_model(self):  # 函数名称前加一个下划线表示私有函数，不希望在外部进行调用
        # .float(): 将模型的参数和张量转换为浮点数类型
        cfg = self.cfg
        model = self.model_dict[self.args.model].STEP(**cfg.MODEL.PARAM).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            # nn.DataParallel: 这是 PyTorch 中的一个模块，用于在多个 GPU 上并行地运行模型。
            # 它将输入模型封装在一个新的 DataParallel 模型中。
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def build_train_dataset(self):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        cfg = self.cfg
        data_file_path = "./{0}/data_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "./{0}/index_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "train"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True)

        return dataset, data_loader

    def build_val_dataset(self):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            validation dataset (Dataset)
        """
        cfg = self.cfg
        data_file_path = "./{0}/data_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "./{0}/index_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "valid"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("val len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True)

        return dataset, data_loader

    def build_test_dataset(self):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        cfg = self.cfg
        data_file_path = "./{0}/data_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "./{0}/index_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "test"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("test len: {0}".format(len(dataset)))
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True)

        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self.build_train_dataset()
        vali_data, vali_loader = self.build_val_dataset()

        # path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        path = '/kaggle/working/'  # 使用kaggle跑代码的路径

        time_now = time.time()

        train_steps = len(train_loader)
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

        for epoch in range(5):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            train_pbar = tqdm(train_loader, position=0, leave=True)  # 可视化训练的过程

            for i, (future_data, history_data, long_history_data) in enumerate(train_pbar):
                iter_count += 1
                step += 1
                model_optim.zero_grad()
                future_data = future_data.float().to(self.device)
                history_data = history_data.float().to(self.device)
                long_history_data = long_history_data.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, bernoulli_unnorm, adj_knn, gsl_coefficient = \
                                self.model(history_data=history_data, long_history_data=long_history_data, future_data=None, batch_seen=i, epoch=epoch)
                        else:
                            outputs, _, _, _ = self.model(history_data=history_data, long_history_data=long_history_data, future_data=None, batch_seen=i, epoch=epoch)

                        outputs = outputs.squeeze(-1).squeeze(-1)
                        loss = criterion(outputs, future_data[..., 0])
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs, bernoulli_unnorm, adj_knn, gsl_coefficient = \
                            self.model(history_data=history_data, long_history_data=long_history_data, future_data=None,
                                       batch_seen=i, epoch=epoch)
                    else:
                        outputs, _ = self.model(history_data=history_data, long_history_data=long_history_data,
                                                      future_data=None, batch_seen=i, epoch=epoch)

                    outputs = outputs.squeeze(-1).squeeze(-1)
                    loss = criterion(outputs, future_data[..., 0])
                    train_loss.append(loss.item())

                    # outputs = outputs.detach().cpu().numpy()
                    # y = batch_y[:, self.args.label_len:, :, 0].detach().cpu().numpy()
                    # if train_data.scale and self.args.inverse:
                    #     batch_size, pred_len, n_nodes = outputs.shape
                    #     outputs = train_data.inverse_transform(outputs.reshape(-1, n_nodes)).reshape(batch_size,
                    #                                                                                 pred_len, n_nodes)
                    #     y = train_data.inverse_transform(y.reshape(-1, n_nodes)).reshape(batch_size,
                    #                                                                                 pred_len, n_nodes)
                    # print('pred:', outputs)
                    # print(outputs.shape)
                    # print('true:', y)
                    # print(y.shape)

                train_pbar.set_description(f'Epoch [{epoch + 1}/{self.args.train_epochs}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
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
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.valid(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            writer.add_scalar(scalar_value=train_loss, global_step=step, tag='Loss/train')
            writer.add_scalar(scalar_value=vali_loss, global_step=step, tag='Loss/valid')
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def valid(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (future_data, history_data, long_history_data) in enumerate(vali_loader):
                future_data = future_data.float().to(self.device)
                history_data = history_data.float().to(self.device)
                long_history_data = long_history_data.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, bernoulli_unnorm, adj_knn, gsl_coefficient = \
                                self.model(history_data=history_data, long_history_data=long_history_data,
                                           future_data=None, batch_seen=i, epoch=None)
                        else:
                            outputs, _, _, _ = self.model(history_data=history_data, long_history_data=long_history_data,
                                                          future_data=None, batch_seen=i, epoch=None)
                        outputs = outputs.squeeze(-1).squeeze(-1)
                        loss = criterion(outputs, future_data[..., 0])
                        total_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs, bernoulli_unnorm, adj_knn, gsl_coefficient = \
                            self.model(history_data=history_data, long_history_data=long_history_data,
                                       future_data=None, batch_seen=i, epoch=None)
                    else:
                        outputs, _, _, _ = self.model(history_data=history_data, long_history_data=long_history_data,
                                                      future_data=None, batch_seen=i, epoch=None)
                    outputs = outputs.squeeze(-1).squeeze(-1)
                    loss = criterion(outputs, future_data[..., 0])
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self.build_test_dataset()
        scaler_0 = "./{0}/scaler_in{1}_out{2}.pkl".format(self.cfg["TEST"]["DATA"]["DIR"], self.cfg["DATASET_INPUT_LEN"],
                                                              self.cfg["DATASET_OUTPUT_LEN"])
        scaler_1 = "./{0}/scaler_in{1}_out{2}.pkl".format(self.cfg["TEST"]["DATA"]["DIR"], self.cfg["DATASET_ARGS"]["seq_len"],
                                                          self.cfg["DATASET_OUTPUT_LEN"])
        scaler0 = load_pkl(scaler_0)
        scaler1 = load_pkl(scaler_1)
        mean0 = scaler0['args']['mean']
        std0 = scaler0['args']['std']

        mean1 = scaler1['args']['mean']
        std1 = scaler1['args']['std']

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        histroy_trues = []
        long_histroy_trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (future_data, history_data, long_history_data) in enumerate(test_loader):
                future_data = future_data.float().to(self.device)
                history_data = history_data.float().to(self.device)
                long_history_data = long_history_data.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, bernoulli_unnorm, adj_knn, gsl_coefficient = \
                                self.model(history_data=history_data, long_history_data=long_history_data,
                                           future_data=None, batch_seen=i, epoch=None)
                        else:
                            outputs, _, _, _ = self.model(history_data=history_data, long_history_data=long_history_data,
                                                          future_data=None, batch_seen=i, epoch=None)
                        outputs = outputs.squeeze(-1).squeeze(-1)
                else:
                    if self.args.output_attention:
                        outputs, bernoulli_unnorm, adj_knn, gsl_coefficient = \
                            self.model(history_data=history_data, long_history_data=long_history_data,
                                       future_data=None, batch_seen=i, epoch=None)
                    else:
                        outputs, _, _, _ = self.model(history_data=history_data, long_history_data=long_history_data,
                                                      future_data=None, batch_seen=i, epoch=None)
                    outputs = outputs.squeeze(-1).squeeze(-1)
                if self.args.inverse:
                    outputs = outputs*std0 + mean0
                    future_data = future_data*std0 + mean0
                # print('pred:', outputs)
                # print('true:', future_data)

                pred = outputs.detach().cpu().numpy()
                true = future_data.detach().cpu().numpy()[..., 0]
                mae, mse, rmse, mape, mspe = metric(pred, true)
                print('mse:{}, mae:{}, rmse{}, mape{}, mspe{}'.format(mse, mae, rmse, mape, mspe))

                preds.append(pred)
                trues.append(true)

                if self.args.inverse:
                    history_data = history_data.detach().cpu().numpy()[..., 0]
                    history_data = history_data*std0 + mean0
                    # long_history_data = long_history_data*std1 + mean1
                # long_histroy_trues.append(long_history_data.detach().cpu().numpy())
                histroy_trues.append(history_data)
        preds = np.array(preds)
        trues = np.array(trues)
        histroy_trues = np.array(histroy_trues)
        # long_histroy_trues = np.array(long_histroy_trues)
        print('shape:', preds.shape, trues.shape, histroy_trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        histroy_trues = histroy_trues.reshape(-1, histroy_trues.shape[-2], histroy_trues.shape[-1])
        # long_histroy_trues = long_histroy_trues.reshape(-1, long_histroy_trues.shape[-2], long_histroy_trues.shape[-1])
        print('shape:', preds.shape, trues.shape, histroy_trues.shape)
        # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        folder_path = '/kaggle/working/'  # 使用kaggle跑代码时的输出路径

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_STEP.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'histroy_trues.npy', histroy_trues)
        # np.save(folder_path + 'long_histroy_trues.npy', long_histroy_trues)

        return




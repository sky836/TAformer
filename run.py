import argparse
from datetime import datetime

import torch
from exp.exp_forcast import Exp_Forecast
from exp.exp_STEP import Exp_STEP
from exp.exp_timeLinear import Exp_TimeLinear
from exp.exp_GWNET import Exp_GWNET
# from utils.print_args import print_args
import random
import numpy as np
from step import STEP_PEMS04
cfg = STEP_PEMS04.CFG
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Taformer')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='forcast',
                        help='task name, options:[forcast, STEP, timeLinear, GWNET]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model', type=str, required=False, default='Taformer',
                        help='model name, options: [Taformer, STEP, timeLinear, GWNET]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='METR-LA', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets/raw_data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='METR-LA/METR-LA.h5', help='data file')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--time_to_feature', type=int, default=1,
                        help='Adding time features to the data. options: [0, 1], 0 stands for 2 features. 1 stands for 4 features')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=12*24, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)

    # model define
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--encoder_embed_dim', type=int, default=96, help='dimension of model')
    parser.add_argument('--d_model', type=int, default=96, help='dimension of timeLinear')
    parser.add_argument('--decoder_embed_dim', type=int, default=96, help='dimension of decoder')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--decoder_num_heads', type=int, default=8, help='num of decoder heads')
    parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=6, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--patch_size', type=int, default=12, help='The size of one patch')
    parser.add_argument('--label_patch_size', type=int, default=12, help='The size of one  decoder input patch')
    parser.add_argument('--time_channel', type=int, default=4, help='The channel of time inputs')
    parser.add_argument('--target_channel', type=int, default=1, help='The channel of target inputs')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    # print_args(args)

    if args.task_name == 'forecast':
        Exp = Exp_Forecast
    elif args.task_name == 'STEP':
        Exp = Exp_STEP
    elif args.task_name == 'timeLinear':
        Exp = Exp_TimeLinear
    elif args.task_name == 'GWNET':
        Exp = Exp_GWNET
    else:
        Exp = Exp_Forecast

    # 获取当前日期和时间
    current_datetime = datetime.now()

    # 格式化为字符串
    formatted_string = current_datetime.strftime("%Y_%m_%d_%H_%M")

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            if args.task_name == 'STEP':
                exp = Exp(args=args, cfg=cfg)  # set experiments
            else:
                exp = Exp(args)
            setting = '{}_{}_sl{}_pl{}_el{}_dl{}_eh{}_dh{}_de{}_ee{}_{}_{}'.format(
                args.task_name,
                args.data,
                args.seq_len,
                args.pred_len,
                args.e_layers,
                args.d_layers,
                args.n_heads,
                args.decoder_num_heads,
                args.decoder_embed_dim,
                args.encoder_embed_dim,
                formatted_string,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if args.task_name != 'pretrain':
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_sl{}_pl{}_el{}_dl{}_eh{}_dh{}_de{}_ee{}_{}_{}'.format(
            args.task_name,
            args.data,
            args.seq_len,
            args.pred_len,
            args.e_layers,
            args.d_layers,
            args.n_heads,
            args.decoder_num_heads,
            args.decoder_embed_dim,
            args.encoder_embed_dim,
            formatted_string,
            ii)

        setting = 'GWNET_METR-LA_sl12_pl12_el2_dl2_eh4_dh8_de96_ee96_2024_03_07_11_58_0'

        if args.task_name == 'STEP':
            exp = Exp(args=args, cfg=cfg)  # set experiments
        else:
            exp = Exp(args)
        if args.task_name != 'pretrain':
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()

from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_h5, Dataset_PEMS04

data_dict = {
    'METR-LA': Dataset_h5,
    'PEMS-BAY': Dataset_h5,
    'PEMSO4': Dataset_PEMS04
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    time_to_feature = args.time_to_feature
    if flag == 'test':
        shuffle_flag = False
        # drop_last 是 DataLoader 类的一个参数，用于指定在数据集大小
        # 不能被批次大小整除时是否**丢弃最后一个小于批次大小的 batch**。
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        time_to_feature=time_to_feature
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

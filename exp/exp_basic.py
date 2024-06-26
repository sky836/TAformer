import os
import torch
from models import timeLinear, GWNET, taformerPredict, taformerPretrain
from step.step_arch import step

class Exp_Basic(object):
    def __init__(self, args, cfg=None):
        self.args = args
        self.model_dict = {
            'Taformer': taformerPredict,
            'STEP': step,
            'timeLinear': timeLinear,
            'GWNET': GWNET,
            'Pretrain': taformerPretrain
        }
        self.clip = 5
        self.device = self._acquire_device()
        self.cfg = cfg
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"]为获取名为"CUDA_VISIBLE_DEVICES"的环境变量的值
            # "CUDA_VISIBLE_DEVICES"环境变量用于指定程序在执行时可以使用哪些GPU设备。它的值是一个逗号
            # 分隔的设备索引列表，例如："0,1,2" 表示程序将使用索引为 0、1 和 2 的GPU。
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

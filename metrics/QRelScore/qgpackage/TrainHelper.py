import os
import importlib
import logging
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable, Function
import numpy as np

def sort_by_order_file(train_examples, order_filename):
    # list to dict
    source = { example.qas_id: example for example in train_examples }
    ret = [ ]
    with open(order_filename) as f:
        order_json = json.load(f)
    for qid, difficulty_score in order_json:
        ret.append(source[qid])

    return ret

class LoggerPather:
    def __init__(self, cfg):
        # rootpath / experiement_key
        self.root_output_dir = Path(cfg.SUMMARY_DIR) / cfg.NAME
        self.root_output_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.root_output_dir / 'log'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_file = '{}_{}.log'.format(cfg.NAME, time_str)
        log_file_full_name = self.log_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(log_file_full_name),
                            format=head)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        self.tensorboard_path = self.root_output_dir / 'tensorboard'
        self.tensorboard_path.mkdir(parents=True, exist_ok=True)

        self.snapshot_path = self.root_output_dir / 'snapshot'
        self.snapshot_path.mkdir(parents=True, exist_ok=True)

        self.prediction_path = self.root_output_dir / 'prediction'
        self.prediction_path.mkdir(parents=True, exist_ok=True)

    def get_logger(self):
        return self.logger
    
    def get_log_path(self):
        return str(self.log_dir)

    def get_snapshot_path(self):
        return str(self.snapshot_path)

    def get_tb_path(self):
        return str(self.tensorboard_path)

    def get_prediction_path(self):
        return str(self.prediction_path)

    def get_prediction_csv_file_name(self):
        return str(self.prediction_csv_file_name)

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

class ModelWrapper:
    def __init__(self):
        pass
    def __call__(self, model, ema = False, wrapped_device = 'cpu'):
        if type(wrapped_device) == list and len(wrapped_device) > 1:
            model = nn.DataParallel(model, device_ids = wrapped_device)
        model.to(torch.device(wrapped_device if wrapped_device == 'cpu' else 'cuda:' + str(wrapped_device[0])))

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

class DeviceWrapper:
    '''
        make sure device without any spaces or tabs
        returned device is either [0, 1, ...] or cpu
    '''
    def __init__(self):
        pass
    def __call__(self, device):
        if 'cuda' in device:
            if ',' in device:
                device = list(map(int, device.split(':')[1].split(',')))
            else:
                device = [ int(device.split(':')[1]) ]
        return device

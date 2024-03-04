import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pytorch_lightning as pl

import os
import json

from .SquadV1Dataset import *

class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        
        D = eval(self.config.TRAIN.DATASET)

        self.train_dataset = D(self.config, 'train', self.tokenizer)
        self.val_dataset = D(self.config, 'val', self.tokenizer)
        self.test_dataset = D(self.config, 'test', self.tokenizer)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size = self.config.TRAIN.BATCH_SIZE,
                          num_workers = self.config.TRAIN.WORKERS,
                          shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size = self.config.VAL.BATCH_SIZE,
                          num_workers = self.config.VAL.WORKERS,
                          shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size = self.config.TEST.BATCH_SIZE,
                          num_workers = self.config.TEST.WORKERS,
                          shuffle = False)

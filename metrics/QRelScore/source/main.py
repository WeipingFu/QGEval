import sys
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
import argparse

from qgpackage import Architecture
from qgpackage.DataModule import DataModule
from qgpackage.TrainerModule import *
from qgpackage.TrainHelper import LoggerPather

from configure.default import config, update_config

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything
seed_everything(config.SEED, workers = True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line interface",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    parse_args()
    # instantiate a vanilla model and a tokenizer
    model, tokenizer = Architecture.get_model(config)

    # DataModule
    dm = DataModule(tokenizer, config)

    # LoggerPather
    loggerpather = LoggerPather(config)

    # tm is aware of the existence of dm
    train_dataloader, val_dataloader = dm.train_dataloader(), dm.val_dataloader()

    # get class of LightningModule
    TM = eval(config.TRAIN.TRAINER_MODULE)
    # instantiation
    tm = TM(model, tokenizer, train_dataloader, val_dataloader, config)

    # latest snapshot to resume the state dict of the model, optimizer and lr_scheduler
    latest_snapshot = os.path.join(loggerpather.get_snapshot_path(), 'last.ckpt')
    checkpoint_to_resume = latest_snapshot if config.TRAIN.RESUME and os.path.isfile(latest_snapshot) \
                                           else None
    # trainer config
    trainer = pl.Trainer(
        resume_from_checkpoint = checkpoint_to_resume,
        gpus = config.DEVICE[5:],
        # gpus = 0,
        accelerator = config.ACCELERATOR,
        benchmark = config.CUDNN.BENCHMARK,
        deterministic = config.CUDNN.DETERMINISTIC,
        default_root_dir = loggerpather.get_log_path(),
        max_epochs = config.TRAIN.NUM_EPOCHS,
        logger = TensorBoardLogger(loggerpather.get_tb_path(), name = '', version = '', default_hp_metric = False),
        callbacks = [
            EarlyStopping(monitor = 'val/loss', patience = config.TRAIN.PATIENCE),
            ModelCheckpoint(monitor = 'val/loss', dirpath = loggerpather.get_snapshot_path(), filename = 'best', save_last = True),
        ]
    )
    
    # train
    trainer.fit(model = tm, datamodule = dm)

    # best snapshot to perform test or inference, omitting optimizer and lr_scheduler state
    best_snapshot = os.path.join(loggerpather.get_snapshot_path(), 'best.ckpt')
    trainer.test(ckpt_path = best_snapshot)

if __name__ == '__main__':
    main()
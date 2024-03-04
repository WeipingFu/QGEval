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

    # best snapshot
    best_snapshot = os.path.join(loggerpather.get_snapshot_path(), 'best.ckpt')
    assert os.path.isfile(best_snapshot), 'Cannot find the best checkpoint to resume or evaluate'

    # get class of LightningModule
    TM = eval(config.TRAIN.TRAINER_MODULE)
    # instantiation
    tm = TM.load_from_checkpoint(best_snapshot,
                                 map_location = 'cpu',
                                 hparams_file = None,
                                 strict = True,
                                 model = model,
                                 tokenizer = tokenizer,
                                 train_dataloader = train_dataloader,
                                 val_dataloader = val_dataloader,
                                 config = config)

    # trainer config
    trainer = pl.Trainer(
        resume_from_checkpoint = None,
        # gpus = config.DEVICE[5:6], # here we always test it in only one rank_node to ensure write only one prediction file
        gpus = config.DEVICE[5:],
        accelerator = config.ACCELERATOR,
        benchmark = config.CUDNN.BENCHMARK,
        deterministic = config.CUDNN.DETERMINISTIC,
        default_root_dir = loggerpather.get_log_path(),
        logger = TensorBoardLogger(loggerpather.get_tb_path(), name = '', version = '', default_hp_metric = False)
    )

    # test or inference with the best snapshot, containing the best state dict of the model (nn.Module)
    # before that, the model (pl.LightningModule) must been set in some run !!!!
    trainer.test(model = tm, datamodule = dm)

if __name__ == '__main__':
    main()
import os

from yacs.config import CfgNode as CN

_C = CN()

_C.NAME = ''
_C.DEVICE = ''
_C.ACCELERATOR = ''
_C.SEED = 32767
_C.SUMMARY_DIR = ''

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# pretext or surrgate task pretraining
_C.PRETEXT = CN(new_allowed = True)

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.LM_TYPE = 'AutoModelForCausalLM'
_C.MODEL.PRETRAINED_MODEL_NAME_OR_PATH = 'gpt2'
_C.MODEL.DO_LOWER_CASE = False
_C.MODEL.MAX_INPUT_LENGTH = 484
_C.MODEL.DOC_STRIDE = 356
_C.MODEL.MAX_QUERY_LENGTH = 128
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.SPECIAL_TOKENS = CN()
_C.MODEL.SPECIAL_TOKENS.PAD_TOKEN = '<pad>'
_C.MODEL.SPECIAL_TOKENS.SEP_TOKEN = '<sep>'
_C.MODEL.SPECIAL_TOKENS.CLS_TOKEN = '<cls>'
_C.MODEL.SPECIAL_TOKENS.EOS_TOKEN = '<eos>'
_C.MODEL.SPECIAL_TOKENS.BOS_TOKEN = '<bos>'
_C.MODEL.SPECIAL_TOKENS.CXT_TOKEN = '<cxt>'
_C.MODEL.SPECIAL_TOKENS.ANS_TOKEN = '<ans>'
_C.MODEL.SPECIAL_TOKENS.QUE_TOKEN = '<que>'

# contrastive learning
_C.CONTRASTIVE = CN(new_allowed=True)

# curriulum learning
_C.CURRICULUM = CN(new_allowed=True)

# reinforcement learning
_C.RL = CN(new_allowed=True)

# training
_C.TRAIN = CN()
_C.TRAIN.TRAINER_MODULE = ''
_C.TRAIN.DATASET = ''
_C.TRAIN.DATASET_FILENAME = ''

_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.WORKERS = 4
_C.TRAIN.NUM_EPOCHS = 288
_C.TRAIN.NUM_WARMUP_STEPS = 320
_C.TRAIN.PATIENCE = 32
_C.TRAIN.RESUME = True
_C.TRAIN.LOSS_FREQ = 10
_C.TRAIN.TB_FREQ = 10
_C.TRAIN.DEV_FREQ = 10
_C.TRAIN.UNLABELED = CN(new_allowed=True)

_C.TRAIN.LR = 0.001
_C.TRAIN.EXTRA_LR = 0.0001
_C.TRAIN.LD = 0.9
_C.TRAIN.WD = 5.0e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NESTEROV = False
_C.TRAIN.REDUCTION = 'mean'

# validating
_C.VAL = CN()
_C.VAL.DATASET_FILENAME = ''
_C.VAL.BATCH_SIZE = 8
_C.VAL.WORKERS = 32

# testing
_C.TEST = CN()
_C.TEST.DATASET_FILENAME = ''
_C.TEST.BATCH_SIZE = 32
_C.TEST.WORKERS = 32

# generating
_C.GENERATE = CN(new_allowed=True)

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False

config = _C

def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

if __name__ == '__main__':
    import sys
    import os
    filename = os.path.join(os.path.dirname(__file__), sys.argv[1])
    with open(filename, 'w') as f:
        print(_C, file=f)

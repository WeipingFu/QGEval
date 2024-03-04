import os

from pprint import pprint
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .TrainHelper import AverageMeter, LoggerPather

def get_tokenizer(pretrained_model_name_or_path, config):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        do_lower_case = config.MODEL.DO_LOWER_CASE
    )

    # add the special token with well-known and common attribute name
    if tokenizer.pad_token is None:
        print('set pad_token...')
        tokenizer.add_special_tokens({ 'pad_token': config.MODEL.SPECIAL_TOKENS.PAD_TOKEN })
    if tokenizer.cls_token is None:
        print('set cls_token...')
        tokenizer.add_special_tokens({ 'cls_token': config.MODEL.SPECIAL_TOKENS.CLS_TOKEN })
    if tokenizer.sep_token is None:
        print('set sep_token...')
        tokenizer.add_special_tokens({ 'sep_token': config.MODEL.SPECIAL_TOKENS.SEP_TOKEN })
    if tokenizer.bos_token is None:
        print('set bos_token...')
        tokenizer.add_special_tokens({ 'bos_token': config.MODEL.SPECIAL_TOKENS.BOS_TOKEN })
    if tokenizer.eos_token is None:
        print('set eos_token...')
        tokenizer.add_special_tokens({ 'eos_token': config.MODEL.SPECIAL_TOKENS.EOS_TOKEN })

    # add other special token here
    tokenizer.add_tokens([ config.MODEL.SPECIAL_TOKENS.CXT_TOKEN, config.MODEL.SPECIAL_TOKENS.ANS_TOKEN, config.MODEL.SPECIAL_TOKENS.QUE_TOKEN ])

    return tokenizer


def get_model(config):
    pretrained_model_name_or_path = config.MODEL.PRETRAINED_MODEL_NAME_OR_PATH

    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    tokenizer = get_tokenizer(pretrained_model_name_or_path, config)
    LM = eval(config.MODEL.LM_TYPE)
    model = LM.from_pretrained(
        pretrained_model_name_or_path,
        config = model_config
    )
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

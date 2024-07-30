import logging
import os
from dataclasses import dataclass, field
from datasets import load_dataset
from typing import Dict, List, Optional
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from utils import load_json, save_json

logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = '5'

# dataset
class BartDataset():
    def __init__(self, config_path, pretrained_name_or_path=None):
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.config = load_json(config_path)
        if pretrained_name_or_path is None:
            self.tokenizer = BartTokenizer.from_pretrained(self.config['tokenizer_name_or_path'])
        else:
            self.tokenizer = BartTokenizer.from_pretrained(pretrained_name_or_path)

    def load_dataset_from_json(self, train_path, dev_path, test_path=''):
        if test_path != '':
            dataset = load_dataset('json', data_files={'train': train_path, 'dev': dev_path, 'test': test_path})
            self.train_dataset = dataset['train']
            self.dev_dataset = dataset['dev']
            self.test_dataset = dataset['test']
        else:
            dataset = load_dataset('json', data_files={'train': train_path, 'dev': dev_path})
            self.train_dataset = dataset['train']
            self.dev_dataset = dataset['dev']

    def load_dataset_by_name(self, name):
        dataset = load_dataset(name)

    # tokenize the examples
    def convert_to_features(self, example_batch):
        input_encodings = self.tokenizer.batch_encode_plus(example_batch['source'], truncation=True,
                                                           max_length=self.config['max_len'],
                                                           padding='max_length',
        )
        target_encodings = self.tokenizer.batch_encode_plus(example_batch['target'], truncation=True,
                                                            padding='max_length',
                                                            max_length=self.config['target_max_len'],
        )
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'target_ids': target_encodings['input_ids'],
            'target_attention_mask': target_encodings['attention_mask']
        }
        return encodings

    def save_tokenized_data(self, save_dir):
        columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
        self.train_dataset.set_format(type='torch', columns=columns)
        self.dev_dataset.set_format(type='torch', columns=columns)
        if self.test_dataset is not None:
            self.test_dataset.set_format(type='torch', columns=columns)
        print('dataset count')
        if self.test_dataset is not None:
            print("train: {}, dev: {}, test: {}".format(len(self.train_dataset), len(self.dev_dataset),
                                                        len(self.test_dataset)))
            torch.save(self.test_dataset, save_dir + '/test.pt')
        else:
            print("train: {}, dev: {}".format(len(self.train_dataset), len(self.dev_dataset)))
        torch.save(self.train_dataset, save_dir + '/train.pt')
        torch.save(self.dev_dataset, save_dir + '/dev.pt')
        print('save data success!')

    def apply(self, save_dir, train_path, dev_path, test_path=''):
        # load data
        self.load_dataset_from_json(train_path, dev_path, test_path)
        # encode data
        self.train_dataset = self.train_dataset.map(self.convert_to_features, batched=True)
        self.dev_dataset = self.dev_dataset.map(self.convert_to_features, batched=True)
        if self.test_dataset is not None:
            self.test_dataset = self.test_dataset.map(self.convert_to_features, batched=True)
       
        # save data
        self.save_tokenized_data(save_dir)

@dataclass
class BartDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        # for example in batch:
        #     print(example.keys())
        #     break
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': lm_labels,
            'decoder_attention_mask': decoder_attention_mask
        }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='train.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='dev.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )


def setup_logs(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)


def train_model(arg_path):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=arg_path)
    setup_logs(training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer, model = None, None
    if model_args.tokenizer_name_or_path:
        tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name_or_path, cache_dir=model_args.cache_dir)
    if model_args.model_name_or_path:
        if tokenizer is None:
            tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    if model is None:
        logging.warning('no model')
        return
    if tokenizer is None:
        logging.warning('no tokenizer')
        return

    # Get datasets
    logging.info('loading data')
    train_dataset = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    logging.info('loading done')
    # print(train_dataset)
    # print(valid_dataset)

    data_collator = BartDataCollator()
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Training
    if training_args.do_train:
        logger.info('#' * 50)
        logger.info('start training')
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir + '/tokenizer')

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")
        eval_output = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results

# train QG model
def train_qg(arg_path, train_path, dev_path, test_path, arg_dict=None):
    if arg_dict is not None:
        save_json(arg_dict, arg_path)
    print(arg_path)
    arg_dict = load_json(arg_path)
    # if not os.path.exists(arg_dict['train_file_path']):
    print('start to process data')
    dataset = BartDataset(arg_path, arg_dict['model_name_or_path'])
    save_dir = os.path.dirname(arg_dict['train_file_path'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset.apply(save_dir, train_path, dev_path, test_path)
    print('processed data saved to {}'.format(save_dir))
    train_model(arg_path)

# predict with QG model
def predict(model_dir, tokenizer_dir, test_pt_path, result_save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    test_dataset = torch.load(test_pt_path)
    model = BartForConditionalGeneration.from_pretrained(model_dir).to(device)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_dir)
    print('data and model loaded')
    print(model)
    print('data count: {}'.format(len(test_dataset['input_ids'])))
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    decoded_texts = []
    print('Start predict')
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        output_ids = model.generate(input_ids=input_ids, max_length=153)
        batch_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        decoded_texts.extend(batch_texts)
    assert len(test_dataset['source']) == len(decoded_texts)
    res_dict = {}
    for key in test_dataset.features:
        if key not in ['input_ids', 'attention_mask', 'target_ids', 'target_attention_mask']:
            res_dict[key] = test_dataset[key]
    res_dict['prediction'] = decoded_texts
    result = pd.DataFrame(res_dict)
    if result_save_path is not None:
        result.to_csv(result_save_path, encoding='utf-8', index=False)
       
        print('result saved to {}'.format(result_save_path))
    print('done {} samples'.format(len(decoded_texts)))
    return decoded_texts


if __name__ == "__main__":
    # train QG model
    arg_path = './args/bart_large_hotpot.json'
    data_dir = './data/data_bart/hotpotqa/'
    train_path = data_dir + 'train.json'
    dev_path = data_dir + 'dev.json'
    test_path = data_dir + 'test.json'
    train_qg(arg_path, train_path, dev_path, test_path)
    
    # # test
    # arg_path = './args/bart_large_hotpot.json'
    # test_pt_path = './data/data_bart/hotpotqa/pt/test.pt'
    # model_dir = './model/bart-large/hotpotqa/'
    # tokenizer_dir = model_dir + 'tokenizer/'
    # result_save_path = './model/bart-large/hotpotqa/result/prediction.csv'
    # decoded_texts = predict(model_dir, tokenizer_dir, test_pt_path, result_save_path)

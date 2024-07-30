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
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from utils import load_json, save_json
from peft import LoraConfig, TaskType
from peft import get_peft_model

logger = logging.getLogger(__name__)


# dataset
class FlanT5Dataset():
    def __init__(self, config_path, pretrained_name_or_path=None):
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.config = load_json(config_path)
        if pretrained_name_or_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path)

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
        input_encodings = self.tokenizer.batch_encode_plus(example_batch['source'],
                                                           truncation=True,
                                                           padding='max_length',
                                                           max_length=self.config['max_len'],
                        )
        target_encodings = self.tokenizer.batch_encode_plus(example_batch['target'],
                                                            truncation=True,
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

    def length_count(self, name, lengths):
        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)
        k = int(n * 0.9)
        # print(k, n)
        max_length_9 = max(sorted_lengths[:k])
        print("{}: max_length={}, 90%_length={}, mean_length={}".format(
            name, max(lengths), max_length_9, sum(lengths)/n
        ))

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
class T5DataCollator:
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


class BestModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric = float('inf')  # Initialize with a high value

    def save_model(self, output_dir=None):
        if self.state.is_world_process_zero:
            # Check if the current model is the best based on the desired metric
            if self.best_metric > self.state.eval_metric:
                self.best_metric = self.state.eval_metric
                super().save_model(output_dir)  # Save the model

def train_model(arg_path):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=arg_path)
    setup_logs(training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer, model = None, None
    if model_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, cache_dir=model_args.cache_dir)
    if model_args.model_name_or_path:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    if model is None:
        logging.warning('no model')
        return
    if tokenizer is None:
        logging.warning('no tokenizer')
        return
    
    # peft
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


    # Get datasets
    logging.info('loading data')
    train_dataset = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    logging.info('loading done')
    # print(train_dataset)
    # print(valid_dataset)

    data_collator = T5DataCollator()
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Training
    if training_args.do_train:
        logger.info('#' * 50)
        logger.info('start training')
        trainer.train()
        model.save_pretrained(training_args.output_dir)
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
    if not os.path.exists(arg_dict['train_file_path']):
        print('start to process data')
        dataset = FlanT5Dataset(arg_path, arg_dict['model_name_or_path'])
        save_dir = os.path.dirname(arg_dict['train_file_path'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dataset.apply(save_dir, train_path, dev_path, test_path)
        print('processed data saved to {}'.format(save_dir))
    train_model(arg_path)

# predict with QG model
def predict(pretraied_dir, model_dir, tokenizer_dir, test_pt_path, result_save_path=None):
    from peft import PeftModel, PeftConfig
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    test_dataset = torch.load(test_pt_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretraied_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = PeftModel.from_pretrained(model, model_dir).to(device)
    print('data and model loaded')
    print(model)
    print('data count: {}'.format(len(test_dataset['input_ids'])))
    dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    decoded_texts = []
    print('Start predict')
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        output_ids = model.generate(input_ids=input_ids)
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
        save_dir = os.path.dirname(result_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        result.to_csv(result_save_path, encoding='utf-8', index=False)
        
        print('result saved to {}'.format(result_save_path))
    print('done {} samples'.format(len(decoded_texts)))
    return decoded_texts


# few-shot prediction
def predict_few(model_path, few_shot_path, test_path, result_save_path, name, k=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_tar_length = {
        'squad1.1': 67,
        'hotpotqa': 163
    }
    max_length = 4096
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inst = 'Generate a question based on the given answer and context.\n'
    few_data = load_json(few_shot_path)
    print('total few-shot examples count={}'.format(len(few_data)))
    test_data = load_json(test_path)
    result = []
    for one in tqdm(test_data):
        # instruction
        for_pred = 'Answer: {}\nContext: {}\nQuestion: '.format(
            one['answer'], one['context']
        )
        
        examples = []
        input_text = inst + '\n'.join(examples) + for_pred
        for i in range(k):
            ness_len = tokenizer(input_text, return_tensors="pt")['input_ids'].size(-1)
            if ness_len > max_length:
                if len(examples) == 0:
                    input_text = inst + for_pred
                else:
                    input_text = inst + '\n'.join(examples[:-1]) + '\n' + for_pred
                break
            example = 'Answer: {}\nContext: {}\nQuestion: {}'.format(
                few_data[i]['answer'], few_data[i]['context'], few_data[i]['question']
            )
            examples.append(example)
            input_text = inst + '\n'.join(examples) + '\n' +for_pred

        if len(result) == 0:
            print(input_text)
        # tokenize
        inputs = tokenizer(input_text, max_length=max_length, truncation=True, return_tensors="pt")
        print('example count={}, input token length={}'.format(
            len(examples), inputs['input_ids'].size(-1)
        ))
        input_ids = inputs['input_ids'].to(device)
        # generate
        outputs = model.generate(input_ids=input_ids, max_length=max_tar_length[name])
        decoded_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print('prediction: ', decoded_text)
        one['prediction'] = decoded_text
        result.append(one)
    # save result
    pd.DataFrame(result).to_csv(result_save_path, index=False)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # train QG model
    arg_path = 'args/flant5_xxl_squad_lora.json'
    data_dir = './data/data_flant5/squad1.1/'
    train_path = data_dir + 'train.json'
    dev_path = data_dir + 'dev.json'
    test_path = data_dir + 'test.json'
    train_qg(arg_path, train_path, dev_path, test_path)

    # # test
    # test_pt_path = './data/data_flant5/squad1.1/pt-xxl/test.pt'
    # pretrained_dir = 'google/flan-t5-xxl'
    # model_dir = './model/flant5-xxl/squad1.1/'
    # tokenizer_dir = model_dir + 'tokenizer/'
    # result_save_path = model_dir + 'result/prediction.csv'
    # decoded_texts = predict(pretrained_dir, model_dir, tokenizer_dir, test_pt_path, result_save_path)

    # # few-shot prediction
    # model_path = 'google/flan-t5-xxl'
    # few_shot_path = './data/squad1.1-few.json'
    # test_path = './data/squad-dev.json'
    # result_save_path = './data/predictions/squad-flant5-xxl-few.csv'
    # predict_few(model_path, few_shot_path, test_path, result_save_path, 'squad1.1')

    
   
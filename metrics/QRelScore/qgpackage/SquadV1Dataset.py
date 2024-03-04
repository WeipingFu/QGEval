import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pytorch_lightning as pl

import os
import json

from .DataCommon import DatasetMixin, CLMMixin, Seq2SeqLMMixin

class SquadV1DatasetMixin(Dataset):
    '''
        Prerequisite object/instance-related attributes: self.config, self.split_set, self.tokenizer
    '''
    def parse_and_build(self):
        if self.split_set == 'train':
            with open(self.config.TRAIN.DATASET_FILENAME) as f_train:
                train_set = json.load(f_train)
                self.data_file = train_set
        elif self.split_set == 'val':
            with open(self.config.VAL.DATASET_FILENAME) as f_val:
                val_set = json.load(f_val)
                self.data_file = val_set
        elif self.split_set == 'test':
            with open(self.config.TEST.DATASET_FILENAME) as f_test:
                test_set = json.load(f_test)
                self.data_file = test_set
        else:
            self.data_file = None

        cxt_token = self.config.MODEL.SPECIAL_TOKENS.CXT_TOKEN
        ans_token = self.config.MODEL.SPECIAL_TOKENS.ANS_TOKEN

        # convert
        self.contexts = [ ]
        self.questions = [ ]
        self.answers = [ ]
        self.answer_starts = [ ]
        input_contexts = [ ]
        self.qids = [ ]

        for doc in self.data_file['data']:
            title = doc['title']
            for par in doc['paragraphs']:
                context = par['context']
                for qa in par['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]['text']
                    answer_start = qa['answers'][0]['answer_start']
                    qid = qa['id']

                    self.contexts.append(context)
                    self.questions.append(question)
                    self.answers.append(answer)
                    self.answer_starts.append(answer_start)

                    # trailing hint/suffix tokens for answer-aware QG
                    # input_context = context + ' ' + cxt_token + ' ' + \
                    #                 answer + ' ' + ans_token

                    # leading prefix tokens for answer-aware QG
                    input_context = cxt_token + ' ' + context + ' ' + \
                                    ans_token + ' ' + answer
                    
                    input_contexts.append(input_context)
                    self.qids.append(qid)
        
        if self.split_set in { 'train', 'val' }:
            train_inputs, test_inputs = self.prepare_input(input_contexts, self.questions)
        else:
            train_inputs, test_inputs = self.prepare_input(input_contexts)

        self.train_inputs, self.test_inputs = train_inputs, test_inputs

    def __getitem__(self, index):
        train_input = { }
        test_input = { }
        question_text = self.questions[index]
        qid = self.qids[index]

        if self.split_set in { 'train', 'val' }:
            for key in self.train_inputs.keys():
                train_input[key] = self.train_inputs[key][index]
            for key in self.test_inputs.keys():
                test_input[key] = self.test_inputs[key][index]

            if self.split_set == 'train':
                return train_input, test_input, question_text
            elif self.split_set == 'val':
                return train_input, test_input, question_text, qid
        else:
            for key in self.test_inputs.keys():
                test_input[key] = self.test_inputs[key][index]

            # fake test in our experiments, because they contain the ground truth sequence
            return test_input, question_text, qid

    def __len__(self):
        return len(self.qids)

class SquadV1CLMDataset(SquadV1DatasetMixin, CLMMixin):
    def __init__(self, config, split_set, tokenizer):
        super().__init__()
        self.config = config
        self.split_set = split_set
        self.tokenizer = tokenizer

        self.parse_and_build()


class SquadV1Seq2SeqLMDataset(SquadV1DatasetMixin, Seq2SeqLMMixin):
    def __init__(self, config, split_set, tokenizer):
        super().__init__()
        self.config = config
        self.split_set = split_set
        self.tokenizer = tokenizer

        self.parse_and_build()
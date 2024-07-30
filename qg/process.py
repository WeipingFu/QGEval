from utils import load_json, save_json
import os
from collections import Counter
import pandas as pd
import re
import random

# prepare squad data for T5 series models
class T5Squad:
    def deal_squad(self, save_dir = './data/data_t5/squad1.1/'):
        tr = load_json('./data/squad1.1/train-v1.1.json')
        dev = load_json('./data/squad1.1/dev-v1.1.json')

        print(len(tr['data']), len(dev['data']))
        valid = dev['data'][:24]
        test = dev['data'][24:]
        train = tr['data']
        print(len(train), len(valid), len(test))
        train_data, valid_data, test_data = [], [], []
        for one in train:
            train_data += self.deal_one_squad(one)
        for one in valid:
            valid_data += self.deal_one_squad(one)
        for one in test:
            test_data += self.deal_one_squad(one)
        print('sample count, train={}, valid={}, test={}'.format(
            len(train_data), len(valid_data), len(test_data)))
        
        # save processed data
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_json(train_data, save_dir + 'train.json')
        save_json(valid_data, save_dir + 'dev.json')
        save_json(test_data, save_dir + 'test.json')

    def deal_one_squad(self, one):
        samples = []
        for para in one['paragraphs']:
            context = para['context']
            for item in para['qas']:
                question = item['question']
                answers = [x['text'] for x in item['answers']]
                answer = Counter(answers).most_common(1)[0][0]
                source = 'answer: {}  context: {}'.format(answer, context)
                one_sample = {
                    'id': item['id'],
                    'source': source,
                    'target': question
                }
                samples.append(one_sample)
        return samples

# prepare squad data for BART series models
class BartSquad:
    def deal_squad(self, save_dir = './data/data_bart/squad1.1/'):
        tr = load_json('./data/squad1.1/train-v1.1.json')
        dev = load_json('./data/squad1.1/dev-v1.1.json')

        print(len(tr['data']), len(dev['data']))
        valid = dev['data'][:24]
        test = dev['data'][24:]
        train = tr['data']
        print(len(train), len(valid), len(test))
        train_data, valid_data, test_data = [], [], []
        for one in train:
            train_data += self.deal_one_squad(one)
        for one in valid:
            valid_data += self.deal_one_squad(one)
        for one in test:
            test_data += self.deal_one_squad(one)
        print('sample count, train={}, valid={}, test={}'.format(
            len(train_data), len(valid_data), len(test_data)))
        
        # save processed data
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_json(train_data, save_dir + 'train.json')
        save_json(valid_data, save_dir + 'dev.json')
        save_json(test_data, save_dir + 'test.json')
        print('saved data to {}'.format(save_dir))

    def deal_one_squad(self, one):
        samples = []
        for para in one['paragraphs']:
            context = para['context']
            for item in para['qas']:
                question = item['question']
                answers = [x['text'] for x in item['answers']]
                answer = Counter(answers).most_common(1)[0][0]
                source = '{} </s> {}'.format(answer, context)
                one_sample = {
                    'id': item['id'],
                    'source': source,
                    'target': question
                }
                samples.append(one_sample)
        return samples

# prepare squad data for Flan-T5 series models
class FlanT5Squad:
    def deal_squad(self, save_dir = './data/data_flant5/squad1.1/'):
        tr = load_json('./data/squad1.1/train-v1.1.json')
        dev = load_json('./data/squad1.1/dev-v1.1.json')

        print(len(tr['data']), len(dev['data']))
        valid = dev['data'][:24]
        test = dev['data'][24:]
        train = tr['data']
        print(len(train), len(valid), len(test))
        train_data, valid_data, test_data = [], [], []
        for one in train:
            train_data += self.deal_one_squad(one)
        for one in valid:
            valid_data += self.deal_one_squad(one)
        for one in test:
            test_data += self.deal_one_squad(one)
        print('sample count, train={}, valid={}, test={}'.format(
            len(train_data), len(valid_data), len(test_data)))
        
        # save processed data
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_json(train_data, save_dir + 'train.json')
        save_json(valid_data, save_dir + 'dev.json')
        save_json(test_data, save_dir + 'test.json')

    def deal_one_squad(self, one):
        samples = []
        for para in one['paragraphs']:
            context = para['context']
            for item in para['qas']:
                question = item['question']
                answers = [x['text'] for x in item['answers']]
                answer = Counter(answers).most_common(1)[0][0]
                source = 'Generate a question based on the given answer and context. Answer: {}  Context: {}'.format(answer, context)
                one_sample = {
                    'id': item['id'],
                    'source': source,
                    'target': question
                }
                samples.append(one_sample)
        return samples

# prepare hotpotqa data for T5 series models
class T5Hotpot:
    def deal_hotpot(self, save_dir = './data/data_t5/hotpotqa/'):
        tr = load_json('./data/hotpotqa/hotpot_train_v1.1.json')
        dev = load_json('./data/hotpotqa/hotpot_dev_distractor_v1.json')

        print(len(tr), len(dev))
        valid = dev[:3700]
        test = dev[3700:]
        train = tr
        print(len(train), len(valid), len(test))
        train_data, valid_data, test_data = [], [], []
        for one in train:
            train_data.append(self.deal_one_hotpot(one))
        for one in valid:
            valid_data.append(self.deal_one_hotpot(one))
        for one in test:
            test_data.append(self.deal_one_hotpot(one))
        print('sample count, train={}, valid={}, test={}'.format(
            len(train_data), len(valid_data), len(test_data)))
        
        # save processed data
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_json(train_data, save_dir + 'train.json')
        save_json(valid_data, save_dir + 'dev.json')
        save_json(test_data, save_dir + 'test.json')

    def deal_one_hotpot(self, one):
        para_names = list(set([x[0] for x in one['supporting_facts']]))
        # map_dict = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':}
        paras = []
        for i in range(len(para_names)):
            para = [' '.join(x[1]) for x in one['context'] if x[0] == para_names[i]][0]
            paras.append(para)
        context = '\n'.join(paras)
        source = 'answer: {}  context: {}'.format(one['answer'], context)
        one_sample = {
            '_id': one['_id'],
            'type': one['type'],
            'level': one['level'],
            'passage': context,
            'answer': one['answer'],
            'source': source,
            'target': one['question']
        }
        return one_sample

# prepare hotpotqa data for BART series models
class BartHotpot:
    def deal_hotpot(self, save_dir = './data/data_bart/hotpotqa/'):
        tr = load_json('./data/hotpotqa/hotpot_train_v1.1.json')
        dev = load_json('./data/hotpotqa/hotpot_dev_distractor_v1.json')
        print(len(tr), len(dev))
        valid = dev[:3700]
        test = dev[3700:]
        train = tr
        print(len(train), len(valid), len(test))
        train_data, valid_data, test_data = [], [], []
        for one in train:
            train_data.append(self.deal_one_hotpot(one))
        for one in valid:
            valid_data.append(self.deal_one_hotpot(one))
        for one in test:
            test_data.append(self.deal_one_hotpot(one))
        print('sample count, train={}, valid={}, test={}'.format(
            len(train_data), len(valid_data), len(test_data)))
        # save processed data
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_json(train_data, save_dir + 'train.json')
        save_json(valid_data, save_dir + 'dev.json')
        save_json(test_data, save_dir + 'test.json')

    def deal_one_hotpot(self, one):
        para_names = list(set([x[0] for x in one['supporting_facts']]))
        # map_dict = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':}
        paras = []
        for i in range(len(para_names)):
            para = [' '.join(x[1]) for x in one['context'] if x[0] == para_names[i]][0]
            paras.append('paragraph {}: {}'.format(str(i + 1), para))
        context = '  '.join(paras)
        source = '{} </s> {}'.format(one['answer'], context)
        one_sample = {
            '_id': one['_id'],
            'type': one['type'],
            'level': one['level'],
            'passage': context,
            'answer': one['answer'],
            'source': source,
            'target': one['question']
        }
        return one_sample

# prepare hotpotqa data for Flan-T5 series models
class FlanT5Hotpot:
    def deal_hotpot(self, save_dir = './data/data_flant5/hotpotqa/'):
        tr = load_json('./data/hotpotqa/hotpot_train_v1.1.json')
        dev = load_json('./data/hotpotqa/hotpot_dev_distractor_v1.json')

        print(len(tr), len(dev))
        valid = dev[:3700]
        test = dev[3700:]
        train = tr
        print(len(train), len(valid), len(test))
        train_data, valid_data, test_data = [], [], []
        for one in train:
            train_data.append(self.deal_one_hotpot(one))
        for one in valid:
            valid_data.append(self.deal_one_hotpot(one))
        for one in test:
            test_data.append(self.deal_one_hotpot(one))
        print('sample count, train={}, valid={}, test={}'.format(
            len(train_data), len(valid_data), len(test_data)))
        
        # save processed data
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_json(train_data, save_dir + 'train.json')
        save_json(valid_data, save_dir + 'dev.json')
        save_json(test_data, save_dir + 'test.json')

    def deal_one_hotpot(self, one):
        para_names = list(set([x[0] for x in one['supporting_facts']]))
        # map_dict = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':}
        paras = []
        for i in range(len(para_names)):
            para = [' '.join(x[1]) for x in one['context'] if x[0] == para_names[i]][0]
            paras.append(para)
        context = '\n'.join(paras)
        source = 'Generate a question based on the given answer and context. Answer: {}  Context: {}'.format(
            one['answer'], context)
        one_sample = {
            '_id': one['_id'],
            'type': one['type'],
            'level': one['level'],
            'passage': context,
            'answer': one['answer'],
            'source': source,
            'target': one['question']
        }
        return one_sample


if __name__ == '__main__':
    # prepare data
    save_dir = './data/data_t5/hotpotqa/'
    hot = T5Hotpot()
    hot.deal_hotpot(save_dir)
    
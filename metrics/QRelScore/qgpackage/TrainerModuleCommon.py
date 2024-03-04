import torch
import pytorch_lightning as pl
import re
import os
import json
import glob

from evalpackage import Evaluator, Bleu, Rouge, Meteor

class TrainerModuleEvalMixin:
    bleu_scorer = Bleu(4)
    rouge_scorer = Rouge()

    def write_rank_prediction(self, hyp_questions, gold_standards, qids, subdir):
        prediction_dict = { }
        sum_dict = { }
        for hyp_question, gold_standard, qid in zip(hyp_questions, gold_standards, qids):
            prediction_dict[qid] = {
                'hyp': hyp_question,
                'ref': gold_standard
            }
            scores_dict = Evaluator.compute_individual_metrics(gold_standard, hyp_question)
            prediction_dict[qid].update(scores_dict)

            for k, v in scores_dict.items():
                sum_dict[k] = sum_dict.get(k, 0.0) + v
        
        average_dict = { }
        for k, v in sum_dict.items():
            average_dict[k] = v / float(len(hyp_questions))

        prediction_dirname = os.path.join(self.prediction_path, subdir)
        os.makedirs(prediction_dirname, exist_ok = True)
        prediction_filename = os.path.join(prediction_dirname, 'prediction_{}.json'.format(torch.distributed.get_rank()))
        average_filename = os.path.join(prediction_dirname, 'average_{}.json'.format(torch.distributed.get_rank()))
        with open(prediction_filename, 'w') as f:
            json.dump(prediction_dict, f,  ensure_ascii = True, indent = 4)
        with open(average_filename, 'w') as f:
            json.dump(average_dict, f, indent = 4)

        return average_dict

    def gather_rank_prediction(self, subdir):
        prediction_dirname = os.path.join(self.prediction_path, subdir)

        target_prediction = { }
        for source_prediction_filename in glob.glob(os.path.join(prediction_dirname, 'prediction_[0-9]*.json')):
            with open(source_prediction_filename) as f:
                source_prediction = json.load(f)
            target_prediction.update(source_prediction)
        
        target_average = { }
        for source_average_filename in glob.glob(os.path.join(prediction_dirname, 'average_[0-9]*.json')):
            with open(source_average_filename) as f:
                source_average = json.load(f)
            for k, v in source_average.items():
                l = target_average.setdefault(k, [ ])
                l.append(v)
        for k, l in target_average.items():
            if len(l):
                target_average[k] = sum(l) / len(l)

        target_prediction_filename = os.path.join(prediction_dirname, 'prediction_gather.json')
        target_average_filename = os.path.join(prediction_dirname, 'average_gather.json')

        with open(target_prediction_filename, 'w') as f:
            json.dump(target_prediction, f, ensure_ascii = True, indent = 4)
        with open(target_average_filename, 'w') as f:
            json.dump(target_average, f, indent = 4)

    def evaluate_training_batch(self, hyp_questions, gold_standards):
        hyp_input = { }
        gold_input = { }

        score, scores = TrainerModuleEvalMixin.bleu_scorer.compute_score_flatten(gold_standards, hyp_questions)

        return score[3]
    
    def save_huggingface_model(self, snapshot_key):
        # the typical snapshot_keys consist of best, last, epoch_#, iteration_# ...
        model_dirname = os.path.join(self.snapshot_path, snapshot_key)
        os.makedirs(model_dirname, exist_ok = True)

        self.model.save_pretrained(model_dirname)
        self.tokenizer.save_pretrained(model_dirname)

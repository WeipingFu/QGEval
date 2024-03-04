import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from tqdm import tqdm

import re
import os
import json
import time

from .TrainHelper import AverageMeter, LoggerPather
from .TrainerModuleCommon import TrainerModuleEvalMixin


class TrainerModule(pl.LightningModule, TrainerModuleEvalMixin):
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader, config):
        super().__init__()
        self.config = config
        # self.save_hyperparameters(self.config)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        '''
        According to the document of pytorch-lightning, the dataloader will be wrapped with DistributedSampler automatically when using the `DDP/DDP2` acclerator. However, this implementation is also based on the callback mechanism which happens after `trainer.fit(model, datamodule)`. In our construction of the `TrainerModule(pl.LightningModule)`, the dataloader is just passed by the main function and still the vanilla dataloader. We have to divide the len(dataloader) by the #rank_node to obtain the train_batch_per_epoch, which is also known as the number of parallel iterations per epoch under the scenario of multi-processes (GPUs) training.
        By the way, if we do not perform the division, we would obtain the number of sequential iterations per epoch. It is not appliable to the lr_scheduler.
        '''
        self.num_train_batch_per_epoch = len(self.train_dataloader) // len(self.config.DEVICE[5:].split(','))
        self.num_epochs = self.config.TRAIN.NUM_EPOCHS
        self.num_iterations = self.num_epochs * self.num_train_batch_per_epoch

        loggerpather = LoggerPather(self.config)
        # self.logger = loggerpather.get_logger()
        self.log_path = loggerpather.get_log_path()
        self.snapshot_path = loggerpather.get_snapshot_path()
        self.tb_path = loggerpather.get_tb_path()
        self.prediction_path = loggerpather.get_prediction_path()
        # self.writer = SummaryWriter(self.tb_path)

        self.loss_avg_meter = AverageMeter()

    def forward(self, **args):
        return self.model(**args)
    
    def training_step_feeding(self, batch, batch_idx):
        train_input, test_input, question_text = batch
        model_input = train_input
        outputs = self(**model_input)
        loss = outputs['loss']

        return loss

    def training_step_logging(self, batch, batch_idx, loss):
        self.loss_avg_meter.update(loss.item())
        self.log('train/loss_cur', loss.item(), on_step = True, prog_bar = True, logger = False)
        self.log('train/loss_avg', self.loss_avg_meter.average(), on_step = True, prog_bar = True, logger = False)
        
        iteration = self.current_epoch * self.num_train_batch_per_epoch + batch_idx + 1
        self.logger.experiment.add_scalar('train/loss_cur', loss.item(), iteration)
        self.logger.experiment.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.training_step_feeding(batch, batch_idx)
        bp_loss = self.training_step_logging(batch, batch_idx, loss)
        return bp_loss
  
    # # For DP/DDP2, a single process handles the batch split and they (splitted samples) must be gathered by yourself.
    # # Here, we are strongly recommended to use the DDP/DDP-spawn
    # def training_step_end(self, training_step_outputs):
    #     gpu_0_pred = training_step_outputs[0]['pred']
    #     gpu_1_pred = training_step_outputs[1]['pred']
    #     gpu_n_pred = training_step_outputs[n]['pred']

    #     # this softmax now uses the full batch
    #     loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
    #     return loss

    def validation_step(self, batch, batch_idx):
        train_input, test_input, gold_standards, qids = batch
        model_input = train_input
        with torch.no_grad():
            outputs = self(**model_input)
        loss = outputs['loss']
        return loss.item()

    def validation_epoch_end(self, validation_step_outputs):
        loss = sum(validation_step_outputs) / len(validation_step_outputs)

        self.log('val/loss', loss, on_epoch = True, prog_bar = True, logger = False)
        self.logger.experiment.add_scalar('val/loss', loss, self.current_epoch)

    def generate(self, input_ids, num_return_sequences, do_sample, output_scores):
        sample_outputs = self.model.generate(
            input_ids = input_ids,
            max_length = self.config.MODEL.MAX_INPUT_LENGTH,
            early_stopping = True,
            temperature = self.config.GENERATE.TEMPERATURE,
            do_sample = do_sample,
            top_p = self.config.GENERATE.TOP_P,
            top_k = self.config.GENERATE.TOP_K,
            num_beams = self.config.GENERATE.BEAM_SIZE,
            num_return_sequences = num_return_sequences,

            output_scores = output_scores,

            bos_token_id = self.tokenizer.bos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id,

            return_dict_in_generate = True
        )
        
        sample_outputs = list(sample_outputs.values())
        if do_sample and output_scores:
            # sequences, sequences_scores, scores    
            return sample_outputs[0], sample_outputs[2]
        else:
            # sequences
            return sample_outputs[0]

    def decode(self, input_ids, sample_outputs):
        input_ids_len = input_ids.shape[-1] # have been padded as the same length of DOC_STRIDE (yet MAX_INPUT_LENGTH in training and validation step)
        
        # B*num_return_sequences x max_seq_len
        hyp_questions = self.tokenizer.batch_decode(sample_outputs[:, input_ids_len:], skip_special_tokens = True)

        return hyp_questions

    def test_step(self, batch, batch_idx):
        test_input, gold_standards, qids = batch
        model_input = test_input
        input_ids = model_input['input_ids']

        num_return_sequences = 1 # sequentialize not parallelize
        with torch.no_grad():
            sample_outputs = self.generate(input_ids, num_return_sequences, do_sample = False, output_scores = False)

        # B*num_return_sequences x max_seq_len
        hyp_questions = self.decode(input_ids, sample_outputs)

        return hyp_questions, gold_standards, qids

    def test_epoch_end(self, test_step_outputs):
        hyp_questions = [ ]
        gold_standards = [ ]
        qids = [ ]

        for batch_hyp_questions, batch_gold_standards, batch_qids in test_step_outputs:
            hyp_questions.extend(list(batch_hyp_questions))
            gold_standards.extend(list(batch_gold_standards))
            qids.extend(list(batch_qids))

        subdir = 'best'
        average_dict = self.write_rank_prediction(hyp_questions, gold_standards, qids, subdir)

        self.save_huggingface_model('huggingface_best')

        # wait for other processes and gather the distributed results using rank 0 node / process / GPU
        if not torch.distributed.get_rank():
            time.sleep(1)
            self.gather_rank_prediction(subdir)

        return average_dict
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.TRAIN.WD,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.config.TRAIN.LR, correct_bias = True)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps = self.config.TRAIN.NUM_WARMUP_STEPS, num_training_steps = self.num_iterations
        )

        # Before pl v1.3, lr_scheduler is invoked automatically according to the interval (step or epoch)
        # After pl v1.3, it is a manual operation (manually)
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step'
        }

        return [ optimizer ], [ lr_dict ]


class Seq2SeqTrainerModule(TrainerModule):
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader, config):
        super().__init__(model, tokenizer, train_dataloader, val_dataloader, config)

    def decode(self, input_ids, sample_outputs):
        # The output of Seq2SeqLM is just what we want, and it is unnecessary to filter out the previous part
        # This is just conditional generation and the model does not watch other things like CausalLM, such as old words, input words and previous words
        # B*num_return_sequences x max_query_length
        hyp_questions = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens = True)

        return hyp_questions


class RLTrainerModule(TrainerModule):
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader, config):
        super().__init__(model, tokenizer, train_dataloader, val_dataloader, config)
        self.criterion = nn.CrossEntropyLoss()

    def get_rl_ratio(self, epoch):
        ret = 0.0
        
        epoch_diff = epoch - self.config.RL.START_EPOCH
        
        if epoch_diff >= 0:
            ret = max(self.config.RL.MAX_RL_RATIO, self.config.RL.RL_RATIO_BASE * (epoch_diff + 1))

        return ret

    def training_step_feeding(self, batch, batch_idx):
        lm_loss = super().training_step_feeding(batch, batch_idx)

        train_input, test_input, gold_questions = batch
        model_input = test_input
        input_ids = model_input['input_ids']
        input_ids_len = input_ids.size(-1)

        pg_loss = 0.0
        rl_ratio = self.get_rl_ratio(self.current_epoch)

        if rl_ratio > 0.001:
            sample_logits = self.model(**model_input)['logits']        
            # B x max_input_len, max_input_len-input_ids.shape[-1] shaped tuple of (B*numreturnsequences*beamsize x vocabsize)
            sample_ids, sample_logits = self.generate(input_ids, num_return_sequences = 1, do_sample = True, output_scores = True)
            sample_questions = self.decode(input_ids, sample_ids)

            sample_reward = self.evaluate_training_batch(sample_questions, list(gold_questions))
            sample_logits = torch.stack(sample_logits, dim = 0).permute(1, 0, 2) # B*numreturnsequences*beamsize x max_output_len x vocabsize
            
            sample_logits = sample_logits.view(-1, self.config.GENERATE.BEAM_SIZE, *sample_logits.shape[1:])[:, 0]
            sample_labels = train_input['labels'][:, input_ids_len:]
            sample_loss = self.criterion(sample_logits.reshape(-1, sample_logits.size(-1)), sample_labels.reshape(-1))

            with torch.no_grad():
                baseline_ids = self.generate(input_ids, num_return_sequences = 1, do_sample = False, output_scores = False)
            baseline_questions = self.decode(input_ids, baseline_ids)
            greedy_reward = self.evaluate_training_batch(baseline_questions, list(gold_questions))

            # SCST policy gradient
            # reduction == mean
            pg_loss = (sample_reward - greedy_reward) * sample_loss

        ret_loss = rl_ratio * pg_loss + (1 - rl_ratio) * lm_loss
        return [ ret_loss, lm_loss, pg_loss, rl_ratio, sample_questions, baseline_questions ]

    def training_step_logging(self, batch, batch_idx, loss):
        ret_loss, lm_loss, pg_loss, rl_ratio, sample_questions, baseline_questions = loss

        self.loss_avg_meter.update(ret_loss.item())

        self.log('train/loss_cur', ret_loss.item(), on_step = True, prog_bar = True, logger = False)
        self.log('train/loss_avg', self.loss_avg_meter.average(), on_step = True, prog_bar = True, logger = False)
        self.log('train/loss_lm', lm_loss.item(), on_step = True, prog_bar = True, logger = False)
        self.log('train/loss_rl', pg_loss.item(), on_step = True, prog_bar = True, logger = False)
        
        iteration = self.current_epoch * self.num_train_batch_per_epoch + batch_idx + 1
        self.logger.experiment.add_scalar('train/loss_cur', ret_loss.item(), iteration)
        self.logger.experiment.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)
        self.logger.experiment.add_scalar('train/loss_lm', lm_loss.item(), iteration)
        self.logger.experiment.add_scalar('train/loss_rl', pg_loss.item(), iteration)

        if not iteration % self.config.TRAIN.TB_FREQ:
            self.logger.experiment.add_text('train/sample_questions', sample_questions[0], iteration)
            self.logger.experiment.add_text('train/baseline_questions', baseline_questions[0], iteration)
        
        return ret_loss
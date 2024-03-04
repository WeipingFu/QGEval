import torch
import pytorch_lightning as pl

import os
import copy
import json

class LMMixin:
    def convert_to_tensor(self, model_input):
        # Now all tensors are contructed in the CPU and will be manually moved to other devides later.
        # key -> nested python list of python integers or LongTensor
        for key, value in model_input.items():
            if not torch.is_tensor(value):
                model_input[key] = torch.LongTensor(value)
        return model_input


# GPT2 or GPT3-style input (decoder-only, autoregressive-only generation)
class CLMMixin(LMMixin):
    def prepare_input(self, context, label=None):
        tokenizer = self.tokenizer
        pad_token_id = self.tokenizer.pad_token_id
        
        # we mark the label using an extra special token <que> besides the <cxt> and <ans> used in the context because the context and label will be wrapped as a single sequence in the construction of CLM input
        que_token = self.config.MODEL.SPECIAL_TOKENS.QUE_TOKEN
        question_prefix = que_token + ' Ask a question:'
        blank_or_space_token = ' '
        
        '''
        Please note that Tokenizers of the GPT2 models will not wrap the input sequence with bos/eos automatically. It's not really a bug because the default behavior (in original paper) of GPT2 is to just not add bos or eos tokens. These special tokens are just used for fine-tuning unconditional or conditional generation tasks, not pre-training on the original corpora.
        '''

        # firstly, wrap the context with bos token
        test_context = [ tokenizer.bos_token +
                         blank_or_space_token + c +
                         blank_or_space_token + question_prefix
                         for c in context ]

        # padding, tokenization, convert tokens to ids and wrap with torch.LongTensor
        test_input = tokenizer(test_context, padding = 'max_length', truncation = True, max_length = self.config.MODEL.DOC_STRIDE, return_attention_mask = True, return_token_type_ids = True, return_tensors = 'pt')
        
        train_input = None
        if label is not None:
            # input of Causal Language Models (CLM) GPT2
            # first wrap the context and the label
            # For the same reason as the above, GPT2Tokenizer will not add the special tokens bos/eos automatically
            train_context = [ tokenizer.bos_token +
                              blank_or_space_token + c
                              for c in context ]
            train_label = [ question_prefix + 
                            blank_or_space_token + l +
                            blank_or_space_token + tokenizer.eos_token
                            for l in label ]

            # for tc, tl in zip(train_context, train_label):
            #     print(tc + ' ' + tl)

            train_input = tokenizer(train_context, train_label, padding = 'max_length', truncation = True, max_length = self.config.MODEL.MAX_INPUT_LENGTH, return_attention_mask = True, return_token_type_ids = True)

            # CLM such as GPT2 requre no decoder_input_ids

            # then copy.deepcopy input_ids
            # last build the loss masks for the padding tokens
            train_input['labels'] = copy.deepcopy(train_input['input_ids'])
            for sample_index, sample_token_type_ids in enumerate(train_input['token_type_ids']):
                for position_index, token_type_id in enumerate(sample_token_type_ids):
                    if not token_type_id:
                        # set **the context part and padding part** to -100 for ignoring the loss computation and its back-propagation
                        # this is implemented by `class torch.nn.CrossEntropyLoss`
                        train_input['labels'][sample_index][position_index] = -100
            
            train_input = self.convert_to_tensor(train_input)

        return train_input, test_input


# BART, T5 or MASS-style input (classical encoder-decoder generation)
class Seq2SeqLMMixin(LMMixin):
    def prepare_input(self, context, label=None):
        tokenizer = self.tokenizer
        PAD_TOKEN_ID = self.tokenizer.pad_token_id

        '''
        Given S to BARTTokenizer, we get <s> S </s>.
        Given S1 and S2 to BARTTokenizer, we get <s> S1 </s> </s> S2 </s>.
        Obviously, <s> plays the role of cls_token, and </s> is for the role of sep_token and is different BERT behaviors, because </s> also leads all the sequences except the first sequence.
        Actually, BART cannot handle the input with both text and text pair.

        However, given the same input as the above, T5Tokenizer will output S </s> and S1 </s> S2 </s>, respectively.
        Obviously, T5 doesn't need special token to mark the start of a sequene and can deal with the input with both text and text pair primitively.
        In general, T5 needs an extra task-specific prefix (TSP) to specify different fine-tuning tasks when fine-tuning its encoders and decoders on the corresponding downstream tasks.

        For a generation task with multiple input sequences, we use differnt special tokens to mark every of them (e.g. cxt token and ans token in our QG task).
        However, prefix tokens of T5 model serve as the same function as the above, but they usually lead every of them or lead the whole task input.
        Therefore, although there are essential difference in their formats, I think we can employ an unified format to deal with the input for all generation-oriented tasks (i.e. CLM and Seq2Seq model).
        '''
        # Tokenizers of the Seq2Seq models will wrap the input sequence with bos/eos or bos/eos-styled special tokens besides the essential padding tokens
        test_input = tokenizer(context, padding = 'max_length', truncation = True, max_length = self.config.MODEL.DOC_STRIDE, return_attention_mask = True, return_tensors = 'pt')

        train_input = None
        if label is not None:
            # input of Seq2Seq Language Models (Seq2Seq)
            # first copy.deepcopy input_ids, attention_masks
            # torch.*Tensor.clone() can also make it and perserve autograd, but we don't need this operation in the computation graph
            train_input = copy.deepcopy(test_input)
            
            # then build labels for train_input
            # BARTTokenizer will also wrap the label text with bos/eos
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Thus, both input_ids and labels of BART are enclosed with bos/eos
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            train_input['labels'] = tokenizer(label, padding = 'max_length', truncation = True, max_length = self.config.MODEL.MAX_QUERY_LENGTH, return_attention_mask = True)['input_ids']
            
            # then build decoder_input_ids
            # decoder_input_ids will be built by shifting labels right for BartForConditionalGeneration and adding decoder_start_token_id (eos_token_id).
            # on the contrary, BartModel will not accept the argument labels and employ input_ids as the decoder_input_ids to perform denosing auto-encoding pre-training.

            # last build the loss masks for the padding tokens
            for sample_index, sample_labels in enumerate(train_input['labels']):
                for position_index, label in enumerate(sample_labels):
                    if label == PAD_TOKEN_ID:
                        # set **padding part** to -100 for ignoring the loss computation and its back-propagation
                        # this is implemented by `class torch.nn.CrossEntropyLoss`
                        train_input['labels'][sample_index][position_index] = -100

            train_input = self.convert_to_tensor(train_input)

        return train_input, test_input

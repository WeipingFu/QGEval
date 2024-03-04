import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

class CLMScore:
    def __init__(self, model_type='gpt2', batch_size = 32, nthreads = 4, device = None):
        self.batch_size = batch_size
        self.nthreads = nthreads
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model, self.tokenizer = self.get_model(model_type)
        self.model.eval()
        self.model.to(self.device)

    def get_model(self, model_type = 'gpt2'):
        # model_config = AutoConfig.from_pretrained(model_type)
        tokenizer = self.get_tokenizer(model_type)
        model = AutoModelForCausalLM.from_pretrained(
            model_type,
            # config = model_config
        )
        # model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def get_tokenizer(self, model_type):
        tokenizer = AutoTokenizer.from_pretrained(
            model_type,
            do_lower_case = False
        )

        return tokenizer

    def sent_encode(self, sent):
        '''
        Wrap the input sentence as for the transformers model
        
        Args:
            - :param: `sentence` (str) : sentnece to encode
        
        
        Return:
            - `encoded_sentnece` (dict of str, torch.LongTensor) : input_ids, attention_mask, token_type_id, return_dict ...
        '''
        # input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(sent, max_length=1022, truncation=True) + [self.tokenizer.eos_token_id]
        input_ids = self.tokenizer.encode(sent, max_length=1024, truncation=True, add_special_tokens=True)
        input_dict = { 'input_ids': input_ids }
        for key in input_dict.keys():
            input_dict[key] = torch.LongTensor(input_dict[key]).unsqueeze(dim = 0).to(self.device)
        return input_dict

    def clm_encode(self, encoded_sent):
        with torch.no_grad():
            outputs = self.model(**encoded_sent)
        return outputs['logits']

    def forward_pass_score(self, sent):
        encoded_sent = self.sent_encode(sent)
        # print('*'*100, 'encoded_sent')
        # print(self.tokenizer.model_max_length)
        # print(encoded_sent)
        labels = encoded_sent['input_ids']
        lm_logits = self.clm_encode(encoded_sent)

        shift_logits = lm_logits[:, :-1].contiguous() # B x MaxSeqLen-1 x VocabSize
        shift_labels = labels[:, 1:].contiguous() # B x MaxSeqLen-1

        # print(torch.argmax(shift_logits, dim = 2), shift_labels)
        # Use the cross entropy loss in an 1-d way (flatten the MaxSeqLen dimension or axis)
        loss_fct = nn.CrossEntropyLoss(reduction = 'none')
        tokenwise_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # because B (batch size) is always equal to 1 (see sent_encode function), we don't need to re-shape the tensor of tokenwise_loss

        tokenwise_prob = torch.exp(-tokenwise_loss)

        score = torch.mean(tokenwise_prob).cpu().detach().item()

        return score

    def diff_score(self, context, question):
        baseline = self.forward_pass_score(context)
        enhancement = self.forward_pass_score(question + ' ' + context)

        score = (enhancement - baseline) / baseline
        score = score if score >= 0.0 else 0.0
        
        # baseline re-scaling
        # score = (score + 0.21804565) / (0.55268557 + 0.21804565)

        return score

    '''
        Compute the CLM scores for the list of candidates or hypotheses

        Args:
            - :param: `gts` (list of str) : contexts of the samples
            - :param: `res` (list of str) : generated questions of the samples (hypotheses or candidates)

        Return:
            - `clm_score` (float) : causal language model scores for this batch input
    '''
    def compute_score_flatten(self, gts, res):
        assert type(gts) == type(res) == list
        assert len(gts) == len(res)

        scores = [ ]

        for g, r in tqdm(zip(gts, res), total=len(gts)):
            # try:
            score = self.diff_score(g, r)
            # except Exception as e:
            #     print(e)
            #     score = -1
            #     print(g, r)
            scores.append(score)

        return scores 

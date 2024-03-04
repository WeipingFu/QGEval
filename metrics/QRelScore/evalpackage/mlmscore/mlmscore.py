import csv
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm

class MLMScore:
    def __init__(self, model_type='bert-base-cased', batch_size = 32, nthreads = 4, device = None):
        self.batch_size = batch_size
        self.nthreads = nthreads
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model, self.tokenizer = self.get_model(model_type)
        self.model.eval()
        self.model.to(self.device)

    def get_model(self, model_type = 'bert-base-cased'):
        # model_config = AutoConfig.from_pretrained(model_type)
        tokenizer = self.get_tokenizer(model_type)
        model = AutoModel.from_pretrained(
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

    def sent_encode(self, question, context):
        input_dict = self.tokenizer.encode_plus(question, context, truncation = 'only_second', max_length = 512)

        for key in input_dict.keys():
                input_dict[key] = torch.LongTensor(input_dict[key]).unsqueeze(dim = 0).to(self.device)
        input_dict.update({ 'output_hidden_states': True,
                            'output_attentions': True })

        question_length, context_length = len(self.tokenizer.tokenize(question)), len(self.tokenizer.tokenize(context))
        return input_dict, question_length, context_length

    def mlm_encode(self, input_dict):
        with torch.no_grad():
            outputs = self.model(**input_dict)
        # return_dict is set by default
        return outputs['hidden_states'], outputs['attentions']

    def weighted_cosine_similarity(self, lhs_hidden, rhs_hidden, l2r_attention):
        '''
            lhs_hidden, rhs_hidden, l2r_attention (K heads self-attention)
            B x #Layers x L1 x d, B x #Layers x L2 x d, B x #Layers x K x L1 x L2
        '''
        # max pooling over heads distribution (avg pooling may be inappropriate)
        max_l2r_attention = torch.max(l2r_attention, dim = 2)[0]
        mean_l2r_attention = max_l2r_attention / torch.sum(max_l2r_attention, dim = -1, keepdim = True)
        
        # for l in range(mean_l2r_attention.size(1)):
        #     self.write_demo_csv(mean_l2r_attention[0, l, [0, 2, 3]], 'demo-layer-{}.csv'.format(l))

        max_values, max_indices = torch.max(mean_l2r_attention, dim = -1, keepdim = True)

        # preserve the maximum heads only
        mean_l2r_attention = torch.zeros_like(mean_l2r_attention).scatter_(dim = -1, index = max_indices, value = 1.0) # or src = max_values

        lhs_hidden.div_(torch.norm(lhs_hidden, p = 2, dim = -1, keepdim = True))
        rhs_hidden.div_(torch.norm(rhs_hidden, p = 2, dim = -1, keepdim = True ))

        raw_pairwise_inner_product_similarity = torch.matmul(lhs_hidden, rhs_hidden.permute(0, 1, 3, 2))
        lhs_lengths = torch.norm(lhs_hidden, p = 2, dim = 3, keepdim = True)
        rhs_lengths = torch.norm(rhs_hidden, p = 2, dim = 3).unsqueeze(dim = 2)
        denominator_lengths = lhs_lengths * rhs_lengths
        stable_lengths = torch.max(torch.full_like(denominator_lengths, 1.0e-08), denominator_lengths)
        raw_pairwise_cosine_similarity = raw_pairwise_inner_product_similarity / stable_lengths
        
        weighted_pairwise_similarity = raw_pairwise_cosine_similarity * mean_l2r_attention
        
        weighted_pairwise_similarity = torch.sum(weighted_pairwise_similarity, dim = 3)

        # average all layers of all tokens
        weighted_score = torch.mean(weighted_pairwise_similarity, dim = (1, 2))
        # average all tokens along each layer (return layerwise results)
        # weighted_score = torch.mean(weighted_pairwise_similarity, dim = 2)

        if len(weighted_score.size()) == 1 and weighted_score.size(dim = 0) == 1:
            weighted_score = weighted_score.item()

        return weighted_score

    def write_demo_csv(self, tensor_to_write, filename):
        assert len(tensor_to_write.shape) == 2
        
        results = tensor_to_write.cpu().data.numpy().tolist()

        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(results)

    def forward_pass_score(self, question, context):
        input_dict, ql, cl = self.sent_encode(question, context)
        
        hidden_states, attentions = self.mlm_encode(input_dict)

        hidden_states = torch.stack(hidden_states[1:], dim = 1)
        attentions = torch.stack(attentions, dim = 1)
        
        l1_slice = slice(1, ql) # ql + 1 === punctuation (?)
        l2_slice = slice(ql + 2, -2) # === ql + 2 + cl
        question_hidden = hidden_states[:, :, l1_slice]
        context_hidden = hidden_states[:, :, l2_slice]
        q2c_attention = attentions[:, :, :, l1_slice, l2_slice] 
        c2q_attention = attentions[:, :, :, l2_slice, l1_slice]

        precision = self.weighted_cosine_similarity(question_hidden, context_hidden, q2c_attention)
        # recall = self.weighted_cosine_similarity(context_hidden, question_hidden, c2q_attention)
        ret = precision
        
        # baseline re-scaling
        # ret = (ret - 0.26209762) / (0.68293935 - 0.26209762)

        return ret

    '''
        Compute the CLM scores for the list of candidates or hypotheses

        Args:
            - :param: `gts` (list of str) : contexts of the samples
            - :param: `res` (list of str) : generated questions of the samples (hypotheses or candidates)

        Return:
            - `clm_score` (float) : causal language model for this batch input
    '''
    def compute_score_flatten(self, gts, res):
        assert type(gts) == type(res) == list
        assert len(gts) == len(res)

        scores = [ ]

        for g, r in tqdm(zip(gts, res), total=len(gts)):
            score = self.forward_pass_score(r, g)
            scores.append(score)

        return scores 

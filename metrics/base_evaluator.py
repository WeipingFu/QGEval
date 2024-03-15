import os
from tqdm import tqdm
# from nlgeval import compute_metrics
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def read_text(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip('\n').strip())
    return data

class NLGEvaluator:
    def __init__(self, hypo_path, ref_path_list, hypos=None, refs_list=None):
        self.hypo_path = hypo_path
        self.ref_path_list = ref_path_list
        self.hypos = None
        self.refs_list = None
        if hypos is not None:
            self.hypos = hypos
        if refs_list is not None:
            self.refs_list = refs_list
    
    def __load_data(self):
        print('Loading data')
        self.hypos = read_text(self.hypo_path)
        self.hypos = [x.strip() for x in self.hypos if len(x.strip())>0]
        refs_list = []
        for ref_path in self.ref_path_list:
            refs = read_text(ref_path)
            refs = [x.strip() for x in refs if len(x.strip())>0]
            assert len(self.hypos) == len(refs)
            refs_list.append(refs)
        self.refs_list = refs_list
        print('Load data success! Get {} refs, sentence count={}'.format(len(self.refs_list), len(self.hypos)))

    # def get_nlgeval(self):  
        
    #     metrics_dict = compute_metrics(hypothesis=self.hypo_path,
    #                            references=self.ref_path_list)
    #     print('#'*25, 'nlg-eval result', '#'*25)
    #     print(metrics_dict)
    #     print()
    #     return metrics_dict

    def get_bert_score(self, lang='en'):
        if self.hypos is None:
            self.__load_data()
        print('Start calculate bert_score')
        bert_score_dict = {'ref_'+str(i+1):None for i in range(len(self.refs_list))}
        for i in tqdm(range(len(self.refs_list))):
            P, R, F1 = score(self.hypos, self.refs_list[i], lang=lang)
            # tensor to list
            P = list(map(lambda x: round(x, 4), P.tolist()))
            R = list(map(lambda x: round(x, 4), R.tolist()))
            F1 = list(map(lambda x: round(x, 4), F1.tolist()))
            bert_score_dict['ref_'+str(i+1)] = {
                'P': P,
                'R': R,
                'F1': F1
            }
        print('Bert_score calculated!')
        return bert_score_dict

    def get_sentence_bleu(self):
        # bleu-4
        bleu_list = []
        if self.hypos is None:
            self.__load_data()
        chencherry = SmoothingFunction()
        print('Start Calculating Sentence-BLEU')
        for i in tqdm(range(len(self.hypos))):
            hypo = self.hypos[i]
            cand_words = hypo.strip().split(' ')
            refs = [refs_one[i] for refs_one in self.refs_list]
            ref_words_list = list(map(lambda x: x.strip().split(' '), refs))
            # print(i)
            # print(cand_words)
            # print(ref_words_list)
            one_bleu = round(sentence_bleu(ref_words_list,cand_words,smoothing_function=chencherry.method1), 4)
            bleu_list.append(one_bleu)
        print('Sentence-BLEU calculated!')
        return bleu_list

    def get_meteor_score(self):
        # bleu-4
        meteor_list = []
        if self.hypos is None:
            self.__load_data()
        print('Start Calculating meteor_score')
        for i in tqdm(range(len(self.hypos))):
            hypo = self.hypos[i]
            cand_words = hypo.strip().split(' ')
            refs = [refs_one[i] for refs_one in self.refs_list]
            ref_words_list = list(map(lambda x: x.strip().split(' '), refs))
            meteor = round(meteor_score(ref_words_list, cand_words),4)
            meteor_list.append(meteor)
        print('meteor_score calculated!')
        return meteor_list

    def get_rouge_L(self):
        if self.hypos is None:
            self.__load_data()
        print('Start calculating rougeL')
        rouge_score_dict = {'ref_'+str(i+1):None for i in range(len(self.refs_list))}
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        for j in range(len(self.refs_list)):
            scores = []
            for i in tqdm(range(len(self.hypos))):
                hypo = self.hypos[i]
                ref = self.refs_list[j][i]
                r_score = scorer.score(hypo, ref)['rougeL']
                pre, rec, f1 = round(r_score[0], 4), round(r_score[1], 4), round(r_score[2], 4)
                scores.append({'precision':pre, 'recall':rec, 'f1':f1})
            rouge_score_dict['ref_'+str(j+1)] = scores
        print('rougeL calculated!')
        return rouge_score_dict

    def apply(self, lang='en', score_names=['BLEU-4','METEOR','ROUGE-L','BERTScore']):
        metrics = {}
        # support = ['s-bleu','meteor','rouge_L','bert_score','nlg-eval']
        support = ['BLEU-4','METEOR','ROUGE-L','BERTScore']
        not_in = [x for x in score_names if x not in support]
        if 'BLEU-4' in score_names:
            metrics['BLEU-4'] = self.get_sentence_bleu()
        if 'METEOR' in score_names:
            metrics['METEOR'] = self.get_meteor_score()
        if 'ROUGE-L' in score_names:
            metrics['ROUGE-L'] = self.get_rouge_L()
        if 'BERTScore' in score_names:
            metrics['BERTScore'] = self.get_bert_score(lang=lang)
        # if 'nlg-eval' in score_names:
        #     metrics['nlg-eval'] = self.get_nlgeval()
        if len(not_in) > 0:
            print('Metric {} is not support yet in base_evaluator'.format(not_in))
        return metrics

def main():
    import pandas as pd
    data = pd.read_excel('test.xlsx')
    hypos = data['prediction'].tolist()
    refs_list = [data['question'].tolist()]
    evaluator = NLGEvaluator(None, None, hypos, refs_list)
    result = evaluator.apply(score_names=['s-bleu', 'meteor', 'rouge_L', 'bert_score'])
    # only store f1
    bleu_res = [round(x, 4) for x in result['bleu']]
    meteor_res = [round(x, 4) for x in result['meteor']]
    rouge_res = [round(x['f1'], 4) for x in result['rouge_L']['ref_1']]
    berts_res = [round(x, 4) for x in result['bert_score']['ref_1']['F1']]
    res = data.to_dict(orient='list')
    res['BLEU'] = bleu_res
    res['METEOR'] = meteor_res
    res['ROUGE'] = rouge_res
    res['BERTScore'] = berts_res
    pd.DataFrame(res).to_excel('result.xlsx', index=False)

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    main()

# Bleu_1: 0.537177
# Bleu_2: 0.404950
# Bleu_3: 0.320807
# Bleu_4: 0.256220
# METEOR: 0.266451
# ROUGE_L: 0.555784
# CIDEr: 2.119157
# SkipThoughtsCosineSimilarity: 0.879024
# EmbeddingAverageCosineSimilarity: 0.920001
# EmbeddingAverageCosineSimilairty: 0.920001
# VectorExtremaCosineSimilarity: 0.626405
# GreedyMatchingScore: 0.811177
from moverscore.moverscore_v2 import get_idf_dict, word_mover_score 
from collections import defaultdict
from itertools import zip_longest
import numpy as np


def sentence_score(hypothesis, references, trace=0):
    
    # idf_dict_hyp = get_idf_dict(hypothesis) 
    idf_dict_hyp = defaultdict(lambda: 1.)
    # idf_dict_ref = get_idf_dict(references) 
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 
    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
          
    return sentence_score

def corpus_mover(hypos, refs_list, trace=0):
    score_list = []
    assert len(hypos) == len(refs_list[0])
    for i in range(len(hypos)):
        refs = [refs_one[i] for refs_one in refs_list]
        one_score = round(sentence_score(hypos[i], refs, trace=trace), 4)
        score_list.append(one_score)
    return score_list
        


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    res_path = 'test.xlsx'
    data = pd.read_excel(res_path)
    # scores = []
    # for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
    #     score = sentence_score(row['prediction'], [row['target']])
    #     # score = sentence_score(row['prediction'], [row['question']])
    #     scores.append(round(score, 4))
    # data['MoverScore'] = scores
    # data.to_excel(res_path, index=False)
    scores = corpus_mover(data['prediction'].tolist(), [data['target'].tolist()])
    print(len(scores), sum(scores)/len(scores))
    # hypothesis = 'How many men did William Trent send to Fort Duquesne?'
    # references = ['How many men did Duquesne send to relieve Saint-Pierre?']
    # print(sentence_score(hypothesis, references, 1))


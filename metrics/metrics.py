import torch
from tqdm import tqdm

def prefix_prompt(l, p):
    new_l = []
    for x in l:
        new_l.append(p + ', ' + x)
    return new_l 

def prefix_prompt_qg(l, p):
    new_l = []
    for x in l:
        new_l.append('{}\n\nQuestion: {}'.format(p, x))
    return new_l

def get_metrics(hypos, refs_list, contexts, answers, score_names):
    if len(score_names) <= 0:
        raise ValueError('score_names is empty!')
    res = {sn:[] for sn in score_names}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: {}'.format(device), '#'*50)
    # bleu, rouge, meteor, bert_score
    all_basic = ['BLEU-4','METEOR','ROUGE-L','BERTScore']
    basic_scores = [x for x in all_basic if x in score_names]
    if len(basic_scores) > 0:
        print('Basic Scores', '#'*50)
        from base_evaluator import NLGEvaluator
        evaluator = NLGEvaluator(None, None, hypos, refs_list)
        result = evaluator.apply(score_names=basic_scores)
        if 'BLEU-4' in score_names:
            bleu_res = [round(x, 4) for x in result['BLEU-4']]
            res['BLEU-4'] = bleu_res
        if 'METEOR' in score_names:
            meteor_res = [round(x, 4) for x in result['METEOR']]
            res['METEOR'] = meteor_res
        if 'ROUGE-L' in score_names:
            rouge_res = [round(x['f1'], 4) for x in result['ROUGE-L']['ref_1']]
            res['ROUGE-L'] = rouge_res
        if 'BERTScore' in score_names:
            bert_res = [round(x, 4) for x in result['BERTScore']['ref_1']['F1']]
            res['BERTScore'] = bert_res
        torch.cuda.empty_cache()
    # bleurt
    if 'BLEURT' in score_names:
        print('BLEURT', '#'*50)
        from bleurt.score import BleurtScorer
        checkpoint = "bleurt/BLEURT-20"
        scorer = BleurtScorer(checkpoint)
        res['BLEURT'] = [round(x, 4) for x in scorer.score(references=refs_list[0], candidates=hypos)]
        torch.cuda.empty_cache()
    # moverscore
    if 'MoverScore' in score_names:
        print('MoverScore', '#'*50)
        from MoverScore import corpus_mover
        res['MoverScore'] = corpus_mover(hypos, refs_list)
        torch.cuda.empty_cache()
    # qbleu
    if 'Q-BLEU4' in score_names:
        print('Q-BLEU4', '#'*50)
        from QBLEU.answerability_score import get_answerability_scores
        mean_answerability_score, mean_fluent_score, new_scores, fluent_scores = get_answerability_scores(
            hypotheses=hypos,
            references=refs_list[0],
            ngram_metric='Bleu_4',
            delta=0.7,
            ner_weight=0.6,
            qt_weight=0.2,
            re_weight=0.1)
        print('mean_answerability_score={}'.format(mean_answerability_score))
        res['Q-BLEU4'] = [round(x, 4) for x in new_scores]
        torch.cuda.empty_cache()
    # qsts
    if 'QSTS' in score_names:
        print('QSTS', '#'*50)
        from qsts import corpus_qsts
        res['QSTS'] = corpus_qsts(hypos, refs_list[0], device=device)
        torch.cuda.empty_cache()
    # bartscore
    if 'BARTScore-ref' in score_names or 'BARTScore-src' in score_names:
        print('BARTScore', '#'*50)
        from BARTScore.bart_score import BARTScorer
        scorer = BARTScorer(device=device)
        scorer.load()
        if 'BARTScore-ref' in score_names:
            print('BARTScore-ref', '#'*50)
            prompt_hypos = prefix_prompt(hypos, 'To rephrase it')
            scores = scorer.score(refs_list[0], prompt_hypos, batch_size=4)
            res['BARTScore-ref'] = [round(x, 4) for x in scores]
        if 'BARTScore-src' in score_names:
            print('BARTScore-src', '#'*50)
            srcs = ['Passage: {}\tAnswer: {}\t'.format(p, a) for p, a in zip(contexts, answers)]
            prompt_hypos = prefix_prompt(hypos, 'Generate a question based on the given passage and answer')
            scores = scorer.score(srcs, prompt_hypos, batch_size=4)
            res['BARTScore-src'] = [round(x, 4) for x in scores]
        torch.cuda.empty_cache()
    # unieval
    if 'UniEval' in score_names:
        print('UniEval', '#'*50)
        from UniEval.unieval import evaluate_qg
        dims = ['fluency', 'clarity', 'conciseness', 'relevance', 'consistency',
                'answerability', 'acceptance', 'answer_consistency']
        score_dict_list = evaluate_qg(hypos, answers, contexts, device, dims=dims)
        new_scores = []
        for one in score_dict_list:
            for k, v in one.items():
                one[k] = round(v, 4)
            new_scores.append(one)
        res['UniEval'] = new_scores
        torch.cuda.empty_cache()
    # gptscore
    if 'GPTScore-ref' in score_names or 'GPTScore-src' in score_names:
        print('GPTScore', '#'*50)
        from gptscore import FlanScorerMod
        model_path = 'google/flan-t5-xxl'
        scorer = FlanScorerMod(model_path, device=device)
        if 'GPTScore-ref' in score_names:
            print('GPTScore-ref', '#'*50)
            template_dict = {
                'fluency': 'Rewrite the following question into a fluent and grammatical question. {} In other words, ',
                'clarity': 'Rewrite the following question into a clear question, without any ambiguity. {} In other words, ',
                'conciseness': 'Rewrite the following question into a concise question, without redundant modifiers. {} In other words, ',
                'relevance': 'Rewrite the following question with consistent details. {} In other words, ',
                'consistency': 'Rewrite the following question with consistent facts. {} In other words, ',
                'answerability': 'Rewrite the following question into an answerable question. {} In other words, ',
                'acceptance': 'Rewrite the following question into a fluent, concise, and answerable question with consistent facts. {} In other words, ',
                'answer_consistency': 'Rewrite the following question with the same answer. {} In other words, '
            }
            score_dict_list = scorer.score(refs_list[0], hypos, None, template_dict, batch_size=4)
            res['GPTScore-ref'] = score_dict_list
        if 'GPTScore-src' in score_names:
            print('GPTScore-src', '#'*50)
            template_dict = {
                'fluency': 'Generate a fluent and grammatical question based on the given passage and answer.\n\n Passage: {}\n\nAnswer: {}\n\nQuestion: ',
                'clarity': 'Generate a clear question without any ambiguity based on the given passage and answer.\n\n Passage: {}\n\nAnswer: {}\n\nQuestion: ',
                'conciseness': 'Generate a concise question without redundant modifiers based on the given passage and answer.\n\n Passage: {}\n\nAnswer: {}\n\nQuestion: ',
                'relevance': 'Generate a relevant question based on the given passage and answer.\n\n Passage: {}\n\nAnswer: {}\n\nQuestion: ',
                'consistency': 'Generate a question whose facts are consistent with the context, based on the given passage and answer.\n\n Passage: {}\n\nAnswer: {}\n\nQuestion: ',
                'answerability': 'Generate an answerable question based on the given passage and answer.\n\n Passage: {}\n\nAnswer: {}\n\nQuestion: ',
                'acceptance': 'Generate a fluent, concise, and answerable question whose facts are consistent with the context, based on the given passage and answer.\n\n Passage: {}\n\nAnswer: {}\n\nQuestion: ',
                'answer_consistency': 'Generate a question that can be answered by the provided answer based on the given passage.\n\n Passage: {}\n\nAnswer: {}\n\nQuestion: '
            }
            score_dict_list = scorer.score(contexts, hypos, answers, template_dict, batch_size=4)
            res['GPTScore-src'] = score_dict_list
        torch.cuda.empty_cache()
    # qrel_score
    if 'QRelScore' in score_names:
        print('QRelScore', '#'*50)
        from qrel import corpus_qrel
        res['QRelScore'] = [round(x, 4) for x in corpus_qrel(hypos, contexts, device)]
        torch.cuda.empty_cache()
    # rquge
    if 'RQUGE' in score_names:
        print('RQUGE', '#'*50)
        from RQUGE.rquge_score import RQUGE
        rquge_model = RQUGE(device=device)
        total = 0
        scores = []
        for c, h, a in tqdm(zip(contexts, hypos, answers), total=len(contexts)):
            score = rquge_model.scorer(c, h, a)
            total += score
            scores.append(round(score, 4))
        res['RQUGE'] = scores
        print(f'Average RQUGE score: {total/len(scores)*1.0}')
        torch.cuda.empty_cache()
    
    return res


if __name__ == "__main__":
    import os
    import pandas as pd
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    # load data
    data_path = './test_data/test.xlsx'
    save_path = data_path
    data = pd.read_excel(data_path)
    # prepare parameters
    hypos = data['prediction'].tolist()
    refs_list = [data['target'].tolist()]
    contexts = data['passage'].tolist()
    answers = data['answer'].tolist()
    # metrics to use
    score_names = [
        # 'BLEU-4', 'METEOR','ROUGE-L','BERTScore', 'MoverScore', 
        # 'BLEURT', 'Q-BLEU4', 
        # 'QSTS',  
        # 'BARTScore-ref', 'BARTScore-src',
        # 'UniEval', 
        'QRelScore', 
        # 'RQUGE'
    ]
    # run metric one by one
    res = get_metrics(hypos, refs_list, contexts, answers, score_names=score_names)
    # handle results
    for k, v in res.items():
        data[k] = v
    print(data.columns)
    # save results
    data.to_excel(save_path, index=False)
    print('Metrics saved to {}'.format(save_path))
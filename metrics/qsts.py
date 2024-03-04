import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir+'/QSTS/code')

from TypeSensitiveQSimilarity import QSTS
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

def get_qtype(ques, tokenizer, model, device, **generator_args):
    ques = ques.strip().lower()
    input_ids = tokenizer.encode(ques, return_tensors="pt").to(device)
    res = model.generate(input_ids, **generator_args)
    t = tokenizer.batch_decode(res, skip_special_tokens=True)[0].replace("<pad>","").replace("</s>","").strip()
    if ":" not in t:
        return "ENTY:other"
    return t


def corpus_qsts(preds, refs, device, tcmodel_path=current_dir+'/QSTS/model/qct5', 
                glve_path=current_dir+'/QSTS/embeddings/glove.840B.300d.wiki_ss.pkl'):
    scores = []
    assert len(preds) == len(refs)
    qsts = QSTS(glve_path)
    print('device: {}'.format(device), '#'*50)
    tc_tokenizer = AutoTokenizer.from_pretrained(tcmodel_path)
    tc_model = T5ForConditionalGeneration.from_pretrained(tcmodel_path).to(device)
    for idx in tqdm(range(len(preds))):
        pred = preds[idx]
        target = refs[idx]
        goldqc = get_qtype(target, tc_tokenizer, tc_model, device=device)
        predqc = get_qtype(pred, tc_tokenizer, tc_model, device=device)
        # print(goldqc,predqc)
        score = qsts.scoreQPair(
            goldq=target, predq=pred, ignoreQC=False, goldqc=goldqc, predqc=predqc)
        scores.append(round(score, 4))
    return scores

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    preds = ['what is it?']
    refs = ['what is it?']
    scores = corpus_qsts(preds, refs)
    print(scores)
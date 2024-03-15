import torch
from transformers import AutoTokenizer
from transformers import  AutoModelForSeq2SeqLM
import json 
import pandas as pd
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("mamlong34/t5_large_race_cosmos_qa")
model = AutoModelForSeq2SeqLM.from_pretrained("mamlong34/t5_large_race_cosmos_qa")

file_path = "./train.jsonl"
with open(file_path) as f:
    lines = f.readlines()
eval_examples = [json.loads(line) for line in lines]


def predict_xxl(few_shot_path, test_path, result_save_path, name, k=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_tar_length = {
        'squad1.1': 67,
        'hotpotqa': 163,
        'narrativeqa': 32
    }
    max_length = 4096
    # model = AutoModelForSeq2SeqLM.from_pretrained("t5-11b")
    # tokenizer = AutoTokenizer.from_pretrained("t5-11b")
    inst = 'question:'
    few_data = load_json(few_shot_path)
    print('total few-shot examples count={}'.format(len(few_data)))
    test_data = load_json(test_path)
    result = []
    for one in tqdm(test_data):
        for_pred = '{}'.format(
            one['answer'], one['context']
        )
        # ness_input = inst + for_pred
        # ness_len = tokenizer(ness_input, return_tensors="pt")['input_ids'].size(-1)
        examples = []
        input_text = inst + '</s>'.join(examples) + for_pred
        for i in range(k):
            ness_len = tokenizer(input_text, return_tensors="pt")['input_ids'].size(-1)
            if ness_len > max_length:
                if len(examples) == 0:
                    input_text = inst + for_pred
                else:
                    input_text = inst + '</s>'.join(examples[:-1]) + '</s>' + for_pred
                break
            example = 'answer: {}\tcontext: {}\tquestion: {}'.format(
                few_data[i]['answer'], few_data[i]['context'], few_data[i]['question']
            )
            examples.append(example)
            input_text = inst + '</s>'.join(examples) + '</s>' + for_pred
        if len(result) == 0:
            print(input_text)
        input_ids = tokenizer(input_text, max_length=max_length, truncation=True, return_tensors="pt").input_ids
        print('example count={}, input token length={}'.format(
            len(examples), input_ids.size(-1)
        ))
        outputs = model.generate(input_ids)
        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('prediction: ', decoded_text)
        one['prediction'] = decoded_text
        result.append(one)
    pd.DataFrame(result).to_csv(result_save_path, index=False)



import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from openai import OpenAI
import json


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def completion(model, prompt, max_try=3, prt=False):
    client = OpenAI()
    message = ''
    for i in range(max_try):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                n=1,
                stop=None
            )
            if prt:
                print('{} response:'.format(model))
                print(response)
            message = response.choices[0].message.content
            break
        except Exception as e:
            print(e)
            time.sleep(2)
    return message


# QG prompt
def generate_with_ans_prompt(answer, context):
    prompt = 'Generate a question based on the given answer and context, the generated question must be answered by the given answer.\n\n'
    # prompt = 'Generate a multi-hop question based on the given answer and context, the generated question must be answered by the given answer.\n'
    prompt += 'Answer: {}\nContext: {}\n'.format(answer, context)
    prompt += 'Question:'
    return prompt

def generate_with_ans_fewshot_prompt(data, answer, context):
    prompt = 'Generate a question based on the given answer and context, the generated question must be answered by the given answer.\n\n'
    # prompt = 'Generate a multi-hop question based on the given answer and context, the generated question must be answered by the given answer.\n'
    prompt += 'Examples:\n\n'
    for one in data:
        prompt += 'Answer: {}\nContext: {}\nQuestion: {}\n\n'.format(one['answer'], one['context'], one['question'])
    prompt += 'Answer: {}\nContext: {}\n'.format(answer, context)
    prompt += 'Question:'
    return prompt


# batch QG
def question_generation(data_path, model='gpt-3.5-turbo', save_dir='./result/', few_shot_path=None):
    df = pd.read_excel(data_path)
    df = df.drop_duplicates(subset=['passage','answer'])
    data = df.to_dict(orient='records')
    count = 0
    setting = 'fewshot' if few_shot_path else 'zeroshot'
    few_data = []
    if setting == 'fewshot':
        few_data = load_json(few_shot_path)
    new_data = []
    for item in tqdm(data):
        answer = item['answer']
        context = item['passage']
        message = ''
        prompt = ''
        # settings
        if setting == 'fewshot':
            prompt = generate_with_ans_fewshot_prompt(few_data, answer, context)
        elif setting == 'zeroshot':
            prompt = generate_with_ans_prompt(answer, context)
        else:
            raise ValueError('setting type error: {}'.format(setting))
        if count == 0:
            print('*'*30, 'prompt example', '*'*30)
            print(prompt)
        # apply api
        message = completion(model, prompt)
        item[model+'_'+setting] = message.strip()
        new_data.append(item)
        count += 1
        if save_dir:
            pd.DataFrame(new_data).to_excel(save_dir+'/{}-{}.xlsx'.format(model, setting), index=False)
    pd.DataFrame(new_data).to_excel(save_dir+'/{}-{}.xlsx'.format(model, setting), index=False)
    

    
if __name__ == "__main__":
    data_path = './test.xlsx'
    model = 'gpt-3.5-turbo'
    save_dir = './result/'
    few_shot_path = './data/hotpotqa-few.json'
    question_generation(
        data_path=data_path,
        model=model,
        save_dir=save_dir,
        few_shot_path=few_shot_path
    )
    
   
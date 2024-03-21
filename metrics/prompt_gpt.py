import openai
from tqdm import tqdm
import time
import pandas as pd
import os
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# 旧的调用方式
def completion_old(model, prompt, max_try=1):
    count = 0
    message = ''
    openai.api_base = ''
    openai.api_key = ''
    for i in range(max_try):
        try:
            _response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "you are a professional evaluator"},
                        {"role": "user", "content": prompt}
                    ],
                    # temperature=2,
                    # max_tokens=5,
                    # top_p=1,
                    # frequency_penalty=0,
                    # presence_penalty=0,
                    # stop=None,
                    # # logprobs=40,
                    # n=20
                )
            # time.sleep(0.5)
            # print('resp')
            # print(_response)
            # all_responses = [_response['choices'][i]['message']['content'] for i in
            #                      range(len(_response['choices']))]
            message = _response.choices[0].message.content
            break
        except Exception as e:
            print(e)
            continue
    return message


def prompt_one(p, q):
    prompt = 'You will be given one context and one question ask about information in the context. Your task is to rate the answerability of this question based on the given context, and the score is a floating-point number between 0 and 1, where 0 indicates that the question is completely unanswerable, and 1 indicates that the question is answerable.\n\n'
    prompt += 'The answerability of a question is a measure of whether the question can be answered using only the information provided in the given context, without requiring additional external information.\n\n'
    # 输入
    context = 'Context:\n\n{}\n\n'.format(p)
    question = 'Question:\n\n{}\n\n'.format(q)
    prompt += context + question
    prompt += 'Answerability (score ONLY):'
    # print(prompt)
    model = 'gpt-4-1106-preview'
    # model = 'gpt-3.5-turbo'
    message = completion_old(model, prompt)
    # print(message)
    time.sleep(1)
    return message

def prompt_all(data, json_path):
    new_data = []
    print(len(data))
    for one in tqdm(data):
        new_one = {k:v for k,v in one.items()}
        p = one['passage']
        q = one['prediction']
        message = prompt_one(p, q)
        new_one['response'] = message
        new_data.append(new_one)
        if len(new_data) % 5 == 0:
            save_json(new_data, json_path)
    save_json(new_data, json_path)
    return new_data

if __name__ == "__main__":
    
    p = '"Bryan Charles Kocis (May 28, 1962 – January 24, 2007), also known as Bryan Phillips, was a director of gay pornographic films and founder of Cobra Video, a gay porn film studio.  Kocis was murdered at his Dallas Township, Pennsylvania home on January 24, 2007; arson was used in an attempt to disguise the circumstances of his death.  Two escorts, Harlow Cuadra and Joseph Kerekes, were charged and convicted for Kocis\' murder and subsequently given a sentence of life imprisonment without any possibility of parole.\nSchoolboy Crush (2004) is a controversial gay pornographic film directed by Bryan Kocis (under the industry name ""Bryan Phillips""), released on Cobra Video, and cast with Brent Everett and Sean Paul Lockhart under the stage name ""Brent Corrigan"".  Corrigan being underage at the time of filming led to legal actions against Phillips and the withdrawal of the film ""Schoolboy Crush"" from the Cobra Video film catalog."'
    q = 'Who were the two escorts charged and convicted for the murder of Bryan Charles Kocis, the founder of Cobra Video?'
    print(prompt_one(p, q))
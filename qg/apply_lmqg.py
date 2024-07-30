from lmqg import TransformersQG
from utils import load_json
import pandas as pd
import os


def apply_lmqg(model_name, data_json_path='./data/squad-test.json', save_path='./result/squad-test.csv', batch_size=8):
    # initialize model
    print('#'*30, 'load model', '#'*30)
    model = TransformersQG(model=model_name, max_length=1024)
    # load test data
    data_dir = './data/'
    print('#'*30, 'load data', '#'*30)
    data = load_json(data_json_path)
    print('data size={}'.format(len(data)))

    # a list of passages and answers
    context = [x['context'] for x in data]
    answer = [x['answer'] for x in data]

    # model prediction
    print('#'*30, 'generate', '#'*30)
    if len(data) % batch_size == 0:
        iters = len(data) // batch_size
    else:
        iters = len(data)//batch_size + 1
    st = 0
    question = []
    for iter in range(iters):
        ed = min(len(data), (iter+1)*batch_size)
        print('start={}, end={}'.format(st, ed))
        temp = model.generate_q(list_context=context[st:ed], list_answer=answer[st:ed])
        st = ed
        question += temp

    # save result
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df = pd.DataFrame(data)
        df['prediction'] = question
        df.to_csv(save_path, index=False)
        print('save result to {}'.format(save_path))
    
    return question


if __name__ == "__main__":
    model_name = 'lmqg/flan-t5-base-squad-qg'
    data_json_path='./data/squad-test.json'
    save_path='./result/squad-test.csv'
    questions = apply_lmqg(
        model_name=model_name,
        data_json_path=data_json_path,
        save_path=save_path,
        batch_size=8
    )






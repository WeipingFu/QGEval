from lmqg import TransformersQG
from utils import load_json
import pandas as pd
import os


model_path = 'lmqg/flan-t5-base-squad-qg'

# initialize model
print('#'*30, 'load model', '#'*30)
model = TransformersQG(model=model_path, max_length=1024)

# load test data
data_dir = './data/'
print('#'*30, 'load data', '#'*30)
# data_df = pd.read_csv(data_dir + 'squad-dev.csv')
# data = data_df.to_dict(orient='records')
data = load_json(data_dir+'squad-test.json')
print('data size={}'.format(len(data)))

# a list of paragraph
context = [x['context'] for x in data]

# a list of answer (same size as the context)
answer = [x['answer'] for x in data]

# model prediction
print('#'*30, 'predict', '#'*30)
batch_size = 8
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
df = pd.DataFrame(data)
df['prediction'] = question
df['source'] = ['SQuAD_FlanT5-large_finetune']*len(question)
save_path = data_dir + 'predictions/squad-flant5-large.csv'
df.to_csv(save_path, index=False)
print('save result to {}'.format(save_path))

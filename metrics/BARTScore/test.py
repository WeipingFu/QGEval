from bart_score import BARTScorer
import pandas as pd
bart_scorer = BARTScorer(device='cuda:0', checkpoint='./model')
bart_scorer.load(path='bart_score.pth')
file_path = './score.xlsx'
df = pd.read_excel(file_path, sheet_name = "Sheet1")
with open('bs.txt','w') as f:
    for i in range(len(df['prediction'])):
        
        s = bart_scorer.score(["Generate a question according to the passage and answer. Passage:" + df["passage"][i]+ "Answer:" + df["answer"][i]], [df["prediction"][i]], batch_size=4)
        f.write(str(s)[1:-2]+'\n')


# print(df["prediction"][0])
# print(bart_scorer.score(['This is interesting.'], ['This is fun.'], batch_size=4)) # generation scores from the first list of texts to the second list of texts.

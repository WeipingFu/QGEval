## How to use
Our codes provide the ability to `evaluate automatic metrics` which concludes the ability to `calculate automatic metrics`. Please follow these steps to calculate automatic QG metrics and evaluate automatic metrics on our benchmark.

1. prepare data for Question Generation, you can use the hotpotQA dataset and squad1.1 dataset which are provided at [./data](./data), or you can use your dataset which provides passages, questions, and answers.
2. check your Python environment or just run `pip install -r requirements.txt` to install the required packages.
3. the data should be processed first, run `python process.py` to  process the data, and you may change the data dictionary in the specific class in [./process.py](./process.py) if you change the dataset dictionary, and change the model you want to train in the main function in [./process.py](./process.py).
4. after processing the data, try to train your model. We provide methods to train T5/FLAN-T5/BART-based QG model. You may run the code file for specific models to train. For example, run `python T5.py` to train your T5-based QG model

## How to use
Our codes provide the ability to `evaluate automatic metrics` `calculate automatic metrics`.
### Evaluation of Automatic Metrics
- The codes for **Evaluation of Automatic Metrics** are in [metric](./metric)
- Take the evaluation of QRelScore as an example, you can use the QGEval benchmark to evaluate QRelScore by these steps:
1. prepare data, you can get the QGEval benchmark at [data/scores.xlsx](./data/scores.xlsx)
2. cd ./metric
3. run `pip install -r requirements.txt` to install the required packages
4. run `python grel.py` or `python metrics.py` to get QRelScore metric result
5. run `python pearson.py` to obtain the Pearson correlation coefficient between the generated results and the labeled results.

Find more details in [metric/readme](./metric/readme).

### Question Generation
- The codes and the data for **Question Generation** are in [qg](./qg), train your own QG model by these steps:
1. cd ./qg
2. run `pip install -r requirements.txt` to install the required packages
3. run `python process.py` to process data
4. run the code file for specific models to train. For example, run `python T5.py` to train your T5-based QG model.

Find more details in [qg/readme](./qg/readme).

###  Automatic Metrics Calculation
- The codes for **Automatic metrics Calculation(e.g. BLEU-4)** are in [metric](./metric), calculate automatic metrics by these steps:
1. prepare data, you can get the Question Generation dataset at [qg/data](./qg/data) or you can prepare data yourself
2. cd ./metric
3. run `pip install -r requirements.txt` to install the required packages
4. run `python metrics.py` to get your chosen metrics evaluation result.

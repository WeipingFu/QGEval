## How to use
Our codes provide the ability to  `train Question Generation model`, please follow these steps to train your QG model.

- The codes for **Evaluation of Automatic Metrics** are in [metric](./metric)
- Take the evaluation of QRelScore as an example, you can use the QGEval benchmark to evaluate QRelScore by these steps:
1. prepare data, you can get the QGEval benchmark at [data/scores.xlsx](./data/scores.xlsx)
2. cd ./metric
3. run `pip install -r requirements.txt` to install the required packages
4. run `python grel.py` or `python metrics.py` to get QRelScore metric result
5. run `python pearson.py` to obtain the Pearson correlation coefficient between the generated results and the labeled results.

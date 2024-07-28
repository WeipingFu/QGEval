## How to use
Our codes provide the ability to `evaluate automatic metrics` which concludes the ability to `calculate automatic metrics`. Please follow these steps to calculate automatic QG metrics and evaluate automatic metrics on our benchmark.

1. prepare data for Question Generation evaluation, you can use your QG model to generate the data, which should provide passages, answers, and targets from the our QG dataset(data in [data/original_data.xlsx](data/original_data.xlsx)) and predictions from the generated results. The format can refer to [data/scores.xlsx](data/scores.xlsx).
2. check your Python environment or just run `pip install -r requirements.txt` to install the required packages.
3. calculate automatic metrics. You may run the code file for specific metrics to calculate. For example, run `python QRel.py` to calculate the QRel result. You can also run `python metrics.py` to calculate your assigned metrics results by changing `score_names` in `metrics.py`. (`data_path` in each file should be changed into your data path)
4. run `python coeff.py` to obtain the Pearson, Spearman, and Kendall correlation coefficient between the generated results and the labeled results. (`df` in `coeff.py` should be changed into your data path, and your data should provide two lines of the generated results and the labeled results so that to calculate correlation coefficient)

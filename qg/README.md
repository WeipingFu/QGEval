## How to use
Our codes provide the ability to  `train Question Generation model`, please follow these steps to train your QG model.

1. prepare data for Question Generation, you can use the hotpotQA dataset and squad1.1 dataset which are provided at [./data](./data), or you can use your dataset which provides passages, questions, and answers. You may 
2. check your Python environment or just run `pip install -r requirements.txt` to install the required packages
5. run `python grel.py` or `python metrics.py` to get QRelScore metric result
6. run `python pearson.py` to obtain the Pearson correlation coefficient between the generated results and the labeled results.

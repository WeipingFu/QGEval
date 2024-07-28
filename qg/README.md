## How to use
Our codes provide the ability to  `train Question Generation model`, please follow these steps to train your QG model.

1. prepare data for Question Generation, you can use the hotpotQA dataset and squad1.1 dataset which are provided at [./data](./data), or you can use your dataset which provides passages, questions, and answers.
2. check your Python environment or just run `pip install -r requirements.txt` to install the required packages.
3. the data should be processed first, run `python process.py` to  process the data, and you may change the data dictionary in the specific class in [./process.py](./process.py) if you change the dataset dictionary, and change the model you want to train in the main function in [./process.py](./process.py).
4. after processing the data, try to train your model. we provide methods to train T5/FLAN-T5/BART-based QG model and run the code file for specific models to train. For example, run `python T5.py` to train your T5-based QG model

# QGEval
Resources for paper - QGEval: A Benchmark for Question Generation Evaluation

## Data
We share the generated questions from 15 QG systems with averaged annotation scores of three annotators in [data/scores.xlsx](./data/scores.xlsx), and the instances integrated by passages are in [data/instances.json](./data/instances.json).
We also share the annotation result of each annotator in [data/annotation result](./data/annotation%20result).

Example of instances.
```json
{
  "id": "572882242ca10214002da423",
  "passage": "... The publication of a Taoist text inscribed with the name of Töregene Khatun, Ögedei's wife, ...",
  "reference": "Who was Ögedei's wife?"
  "answer": "Töregene Khatun",
  "questions": [
      {
        "prediction": "Who was the author of the Taoist text inscribed with the name of?",
        "source": "SQuAD_BART-base_finetune",
        "fluency": 3.0,
        "clarity": 2.6667,
        "conciseness": 3.0,
        "relevance": 3.0,
        "consistency": 2.0,
        "answerability": 1.0,
        "answer_consistency": 1.0
      },
      // ... 14 more questions
  ]
}
```

The average annotation scores of each QG system over 7 dimensions are shown in the below table.
| **Systems**                | **Flu.** | **Clar.** | **Conc.** | **Rel.** | **Cons.** | **Ans.** | **AnsC.** | **Avg.** | 
|-----------------------------|----------|-----------|-----------|----------|-----------|----------|----------|-----------|
| M1 - Reference              | 2.968    | 2.930     | **2.998** | 2.993    | 2.923     | 2.832    | **2.768** | **2.916** |
| M2 - BART-base-finetune     | 2.958    | 2.882     | 2.898     | 2.995    | 2.920     | <u>2.732</u>  | 2.588     | 2.853 |
| M3 - BART-large-finetune    | <u>2.932</u>  | 2.915     | <u>2.828</u>   | 2.995    | 2.935    | 2.825    | **2.737** | 2.881  |
| M4 - T5-base-finetune       | 2.972    | 2.923     | 2.922     | **3.000**| <u>2.917</u>   | 2.788    | 2.652     | 2.882 |
| M5 - T5-large-finetune      | 2.978    | 2.930     | 2.907     | 2.995    | 2.933     | 2.795    |  2.720    | 2.894 |
| M6 - Flan-T5-base-finetune | 2.963    | 2.888     | 2.938     | **2.998**| 2.925     | 2.775    | 2.665     | 2.879 |
| M7 - Flan-T5-large-finetune| 2.982    | 2.902     | 2.895     | 2.995    | **2.950**| 2.818    | 2.727     | 2.895 |
| M8 - Flan-T5-XL-LoRA        | <u>2.913</u>  | <u>2.843</u>   | <u>2.880</u>   | 2.997    | 2.928     | 2.772    | 2.667     | 2.857 |
| M9 - Flan-T5-XXL-LoRA       | <u>2.938</u>  | <u>2.848</u>   | 2.907     | **3.000**| 2.943    | 2.757    | 2.678     | 2.867 |
| M10 - Flan-T5-XL-fewshot    | 2.975    | <u>2.820</u>   | **2.985** | <u>2.955</u>  | <u>2.908</u>   | <u>2.652</u>  | <u>2.193</u>   | <u>2.784</u> |
| M11 - Flan-T5-XXL-fewshot   | **2.987**| 2.882     | **2.990** | <u>2.988</u>  | 2.920     | <u>2.687</u>  |  2.432     | <u>2.841</u> |
| M12 - GPT-3.5-Turbo-fewshot | 2.972    | 2.927     | <u>2.858</u>   | 2.995    | **2.955**| **2.850**|  <u>2.335</u>   | 2.842 |
| M13 - GPT-4-Turbo-fewshot   | **2.988**| **2.987** | 2.897     | 2.992    | **2.947**| **2.922**| **2.772** | **2.929** |
| M14 - GPT-3.5-Turbo-zeroshot| **2.995**| **2.977** | 2.913     | 2.992    | <u>2.917</u>   | 2.823    | <u>2.157</u>   | <u>2.825</u> |
| M15 - GPT-4-Turbo-zeroshot  | 2.983    | **2.990** | 2.943     | <u>2.970</u>  | 2.932     | **2.883**| 2.723     | **2.918** |
| Avg.                  | 2.967    | 2.910     | 2.917     | 2.991    | 2.930     | 2.794   | 2.588 |


## Metrics
We implemented 15 metrics for re-evaluation, they are:
| **Metrics**                | **Paper** | **Code** |
|-----------------------------|----------|-----------|
| BLEU-4            | [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf)  |  [link](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)   |
| ROUGE-L     | [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)   | [link](https://github.com/google-research/google-research/tree/master/rouge)   | 
| METEOR    | [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://aclanthology.org/W05-0909.pdf)  | [link](https://www.nltk.org/api/nltk.translate.meteor_score.html) | 
| BERTScore       | [BERTScore: Evaluating Text Generation with BERT](https://openreview.net/pdf?id=SkeHuCVFDr)   | [link](https://github.com/Tiiiger/bert_score)     | 
| MoverScore      | [MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://aclanthology.org/D19-1053.pdf)   | [link](https://github.com/AIPHES/emnlp19-moverscore)     | 
| BLEURT | [BLEURT: Learning Robust Metrics for Text Generation](https://aclanthology.org/2020.acl-main.704.pdf)    | [link](https://github.com/google-research/bleurt)     |
| BARTScore-ref| [BARTScore: Evaluating Generated Text as Text Generation](https://arxiv.org/pdf/2106.11520.pdf)    | [link](https://github.com/neulab/BARTScore)     |
| GPTScore-ref        | [GPTScore: Evaluate as You Desire](https://arxiv.org/pdf/2302.04166.pdf)  | [link](https://github.com/jinlanfu/GPTScore)   | 
| Q-BLEU4       | [Towards a Better Metric for Evaluating Question Generation Systems](https://aclanthology.org/D18-1429.pdf)  | [link](https://github.com/PrekshaNema25/Answerability-Metric)   |
| QSTS    | [QSTS: A Question-Sensitive Text Similarity Measure for Question Generation](https://aclanthology.org/2022.coling-1.337.pdf)    | [link](./metrics/QSTS)   | 
| BARTScore-src   | [BARTScore: Evaluating Generated Text as Text Generation](https://arxiv.org/pdf/2106.11520.pdf) | [link](https://github.com/neulab/BARTScore)     | 
| GPTScore-src | [GPTScore: Evaluate as You Desire](https://arxiv.org/pdf/2302.04166.pdf)    | [link](https://github.com/jinlanfu/GPTScore)     |
| QRelScore   | [QRelScore: Better Evaluating Generated Questions with Deeper Understanding of Context-aware Relevance](https://aclanthology.org/2022.emnlp-main.37.pdf) | [link](https://github.com/Robert-xiaoqiang/QRelScore) | 
| UniEval| [Towards a Unified Multi-Dimensional Evaluator for Text Generation](https://aclanthology.org/2022.emnlp-main.131.pdf) | [link](https://github.com/maszhongming/UniEval) | 
| RQUGE  | [RQUGE: Reference-Free Metric for Evaluating Question Generation by Answering the Question](https://aclanthology.org/2023.findings-acl.428.pdf)   | [link](https://github.com/alirezamshi/RQUGE) | 

We share the results of each metric on each generated question in [data/metric_result.xlsx](https://github.com/WeipingFu/QGEval/blob/main/data/metric_result.xlsx).
Results of LLM-based metrics on answerability are in [data/test_answerability.xlsx](./data/test_answerability.xlsx).

## Models
You can find our trained model at (https://huggingface.co/hujianxiang/QG_EVAL).

## Quick Start
### Question Generation
The codes and the data for Question Generation are in [qg](./qg), you can train your own QG model by these steps:
1. cd ./qg
2. run `pip install -r requirements.txt` to install the required packages
3. run `python process.py` to process data
4. run the code file for specific models to train. For example, run `python T5.py` to train your T5-based QG model.


### Automatic Metrics
The codes for Automatic Metrics are in [metric](./metric). 

-use **our metrics(QGeval)** to evaluate your questions:
1. cd ./metric
2. run `pip install -r requirements.txt` to install the required packages
3. run `python geval.py` to get our metric evaluation result


-use **other automatic metrics**(e.g. BLEU-4) to evaluate your questions:
1. cd ./metric
2. run `pip install -r requirements.txt` to install the required packages
3. run `python metrics.py` to get your chosen metrics evaluation result
[Notice] When applying metric codes, please modify the model file location of each metric in the code.

## Citation
Please cite:
```python
@misc{fu2024qgeval,
      title={QGEval: A Benchmark for Question Generation Evaluation}, 
      author={Weiping Fu and Bifan Wei and Jianxiang Hu and Zhongmin Cai and Jun Liu},
      year={2024},
      eprint={2406.05707},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

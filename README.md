# QGEval
Resources for paper - QGEval: A Benchmark for Question Generation Evaluation

## Data
We share the generated questions from 15 QG systems with averaged annotation scores of three annotators in [data/scores.xlsx](https://github.com/WeipingFu/QGEval/blob/main/data/scores.xlsx).
We also share the annotation result of each annotator in [data/annotation result](https://github.com/WeipingFu/QGEval/tree/main/data/annotation%20result).

The average annotation scores of each QG system over eight dimensions are shown in the below table.
| **Systems**                | **Flu.** | **Clar.** | **Conc.** | **Rel.** | **Cons.** | **Ans.** | **Acc.** | **AnsC.** |
|-----------------------------|----------|-----------|-----------|----------|-----------|----------|----------|-----------|
| M1 - Reference              | 2.968    | 2.930     | **2.998** | 2.993    | 2.923     | 2.832    | **2.785** | **2.768** |
| M2 - BART-base-finetune     | 2.958    | 2.882     | 2.898     | 2.995    | 2.923     | *2.732*  | *2.638*  | 2.588     |
| M3 - BART-large-finetune    | *2.933*  | 2.915     | *2.828*   | 2.995    | 2.935     | 2.825    | 2.662    | **2.737** |
| M4 - T5-base-finetune       | 2.972    | 2.923     | 2.922     | **3.000**| *2.917*   | 2.788    | 2.685    | 2.652     |
| M5 - T5-large-finetune      | 2.978    | 2.930     | 2.907     | 2.995    | 2.933     | 2.795    | 2.695    | 2.720     |
| M6 - Flan-T5-base-finetune | 2.963    | 2.888     | 2.938     | **2.998**| 2.925     | 2.775    | 2.708    | 2.665     |
| M7 - Flan-T5-large-finetune| 2.982    | 2.902     | 2.895     | 2.995    | **2.950**| 2.818    | 2.717    | 2.727     |
| M8 - Flan-T5-XL-LoRA        | *2.913*  | *2.843*   | *2.880*   | 2.997    | 2.928     | 2.770    | 2.643    | 2.667     |
| M9 - Flan-T5-XXL-LoRA       | *2.938*  | *2.848*   | 2.907     | **3.000**| 2.942     | 2.755    | 2.677    | 2.677     |
| M10 - Flan-T5-XL-fewshot    | 2.975    | *2.820*   | **2.985** | *2.955*  | *2.908*   | *2.652*  | *2.602*  | *2.193*   |
| M11 - Flan-T5-XXL-fewshot   | **2.987**| 2.882     | **2.990** | *2.988*  | 2.918     | *2.685*  | *2.623*  | 2.430     |
| M12 - GPT-3.5-Turbo-fewshot | 2.972    | 2.927     | *2.858*   | 2.995    | **2.962**| **2.852**| 2.757    | *2.330*   |
| M13 - GPT-4-Turbo-fewshot   | **2.988**| **2.987** | 2.897     | 2.992    | **2.947**| **2.922**| **2.813**| **2.772** |
| M14 - GPT-3.5-Turbo-zeroshot| **2.995**| **2.977** | 2.915     | 2.992    | *2.913*   | 2.823    | 2.712    | *2.157*   |
| M15 - GPT-4-Turbo-zeroshot  | 2.983    | **2.990** | 2.943     | *2.970*  | 2.932     | **2.883**| **2.785**| 2.723     |
| **Avg.**                    | 2.967    | 2.910     | 2.917     | 2.991    | 2.930     | 2.794   


## Metrics
We implemented 15 metrics for re-evaluation, they are:
| **Metrics**                | **Paper** | **Code** |
|-----------------------------|----------|-----------|
| BLEU-4            | link  |  link   |
| ROUGE-L     | link   | link   | 
| METEOR    | link  | link     | 
| BERTScore       | link   | link     | 
| MoverScore      | link   | link     | 
| BLEURT | link    | link     |
| BARTScore-ref| link    | link     |
| GPTScore-ref        | link  | link   | 
| Q-BLEU4       | link  | link   |
| QSTS    | link    | link   | 
| BARTScore-src   | link| link     | 
| GPTScore-src | link    | link     |
| QRelScore   | link | link | 
| UniEval| link | link | 
| RQUGE  | link   | link | 

We share the results of each metric on each generated question in [data/metric_result.xlsx](https://github.com/WeipingFu/QGEval/blob/main/data/metric_result.xlsx).

## Citation

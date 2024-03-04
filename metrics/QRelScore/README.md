# QRelScore
This repository is the official implementation of the EMNLP 2022 paper
[QRelScore: Better Evaluating Generated Questions with Deeper Understanding of Context-aware Relevance](https://arxiv.org/abs/2204.13921)

## Prerequisites
```bash
pip install -r requirements.txt
```

## Run specification for QG
```bash
python source/main.py --cfg configure/configure_filename
# example configure files are included in the configure directory
```

## Example usage for QG evaluation (QRelScore)
```python
from evalpackage.qrelscore import QRelScore

# instantiate the scorer class by loading required pre-trained language models
scorer = QRelScore()

context_str = 'The fight scene finale between Sharon and the character played by Ali Larter, from the movie Obsessed, won the 2010 MTV Movie Award for Best Fight.'
candidate_str = 'Which award did the fight scene between Sharon and the role of Ali Larter win?'

# a list of input passages (contexts)
gts = [ context_str ]
# a list of generated questions (candidates)
res = [ candidate_str ]

# a list of relevance scores
scores = scorer.compute_score_flatten(gts, res)
print(scores)

```

## License
Copyright 2022 Author of this paper

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Citation
```latex
@article{wang2022qrelscore,
  title={QRelScore: Better Evaluating Generated Questions with Deeper Understanding of Context-aware Relevance},
  author={Wang, Xiaoqiang and Liu, Bang and Tang, Siliang and Wu, Lingfei},
  journal={arXiv preprint arXiv:2204.13921},
  year={2022}
}
```
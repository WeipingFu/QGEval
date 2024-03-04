from ..mlmscore import MLMScore
from ..clmscore import CLMScore

class QRelScore:
    def __init__(self, mlm_model='bert-base-cased', clm_model='gpt2', batch_size = 32, nthreads = 4, device = None):
        self.mlm_scorer = MLMScore(mlm_model, batch_size, nthreads, device)
        self.clm_scorer = CLMScore(clm_model, batch_size, nthreads, device)

    '''
        Compute the QRelScore for the list of candidates or hypotheses

        Args:
            - :param: `gts` (list of str) : contexts of the samples
            - :param: `res` (list of str) : generated questions of the samples (hypotheses or candidates)

        Return:
            - `qrelscore` (float) : relevance score of generated questions
    '''
    def compute_score_flatten(self, gts, res):
        mlm_scores = self.mlm_scorer.compute_score_flatten(gts, res)
        clm_scores = self.clm_scorer.compute_score_flatten(gts, res)

        epsilon = 1.0e-8

        # scores = list(map(lambda t: 2 * t[0] * t[1] / (t[0] + t[1] + epsilon), zip(mlm_scores, clm_scores)))
        scores = []
        for mlm_score, clm_score in zip(mlm_scores, clm_scores):
            if clm_score == -1:
                score = -1
            else:
                score = 2 * mlm_score * clm_score / (mlm_score + clm_score + epsilon)
            scores.append(score)
        return scores

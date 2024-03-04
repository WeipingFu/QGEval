
from GPTScore.flan_score import FLANScorer

class FlanScorerMod:
    def __init__(self, model_path, max_length=1024, device='cuda:0'):
        self.flan_scorer = FLANScorer(device=device, 
                                      max_length=max_length, 
                                      checkpoint=model_path)
        print('FLANScorer setup finished. Begin calculating FLANScorer.')
        
    def add_prompt(self, srcs, adds, template):
        prompts = []
        if adds is None:
            for src in srcs:
                prompts.append(template.format(src))
        else:
            for src, add in zip(srcs, adds):
                prompts.append(template.format(src, add))
        return prompts
    
    def reform_scores(self, score_dict):
        score_list = []
        dims = list(score_dict.keys())
        length = len(score_dict[dims[0]])
        for i in range(length):
            one = {dim:score_dict[dim][i] for dim in dims}
            score_list.append(one)
        return score_list
    
    def score(self, srcs, tgts, adds, template_dict, batch_size=1):
        score_dict = {dim:0 for dim in template_dict.keys()}
        for dim, template in template_dict.items():
            prompts = self.add_prompt(srcs, adds, template)
            score_list = self.flan_scorer.score(prompts, tgts, batch_size)
            score_list = [round(x, 4) for x in score_list]
            score_dict[dim] = score_list
            avg_score = sum(score_list) / len(score_list)
            print('{} avg score is: {}'.format(dim, avg_score))
        final_scores = self.reform_scores(score_dict)
        return final_scores

    
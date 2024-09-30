import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .utils import convert_to_json
from metric.evaluator import get_evaluator

# Example for qg
# task = 'qg'

# # a list of model outputs to be evaluataed
# output_list = ['Sophocles demonstrated civil disobedience in a play that was called?']
# # a list of answers
# ref_list = ['Antigone']
# # a list of source passage
# src_list = ['One of the oldest depictions of civil disobedience is in Sophocles\' play Antigone, in which Antigone, one of the daughters of former King of Thebes, Oedipus, defies Creon, the current King of Thebes, who is trying to stop her from giving her brother Polynices a proper burial. She gives a stirring speech in which she tells him that she must obey her conscience rather than human law. She is not at all afraid of the death he threatens her with (and eventually carries out), but she is afraid of how her conscience will smite her if she does not do this.']

# # Prepare data for pre-trained evaluators
# data = convert_to_json(output_list=output_list, ref_list=ref_list, src_list=src_list)
# # Initialize evaluator for a specific task
# evaluator = get_evaluator(task, device='cuda:6')
# # Get multi-dimensional evaluation scores
# eval_scores = evaluator.evaluate(data, print_result=True)



'''
# Example for summarization
task = 'summarization'

# a list of source documents
src_list = ['Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital.']
# a list of human-annotated reference summaries
ref_list = ['Elizabeth was hospitalized after attending a party with Peter.']
# a list of model outputs to be evaluataed
output_list = ['Peter and Elizabeth attend party city. Elizabeth rushed hospital.']

# Prepare data for pre-trained evaluators
data = convert_to_json(output_list=output_list, 
                       src_list=src_list, ref_list=ref_list)
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# Get multi-dimensional evaluation scores
eval_scores = evaluator.evaluate(data, print_result=True)
# eval_scores = evaluator.evaluate(data, dims=['coherence', 'consistency', 'fluency'], 
#                                  overall=False, print_result=True)




# Example for dialogue response generation
task = 'dialogue'

# a list of dialogue histories
src_list = ['hi , do you know much about the internet ? \n i know a lot about different sites and some website design , how about you ? \n\n']
# a list of additional context that should be included into the generated response
context_list = ['the 3 horizontal line menu on apps and websites is called a hamburger button .\n']
# a list of model outputs to be evaluated
output_list = ['i do too . did you know the 3 horizontal line menu on apps and websites is called the hamburger button ?']

# Prepare data for pre-trained evaluators
data = convert_to_json(output_list=output_list, 
                       src_list=src_list, context_list=context_list)
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# Get multi-dimensional evaluation scores
eval_scores = evaluator.evaluate(data, print_result=True)



# Example for factual consistency detection
task = 'fact'

# a list of source documents
src_list = ['Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital.']
# a list of model outputs (claims) to be evaluataed
output_list = ['Tom was rushed to hospital.']

# Prepare data for pre-trained evaluators
data = convert_to_json(output_list=output_list, src_list=src_list)
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# Get factual consistency scores
eval_scores = evaluator.evaluate(data, print_result=True)
'''

def evaluate_qg(output_list, ref_list, src_list, device, dims=None):
    task = 'qg'
    # Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=output_list, ref_list=ref_list, src_list=src_list)
    # Initialize evaluator for a specific task
    evaluator = get_evaluator(task, device=device)
    # Get multi-dimensional evaluation scores
    eval_scores = evaluator.evaluate(data, dims=dims, print_result=True)
    return eval_scores

if __name__ == '__main__':
    import pandas as pd
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: {}'.format(device), '#'*50)
    data_path = ''
    save_path = data_path
    data = pd.read_excel(data_path)
    eval_scores = evaluate_qg(data['prediction'].tolist(), data['answer'].tolist(), data['passage'].tolist(), device=device)
    data['UniEval'] = eval_scores
    data.to_excel(save_path, index=False)
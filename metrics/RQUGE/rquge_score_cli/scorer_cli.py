import torch
from rquge_score import RQUGE
import argparse
import os
from tqdm import tqdm

def main():

    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser("Calculate RQUGE score")

    parser.add_argument(
        "--sp_scorer_path",
        type=str,
        default=None,
        help='path to the span scorer model',
    )

    parser.add_argument(
        "--qa_model_path",
        type=str,
        default=None,
        help='path to QA model (either local path or name of the model in huggingface hub',
    )

    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help='The context of generated question',
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help='The generated question',
    )

    parser.add_argument(
        "--answer",
        type=str,
        default=None,
        help='The gold answer span',
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help='The output path for offline mode',
    )

    parser.add_argument(
        "--input_type",
        type=str,
        choices=["sample","offline"],
        default="sample",
        help='The type of input (sample or offline). In the sample mode, "--context", "--question", and '
             '"--answer" commands contain string, while in offline mode, they contain path to files including contexts,'
             ' corresponding questions and answers',
    )

    args = parser.parse_args()

    if not (args.context is not None and args.question is not None and args.answer is not None):
        raise ValueError('None of "--context","--question", and "--answer" commands should be None!')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rquge_model = RQUGE(sp_scorer_path=args.sp_scorer_path, qa_model_path=args.qa_model_path, device=device)

    print("RQUGE model is created....\n"
          "Computing the score....")

    if args.input_type == "sample":
        print("Sample Mode is initiated...")
        print(f'RQUGE score: {rquge_model.scorer(args.context, args.question, args.answer)}')
    else:
        contexts = []
        with open(args.context, 'r') as f:
            for line in f:
                contexts.append(line.strip())

        questions = []
        with open(args.question, 'r') as f:
            for line in f:
                questions.append(line.strip())

        answers = []
        with open(args.answer, 'r') as f:
            for line in f:
                answers.append(line.strip())

        output = []
        total = 0
        for context, question, answer in zip(contexts, questions, answers):
            score = rquge_model.scorer(context, question, answer)
            total += score
            output.append(score)

        with open(args.output_path,'w') as f:
            for num in output:
                f.write(str(num))
                f.write("\n")

        print(f'Output saved in {args.output_path}')
        print(f'Average RQUGE score: {total/len(output)*1.0}')

def apply(data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sp_scorer_path = '../quip-512-mocha'
    qa_model_path = '../unifiedqa-t5-large'
    rquge_model = RQUGE(sp_scorer_path=sp_scorer_path, 
                        qa_model_path=qa_model_path, 
                        device=device
                )
    print("RQUGE model is created....\n"
          "Computing the score....")
    
    total = 0
    new_data = []
    for one in tqdm(data):
        score = rquge_model.scorer(one['passage'], one['prediction'], one['answer'])
        total += score
        one['RQUGE'] = score
        new_data.append(one)
    print(f'Average RQUGE score: {total/len(data)*1.0}')
    return new_data


if __name__ == "__main__":
    # main()
    import pandas as pd
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    data_path = 'test.xlsx'
    save_path = data_path
    data = pd.read_excel(data_path)
    data = data.to_dict(orient='records')
    new_data = apply(data)
    pd.DataFrame(new_data).to_excel(save_path, index=False)
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import pandas as pd

class Coeff:
    def __init__(self):
        pass

    # 皮尔逊相关系数衡量的是两个变量之间的线性相关性，它的取值范围在-1到1之间
    def get_pearson(self, labels, preds):
        labels_array = np.array(labels)
        scores_array = np.array(preds)
        # 计算皮尔逊相关系数和p-value
        correlation, p_value = pearsonr(labels_array, scores_array)
        correlation = round(correlation, 4)
        p_value = round(p_value, 4)
        print("Pearson correlation coefficient: {}, p-value: {}".format(correlation, p_value))
        return correlation, p_value

    # 斯皮尔曼相关系数衡量的是两个变量之间的单调关系，可以是线性关系也可以是非线性关系，它的取值范围在-1到1之间
    def get_spearman(self, labels, preds):
        labels_array = np.array(labels)
        scores_array = np.array(preds)
        # 计算斯皮尔曼相关系数和p-value
        correlation, p_value = spearmanr(labels_array, scores_array)
        correlation = round(correlation, 4)
        p_value = round(p_value, 4)
        print("Spearman correlation coefficient: {}, p-value: {}".format(correlation, p_value))
        return correlation, p_value

    # 肯德尔相关系数衡量的是两个变量之间的等级关系，可以是线性关系也可以是非线性关系，它的取值范围在-1到1之间
    def get_kendall(self, labels, preds):
        labels_array = np.array(labels)
        scores_array = np.array(preds)
        tau, p_value = kendalltau(labels_array, scores_array)
        tau = round(tau, 4)
        p_value = round(p_value, 4)
        print("Kendall correlation coefficient: {}, p-value: {}".format(tau, p_value))
        return tau, p_value

    def apply(self, labels, preds):
        per, per_p = self.get_pearson(labels, preds)
        spea, spea_p = self.get_spearman(labels, preds)
        ken, ken_p = self.get_kendall(labels, preds)
        return per, spea, ken


if __name__ == "__main__":
    import pandas as pd
    import random
    df = pd.read_excel('./dataset/QGEval/llm_test/test_answerability2.xlsx')
    # 计算两列之间的Pearson相关系数
    # df = df.sample(n=600)
    # df_e = df[(df['G-EVAL-gpt4']!=99999)&(df['G-EVAL-gpt3.5']!=99999)]
    df_e = df
    print(df_e.shape)
    cols = ['G-EVAL-gpt4', 'G-EVAL-gpt3.5', 'GPT4', 'GPT3.5', 'GPTScore-flant5xxl', 'UniEval','RQUGE']
    for col in cols:
        if col in df_e:
            correlation = df_e['answerability'].corr(df_e[col])
            print(f"{col}: Pearson Correlation Coefficient: {correlation}")


    # sqd = df[df['source'].str.contains('SQuAD')]
    # hot = df[df['source'].str.contains('HotpotQA')]
    # sqd_passages = list(sqd['passage'].unique())
    # hot_passages = list(hot['passage'].unique())
    # print(len(sqd_passages), len(hot_passages))
    # # random.seed(78)
    # # samples = random.sample(passages, 30)
    # sqd_samples = random.sample(sqd_passages, 15)
    # hot_samples = random.sample(hot_passages, 15)
    # samples = sqd_samples + hot_samples
    # sample_df = df[df['passage'].isin(samples)]
    # # sample_df = df.sample(n=500, random_state=42)
    # # print(sample_df.shape)
    # for col in cols:
    #     if col in df:
    #         correlation = sample_df['answerability'].corr(sample_df[col])
    #         print(f"{col}: Pearson Correlation Coefficient: {correlation}")
    # sample_df.to_excel('./dataset/QGEval/llm_test/test_answerability2.xlsx', index=False)
    # print(sample_df[sample_df['source'].str.contains('SQuAD')].shape)
    # print(sample_df[sample_df['source'].str.contains('HotpotQA')].shape)
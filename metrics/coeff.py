import numpy as np
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr
import pandas as pd

class Coeff:
    def __init__(self):
        pass

    # Pearson, -1~1
    def get_pearson(self, labels, preds, prt=False):
        labels_array = np.array(labels)
        scores_array = np.array(preds)
        # get Pearson
        correlation, p_value = pearsonr(labels_array, scores_array)
        correlation = round(correlation, 3)
        p_value = round(p_value, 3)
        if prt:
            print("Pearson correlation coefficient: {}, p-value: {}".format(correlation, p_value))
        return correlation, p_value

    # Spearman, -1~1
    def get_spearman(self, labels, preds, prt=False):
        labels_array = np.array(labels)
        scores_array = np.array(preds)
        # get Spearman
        correlation, p_value = spearmanr(labels_array, scores_array)
        correlation = round(correlation, 3)
        p_value = round(p_value, 3)
        if prt:
            print("Spearman correlation coefficient: {}, p-value: {}".format(correlation, p_value))
        return correlation, p_value

    # Kendall, -1~1
    def get_kendall(self, labels, preds, prt=False):
        labels_array = np.array(labels)
        scores_array = np.array(preds)
        tau, p_value = kendalltau(labels_array, scores_array)
        tau = round(tau, 3)
        p_value = round(p_value, 3)
        if prt:
            print("Kendall correlation coefficient: {}, p-value: {}".format(tau, p_value))
        return tau, p_value

    def apply(self, labels, preds, prt=False):
        per, per_p = self.get_pearson(labels, preds, prt=prt)
        spea, spea_p = self.get_spearman(labels, preds, prt=prt)
        ken, ken_p = self.get_kendall(labels, preds, prt=prt)
        return per, spea, ken


if __name__ == "__main__":
   import pandas as pd
   result_data_path = 'your result path'
   df = pd.read_excel(result_data_path)
   metrics = ['QRelScore']
   # dimensions to calculate correlation with
   dimensions = ['fluency','clarity','conciseness','relevance','consistency','answerability','answer_consistency']
   # calculate pearson
   coeff = Coeff()
   for metric in metrics:
    print(f"Pearson of {metric}")
    for dimension in dimensions:
      labels = df[dimension].to_list()
      preds = df[metric].to_list()
      per, spea, ken = coeff.apply(labels, preds)
      print(f"{dimension}: Pearson={per}, Spearman={spea}, Kendall={ken}")
      print()
    
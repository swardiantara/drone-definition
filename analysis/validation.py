# Evaluation
from bert_score import score
import numpy as np
import torch
import os
import pandas as pd


def compute_mean_std(precision, recall, f1):
  precision_mean = torch.mean(precision)
  precision_std = torch.std(precision)

  recall_mean = torch.mean(recall)
  recall_std = torch.std(recall)

  f1_mean = torch.mean(f1)
  f1_std = torch.std(f1)
  pre = {
      'mean': (round(float(precision_mean), 5)) * 100,
      'std': (round(float(precision_std), 5)) * 100
  }
  rec = {
      'mean': (round(float(recall_mean), 5)) * 100,
      'std': (round(float(recall_std), 5)) * 100
  }
  f1 = {
      'mean': (round(float(f1_mean), 5)) * 100,
      'std': (round(float(f1_std), 5)) * 100
  }

  return pre, rec, f1


def calculate_bert_score(generated_defs, reference_defs):
    # Calculate BERTScore
    P, R, F1 = score(generated_defs, reference_defs, model_type='microsoft/deberta-v3-large')

    # P: Precision scores
    # R: Recall scores
    # F1: F1 scores

    return P, R, F1


def main():
    deepseek_new = pd.read_excel(os.path.join('experiments', 'deepseek', 'new-deepseek.xlsx'), index_col=0).sort_values('Term')
    deepseek_json = pd.read_excel(os.path.join('experiments', 'deepseek', '2025-12-25_1766661574.xlsx'), index_col=0).sort_values('term')
    reference_df = pd.read_excel(os.path.join('dataset', 'definition.xlsx')).sort_values('term')

    definition_dict = {
        'Model': [],
        'Precision_Mean': [],
        'Recall_Mean': [],
        'F1_Mean': [],
        'Precision_Std': [],
        'Recall_Std': [],
        'F1_Std': [],
    }

    print("Deepseek New Prompt: Previous vs. JSON Evaluation")
    precision, recall, F1 = calculate_bert_score(deepseek_json['definition'].to_list(), deepseek_new['Definition'].to_list())
    pre, rec, f1 = compute_mean_std(precision, recall, F1)
    definition_dict['Model'].append('Deepseek New Prompt: Previous vs. JSON')
    definition_dict['Precision_Mean'].append(pre['mean'])
    definition_dict['Recall_Mean'].append(rec['mean'])
    definition_dict['F1_Mean'].append(f1['mean'])
    definition_dict['Precision_Std'].append(pre['std'])
    definition_dict['Recall_Std'].append(rec['std'])
    definition_dict['F1_Std'].append(f1['std'])

    print("\nDeepseek New Prompt: Previous vs. Reference")
    precision, recall, F1 = calculate_bert_score(reference_df['definition'].to_list(), deepseek_new['Definition'].to_list())
    pre, rec, f1 = compute_mean_std(precision, recall, F1)
    definition_dict['Model'].append('Deepseek New Prompt: Previous vs. Reference')
    definition_dict['Precision_Mean'].append(pre['mean'])
    definition_dict['Recall_Mean'].append(rec['mean'])
    definition_dict['F1_Mean'].append(f1['mean'])
    definition_dict['Precision_Std'].append(pre['std'])
    definition_dict['Recall_Std'].append(rec['std'])
    definition_dict['F1_Std'].append(f1['std'])

    print("\nDeepseek New Prompt: JSON vs. Reference")
    precision, recall, F1 = calculate_bert_score(reference_df['definition'].to_list(), deepseek_json['definition'].to_list())
    pre, rec, f1 = compute_mean_std(precision, recall, F1)
    definition_dict['Model'].append('Deepseek New Prompt: JSON vs. Reference')
    definition_dict['Precision_Mean'].append(pre['mean'])
    definition_dict['Recall_Mean'].append(rec['mean'])
    definition_dict['F1_Mean'].append(f1['mean'])
    definition_dict['Precision_Std'].append(pre['std'])
    definition_dict['Recall_Std'].append(rec['std'])
    definition_dict['F1_Std'].append(f1['std'])

    copilot_new = pd.read_excel(os.path.join('experiments', 'copilot', 'new-copilot.xlsx')).sort_values('Term')
    copilot_json = pd.read_json(os.path.join('experiments', 'copilot', 'web-json.json')).sort_values('Term')

    print("Copilot New Prompt: Previous vs. JSON Evaluation")
    precision, recall, F1 = calculate_bert_score(copilot_json['definition'].to_list(), copilot_new['Definition'].to_list())
    pre, rec, f1 = compute_mean_std(precision, recall, F1)
    definition_dict['Model'].append('Copilot New Prompt: Previous vs. JSON')
    definition_dict['Precision_Mean'].append(pre['mean'])
    definition_dict['Recall_Mean'].append(rec['mean'])
    definition_dict['F1_Mean'].append(f1['mean'])
    definition_dict['Precision_Std'].append(pre['std'])
    definition_dict['Recall_Std'].append(rec['std'])
    definition_dict['F1_Std'].append(f1['std'])

    print("\nCopilot New Prompt: Previous vs. Reference")
    precision, recall, F1 = calculate_bert_score(reference_df['definition'].to_list(), copilot_new['Definition'].to_list())
    pre, rec, f1 = compute_mean_std(precision, recall, F1)
    definition_dict['Model'].append('Copilot New Prompt: Previous vs. Reference')
    definition_dict['Precision_Mean'].append(pre['mean'])
    definition_dict['Recall_Mean'].append(rec['mean'])
    definition_dict['F1_Mean'].append(f1['mean'])
    definition_dict['Precision_Std'].append(pre['std'])
    definition_dict['Recall_Std'].append(rec['std'])
    definition_dict['F1_Std'].append(f1['std'])

    print("\nCopilot New Prompt: JSON vs. Reference")
    precision, recall, F1 = calculate_bert_score(reference_df['definition'].to_list(), copilot_json['definition'].to_list())
    pre, rec, f1 = compute_mean_std(precision, recall, F1)
    definition_dict['Model'].append('Copilot New Prompt: JSON vs. Reference')
    definition_dict['Precision_Mean'].append(pre['mean'])
    definition_dict['Recall_Mean'].append(rec['mean'])
    definition_dict['F1_Mean'].append(f1['mean'])
    definition_dict['Precision_Std'].append(pre['std'])
    definition_dict['Recall_Std'].append(rec['std'])
    definition_dict['F1_Std'].append(f1['std'])

    new_df = pd.DataFrame.from_dict(definition_dict)
    new_df.to_excel(os.path.join('analysis', 'quantitative', 'validation.xlsx'), index=False )


if __name__ == "__main__":
    main()
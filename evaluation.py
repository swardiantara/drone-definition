import os
import torch
import json 

import pandas as pd
import statistics
from bert_score import score
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

from utils.statistics import compute_mean_statistics
# Evaluation

def calculate_bert_score(generated_defs, reference_defs):
    # Calculate BERTScore
    P, R, F1 = score(generated_defs, reference_defs, model_type='microsoft/deberta-v3-large')

    # P: Precision scores
    # R: Recall scores
    # F1: F1 scores

    return P, R, F1

def compute_mean_std(generated_defs, reference_defs):
  precision, recall, f1 = calculate_bert_score(generated_defs, reference_defs)
  precision_mean = torch.mean(precision)
  precision_std = torch.std(precision)

  recall_mean = torch.mean(recall)
  recall_std = torch.std(recall)

  f1_mean = torch.mean(f1)
  f1_std = torch.std(f1)
  eval_score = {
    'precision_mean' :(round(float(precision_mean), 5)) * 100,
    'precision_std': (round(float(precision_std), 5)) * 100,
    'recall_mean': (round(float(recall_mean), 5)) * 100,
    'recall_std': (round(float(recall_std), 5)) * 100,
    'f1_mean': (round(float(f1_mean), 5)) * 100,
    'f1_std': (round(float(f1_std), 5)) * 100
  }

  return eval_score


def compute_distance(prompt, definition, model='instructor'):
    # Instantiate the Embedding Model (T5 Encoder or all-MiniLM-L6-v2)
    embedder = INSTRUCTOR('hkunlp/instructor-base') if model == 'instructor' else SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2')
    query = [['Represent the drone term definition: ', prompt]]
    document = [['Represent the drone term definition: ', definition]]

    # Compute the embedding vector
    query_embedding = embedder.encode(query).reshape(
        1, -1) if model == 'instructor' else embedder.encode(prompt).reshape(1, -1)
    document_embedding = embedder.encode(document).reshape(
        1, -1) if model == 'instructor' else embedder.encode(definition).reshape(1, -1)

    # Compute the distance metrics
    [[euclidean_distance]] = euclidean_distances(
        query_embedding, document_embedding)
    [[cosine_distance]] = cosine_distances(query_embedding, document_embedding)
    [[manhattan_distance]] = manhattan_distances(
        query_embedding, document_embedding)
    return euclidean_distance, cosine_distance, manhattan_distance


def compute_mean_sum_distance(generated_defs, reference_defs):
    sum_distance = []

    for i, (pred_def, true_def) in enumerate(zip(generated_defs, reference_defs)):
        d1, d2, d3 = compute_distance(pred_def, true_def)
        sum_distance.append(d1 + d2 + d3)
    mean_distance = statistics.mean(sum_distance)
    mean_distance = (round(float(mean_distance), 5))
    
    return sum_distance, mean_distance


def main():
    prompt = 'old'
    ref_df = pd.read_excel(os.path.join('dataset', 'definition.xlsx'), index_col=0).sort_values('term', ignore_index=True)
    models = os.listdir('experiments')
    mean_statistics = compute_mean_statistics(ref_df['definition'].to_list())
    recap = {}
    syntax_stats = {}
    syntax_stats['refs'] = mean_statistics
    for model in models:
        file_path = os.path.join('experiments', model, f'{prompt}-{model}.xlsx')
        print(f'current: {file_path}')
        if not os.path.exists(file_path):
            continue
        new_df = pd.read_excel(file_path, index_col=0).sort_values('term', ignore_index=True)
        if model == 'deepseek':
            with open(os.path.join('experiments', model, f'{prompt}-{model}.json'), 'r') as file:
                deepseekdata = json.load(file)
                new_df = pd.DataFrame(deepseekdata).sort_values('term', ignore_index=True)
        bert_score = compute_mean_std(new_df['pred_definition'].to_list(), ref_df['definition'].to_list())
        sum_distance, mean_distance = compute_mean_sum_distance(new_df['pred_definition'].to_list(), ref_df['definition'].to_list())
        mean_statistics = compute_mean_statistics(new_df['pred_definition'].to_list())
        bert_score['sum_distance'] = mean_distance
        recap[model] = bert_score
        syntax_stats[model] = mean_statistics
    
    # Convert and write JSON object to file
    with open(f"recap_{prompt}.json", "w") as outfile: 
        json.dump(recap, outfile)
    with open(f"syntax_stats_{prompt}.json", "w") as outfile: 
        json.dump(syntax_stats, outfile)

if __name__ == "__main__":
    main()

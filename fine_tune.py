import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import argparse
# from tqdm import tqdm, range
from simpletransformers.ner import NERModel, NERArgs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
parser = argparse.ArgumentParser()

# parser.add_argument('--model_name', type=str, default='bert-base-cased',
#                     choices=['bert-base-cased', 'roberta-base', 'distilbert-base-cased', 'distilroberta-base', 'google/electra-base-discriminator', 'xlnet-base-cased'])
parser.add_argument('--model_type', type=str, default='bert',
                    choices=['bert', 'roberta', 'distilbert', 'distilroberta', 'electra', 'xlnet'], help='The model type to fine-tune')


def load_dataset(path):
    train_df = pd.read_csv('{}/train.csv'.format(path))
    test_df = pd.read_csv('{}/test.csv'.format(path))
    return train_df, test_df


def get_model_name(model_type):
    if model_type == "bert":
        model_name = "bert-base-cased"
    elif model_type == "roberta":
        model_name = "roberta-base"
    elif model_type == "distilbert":
        model_name = "distilbert-base-cased"
    elif model_type == "distilroberta":
        model_type = "roberta"
        model_name = "distilroberta-base"
    elif model_type == "electra":
        model_name = "google/electra-base-discriminator"
    elif model_type == "xlnet":
        model_name = "xlnet-base-cased"

    return model_type, model_name


def get_model_args(model_type):
    args = NERArgs()
    model_type, model_name = get_model_name(model_type)
    model_name = model_name.split('/')[-1]
    args.num_train_epochs = 1
    args.learning_rate = 0.0001  # 0.0001  5e-5, 3e-5, 2e-5
    args.overwrite_output_dir = True
    args.train_batch_size = 16
    args.eval_batch_size = 16
    args.max_seq_length = 64
    args.output_dir = f"outputs/{model_name}"
    args.best_model_dir = f"outputs/{model_name}/best_model"

    return args


def check_test_data(test_df, preds_list):
    dataset_test_group = test_df.groupby(['sentence_id'], as_index=False)[
        'words', 'labels'].agg(lambda x: list(x))
    y_test = dataset_test_group['labels']
    compatible = True if len(preds_list) == len(y_test) else False

    return compatible


def build_pred_df(test_df, preds_list, strict=True):
    dataset_test_group = test_df.groupby(['sentence_id'], as_index=False)[
        'words', 'labels'].agg(lambda x: list(x))
    y_test = dataset_test_group['labels']
    x_test = dataset_test_group['words']

    x_test_list = []
    y_pred_list = []
    y_test_list = []
    for row in range(0, len(preds_list)):
        x_test_list = np.concatenate((x_test_list, x_test[row]), axis=0)
        y_pred_list = np.concatenate((y_pred_list, preds_list[row]), axis=0)
        y_test_list = np.concatenate((y_test_list, y_test[row]), axis=0)
    data_prediction = pd.DataFrame(
        {'words': x_test_list, 'actual_class': y_test_list, 'predicted_class': y_pred_list})
    confusion_matrix = pd.crosstab(
        data_prediction['predicted_class'], data_prediction['actual_class'])
    if strict == True:
        data_prediction['strict'] = data_prediction.apply(
            lambda row: True if row.actual_class == row.predicted_class else False, axis=1)
    else:
        data_prediction['predicted_non'] = data_prediction.apply(
            lambda row: row.predicted_class.split('-')[-1], axis=1)
        data_prediction['actual_non'] = data_prediction.apply(
            lambda row: row.actual_class.split('-')[-1], axis=1)
        data_prediction['non_strict'] = data_prediction.apply(lambda row: True if row.actual_class.split(
            '-')[-1] == row.predicted_class.split('-')[-1] else False, axis=1)
        confusion_matrix = pd.crosstab(
            data_prediction['predicted_non'], data_prediction['actual_non'])
    return data_prediction, confusion_matrix


def main():
    args = parser.parse_args()
    device = True if torch.cuda.is_available() else False
    model_type, model_name = get_model_name(args.model_type)
    train_df, test_df = load_dataset('dataset')

    train_stat = pd.Series(train_df["labels"].value_counts()).to_frame()
    test_stat = pd.Series(test_df["labels"].value_counts()).to_frame()

    # Save the train and test statistics into files
    train_stat.to_csv('dataset/train_stat.csv')
    test_stat.to_csv('dataset/test_stat.csv')

    # Create a ClassificationModel
    label = train_df["labels"].unique().tolist()
    model_args = get_model_args(args.model_type)
    model = NERModel(model_type, model_name, labels=label,
                     args=model_args, use_cuda=device)
    output_dir = getattr(model_args, "output_dir")

    # Fine-tune the model using our own dataset
    model.train_model(train_df, eval_data=test_df, acc=accuracy_score)

    # Evaluate the model
    result, model_outputs, preds_list = model.eval_model(test_df)
    compatible = check_test_data(test_df, preds_list)
    if not compatible:
        print('The length of test data and preds data is not compatible')

    preds_df, confusion_matrix = build_pred_df(test_df, preds_list, True)
    preds_df.to_csv("{}/prediction_{}.csv".format(output_dir, model_name))
    confusion_matrix.to_csv(
        "{}/confusion_matrix_{}.csv".format(output_dir, model_name))


if __name__ == "__main__":
    main()

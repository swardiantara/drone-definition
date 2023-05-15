import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from evaluation import build_pred_df, get_confusion_matrix, get_evaluation_score
from simpletransformers.ner import NERModel, NERArgs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    args.num_train_epochs = 3
    args.learning_rate = 0.0001  # 0.0001  5e-5, 3e-5, 2e-5
    args.overwrite_output_dir = True
    args.train_batch_size = 16
    args.eval_batch_size = 16
    args.max_seq_length = 64
    args.output_dir = f"outputs/{model_name}"
    args.best_model_dir = f"outputs/{model_name}/best_model"

    return args


def check_test_data(test_df, preds_list):
    dataset_test_group = test_df.groupby('sentence_id', group_keys=False, as_index=False)[
        'words', 'labels'].agg(lambda x: list(x))
    y_test = dataset_test_group['labels']
    compatible = True if len(preds_list) == len(y_test) else False

    return compatible


def main():
    model_types = ["bert", 'distilbert', 'roberta',
                   'distilroberta', 'electra', 'xlnet']

    for model_type in model_types:
        device = True if torch.cuda.is_available() else False
        model_type, model_name = get_model_name(model_type)
        train_df, test_df = load_dataset('dataset')

        train_stat = pd.Series(train_df["labels"].value_counts()).to_frame()
        test_stat = pd.Series(test_df["labels"].value_counts()).to_frame()

        # Save the train and test statistics into files
        train_stat.to_csv('dataset/train_stat.csv')
        test_stat.to_csv('dataset/test_stat.csv')

        # Create a ClassificationModel
        label = train_df["labels"].unique().tolist()
        model_args = get_model_args(model_type)
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

        # Save the evaluation score to .csv files for error analysis
        preds_df, cm = build_pred_df(test_df, preds_list, True)
        preds_df.to_csv("{}/prediction_{}.csv".format(output_dir, model_name))
        confusion_matrix = get_confusion_matrix(cm)
        confusion_matrix.to_csv(
            "{}/confusion_matrix_{}.csv".format(output_dir, model_name))
        eval_dict, eval_df = get_evaluation_score(confusion_matrix, 'micro')
        eval_df.to_csv(
            "{}/confusion_matrix_{}.csv".format(output_dir, model_name))


if __name__ == "__main__":
    main()

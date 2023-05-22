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
    train_df = pd.read_csv('{}/airdata_train_70.csv'.format(path))
    dev_df = pd.read_csv('{}/dev.csv'.format(path))
    test_df = pd.read_csv('{}/airdata_test_70.csv'.format(path))
    return train_df, dev_df, test_df


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
    args.num_train_epochs = 4
    args.learning_rate = 0.0001  # 0.0001  5e-5, 3e-5, 2e-5
    args.overwrite_output_dir = True
    args.train_batch_size = 8
    args.eval_batch_size = 8
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


def classification(model_type, model_name, train_df, dev_df, test_df):
    device = True if torch.cuda.is_available() else False
     # Create a ClassificationModel
    label = train_df["labels"].unique().tolist()
    model_args = get_model_args(model_type)
    model = NERModel(model_type, model_name, labels=label,
                        args=model_args, use_cuda=device)
    output_dir = getattr(model_args, "output_dir")

    # Fine-tune the model using our own dataset
    model.train_model(train_df, eval_data=test_df, acc=accuracy_score)

    # Evaluate the model on Dev Data
    result, model_outputs, preds_list = model.eval_model(dev_df)
    print("\n=> Evaluating the model on Dev Dataset...")
    print(result)
    result, model_outputs, preds_list = model.eval_model(test_df)
    print("\n=> Evaluating the model on Test Dataset...")
    print(result)
    print("\n=> Saving the evaluation results...")
    # compatible = check_test_data(test_df, preds_list)
    # if not compatible:
    #     print('The length of test data and preds data is not compatible')

    # Save the evaluation score to .csv files for error analysis
    model_name = model_name.split('/')[-1]
    preds_df, cm = build_pred_df(test_df, preds_list, False)
    cm.to_csv(
        "{}/cm_{}.csv".format(output_dir, model_name))
    preds_df.to_csv("{}/prediction_{}.csv".format(output_dir, model_name), index=False)
    load_cm = pd.read_csv("{}/cm_{}.csv".format(output_dir, model_name))
    confusion_matrix = get_confusion_matrix(load_cm)
    confusion_matrix.to_csv(
        "{}/confusion_matrix_{}.csv".format(output_dir, model_name), index=False)
    eval_dict, eval_df = get_evaluation_score(confusion_matrix)
    eval_df.to_csv(
        "{}/evaluation_score{}.csv".format(output_dir, model_name))



def main():
    model_types = ['bert', 'distilbert', 'roberta', 'distilroberta', 'electra', 'xlnet']
    
    for model_type in model_types:
        
        model_type, model_name = get_model_name(model_type)
        train_df, dev_df, test_df = load_dataset('dataset')
        # test_df = test_df.iloc[:100]

        train_stat = pd.Series(train_df["labels"].value_counts()).to_frame()
        dev_stat = pd.Series(dev_df["labels"].value_counts()).to_frame()
        test_stat = pd.Series(test_df["labels"].value_counts()).to_frame()

        # Save the train and test statistics into files
        train_stat.to_csv('dataset/airdata_train_70_stat.csv')
        dev_stat.to_csv('dataset/dev_stat.csv')
        test_stat.to_csv('dataset/airdata_test_70_stat.csv')
        
        classification(model_type, model_name, train_df, dev_df, test_df)
       


if __name__ == "__main__":
    main()

"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import did_dataset
import data_utils
import finetuning_utils
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AutoTokenizer, AdamW
from transformers import TrainingArguments, Trainer
from classifier import Classifier
"""
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch
"""

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a BERT models for Arabic DID."
)
parser.add_argument(
    "level",
    type=str,
    help="Level of dialect, i.e. city, country, region",
)

args = parser.parse_args()
level = args.level
train_batch_size = 32
val_batch_size = 32
test_batch_size = 1
max_seq_length = 64
dropout_prob = 0.1
hidden_size1 = 768
hidden_size2 = 512
num_classes = 26
epochs = 10
learning_rate = 1e-5
early_stop_patience = 2
metric = 'f1'

print('Getting data')
folder = '../data_processed_second/madar_shared_task1/'
train_df = data_utils.get_df_from_files(
    [f'{folder}MADAR-Corpus-26-train.lines', f'{folder}MADAR-Corpus-6-train.lines'])
test_df = data_utils.get_df_from_files(
    [f'{folder}MADAR-Corpus-26-test.lines'])
val_df = data_utils.get_df_from_files(
    [f'{folder}MADAR-Corpus-26-dev.lines', f'{folder}MADAR-Corpus-6-dev.lines'])

print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained(
    'CAMeL-Lab/bert-base-camelbert-mix')
print('Getting Dataset object')
train_data = did_dataset.DIDDataset(
    train_df, tokenizer, level, f'labels/madar_labels_{level}.txt', max_seq_length)
val_data = did_dataset.DIDDataset(
    val_df, tokenizer, level, f'labels/madar_labels_{level}.txt', max_seq_length)
test_data = did_dataset.DIDDataset(
    test_df, tokenizer, level, f'labels/madar_labels_{level}.txt', max_seq_length)

print('Getting DataLoader objects')
# dataLoader for train set
train_dataloader = DataLoader(
    train_data, shuffle=True, batch_size=train_batch_size, num_workers=0)
val_dataloader = DataLoader(
    val_data, shuffle=True, batch_size=val_batch_size, num_workers=0)
test_dataloader = DataLoader(
    test_data, batch_size=test_batch_size, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} device for training')
bert = finetuning_utils.model_init()
hidden_size1 = bert.config.hidden_size

model = Classifier(bert, dropout_prob, hidden_size1, hidden_size2, num_classes)
model.to(device)

# optimizer from hugging face transformers

# define the optimizer
optimizer = AdamW(model.parameters(),
                  lr=learning_rate)
cross_entropy = nn.CrossEntropyLoss()


best_valid_metric = 0
best_valid_epoch = 0
if metric == 'loss':
    best_valid_metric = float('inf')

# empty lists to store training and validation loss of each epoch
train_metrics_list = []
valid_metrics_list = []
train_predictions_list = []
valid_predictions_list = []
n_no_improvement = 0

print('Starting training')

# for each epoch
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    # train model
    train_predictions, train_metrics = finetuning_utils.train(model, train_dataloader,
                                                              cross_entropy, optimizer, device)

    # evaluate model
    valid_predictions, valid_metrics = finetuning_utils.evaluate(model, val_dataloader,
                                                                 cross_entropy, device)
    # get best metric
    if metric == 'loss' and valid_metrics[metric] < best_valid_metric:
        best_valid_epoch = epoch
        best_valid_metric = valid_metrics[metric]
        torch.save(model.state_dict(),
                   f'saved_weights_early_stop_{best_valid_epoch}_{metric}.pt')
        n_no_improvement = 0

    elif metric != 'loss' and valid_metrics[metric] > best_valid_metric:
        best_valid_epoch = epoch
        best_valid_metric = valid_metrics[metric]
        torch.save(model.state_dict(),
                   f'saved_weights_early_stop_{best_valid_epoch}_{metric}.pt')
        n_no_improvement = 0
    else:
        n_no_improvement += 1

    train_metrics_list.append(train_metrics)
    valid_metrics_list.append(valid_metrics)
    train_predictions_list.append(train_predictions)
    valid_predictions_list.append(valid_predictions)

    print(
        f'Training Metrics: {train_metrics}')
    print(
        f'Validation Metrics: {valid_metrics}')
    if n_no_improvement > early_stop_patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

# print best metrics
print(
    f'Best validation epoch is {best_valid_epoch + 1} with metrics: {valid_metrics_list[epoch]}\nModel can be found here saved_weights_early_stop_{best_valid_epoch}_{metric}.pt')

# test at the end after training
bert_test = finetuning_utils.model_init()
model_test = Classifier(bert_test, dropout_prob, hidden_size1,
                        hidden_size2, num_classes)
model_test.to(device)
model_test.load_state_dict(torch.load(
    f'saved_weights_early_stop_{best_valid_epoch}_{metric}.pt'))
test_predictions, test_metrics = finetuning_utils.test(model_test, test_dataloader,
                                                       cross_entropy, device)

print(
    f'Test Metrics: {test_metrics}')

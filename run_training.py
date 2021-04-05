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
train_batch_size = 8
val_batch_size = 32
test_batch_size = 32
max_seq_length = 64
dropout_prob = 0.1
hidden_size1 = 768
hidden_size2 = 512
num_classes = 26
epochs = 3
learning_rate = 1e-5

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
print('In data utils Dataset object')
train_data = did_dataset.DIDDataset(
    train_df, tokenizer, level, 'labels/madar_labels_city.txt', max_seq_length)
val_data = did_dataset.DIDDataset(
    val_df, tokenizer, level, 'labels/madar_labels_city.txt', max_seq_length)
test_data = did_dataset.DIDDataset(
    test_df, tokenizer, level, 'labels/madar_labels_city.txt', max_seq_length)


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
#TODO: hidden_size1  = bert.config.hidden_size

model = Classifier(bert, dropout_prob, hidden_size1, hidden_size2, num_classes)
model.to(device)

# optimizer from hugging face transformers

# define the optimizer
optimizer = AdamW(model.parameters(),
                  lr=learning_rate)
cross_entropy = nn.CrossEntropyLoss()


best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses = []
valid_losses = []

# for each epoch
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    # train model
    train_loss, _, train_metrics = finetuning_utils.train(model, train_dataloader,
                                                          cross_entropy, optimizer, device)

    # evaluate model
    valid_loss, _, valid_metrics = finetuning_utils.evaluate(model, val_dataloader,
                                                             cross_entropy, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(
        f'Training Loss: {train_loss:.3f}, Training Metrics: {train_metrics}')
    print(
        f'Validation Loss: {valid_loss:.3f}, Validation Metrics: {valid_metrics}')

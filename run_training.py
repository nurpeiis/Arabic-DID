"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import did_dataset
import data_utils
import finetuning_utils
import json
import ray
import pandas as pd


from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)

args = parser.parse_args()


folder = '../hierarchical-did/data_processed_second/madar_shared_task1/'
train_df = data_utils.get_df_from_files(
    [f'{folder}MADAR-Corpus-26-train.lines', f'{folder}MADAR-Corpus-6-train.lines'])
test_df = data_utils.get_df_from_files(
    [f'{folder}MADAR-Corpus-26-test.lines'])
val_df = data_utils.get_df_from_files(
    [f'{folder}MADAR-Corpus-26-dev.lines', f'{folder}MADAR-Corpus-6-dev.lines'])


tokenizer = AutoTokenizer.from_pretrained(
    'CAMeL-Lab/bert-base-camelbert-mix')
train_data = did_dataset.DIDDataset(train_df, tokenizer)
val_data = did_dataset.DIDDataset(val_df, tokenizer)
test_data = did_dataset.DIDDataset(test_df, tokenizer)


training_args = TrainingArguments(
    output_dir="./models/",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    num_train_epochs=3,  # due to time/computation constraints
    evaluation_strategy="epoch",
)
trainer = Trainer(
    model_init=finetuning_utils.model_init,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=finetuning_utils.compute_metrics
)
trainer.train()
"""
best_trial = trainer.hyperparameter_search(
    hp_space=lambda _: {
        "learning_rate": tune.uniform(1e-5, 5e-5),
    },
    direction="maximize",
    backend="ray",
    n_trials=5,  # number of hyperparameter samples
    compute_objective=lambda x: x['eval_loss']
)

print(f'Run ID: {best_trial[0]}')
print(f'Objective: {best_trial[1]}')
print(f'Hyperparameters: {best_trial[2]}')
"""

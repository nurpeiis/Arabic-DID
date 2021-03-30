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


train_df = pd.read_json(f"{args.data_dir}/train.jsonl",
                        lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)

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

#
# Choose among schedulers:
# https://docs.ray.io/en/latest/tu
# TODO: Initialize a transformers.Trainer object and run a Bayesian
# hyperparameter search for at least 5 trials (but not too many) on the
# learning rate. Hint: use the model_init() and
# compute_metrics() methods from finetuning_utils.py as arguments to
# trainer.hyperparameter_search(). Use the hp_space parameter to specify
# your hyperparameter search space. (Note that this parameter takes a function
# as its value.)
# Also print out the run ID, objective value,
# and hyperparameters of your best run.

from classifier import Classifier
import os
import json
import torch
import random
import logging
import data_utils
import did_dataset
import finetuning_utils
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import RandomSampler, DataLoader

from transformers import (AutoTokenizer, AdamW, AutoModelForSequenceClassification, AutoConfig,
                          get_linear_schedule_with_warmup)


def run_train(params):
    """
    Run training on the model and get best validation model
    Args:
      params: all parameters to run the experiment
    Returns:
      results: dictionary that contains all data
    """
    # Initialize variables
    print('Initializing variables')
    tokenizer = AutoTokenizer.from_pretrained(params['tokenizer'])
    config = AutoConfig.from_pretrained(
        params['bert'],
        num_labels=params['num_classes'],
        label2id=params['label2id'],
        id2label=params['id2label'],
        finetuning_task='arabic_did'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        params['tokenizer']
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        params['bert'],
        config=config
    )
    seed = params['seed']
    level = params['level']
    train_batch_size = params['train_batch_size']
    val_batch_size = params['val_batch_size']
    max_seq_length = params['max_seq_length']
    epochs = params['epochs']
    learning_rate = params['learning_rate']
    adam_epsilon = params['adam_epsilon']
    metric = params['metric']
    train_files = params['train_files']
    val_files = params['val_files']
    label_space_file = params['label_space_file']
    model_folder = params['model_folder']
    max_grad_norm = params['max_grad_norm']
    save_steps = params['save_steps']
    weight_decay = params['weight_decay']
    warmup_steps = params['warmup_steps']
    gradient_accumulation_steps = params['gradient_accumulation_steps']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get data
    print('Getting Data')
    train_df = data_utils.get_df_from_files(train_files)
    val_df = data_utils.get_df_from_files(val_files)

    train_data = did_dataset.DIDDataset(
        train_df, tokenizer, level, label_space_file, max_seq_length)
    val_data = did_dataset.DIDDataset(
        val_df, tokenizer, level, label_space_file, max_seq_length)

    train_sampler = RandomSampler(train_data)
    val_sampler = RandomSampler(val_data)

    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size)
    val_dataloader = DataLoader(
        val_data, sampler=val_sampler, batch_size=val_batch_size)

    t_total = (len(train_dataloader) // gradient_accumulation_steps*epochs)
    # define the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate,
                      eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=t_total)

    global_step = 0
    epochs_trained = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained,
                            epochs, desc='Epoch')
    finetuning_utils.set_seed(seed)  # Added here for reproductibility
    list_avg_train_metrics = []
    checkpoints = []
    print('Entering Training')
    for _ in train_iterator:
        list_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
        total_metrics = {}
        for m in list_metrics:
            total_metrics[m] = 0
        epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        total_preds = None
        total_true_labels = None

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['labels'] = batch['labels'].to(device)
            labels = batch['labels']
            outputs = model(**batch)
            # model outputs are always tuple in transformers (see doc)
            loss, logits = outputs[:2]

            loss.backward()

            tr_loss += loss.item()
            total_metrics['loss'] += loss.item()
            preds_a = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            if total_preds is None:
                total_preds = preds_a
                total_true_labels = label_ids
            else:
                total_preds = np.append(total_preds, preds_a, axis=0)
                total_true_labels = np.append(
                    total_true_labels,
                    label_ids,
                    axis=0)

            #print(f'Tmp metrics {tmp_metrics}%')

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if (save_steps > 0 and
                    global_step % save_steps == 0):
                # Save model checkpoint
                model_file = f'{model_folder}/{global_step}.pt'
                checkpoints.append(model_file)
                model.save_pretrained(model_file)
                print('Saving model checkpoint to %s', model_file)

        # save model at the end of each epoch
        model_file = f'{model_folder}/{global_step}.pt'
        checkpoints.append(model_file)
        model.save_pretrained(model_file)
        print('Saving model checkpoint to %s', model_file)

        # Calculate the accuracy for this batch of test sentences.
        finetuning_utils.metrics(
            total_preds, total_true_labels,  total_metrics)
        list_avg_train_metrics.append(total_metrics)

    # evaluate through all checkpoints
    best_metrics = {}
    best_checkpoint = ''
    best_predictions = None
    best_predictions_argmax = None
    list_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    for m in list_metrics:
        best_metrics[m] = 0
    best_metrics['loss'] = float('inf')
    for checkpoint in checkpoints:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint)
        valid_predictions, valid_predictions_argmax, true_labels, valid_metrics = finetuning_utils.evaluate(
            model, val_dataloader, device)

        if metric == 'loss' and valid_metrics[metric] < best_metrics[metric]:
            best_metrics = valid_metrics
            best_predictions = valid_predictions
            best_predictions_argmax = valid_predictions_argmax
            model_file = f'{model_folder}/best_model.pt'
            model.save_pretrained(model_file)
            best_checkpoint = checkpoint
        elif metric != 'loss' and valid_metrics[metric] > best_metrics[metric]:
            best_metrics = valid_metrics
            best_predictions = valid_predictions
            best_predictions_argmax = valid_predictions_argmax
            model_file = f'{model_folder}/best_model.pt'
            model.save_pretrained(model_file)
            best_checkpoint = checkpoint

    train_results = {}
    train_results['global_step'] = global_step
    train_results['train_loss'] = tr_loss / global_step
    train_results['model_file'] = model_file
    train_results['list_avg_train_metrics'] = list_avg_train_metrics
    train_results['best_metrics'] = best_metrics
    train_results['best_predictions'] = best_predictions
    train_results['best_predictions_argmax'] = best_predictions_argmax
    train_results['best_checkpoint'] = best_checkpoint

    return train_results


def run_test(params):
    """
    Run test on the model
    Args:
      params: all parameters to run the experiment
    Returns:
      results: dictionary that contains all data
      test_predictions: list with test_predictions
    """
    tokenizer = AutoTokenizer.from_pretrained(params['tokenizer'])
    level = params['level']
    test_batch_size = params['test_batch_size']
    max_seq_length = params['max_seq_length']
    test_files = params['test_files']
    label_space_file = params['label_space_file']
    model_file = params['model_file']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_file)
    # Get Data
    print('Getting Test Data')
    test_df = data_utils.get_df_from_files(test_files)

    test_data = did_dataset.DIDDataset(
        test_df, tokenizer, level, label_space_file, max_seq_length)

    test_sampler = RandomSampler(test_data)

    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=test_batch_size)

    print('Starting Test')

    test_predictions, test_predictions_argmax, true_labels, test_metrics = finetuning_utils.evaluate(
        model, test_dataloader, device)
    print(f'Test Metrics: {test_metrics}')
    results = {}
    for k in test_metrics.keys():
        results[f'test_{k}'] = test_metrics[k]

    return results, test_predictions, test_predictions_argmax, true_labels

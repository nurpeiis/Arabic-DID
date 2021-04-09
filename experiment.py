import torch
import logging
import data_utils
import did_dataset
import finetuning_utils
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer, AdamW
from classifier import Classifier


def run_train(params):
    """
    Run training on the model and get best validation model
    Args:
      params: all parameters to run the experiment
    Returns:
      results: dictionary that contains all data
    """
    # Initialize variables
    logging.info('Initializing variables')
    bert = AutoModel.from_pretrained(params['bert'])
    tokenizer = AutoTokenizer.from_pretrained(params['tokenizer'])
    level = params['level']
    train_batch_size = params['train_batch_size']
    val_batch_size = params['val_batch_size']
    max_seq_length = params['max_seq_length']
    dropout_prob = params['dropout_prob']
    hidden_size1 = bert.config.hidden_size
    hidden_size2 = params['hidden_size2']
    num_classes = params['num_classes']
    epochs = params['epochs']
    learning_rate = params['learning_rate']
    early_stop_patience = params['early_stop_patience']
    metric = params['metric']
    train_files = params['train_files']
    val_files = params['val_files']
    label_space_file = params['label_space_file']
    model_folder = params['model_folder']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Classifier(bert, dropout_prob, hidden_size1,
                       hidden_size2, num_classes)
    model.to(device)

    # Get data
    logging.info('Getting Data')
    train_df = data_utils.get_df_from_files(train_files)
    val_df = data_utils.get_df_from_files(val_files)

    train_data = did_dataset.DIDDataset(
        train_df, tokenizer, level, label_space_file, max_seq_length)
    val_data = did_dataset.DIDDataset(
        val_df, tokenizer, level, label_space_file, max_seq_length)

    train_dataloader = DataLoader(
        train_data, shuffle=True, batch_size=train_batch_size, num_workers=0)
    val_dataloader = DataLoader(
        val_data, shuffle=True, batch_size=val_batch_size, num_workers=0)

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
    early_stop = False
    model_file = ''
    logging.info('Starting training')

    # for each epoch
    for epoch in range(epochs):
        logging.info(f'Epoch {epoch+1}/{epochs}')
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
            model_file = f'{model_folder}/{epoch}.pt'
            torch.save(model.state_dict(), model_file)
            n_no_improvement = 0

        elif metric != 'loss' and valid_metrics[metric] > best_valid_metric:
            best_valid_epoch = epoch
            best_valid_metric = valid_metrics[metric]
            model_file = f'{model_folder}/{epoch}.pt'
            torch.save(model.state_dict(), model_file)
            n_no_improvement = 0
        else:
            n_no_improvement += 1

        train_metrics_list.append(train_metrics)
        valid_metrics_list.append(valid_metrics)
        train_predictions_list.append(train_predictions)
        valid_predictions_list.append(valid_predictions)

        logging.info(
            f'Training Metrics: {train_metrics}')
        logging.info(
            f'Validation Metrics: {valid_metrics}')
        if n_no_improvement > early_stop_patience:
            logging.info(f'Early stopping at epoch {epoch + 1}')
            early_stop = True
            break

    # logging.info best metrics
    logging.info(
        f'Best validation epoch is {best_valid_epoch + 1} with metrics: {valid_metrics_list[epoch]}\nModel can be found here saved_weights_early_stop_{best_valid_epoch}_{metric}.pt')

    results = {}
    results['best_valid_epoch'] = best_valid_epoch
    results['train_metrics_list'] = train_metrics_list
    results['valid_metrics_list'] = valid_metrics_list
    results['best_valid_epoch'] = best_valid_epoch
    results['best_valid_metric'] = valid_metrics_list[best_valid_epoch]
    results['early_stop'] = early_stop
    results['model_file'] = model_file
    return results


def run_test(params):
    """
    Run test on the model
    Args:
      params: all parameters to run the experiment
    Returns:
      results: dictionary that contains all data
      test_predictions: list with test_predictions
    """
    bert = AutoModel.from_pretrained(params['bert'])
    tokenizer = AutoTokenizer.from_pretrained(params['tokenizer'])
    level = params['level']
    test_batch_size = params['test_batch_size']
    max_seq_length = params['max_seq_length']
    dropout_prob = params['dropout_prob']
    hidden_size1 = bert.config.hidden_size
    hidden_size2 = params['hidden_size2']
    num_classes = params['num_classes']
    test_files = params['test_files']
    label_space_file = params['label_space_file']
    model_file = params['model_file']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Classifier(bert, dropout_prob, hidden_size1,
                       hidden_size2, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_file))

    # Get Data
    logging.info('Getting Test Data')
    test_df = data_utils.get_df_from_files(test_files)

    test_data = did_dataset.DIDDataset(
        test_df, tokenizer, level, label_space_file, max_seq_length)

    test_dataloader = DataLoader(
        test_data, batch_size=test_batch_size, num_workers=0)

    # Get cross entropy
    cross_entropy = nn.CrossEntropyLoss()
    logging.info('Starting Test')

    test_predictions, test_metrics = finetuning_utils.test(model, test_dataloader,
                                                           cross_entropy, device)
    logging.info(f'Test Metrics: {test_metrics}')
    results = {}
    for k in test_metrics.keys():
        results[f'test_{k}'] = test_metrics[k]

    return results, test_predictions

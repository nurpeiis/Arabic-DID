import os
import data_utils
import filter_logit
import finetuning_utils
import numpy as np
from datetime import datetime
from experiment import run_train, run_test
from google_sheet_utils import get_sheet, get_all_records, rewrite_page, create_page, find_idx_page


def record_experiment(list_dicts, sheet_title, page_title):
    print('Getting access to Google Sheet')
    sheet = get_sheet(sheet_title)
    index_page = find_idx_page(sheet, page_title)
    if index_page == -1:
        create_page(sheet, page_title, '100', '30')
        index_page = find_idx_page(sheet, page_title)

    df = get_all_records(sheet, index_page)
    final_row = {}
    for single_dict in list_dicts:
        for key in single_dict.keys():
            final_row[key] = str(single_dict[key])
    df = df.append(final_row, ignore_index=True)
    rewrite_page(sheet, index_page, df)


def record_predictions(distribution, argmax, file):
    print(f'Writing predictions into {file}')
    np.savez(f'{file}', distribution=distribution, argmax=argmax)


def run_madar_experiment(level='city'):

    list_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    params = {}

    print('Initializing training variables')

    params['bert'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['tokenizer'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['level'] = level
    params['train_batch_size'] = 32
    params['val_batch_size'] = 32
    params['max_seq_length'] = 128
    params['num_classes'] = 26
    params['epochs'] = 10
    params['learning_rate'] = 3e-5
    params['metric'] = 'f1'
    params['adam_epsilon'] = 1e-08
    params['seed'] = 12345
    params['save_steps'] = 500
    params['weight_decay'] = 0.0
    params['warmup_steps'] = 0
    params['gradient_accumulation_steps'] = 1
    params['max_grad_norm'] = 1.0

    madar_folder = '../data_processed_second/madar_shared_task1/'
    madar_extra = '../data_processed_second/madar_corpus6_extra/'
    params['train_files'] = [
        f'{madar_folder}MADAR-Corpus-26-train.lines', f'{madar_extra}train_BTEC-MSA-8K.lines',
        f'{madar_extra}train_BTEC-Beirut-8K.lines', f'{madar_extra}train_BTEC-Cairo-8K.lines', f'{madar_extra}train_BTEC-Tunis-8K.lines', f'{madar_extra}train_BTEC-Doha-8K.lines']
    params['val_files'] = [
        f'{madar_folder}MADAR-Corpus-26-dev.lines']

    params['label_space_file'] = f'labels/madar_{params["level"]}_label_id.txt'
    labels, label2id, id2label = data_utils.get_label_space(
        params['label_space_file'])
    params['labels'] = labels
    params['label2id'] = label2id
    params['id2label'] = id2label

    print('Entering Training')
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime('%d-%m-%Y-%H:%M:%S')
    params['an experiment name'] = f'{params["metric"]}; train-26, madar extra; dev-26; test-26'
    params['time'] = dt_string
    params['model_folder'] = f'{dt_string}'
    os.mkdir(params['model_folder'])
    train_results = run_train(params)
    print('Entering Testing')
    params['model_file'] = train_results['model_file']
    params['test_files'] = [f'{madar_folder}MADAR-Corpus-26-test.lines']
    params['test_batch_size'] = 32
    test_results, test_predictions, test_predictions_argmax, true_labels = run_test(
        params)

    record_predictions(test_predictions, test_predictions_argmax,
                       f'{params["model_folder"]}/madar_predictions.npz')
    record_experiment([params, test_results, train_results],
                      'experiments', 'madar extra')


def run_madar_pretrained_experiment():

    list_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    params = {}

    print('Initializing training variables')

    params['bert'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['pretrained'] = 'city_21-04-2021-17:17:54/best_model.pt'
    params['pretrained'] = 'country_01-05-2021-10:41:52/best_model.pt'
    params['pretrained'] = 'region_02-05-2021-02:37:27/best_model.pt'
    params['tokenizer'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['level'] = 'city'
    params['train_batch_size'] = 32
    params['val_batch_size'] = 32
    params['max_seq_length'] = 128
    params['num_classes'] = 26
    params['epochs'] = 10
    params['learning_rate'] = 3e-5
    params['metric'] = 'f1'
    params['adam_epsilon'] = 1e-08
    params['seed'] = 12345
    params['save_steps'] = 500
    params['weight_decay'] = 0.0
    params['warmup_steps'] = 0
    params['gradient_accumulation_steps'] = 1
    params['max_grad_norm'] = 1.0

    madar_folder = '../data_processed_second/madar_shared_task1/'
    params['train_files'] = [
        f'{madar_folder}MADAR-Corpus-26-train.lines']
    params['val_files'] = [
        f'{madar_folder}MADAR-Corpus-26-dev.lines']

    params['label_space_file'] = f'labels/madar_{params["level"]}_label_id.txt'
    labels, label2id, id2label = data_utils.get_label_space(
        params['label_space_file'])
    params['labels'] = labels
    params['label2id'] = label2id
    params['id2label'] = id2label

    print('Entering Training')
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime('%d-%m-%Y-%H:%M:%S')
    params['an experiment name'] = f'{params["metric"]}; train-26; dev-26; test-26'
    params['time'] = dt_string
    params['model_folder'] = f'{dt_string}'
    os.mkdir(params['model_folder'])
    train_results = run_train(params)
    print('Entering Testing')
    params['model_file'] = train_results['model_file']
    params['test_files'] = [f'{madar_folder}MADAR-Corpus-26-test.lines']
    params['test_batch_size'] = 32
    test_results, test_predictions, test_predictions_argmax, true_labels = run_test(
        params)

    record_predictions(test_predictions, test_predictions_argmax,
                       f'{params["model_folder"]}/madar_predictions.npz')
    record_experiment([params, test_results, train_results],
                      'experiments', 'madar_pretrained_aggregated')


def run_level_experiment(level):
    list_metrics = ['accuracy', 'loss', 'f1']
    params = {}

    print('Initializing training variables')

    params['bert'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['tokenizer'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['level'] = level
    params['dropout_prob'] = 0.1
    if level == 'region':
        params['num_classes'] = 6
        params['epochs'] = 1
        params['save_steps'] = 200000
    elif level == 'country':
        params['num_classes'] = 22
        params['epochs'] = 2
        params['save_steps'] = 200000
    elif level == 'city':
        params['num_classes'] = 113
        params['epochs'] = 10
        params['save_steps'] = 2000
    else:
        params['num_classes'] = 26
        params['epochs'] = 10
        params['save_steps'] = 1000
    params['train_batch_size'] = 32
    params['val_batch_size'] = 32
    params['max_seq_length'] = 128
    params['num_classes'] = 26
    params['learning_rate'] = 3e-5
    params['metric'] = 'f1'
    params['adam_epsilon'] = 1e-08
    params['seed'] = 12345
    params['weight_decay'] = 0.0
    params['warmup_steps'] = 0
    params['gradient_accumulation_steps'] = 1
    params['max_grad_norm'] = 1.0
    aggregated_folder = 'aggregated_data'
    params['train_files'] = [
        f'{aggregated_folder}/{level}_train.tsv']
    params['val_files'] = [
        f'{aggregated_folder}/{level}_dev.tsv']
    params['label_space_file'] = f'labels/{level}_label_id.txt'
    labels, label2id, id2label = data_utils.get_label_space(
        params['label_space_file'])
    params['labels'] = labels
    params['label2id'] = label2id
    params['id2label'] = id2label

    print('Entering Training')
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime('%d-%m-%Y-%H:%M:%S')
    params['an experiment name'] = f'{params["metric"]}; aggregated_{level}'
    params['time'] = dt_string
    params['model_folder'] = f'{level}_{dt_string}'
    os.mkdir(params['model_folder'])
    train_results = run_train(params)
    """
    print('Entering Testing')
    params['model_file'] = train_results['model_file']
    params['test_files'] = [f'{aggregated_folder}/{level}_test.tsv']
    params['test_batch_size'] = 32
    test_results, test_predictions, test_predictions_argmax, true_labels = run_test(params)

    record_predictions(test_predictions, test_predictions_argmax,
                       f'{params["model_folder"]}/{level}_predictions.npz')
    """
    record_experiment([params, train_results],
                      'experiments', f'{level}_aggregated')


def run_city_aggregate_experiment(level):
    list_metrics = ['accuracy', 'loss', 'f1']
    params = {}

    print('Initializing training variables')

    params['bert'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['tokenizer'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['level'] = level
    params['dropout_prob'] = 0.1
    if level == 'region':
        params['num_classes'] = 6
        params['epochs'] = 10
        params['save_steps'] = 2000
    elif level == 'country':
        params['num_classes'] = 22
        params['epochs'] = 10
        params['save_steps'] = 2000
    elif level == 'city':
        params['num_classes'] = 113
        params['epochs'] = 10
        params['save_steps'] = 2000
    else:
        params['num_classes'] = 26
        params['epochs'] = 10
        params['save_steps'] = 1000
    params['train_batch_size'] = 32
    params['val_batch_size'] = 32
    params['max_seq_length'] = 128
    params['num_classes'] = 26
    params['learning_rate'] = 3e-5
    params['metric'] = 'f1'
    params['adam_epsilon'] = 1e-08
    params['seed'] = 12345
    params['weight_decay'] = 0.0
    params['warmup_steps'] = 0
    params['gradient_accumulation_steps'] = 1
    params['max_grad_norm'] = 1.0
    aggregated_folder = 'aggregated_data'
    params['train_files'] = [
        f'{aggregated_folder}/city_train.tsv']
    params['val_files'] = [
        f'{aggregated_folder}/city_dev.tsv']
    params['label_space_file'] = f'labels/{level}_label_id.txt'
    labels, label2id, id2label = data_utils.get_label_space(
        params['label_space_file'])
    params['labels'] = labels
    params['label2id'] = label2id
    params['id2label'] = id2label

    print('Entering Training')
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime('%d-%m-%Y-%H:%M:%S')
    params['an experiment name'] = f'{params["metric"]}; aggregated_{level} on city'
    params['time'] = dt_string
    params['model_folder'] = f'{level}_{dt_string}'
    os.mkdir(params['model_folder'])
    train_results = run_train(params)
    """
    print('Entering Testing')
    params['model_file'] = train_results['model_file']
    params['test_files'] = [f'{aggregated_folder}/{level}_test.tsv']
    params['test_batch_size'] = 32
    test_results, test_predictions, test_predictions_argmax, true_labels = run_test(params)

    record_predictions(test_predictions, test_predictions_argmax,
                       f'{params["model_folder"]}/{level}_predictions.npz')
    """
    record_experiment([params, train_results],
                      'experiments', f'{level}_aggregated_city_madar')


def get_metrics():
    list_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    total_metrics = {}
    for m in list_metrics:
        total_metrics[m] = 0
    return total_metrics


def run_level_test_madar_experiment(level):
    params = {}

    print('Initializing training variables')
    if level == 'region':
        params['num_classes'] = 6
    elif level == 'country':
        params['num_classes'] = 22
    elif level == 'city':
        params['num_classes'] = 113
    else:
        params['num_classes'] = 26
    params['bert'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['tokenizer'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['level'] = level
    params['max_seq_length'] = 128
    params['num_classes'] = 26
    params['seed'] = 12345
    madar_folder = '../hierarchical-did/data_processed_second/madar_shared_task1/'
    #madar_folder = '../data_processed_second/madar_shared_task1/'
    #params['model_file'] = 'city_21-04-2021-17:17:54/best_model.pt'
    params['model_file'] = 'models/best_model.pt'
    params['test_files'] = [f'{madar_folder}MADAR-Corpus-26-test.lines']
    params['label_space_file'] = f'labels/{level}_label_id.txt'
    params['label_space_madar_file'] = f'labels/madar_{level}_label_id.txt'
    labels, label2id, id2label = data_utils.get_label_space(
        params['label_space_file'])
    params['labels'] = labels
    params['label2id'] = label2id
    params['id2label'] = id2label
    space = filter_logit.get_space(
        params['label_space_file'], params['label_space_madar_file'])
    params['test_batch_size'] = 32
    test_results, test_predictions, test_predictions_argmax, true_labels = run_test(
        params)
    record_predictions(test_predictions, f'madar_{level}.npz')
    nullified_metrics = get_metrics()
    nullified_preds = filter_logit.nullify_weight(test_predictions, space)
    finetuning_utils.metrics(nullified_preds, true_labels, nullified_metrics)
    for k in nullified_metrics.keys():
        test_results[f'nullified_{k}'] = nullified_metrics[k]
    # TODO: Check prediction distribution  filtering
    record_experiment([params, test_results],
                      'experiments', f'madar_{level}_test')


if __name__ == '__main__':
    run_city_aggregate_experiment('country')
    run_city_aggregate_experiment('region')
    # uncomment following line to run high level experiment on madar
    # run_level_test_madar_experiment('city')
    # uncomment following line to run madar 26 experiment using pretrained aggregated model
    # run_madar_pretrained_experiment()
    # uncomment following line to run madar 26 experiment
    # run_madar_experiment()
    # uncomment following line to run city level experiment
    # run_level_experiment('city')
    # uncomment following line to run country level experiment
    # run_level_experiment('country')
    # uncomment following line to run region level experiment
    # run_level_experiment('region')

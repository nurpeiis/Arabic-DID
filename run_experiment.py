import os
import logging
import argparse
from datetime import datetime
from finetuning_utils import set_seed
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


def record_predictions(model_folder, test_predictions):
    print(f'Writing predictions into {model_folder}')
    with open(f'{model_folder}/test_predictions.txt', 'w') as f:
        for p in test_predictions:
            f.write(f'{p}\n')


def run_madar_experiment():

    list_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    params = {}

    print('Initializing training variables')

    params['bert'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['tokenizer'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['level'] = 'city'
    params['train_batch_size'] = 32
    params['val_batch_size'] = 32
    params['max_seq_length'] = 32
    params['dropout_prob'] = 0.1
    params['num_classes'] = 26
    params['epochs'] = 10
    params['learning_rate'] = 1e-5
    params['early_stop_patience'] = 2
    params['metric'] = 'accuracy'
    madar_folder = '../data_processed_second/madar_shared_task1/'
    params['train_files'] = [
        f'{madar_folder}MADAR-Corpus-26-train.lines', f'{madar_folder}MADAR-Corpus-6-train.lines']
    params['val_files'] = [
        f'{madar_folder}MADAR-Corpus-26-dev.lines']
    params['label_space_file'] = f'labels/madar_labels_{params["level"]}.txt'

    for metric in list_metrics:
        print('Entering Training')
        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime('%d-%m-%Y-%H:%M:%S')
        params['an experiment name'] = f'{metric}; train-26,6; dev-26; test-26'
        params['time'] = dt_string
        params['metric'] = metric
        params['model_folder'] = f'{dt_string}'
        os.mkdir(params['model_folder'])
        train_results = run_train(params)
        print('Entering Testing')
        params['model_file'] = train_results['model_file']
        params['test_files'] = [f'{madar_folder}MADAR-Corpus-26-test.lines']
        params['test_batch_size'] = 8
        test_results, test_predictions = run_test(params)

        record_predictions(params['model_folder'], test_predictions)
        record_experiment([params, test_results, train_results],
                          'experiments', 'madar')


def run_level_experiment(level):
    list_metrics = ['accuracy', 'loss', 'f1']
    params = {}

    print('Initializing training variables')

    params['bert'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['tokenizer'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['level'] = level
    params['train_batch_size'] = 32
    params['val_batch_size'] = 32
    params['max_seq_length'] = 32
    params['dropout_prob'] = 0.1
    if level == 'region':
        params['num_classes'] = 6
    elif level == 'country':
        params['num_classes'] = 22
    elif level == 'city':
        params['num_classes'] = 113
    else:
        params['num_classes'] = 26
    params['epochs'] = 10
    params['learning_rate'] = 1e-5
    params['early_stop_patience'] = 2
    params['metric'] = 'accuracy'
    aggregated_folder = 'data_aggregated'
    params['train_files'] = [
        f'{aggregated_folder}/{level}_train.tsv']
    params['val_files'] = [
        f'{aggregated_folder}/{level}_dev.tsv']
    params['label_space_file'] = f'labels/{level}_label_id.txt'

    for metric in list_metrics:
        print('Entering Training')
        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime('%d-%m-%Y-%H:%M:%S')
        params['time'] = dt_string
        params['metric'] = metric
        params['model_folder'] = f'{dt_string}'
        os.mkdir(params['model_folder'])
        train_results = run_train(params)
        #record_predictions(params['model_folder'], test_predictions)
        record_experiment([params, train_results],
                          'experiments', f'{level}')


def run_camelbert_experiment(params):
    params['seed'] = 1234
    set_seed(params[seed])


if __name__ == '__main__':
    run_madar_experiment()

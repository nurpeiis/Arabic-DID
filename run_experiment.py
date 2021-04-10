import os
import logging
import argparse
from datetime import datetime
from experiment import run_train, run_test
from google_sheet_utils import get_sheet, get_all_records, rewrite_page, create_page, find_idx_page


def record_experiment(list_dicts, sheet_title, page_title):
    logging.info('Getting access to Google Sheet')
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
    logging.info(f'Writing predictions into {model_folder}')
    with open(f'{model_folder}/test_predictions.txt', 'w') as f:
        for p in test_predictions:
            f.write(f'{p}\n')


def main():
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter search for finetuning a BERT models for Arabic DID."
    )
    parser.add_argument(
        "logger_file",
        type=str,
        help="Logger file in cluster",
    )

    list_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    params = {}

    logging.info('Initializing training variables')
    args = parser.parse_args()
    params['logger_file'] = args.logger_file

    params['bert'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['tokenizer'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['level'] = 'city'
    params['train_batch_size'] = 32
    params['val_batch_size'] = 32
    params['max_seq_length'] = 32
    params['dropout_prob'] = 0.1
    params['hidden_size2'] = 512
    params['num_classes'] = 26
    params['epochs'] = 10
    params['learning_rate'] = 1e-5
    params['early_stop_patience'] = 2
    params['metric'] = 'accuracy'
    madar_folder = '../data_processed_second/madar_shared_task1/'
    params['train_files'] = [
        f'{madar_folder}MADAR-Corpus-26-train.lines', f'{madar_folder}MADAR-Corpus-6-train.lines']
    params['val_files'] = [
        f'{madar_folder}MADAR-Corpus-26-dev.lines', f'{madar_folder}MADAR-Corpus-6-dev.lines']
    params['label_space_file'] = f'labels/madar_labels_{params["level"]}.txt'

    for metric in list_metrics:
        logging.info('Entering Training')
        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime('%d-%m-%Y-%H:%M:%S')
        params['time'] = dt_string
        params['metric'] = metric
        params['model_folder'] = f'{dt_string}'
        os.mkdir(params['model_folder'])
        train_results = run_train(params)
        logging.info('Entering Testing')
        params['model_file'] = train_results['model_file']
        params['test_files'] = [f'{madar_folder}MADAR-Corpus-26-test.lines']
        params['test_batch_size'] = 8
        test_results, test_predictions = run_test(params)

        record_predictions(params['model_folder'], test_predictions)
        record_experiment([params, test_results, train_results],
                          'experiments', '26 labels 2')


def run_aggregate_experiment(params):
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter search for finetuning a BERT models for Arabic DID."
    )
    parser.add_argument(
        "logger_file",
        type=str,
        help="Logger file in cluster",
    )

    list_metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    params = {}

    logging.info('Initializing training variables')
    args = parser.parse_args()
    params['logger_file'] = args.logger_file

    params['bert'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['tokenizer'] = 'CAMeL-Lab/bert-base-camelbert-mix'
    params['level'] = 'city'
    params['train_batch_size'] = 32
    params['val_batch_size'] = 32
    params['max_seq_length'] = 32
    params['dropout_prob'] = 0.1
    params['hidden_size2'] = 512
    params['num_classes'] = 26
    params['epochs'] = 10
    params['learning_rate'] = 1e-5
    params['early_stop_patience'] = 2
    params['metric'] = 'accuracy'
    aggregated_folder = 'data_aggregated/'
    params['train_files'] = [
        f'{aggregated_folder}/']
    params['val_files'] = [
        f'{aggregated_folder}MADAR-Corpus-26-dev.lines', f'{aggregated_folder}MADAR-Corpus-6-dev.lines']
    params['label_space_file'] = f'labels/madar_labels_{params["level"]}.txt'

    for metric in list_metrics:
        logging.info('Entering Training')
        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime('%d-%m-%Y-%H:%M:%S')
        params['time'] = dt_string
        params['metric'] = metric
        params['model_folder'] = f'{dt_string}'
        os.mkdir(params['model_folder'])
        train_results = run_train(params)
        logging.info('Entering Testing')
        params['model_file'] = train_results['model_file']
        params['test_files'] = [
            f'{aggregated_folder}MADAR-Corpus-26-test.lines']
        params['test_batch_size'] = 8
        test_results, test_predictions = run_test(params)

        record_predictions(params['model_folder'], test_predictions)
        record_experiment([params, test_results, train_results],
                          'experiments', '26 labels 2')


if __name__ == '__main__':
    main()

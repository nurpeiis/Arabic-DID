import json
import pandas as pd


def json_to_tsv(in_file, out_file):

    with open(in_file) as data_file:
        data = json.load(data_file)
    df = pd.json_normalize(data)
    df.to_csv(out_file, sep='\t', index=False)


if __name__ == '__main__':
    json_to_tsv('results.json', 'results.tsv')
    json_to_tsv('results_salameh_repeat.json', 'results_salameh_repeat.tsv')
    json_to_tsv('results_salameh_repeat_train.json',
                'results_salameh_repeat_train.tsv')
    json_to_tsv('results_salameh_repeat_train_gpu.json',
                'results_salameh_repeat_train_gpu.tsv')

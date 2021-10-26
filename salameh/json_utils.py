import json
import pandas as pd


def json_to_tsv(in_file, out_file):

    with open(in_file) as data_file:
        data = json.load(data_file)
    df = pd.json_normalize(data)
    df.to_csv(out_file, sep='\t', index=False)


def compare(cur_val, cur_max, score):
    if cur_val == None:
        cur_val = cur_max
    else:
        if cur_val['levels_score']['city'][score] < cur_max['levels_score']['city'][score]:
            cur_val = cur_max
    return cur_val


def parse_json(file_name):
    with open(file_name) as data_file:
        data = json.load(data_file)
    city = None
    country = None
    region = None

    for d in data:
        if 'layer_0' in d:
            if d['layer_0']['level'] == 'city':
                city = compare(city, d, 'f1_macro')
            elif d['layer_0']['level'] == 'country':
                country = compare(country, d, 'f1_macro')
            elif d['layer_0']['level'] == 'region':
                region = compare(region, d, 'f1_macro')

    # repeat_train_


if __name__ == '__main__':
    """
    json_to_tsv('results.json', 'results.tsv')
    json_to_tsv('results_salameh_repeat.json', 'results_salameh_repeat.tsv')
    json_to_tsv('results_salameh_repeat_train.json',
                'results_salameh_repeat_train.tsv')
    json_to_tsv('results_salameh_plus.json',
                'results_salameh_plus.tsv')
    """
    json_to_tsv('results_sal_agg.json',
                'results_sal_agg.tsv')
# parse_json('results_salameh_repeat_train.json')

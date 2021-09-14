from salameh import DialectIdentifier
from utils import LayerObject


def run_experiment(aggregated_layers):
    print('Running Experiment')
    d = DialectIdentifier(
        aggregated_layers=aggregated_layers)
    d.train()
    scores = d.eval(data_set='TEST')
    d.record_experiment(scores)


def get_kenlm_train(level):
    return f'../aggregated_data/{level}_train.tsv'


def get_train_path(level):
    return ['aggregated_city/MADAR-Corpus-26-train.lines', f'../aggregated_data/{level}_train.tsv']


def get_cols_train(level, file_name):
    cols = ['dialect_city_id', 'dialect_country_id', 'dialect_region_id']
    if 'madar' in file_name.lower() or level == 'city':
        return cols
    elif level == 'country':
        return cols[1:]
    elif level == 'region':
        return cols[2:]


def get_single_layer_list(level, kenlm_train, exclude_list):
    layers = []

    for exclude in exclude_list[level]:
        for train_path in get_train_path(level):
            for lm in use_lm:
                for distr in use_distr:
                    if not lm and not distr:
                        continue
                    dict_repr = {}
                    dict_repr['level'] = level
                    dict_repr['kenlm_train'] = kenlm_train
                    dict_repr['kenlm_train_files'] = get_kenlm_train(level)
                    dict_repr['exclude_list'] = exclude
                    dict_repr['train_path'] = train_path
                    dict_repr['use_lm'] = lm
                    dict_repr['use_distr'] = distr
                    layers.append(dict_repr)

    return layers


def subsets_util(levels, result, subset, index):
    result.append(subset)
    for i in range(index, len(levels)):
        print(i)
        # include the A[i] in subset.
        subset.append(i)
        # move onto the next element.
        subsets_util(level, result, subset, i+1)
        # exclude the A[i] from subset and triggers
        # backtracking.
        subset.pop(-1)


def get_combo(levels):
    subset = []
    result = []
    # keeps track of current element in vector A;
    index = 0
    subsets_util(levels, result, subset, index)
    print(result)
    return result


if __name__ == '__main__':
    levels = ['city', 'country', 'region']
    kenlm_train = False
    exclude_list = {'city': [[], ['msa-msa-msa']],
                    'country': [[], ['msa-msa']],
                    'region': [[], ['msa']]}
    use_lm = [True, False]
    use_distr = [True, False]
    single_layers = []
    for level in levels:
        layers = get_single_layer_list(level, kenlm_train, exclude_list)
        single_layers.append(layers)

    get_combo(levels)

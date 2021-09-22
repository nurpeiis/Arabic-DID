from salameh import DialectIdentifier
from utils import LayerObject


def run_experiment(aggregated_layers=None, repeat_train=0, repeat_eval=0, file_name='results.json'):
    print('Running Experiment')
    d = DialectIdentifier(result_file_name=file_name,
                          aggregated_layers=aggregated_layers,
                          repeat_sentence_eval=repeat_eval,
                          repeat_sentence_train=repeat_train)
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


def get_single_layer_list(level, kenlm_train, exclude_list, use_lm, use_distr):
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
                    dict_repr['kenlm_train_files'] = get_kenlm_train(
                        level)
                    dict_repr['exclude_list'] = exclude
                    dict_repr['train_path'] = train_path
                    dict_repr['use_lm'] = lm
                    dict_repr['use_distr'] = distr
                    dict_repr['cols_train'] = get_cols_train(
                        level, train_path)
                    layers.append(dict_repr)

    return layers


def subsets_util(levels, i, n, result, subset, j):
    # checking if all elements of the array are traverse or not
    if(i == n):
        # print the subset array
        idx = 0
        a = []
        while(idx < j):
            a.append(subset[idx])
            idx += 1

        result.append(a)
        return

    # for each index i, we have 2 options
    # case 1: i is not included in the subset
    # in this case simply increment i and move ahead
    subsets_util(levels, i+1, n, result, subset, j)
    # case 2: i is included in the subset
    # insert arr[i] at the end of subset
    # increment i and j
    subset[j] = i
    subsets_util(levels, i+1, n, result, subset, j+1)


def get_combo(levels):
    subset = [0]*2**len(levels)
    result = []

    subsets_util(levels, 0, len(levels), result, subset, 0)
    result = result[1:]  # exclude empty array
    return result


def get_layers_combinations(combos, single_layers):
    layers_combo = []
    single_layer = []
    for combo in combos:
        single_layer = []
        for i in range(len(combo)):
            if i == 0 and len(combo) == 1:
                for layer in single_layers[combo[i]]:
                    single_layer.append([layer])
            # else:
        if len(single_layer):
            layers_combo.append(single_layer)

    return layers_combo


def run_experiments(layers_combo, file_name='results.json'):
    # just Salameh
    for repeat_train in range(3):
        for repeat_eval in range(3):
            run_experiment(repeat_train=repeat_train,
                           repeat_eval=repeat_eval, file_name=file_name)
    # all different combos
    for combo in layers_combo:
        for layers in combo:
            for repeat_train in range(3):
                for repeat_eval in range(3):
                    aggregated_layers = []
                    for layer in layers:
                        l = LayerObject(layer)
                        aggregated_layers.append(l)
                    run_experiment(
                        aggregated_layers, repeat_train=repeat_train, repeat_eval=repeat_eval, file_name=file_name)


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
        layers = get_single_layer_list(
            level, kenlm_train, exclude_list, use_lm, use_distr)
        single_layers.append(layers)

    combos = get_combo(levels)
    print(combos)
    layers_combo = get_layers_combinations(combos, single_layers)
    file_name = 'results_salameh_repeat_train.json'
    run_experiments(layers_combo, file_name)

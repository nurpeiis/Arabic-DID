from salameh import DialectIdentifier


def run_experiment(aggregated_layers):
    print('Running Experiment')
    d = DialectIdentifier(
        aggregated_layers=aggregated_layers)
    d.train()
    d.eval(data_set='TEST')


def get_kenlm_train(level):
    return f'../aggregated_data/{level}_train.tsv'


def get_train_path(level):
    return ['aggregated_city/MADAR-Corpus-26-train.lines', f'../aggregated_data/{level}_train.tsv']


if __name__ == '__main__':
    levels = ['city', 'country', 'region']
    kenlm_train = False
    exclude_list = {'city': [[], ['msa-msa-msa']],
                    'country': [[], ['msa-msa']],
                    'region': [[], ['msa']]}
    use_lm = [True, False]
    use_distr = [True, False]

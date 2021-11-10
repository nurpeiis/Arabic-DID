import os
from utils import LayerObject
from run_salameh import get_cols_train
from salameh import DialectIdentifier


def create_lm():
    layer_dict = {'level': 'city', 'data_dir': 'nadi_data', 'kenlm_train': True,
                  'kenlm_train_files': ['nadi_data/train_labeled.lines'], 'exclude_list': [],
                  'train_path': 'nadi_data/train_labeled.lines', 'use_lm': False, 'use_distr': True, 'cols_train': get_cols_train('city')}

    layer = LayerObject(layer_dict)


if __name__ == '__main__':
    nadi_dir = 'nadi_data/'
    labels = [i[:-4] for i in os.listdir(f'{nadi_dir}char')]
    print(labels)
    d = DialectIdentifier(
        labels=labels,
        labels_extra=[],
        char_lm_dir=f'{nadi_dir}lm/char',
        word_lm_dir=f'{nadi_dir}lm/word',
        aggregated_layers=None,
        result_file_name=None,
        repeat_sentence_train=0,
        repeat_sentence_eval=0,
        extra_lm=False,
        extra=False)
    d.train(data_path=[f'{nadi_dir}train_labeled.lines'])
    scores = d.eval(data_path=[f'{nadi_dir}dev_labeled.lines'])
    print(scores)

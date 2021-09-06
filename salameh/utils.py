import os
import kenlm
import timeit
import pandas as pd
import collections


class LayerObject:

    def __init__(self, level, kenlm_train, kenlm_train_files, exclude_list, train_path, test_path, val_path):

        self.data_dir = os.path.join(
            os.path.dirname(__file__), f'aggregated_{level}')
        self.kenlm_train = kenlm_train
        self.kenlm_train_files = kenlm_train_files
        self.exclude_list = exclude_list

        # kenlm was not trained yet or we would like to train from scratch
        if not os.path.exists(self.data_dir) or self.kenlm_train:
            self.kenlm_process()

        self.char_lm_dir = os.path.join(self.data_dir, 'lm', 'char')
        self.word_lm_dir = os.path.join(self.data_dir, 'lm', 'word')
        self.get_labels()
        self._char_lms = collections.defaultdict(kenlm.Model)
        self._word_lms = collections.defaultdict(kenlm.Model)
        self.load_lms()

        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path

    def get_labels(self):
        self.labels = sorted([i[:-5]
                              for i in os.listdir(self.char_lm_dir) if i[-4:] == 'arpa' and i[:-5] not in self.exclude_list])

    def kenlm_process(self):
        start = timeit.default_timer()
        print('Creating list of dialects and sentences')
        dialect_list, sentence_list = file2dialectsentence(
            self.kenlm_train_files, f'{self.level}')
        print('Creating dialect dictionary')
        dialect_dict = split_by_dialect(dialect_list, sentence_list)
        create_directory(self.data_dir)
        print('Putting each dialect into file')
        dialect_dict2file(dialect_dict, self.data_dir)
        print('Creating KenLM for each dialect')
        dialect_dict_to_lm(dialect_dict, self.data_dir)
        end = timeit.default_timer()
        print('Finished creating KenLM models in ', end - start)

    def load_lms(self):
        config = kenlm.Config()
        config.show_progress = False

        for label in self.labels:
            char_lm_path = os.path.join(
                self.char_lm_dir, '{}.arpa'.format(label))
            word_lm_path = os.path.join(
                self.word_lm_dir, '{}.arpa'.format(label))
            self.char_lms[label] = kenlm.Model(
                char_lm_path, config)
            self.word_lms[label] = kenlm.Model(
                word_lm_path, config)


def split_by_dialect(dialect_list, sentence_list):
    char_list = []
    for s in sentence_list:
        s = ' '.join(list(s.replace(' ', 'X')))
        char_list.append(s)
    dialect_dict = dict()
    for i in range(len(char_list)):
        if (dialect_list[i] in dialect_dict) == False:
            dialect_dict[dialect_list[i]] = dict()
            dialect_dict[dialect_list[i]]['word'] = []
            dialect_dict[dialect_list[i]]['char'] = []

        dialect_dict[dialect_list[i]]['word'].append(sentence_list[i])
        dialect_dict[dialect_list[i]]['char'].append(char_list[i])

    return dialect_dict


def create_directory(dir_name):
    if not os.path.exists(f'{dir_name}'):
        os.mkdir(f'{dir_name}')


def dialect_dict2file(dialect_dict, folder):
    create_directory(f'{folder}/word')
    create_directory(f'{folder}/char')

    for k in dialect_dict.keys():
        with open(f'{folder}/word/{k}.txt', 'w') as f:
            for item in dialect_dict[k]['word']:
                f.write(f'{item}\n')
        with open(f'{folder}/char/{k}.txt', 'w') as f:
            for item in dialect_dict[k]['char']:
                f.write(f'{item}\n')


def create_lm(in_file, out_file):
    command = f'~/kenlm/build/bin/lmplz -o 5 < {in_file} > {out_file} --discount_fallback'
    os.system(command)


def dialect_dict_to_lm(dialect_dict, folder):
    create_directory(f'{folder}/lm')
    create_directory(f'{folder}/lm/word')
    create_directory(f'{folder}/lm/char')

    for k in dialect_dict.keys():
        create_lm(f'{folder}/word/{k}.txt', f'{folder}/lm/word/{k}.arpa')
        create_lm(f'{folder}/char/{k}.txt', f'{folder}/lm/char/{k}.arpa')


def file2dialectsentence(files, level):
    df = pd.read_csv(files[0], sep='\t', header=0)
    for i in range(1, len(files)):
        df = df.append(pd.read_csv(files[i], sep='\t', header=0))
    sentence_list = df['original_sentence'].tolist()
    cols = []
    if level == 'city':
        cols = ['dialect_city_id', 'dialect_country_id', 'dialect_region_id']
    elif level == 'country':
        cols = ['dialect_country_id', 'dialect_region_id']
    elif level == 'region':
        cols = ['dialect_region_id']
    df['combined'] = df[cols].apply(
        lambda row: '-'.join(row.values.astype(str)), axis=1)

    dialect_list = df['combined'].tolist()
    return dialect_list, sentence_list


def whole_process(level, train_files):
    start = timeit.default_timer()
    print('Creating list of dialects and sentences')
    dialect_list, sentence_list = file2dialectsentence(train_files, f'{level}')
    print('Creating dialect dictionary')
    dialect_dict = split_by_dialect(dialect_list, sentence_list)
    folder = f'aggregated_{level}'
    create_directory(folder)
    print('Putting each dialect into file')
    dialect_dict2file(dialect_dict, folder)
    print('Creating KenLM for each dialect')
    dialect_dict_to_lm(dialect_dict, folder)
    end = timeit.default_timer()
    print('Finished in ', end - start)


if __name__ == '__main__':
    level = 'country'
    train_files = [f'../aggregated_data/{level}_train.tsv']
    whole_process(level, train_files)

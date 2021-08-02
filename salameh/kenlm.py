import os
import pandas as pd


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
    dialect_list = df['original_sentence'].tolist()
    sentence_list = df[f'dialect_{level}_id'].tolist()
    return dialect_list, sentence_list


def whole_process(level, train_files):

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
    print('Finished')

if __name__ == '__main__':
    level = 'city'
    train_files = [f'../aggregated_data/{level}_train.tsv']
    whole_process(level, train_files)

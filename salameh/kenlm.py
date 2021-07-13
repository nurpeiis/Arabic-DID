import os


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


# if __name__ == '__main__':

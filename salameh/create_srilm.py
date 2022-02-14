import os
from utils import create_directory


def create_srilm(in_file, out_file):
    ngram_count_path = '/scratch/nb2577/srilm-1.7.3/bin/i686-m64/ngram-count'
    ngram_count_path_pc = 'ngram-count'
    command = f'{ngram_count_path} -text {in_file} -order 5 -lm {out_file} -wbdiscount'
    os.system(command)


def create_srilm_directory(folder, dialects):
    create_directory(f'{folder}/srilm')
    create_directory(f'{folder}/srilm/word')
    create_directory(f'{folder}/srilm/char')

    for k in dialects:
        create_srilm(f'{folder}/word/{k}.txt', f'{folder}/srilm/word/{k}.lm')
        create_srilm(f'{folder}/char/{k}.txt', f'{folder}/srilm/char/{k}.lm')


if __name__ == '__main__':
    levels = ['city', 'country', 'region']

    for level in levels:
        print(f'Starting level: {level}')
        folder = f'aggregated_{level}'
        dialects = [i[:-4] for i in os.listdir(
            f'{folder}/word') if i != '.DS_Store' and '.txt' in i]
        print('Dialects', dialects)
        create_srilm_directory(folder, dialects)
        print(f'Finished level: {level}')

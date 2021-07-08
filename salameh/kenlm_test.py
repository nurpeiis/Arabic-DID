import os


def create_lm(in_file, out_file):
    command = f'~/kenlm/build/bin/lmplz -o 5 < {in_file} > {out_file}'
    os.system(command)


def main():
    create_lm(
        '../hierarchical-did/data_processed_second/madar_shared_task1/MADAR-Corpus-6-dev.lines', 'aa')


if __name__ == '__main__':
    main()

from data_process import DataProcess

# Specify a directory to the dataset that you want to preprocess.
dp = DataProcess('/Users/alemshaimardanov/PycharmProjects/NLP/arabic_did/data_processed_check/aoc/', 'mechanical_turk_annotators', 'news_comments',
                 'https://www.cis.upenn.edu/~ccb/data/AOC-dialectal-annotations.zip', 'aoc', {}, {}, 0, 'corpus', 'manual')

if dp.split_original_manual == 'manual':
    # Specify a directory to unprocessed dataset which stores AOC-dialectal-annotations.
    dp.save_file('dialect_alghad-segs.tsv', dp.preprocess(
        '/Users/alemshaimardanov/PycharmProjects/NLP/arabic_did/data_raw_main/AOC-dialectal-annotations/dialect_alghad-segs.txt.norm', '', '', '', 'levant', header=None))

    # Specify a directory to unprocessed dataset which stores AOC-dialectal-annotations.
    dp.save_file('dialect_youm7-segs.tsv', dp.preprocess(
        '/Users/alemshaimardanov/PycharmProjects/NLP/arabic_did/data_raw_main/AOC-dialectal-annotations/dialect_youm7-segs.txt.norm', '', '', 'eg', 'nile_basin', header=None))

    # Specify a directory to unprocessed dataset which stores AOC-dialectal-annotations.
    dp.save_file('dialect_alriyadh-segs.tsv', dp.preprocess(
        '/Users/alemshaimardanov/PycharmProjects/NLP/arabic_did/data_raw_main/AOC-dialectal-annotations/dialect_alriyadh-segs.txt.norm', '', '', '', 'gulf', header=None))

    # Specify a directory to unprocessed dataset which stores AOC-dialectal-annotations.
    dp.save_file('MSA_alghad-segs.tsv', dp.preprocess(
        '/Users/alemshaimardanov/PycharmProjects/NLP/arabic_did/data_raw_main/AOC-dialectal-annotations/MSA_alghad-segs.txt.norm', 'msa', 'msa', 'msa', 'msa', header=None))

    # Specify a directory to unprocessed dataset which stores AOC-dialectal-annotations.
    dp.save_file('MSA_youm7-segs.tsv', dp.preprocess('/Users/alemshaimardanov/PycharmProjects/NLP/arabic_did/data_raw_main/AOC-dialectal-annotations/MSA_youm7-segs.txt.norm',
                 'msa', 'msa', 'msa', 'msa', header=None))

    # Specify a directory to unprocessed dataset which stores AOC-dialectal-annotations.
    dp.save_file('MSA_alriyadh-segs.tsv', dp.preprocess(
        '/Users/alemshaimardanov/PycharmProjects/NLP/arabic_did/data_raw_main/AOC-dialectal-annotations/MSA_alriyadh-segs.txt.norm', 'msa', 'msa', 'msa', 'msa', header=None))

else:
    files = ['dialect_alghad-segs.tsv', 'dialect_youm7-segs.tsv', 'dialect_alriyadh-segs.tsv',
         'MSA_alghad-segs.tsv', 'MSA_youm7-segs.tsv', 'MSA_alriyadh-segs.tsv']
    for file in files:
        df_train, df_dev, df_test = dp.split(file, 0.8, 0.1, 0.1)
        dp.save_file('train_'+file, df_train)
        dp.save_file('dev_'+file, df_dev)
        dp.save_file('test_'+file, df_test)
    
# Specify a directory where you want to save datasets_splitted_features file.
dp.save_features('/Users/alemshaimardanov/PycharmProjects/NLP/arabic_did/datasets_splited_features.tsv')
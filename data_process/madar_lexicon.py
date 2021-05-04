from data_process import DataProcess

dp = DataProcess('../data_processed/madar_lexicon/', 'word_level', 'travel_domain', 'https://camel.abudhabi.nyu.edu/madar/', 'madar_lexicon', {},{},6, 'lexicon')

dp.save_file('madar_lexicon_4.0.tsv', dp.preprocess('../../data_raw/madar_lexicon_4.0.xlsx', 8, '', '', '', header=1, excel=True))


dp.save_features('../datasets_features.tsv')
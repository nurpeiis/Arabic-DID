from data_process import DataProcess

dp_egyptian = DataProcess('../data_processed/tharwa/', 'word_level', 'dictionary_mixed', 'https://www.aclweb.org/anthology/L14-1115/', 'tharwa', {},{},[1,3], 'lexicon')
dp_lev = DataProcess('../data_processed/tharwa/', 'word_level', 'dictionary_mixed', 'https://www.aclweb.org/anthology/L14-1115/', 'tharwa', {},{},[24,26], 'lexicon')


dp_egyptian.save_file('Tharwa-v0.2.tsv', dp_egyptian.preprocess('../../data_raw/Tharwa-v0.2+LEV/Tharwa-v0.2.tsv', '', '', 'Egyptian', 'Egyptian', header=0))
dp_lev.save_file('Tharwa+Lev.tsv', dp_lev.preprocess('../../data_raw/Tharwa-v0.2+LEV/Tharwa+Lev.tsv', '', '', '', 'Levantine', header=0))


dp_lev.save_features('../datasets_features.tsv')
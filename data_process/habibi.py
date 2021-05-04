from data_process import DataProcess

#Step 0: Pre-Clean
df = pd.read_csv('../../data_raw/habibi.csv', header=0, encoding='utf-8-sig')
df_new = df.drop_duplicates('Lyrics')
df_new.to_csv('../../data_raw/habibi_preclean.csv', sep=',',
                      index=False)

dp = DataProcess('../data_processed_splited/habibi/', 'user_level', 'song_lyrics', 'https://www.aclweb.org/anthology/2020.lrec-1.165/', 'habibi', {},{},6, 'corpus', 'manual')

dp.save_file('habibi.tsv', dp.preprocess('../../data_raw/habibi_preclean.csv', '', '', 7, 8, header=0, delimiter=','))

dp.save_features('../datasets_splited_features.tsv')

dp.standardize_labels('../data_processed_splited/habibi/habibi.tsv', '../data_processed_splited/habibi/habibi_labels.tsv')

files = ['habibi_labels.tsv']
#Step 1: Split data into Train, Dev and Test datasets
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.7, 0.15, 0.15)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)


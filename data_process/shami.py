from data_process import DataProcess

dp = DataProcess('../data_processed_splited/shami/', 'semi_automatic', 'web_mixed', 'https://www.aclweb.org/anthology/L18-1576/', 'shami', {},{},0, 'corpus', 'manual')


files = os.listdir('../data_processed/shami/')
for file in files:
    df = pd.read_csv('../data_processed/shami/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'manual'
    df.to_csv('../data_processed_splited/shami/'+file, sep='\t')
    

files = os.listdir('../data_processed_splited/shami/')
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.1, 0.1)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)
    

dp.save_features('../datasets_splited_features.tsv')

files = ['jordinian', 'Lebanees', 'Palestinian', 'syrian']
for f in files:
    dp.save_file(f+'.tsv', dp.preprocess('../../data_raw/shami-corpus-master/Data/'+f+'.txt', '', '', f, '', header=None))


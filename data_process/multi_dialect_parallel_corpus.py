from data_process import DataProcess

dp = DataProcess('../data_processed_splited/multi_dialect_parallel_corpus/', 'manual_translation', 'ldc_mixed', 'https://www.aclweb.org/anthology/L14-1435/', 
                 'multi_dialect_parallel_corpus', {},{},0, 'corpus', 'manual')

files = os.listdir('../data_processed/multi_dialect_parallel_corpus/')
for file in files:
    df = pd.read_csv('../data_processed/multi_dialect_parallel_corpus/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'manual'
    df.to_csv('../data_processed_splited/multi_dialect_parallel_corpus/'+file, sep='\t')
    

dp.save_features('../datasets_splited_features.tsv')


files = os.listdir('../data_processed_splited/multi_dialect_parallel_corpus/')
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.1, 0.1)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)
    

files = ['EG', 'JO', 'MSA', 'PA', 'SY', 'TN']
folder_name = '../../data_raw/MultiDialParCorpus/'
for f in files:
    if f == 'MSA':
        dp.save_file(f +'.tsv', dp.preprocess(folder_name + f, f, f, f, f, header=None))
    
    else:
        dp.save_file(f +'.tsv', dp.preprocess(folder_name + f, '', '', f, '', header=None))



        

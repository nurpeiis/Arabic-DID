from data_process import DataProcess

dp = DataProcess('../data_processed_splited/nadi/', 'user_level', 'twitter', 'https://sites.google.com/view/nadi-shared-task', 'nadi', {},{},1, 'corpus', 'original')

files = os.listdir('../data_processed/nadi/')
for file in files:
    df = pd.read_csv('../data_processed/nadi/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'original'
    df.to_csv('../data_processed_splited/nadi/'+file, sep='\t')
    
    
#sa_Ha'il is wrong label
files_wrong_label = ['train_labeled.tsv', 'dev_labeled.tsv']
for file in files_wrong_label:
    df = pd.read_csv('../data_processed_splited/nadi/'+file, delimiter='\t', header=0)
    for i in range(len(df)):
        if df['dialect_province_id'][i] == "sa_Ha'il":
            df['dialect_province_id'][i] = 'hail'
            df['dialect_country_id'][i] = 'sa'        
            df['dialect_region_id'][i] = 'gulf'
    df.to_csv('../data_processed_splited/nadi/'+file, sep='\t')

        


dp.save_features('../datasets_splited_features.tsv')


dp.save_file('dev_labeled.tsv', dp.preprocess('../../data_raw/NADI_shared_task/NADI_release/dev_labeled.tsv', '', 3, 2, '', header=0))
dp.save_file('train_labeled.tsv', dp.preprocess('../../data_raw/NADI_shared_task/NADI_release/train_labeled.tsv', '', 3, 2, '', header=0))
dp.save_file('test_unlabeled.tsv', dp.preprocess('../../data_raw/NADI_shared_task/NADI-2020_TEST_2.0/test_unlabeled.tsv', '', '', '', '', header=0))
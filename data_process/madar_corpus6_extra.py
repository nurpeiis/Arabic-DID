from data_process import DataProcess

dp = DataProcess('../data_processed_splited/madar_corpus6_extra/', 'manual_translation', 'travel_domain', 'https://camel.abudhabi.nyu.edu/madar-shared-task-2019/', 'madar_corpus6_extra', {},{},0, 'corpus', 'original')

files = os.listdir('../data_processed/madar_corpus6_extra/')
for file in files:
    df = pd.read_csv('../data_processed/madar_shared_task1/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'original'
    df.to_csv('../data_processed_splited/madar_shared_task1/'+file, sep='\t')
    

files = !ls ../../data_raw/Madar-Corpus5-extra8K
for f in files:
    dialect = f[5:-7]
    if dialect != 'EN' and dialect != 'FR':
        dp.save_file(f, dp.preprocess('../../data_raw/Madar-Corpus5-extra8K/{}'.format(f), dialect, '', '', '', header=None))



dp.save_features('../datasets_features.tsv')
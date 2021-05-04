from data_process import DataProcess

dp = DataProcess('../data_processed_splited/madar_shared_task1/', 'manual_translation', 'travel_domain', 'https://camel.abudhabi.nyu.edu/madar-shared-task-2019/', 'madar_shared_task1', {},{},0, 'corpus', 'original')

files = os.listdir('../data_processed/madar_shared_task1/')
for file in files:
    df = pd.read_csv('../data_processed/madar_shared_task1/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'original'
    df.to_csv('../data_processed_splited/madar_shared_task1/'+file, sep='\t')
    

files = !ls ../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-1 
for f in files:
    if f[len(f)-3:] == 'tsv':
        dp.save_file(f, dp.preprocess('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-1/{}'.format(f), 1, '', '', '', header=None))

        
dp.save_features('../datasets_splited_features.tsv')

#Change Tripoli to Lybian tripoli rather than Lebanon
folder = '../data_processed/madar_shared_task1/'
files = os.listdir(folder)
for file in files:
    df = pd.read_csv(f'{folder}{file}', delimiter='\t', header=0)
    for index, row in df.iterrows():
        if row['dialect_province_id'] == 'tripoli':
            row['dialect_city_id'] = 'tripoli'
            row['dialect_province_id'] = 'tripoli'
            row['dialect_country_id'] = 'ly'
            row['dialect_region_id'] = 'maghreb'
    df.to_csv(f'{folder}{file}', sep='\t', index=False)


#Change Ad Dahwan to Doha tripoli rather than Lebanon
folder = '../data_processed/madar_shared_task1/'
files = os.listdir(folder)
for file in files:
    df = pd.read_csv(f'{folder}{file}', delimiter='\t', header=0)
    for index, row in df.iterrows():
        if row['dialect_province_id'] == 'ad_dawhah':
            row['dialect_province_id'] = 'doha'
    df.to_csv(f'{folder}{file}', sep='\t', index=False)
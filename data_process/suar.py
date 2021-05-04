from data_process import DataProcess

files = os.listdir('../data_processed/suar/')
for file in files:
    df = pd.read_csv('../data_processed/suar/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'manual'
    df.to_csv('../data_processed_splited/suar/'+file, sep='\t')
    

dp = DataProcess('../data_processed_splited/suar/', "hashtag_level", "twitter", 'https://www.sciencedirect.com/science/article/pii/S187705091832163X', 
                 'suar', {},{},0,'corpus', 'manual') 
dp.save_features('../datasets_splited_features.tsv')


files = os.listdir('../data_processed_splited/suar/')
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.1, 0.1)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)
    
    
    
raw_folder = '../../data_raw/SUAR/Additional_Cleaning/'
def suar_process(data_source, data_annotation_method, start_index, end_index, province_list):
    dp = DataProcess('../data_preprocessed/suar/', data_annotation_method, data_source, 'https://www.sciencedirect.com/science/article/pii/S187705091832163X', 
                 'suar', {},{},0,'corpus') 
    for f in range(start_index, end_index):
        dp.save_file(data_source + '_' + str(f) +'.tsv', dp.preprocess(raw_folder + str(f) +'.txt', '', ','.join(province_list), 'Saudi', '', header=None))


suar_process("twitter", "hashtag_level", 1, 11, ['Najdi'])
suar_process("youtube_transcript", "speech_transcript", 11, 19, ['Gulf', 'Najdi'])
suar_process("whatsapp", "informal_message", 19, 53, ['Najdi'])
suar_process("blog", "manual_search", 53, 72, ['Hijazi', 'Najdi'])
suar_process("instagram", "user_level", 72, 84,  ['Hijazi', 'Najdi'])
suar_process("forum", "manual_search", 84, 89, ['Hijazi', 'Najdi'])



dp = DataProcess('../data_preprocessed/suar/', "hashtag_level", "twitter", 'https://www.sciencedirect.com/science/article/pii/S187705091832163X', 
                 'suar', {},{},0,'corpus') 
    

dp.save_features('../datasets_features.tsv')


    
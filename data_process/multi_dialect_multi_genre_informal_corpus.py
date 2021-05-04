from data_process import DataProcess

dp_news = DataProcess('../data_processed_splited/multi_dialect_multi_genre_informal_corpus/news/', 'mechanical_turk_annotators', 'news_comments', 'https://github.com/ryancotterell/arabic_dialect_annotation', 
                 'multi_dialect_multi_genre_informal_corpus', {},{},1, 'corpus', 'manual')
dp_twitter = DataProcess('../data_processed_splited/multi_dialect_multi_genre_informal_corpus/twitter/', 'mechanical_turk_annotators', 'twitter', 'https://github.com/ryancotterell/arabic_dialect_annotation', 
                 'multi_dialect_multi_genre_informal_corpus', {},{},1, 'corpus', 'manual')


files = os.listdir('../data_processed/multi_dialect_multi_genre_informal_corpus/')
for file in files:
    df = pd.read_csv('../data_processed/multi_dialect_multi_genre_informal_corpus/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'manual'
    if 'twitter' in file:
        df.to_csv('../data_processed_splited/multi_dialect_multi_genre_informal_corpus/twitter/'+file, sep='\t')
    else:
        df.to_csv('../data_processed_splited/multi_dialect_multi_genre_informal_corpus/news/'+file, sep='\t')        
        
        
dp_news.save_features('../datasets_splited_features.tsv')
dp_twitter.save_features('../datasets_splited_features.tsv')


files_twitter = os.listdir('../data_processed_splited/multi_dialect_multi_genre_informal_corpus/twitter')
for file in files_twitter:
    df_train, df_dev, df_test = dp_twitter.split(file, 0.8, 0.1, 0.1)
    dp_twitter.save_file('train_'+file, df_train)    
    dp_twitter.save_file('dev_'+file, df_dev)
    dp_twitter.save_file('test_'+file, df_test)
files_news = os.listdir('../data_processed_splited/multi_dialect_multi_genre_informal_corpus/news')
for file in files_news:
    df_train, df_dev, df_test = dp_news.split(file, 0.8, 0.1, 0.1)
    dp_news.save_file('train_'+file, df_train)    
    dp_news.save_file('dev_'+file, df_dev)
    dp_news.save_file('test_'+file, df_test)
        

files = ['egyptian', 'gulf', 'iraqi', 'msa', 'maghrebi', 'levantine']
folder_name = '../../data_raw/burch_region/'
for f in files:
    if f == 'msa':
        dp_news.save_file(f +'.tsv', dp_news.preprocess(folder_name + f, f, f, f, f, header=None))
    else:
        dp_news.save_file(f +'.tsv', dp_news.preprocess(folder_name + f, '', '', '', f, header=None))
        dp_twitter.save_file('twitter-' + f +'.tsv', dp_twitter.preprocess(folder_name + 'twitter-' + f, '', '', '', f, header=None))
        



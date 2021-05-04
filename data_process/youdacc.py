from data_process import DataProcess

dp = DataProcess('../data_processed_splited/youdacc/', 'user_level', 'youtube_comments', 'https://www.aclweb.org/anthology/L14-1456/', 'youdacc', {},{},0,'corpus', 'original')


files = os.listdir('../data_processed/youdacc/')
for file in files:
    df = pd.read_csv('../data_processed/youdacc/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'original'
    df.to_csv('../data_processed_splited/youdacc/'+file, sep='\t')
    

dp.save_features('../datasets_splited_features.tsv')


files = !ls ../../data_raw/Youdacc-fixed
dict_dialects = {'EG': 'EG', 'Gulf': 'Gulf', 'IQ':'IQ', 'MSA': 'MSA', 'North': 'North'} #North = Levantine

for f in files:
    if f[:f.find('.')] == 'MSA':
        dp.save_file(f[:-3]+'tsv', dp.preprocess('../../data_raw/Youdacc-fixed/{}'.format(f), f[:f.find('.')], f[:f.find('.')], f[:f.find('.')], f[:f.find('.')], header=None))

    else:
        dp.save_file(f[:-3]+'tsv', dp.preprocess('../../data_raw/Youdacc-fixed/{}'.format(f), '', '', '', dict_dialects[f[:f.find('.')]], header=None))




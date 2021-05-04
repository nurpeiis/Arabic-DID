from data_process import DataProcess

dp = DataProcess('../data_processed_splited/padic/', 'manual_translation', 'speech_transcript', 'https://hal.archives-ouvertes.fr/hal-01718858', 'padic', {},{},0, 'corpus', 'manual')

files = os.listdir('../data_processed/padic/')
for file in files:
    df = pd.read_csv('../data_processed/padic/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'manual'
    df.to_csv('../data_processed_splited/padic/'+file, sep='\t')
    
files = os.listdir('../data_processed_splited/padic/')
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.1, 0.1)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)

dp.save_features('../datasets_splited_features.tsv')


dp_msa = DataProcess('../data_processed/padic/', 'manual_translation', 'speech_transcript', 'https://hal.archives-ouvertes.fr/hal-01718858', 'padic', {},{},0, 'corpus')
dp_alg = DataProcess('../data_processed/padic/', 'manual_translation', 'speech_transcript', 'https://hal.archives-ouvertes.fr/hal-01718858', 'padic', {},{},1, 'corpus')
dp_anb = DataProcess('../data_processed/padic/', 'manual_translation', 'speech_transcript', 'https://hal.archives-ouvertes.fr/hal-01718858', 'padic', {},{},2, 'corpus')
dp_tun = DataProcess('../data_processed/padic/', 'manual_translation', 'speech_transcript', 'https://hal.archives-ouvertes.fr/hal-01718858', 'padic', {},{},3, 'corpus')
dp_pal = DataProcess('../data_processed/padic/', 'manual_translation', 'speech_transcript', 'https://hal.archives-ouvertes.fr/hal-01718858', 'padic', {},{},4, 'corpus')
dp_syr = DataProcess('../data_processed/padic/', 'manual_translation', 'speech_transcript', 'https://hal.archives-ouvertes.fr/hal-01718858', 'padic', {},{},5, 'corpus')
dp_mar = DataProcess('../data_processed/padic/', 'manual_translation', 'speech_transcript', 'https://hal.archives-ouvertes.fr/hal-01718858', 'padic', {},{},6, 'corpus')

dp_msa.save_file('padic_msa.tsv', dp_msa.preprocess('../../data_raw/PADIC-31-05SansFranSansAng.xlsx', 'MSA', 'MSA', 'MSA', 'MSA', header=0, excel=True))
dp_alg.save_file('padic_alg.tsv', dp_alg.preprocess('../../data_raw/PADIC-31-05SansFranSansAng.xlsx', '', 'ALG', '', '', header=0, excel=True))
dp_anb.save_file('padic_anb.tsv', dp_anb.preprocess('../../data_raw/PADIC-31-05SansFranSansAng.xlsx', '', 'ANB', '', '', header=0, excel=True))
dp_tun.save_file('padic_tun.tsv', dp_tun.preprocess('../../data_raw/PADIC-31-05SansFranSansAng.xlsx', '', '', 'TUN', '', header=0, excel=True))
dp_pal.save_file('padic_pal.tsv', dp_pal.preprocess('../../data_raw/PADIC-31-05SansFranSansAng.xlsx', '', '', 'PAL', '', header=0, excel=True))
dp_syr.save_file('padic_syr.tsv', dp_syr.preprocess('../../data_raw/PADIC-31-05SansFranSansAng.xlsx', '', '', 'SYR', '', header=0, excel=True))
dp_mar.save_file('padic_mar.tsv', dp_mar.preprocess('../../data_raw/PADIC-31-05SansFranSansAng.xlsx', '', '', 'MAR', '', header=0, excel=True))


dp_msa.save_features('../datasets_features.tsv')
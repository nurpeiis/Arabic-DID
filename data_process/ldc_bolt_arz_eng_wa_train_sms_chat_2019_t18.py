from data_process import DataProcess

#Step 0: Preprocess the data
df = pd.DataFrame()
folder = '../../data_raw/ldc_bolt_arz_eng_wa_train_sms_chat_2019_t18/data/source/tokenized'
files = os.listdir(folder)
for file in files:
    with open(folder + '/' + file) as f:
        lines = f.read().splitlines()
    df_tmp = pd.DataFrame(lines)
    df = df.append(df_tmp, ignore_index=True)
df.to_csv('../../data_raw/ldc_bolt_arz_eng_wa_train_sms_chat_2019_t18/processed.tsv', index=False, header=None, sep='\t')

#Step 1: Create an object of DataProcess class in order to preclean and preprocess the original dataset
dp = DataProcess('../data_processed/ldc_bolt_arz_eng_wa_train_sms_chat_2019_t18/', 'user_level', 'sms', 'https://catalog.ldc.upenn.edu/LDC2019T18', 'ldc_bolt_arz_eng_wa_train_sms_chat_2019_t18', {},{},0, 'corpus')


dp.save_file('processed.tsv', dp.preprocess('../../data_raw/ldc_bolt_arz_eng_wa_train_sms_chat_2019_t18/processed.tsv', '', '', 'Egypt', '', header=None))


dp.save_features('../datasets_features.tsv')
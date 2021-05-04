from data_process import DataProcess

df = pd.DataFrame([])
folders = !ls ../../data_raw/ldc_bab_arabic_2005_s08/data/text
for folder in folders:
    fol = "../../data_raw/ldc_bab_arabic_2005_s08/data/text/" + folder + '/'
    files = os.listdir(fol)
    for f in files:
        df_temp = pd.read_csv(fol+f, header=None)
        df = df.append(df_temp, ignore_index=True)
        

df.to_csv("../../data_raw/ldc_bab_arabic_2005_s08/processed.tsv", sep="\t", index=False, header=None)
dp.save_file('processed.tsv', dp.preprocess('../../data_raw/ldc_bab_arabic_2005_s08/processed.tsv', '', '', 'LEV', '', header=0))

# Create an object of DataProcess class in order to preclean and preprocess the original dataset
dp = DataProcess('../data_processed_splited/ldc_bab_arabic_2005_s08/', 'user_level', 'speech_transcript', 'https://catalog.ldc.upenn.edu/LDC2005S08/', 'ldc_bab_arabic_2005_s08', {},{},0, 'corpus', 'manual')


df = pd.read_csv('../data_processed/ldc_bab_arabic_2005_s08/processed.tsv', delimiter='\t', header=0, index_col=0)
df['split_original_manual'] = 'manual'
df.to_csv('../data_processed_splited/ldc_bab_arabic_2005_s08/processed.tsv', sep='\t')


dp.save_features('../datasets_splited_features.tsv')


files = ['processed.tsv']
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.10, 0.10)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)
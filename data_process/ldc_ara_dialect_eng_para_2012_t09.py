from data_process import DataProcess

#Step 0: From xml to csv
xtree = et.parse('../../data_raw/ldc_ara_dialect_eng_para_2012_t09/data/BBN-Dialect_Arabic-English-Web.xml')
xroot = xtree.getroot()
rows_eg = []
rows_levant = []
cols = ["original_sentence", "dialect_region_id", "dialect_country_id"]
dialects = set()
for node in xroot: 
    if(node.find("DIALECT").text == 'EGYPTIAN'):
        rows_eg.append({'dialect_region_id': 'nile_basin', 'dialect_country_id': 'eg', 'original_sentence': node.find("SOURCE").text})
    else:
        rows_levant.append({'dialect_region_id': 'levant', 'dialect_country_id': '', 'original_sentence': node.find("SOURCE").text})
df_eg = pd.DataFrame(rows_eg, columns=cols)
df_eg.to_csv('../../data_raw/ldc_ara_dialect_eng_para_2012_t09/eg_processed.tsv', sep='\t', index=False)
df_levant = pd.DataFrame(rows_levant, columns=cols)
df_levant.to_csv('../../data_raw/ldc_ara_dialect_eng_para_2012_t09/levant_processed.tsv', sep='\t', index=False)


#Step 1: Create an object of DataProcess class to preclean, preprocess raw dataset
dp = DataProcess('../data_processed_splited/ldc_ara_dialect_eng_para_2012_t09/', 'mechanical_turk_annotators', 'web_mixed', 'https://catalog.ldc.upenn.edu/LDC2012T09', 'ldc_ara_dialect_eng_para_2012_t09', {},{},0, 'corpus', 'manual')

dp.save_file('eg_processed.tsv', dp.preprocess('../../data_raw/ldc_ara_dialect_eng_para_2012_t09/eg_processed.tsv', '', '', 2, 1, header=0))
dp.save_file('levant_processed.tsv', dp.preprocess('../../data_raw/ldc_ara_dialect_eng_para_2012_t09/levant_processed.tsv', '', '', 2, 1, header=0))

dp.save_features('../datasets_splited_features.tsv')

files = ['eg_processed.tsv', 'levant_processed.tsv']

#Step 2: Split the dataset into Train, Dev, Test datasets
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.10, 0.10)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)
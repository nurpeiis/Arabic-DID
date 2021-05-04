from data_process import DataProcess


# Step 0: Preprocess testset
df_test = pd.read_csv('../../data_raw/qadi_country/testset/QADI_test.txt', delimiter='\t', header=None)

df_test.to_csv('../../data_raw/qadi_country/test_processed.tsv', sep='\t', header=None, index=False)    

# Step 1: Preprocess json trainset
files = [i for i in os.listdir('../../data_raw/qadi_country/train_json') if i[-4:] == 'json']
df = pd.DataFrame(columns={'original_sentence', 'dialect_country_id'})
for file in files:
    dialect = file[15:-5]
    with open('../../data_raw/qadi_country/train_json/' + file) as f:
        lines = f.read().splitlines()
    final = []
    sentences = []
    for l in lines:
        sentences.append(' '.join(json.loads(l)['full_text'].split()))
    df_temp = pd.DataFrame(columns={'original_sentence', 'dialect_country_id'})
    df_temp['original_sentence'] = sentences
    df_temp['dialect_country_id'] = dialect
    df = df.append(df_temp, ignore_index=True)


df.to_csv('../../data_raw/qadi_country/train_processed.tsv', index=False, sep='\t')


dp = DataProcess('../data_processed_splited/qadi/', 'user_level', 'twitter', 'http://alt.qcri.org/resources/qadi/', 'qadi', {},{},1, 'corpus', 'original')
dp.save_file('train_processed.tsv', dp.preprocess('../../data_raw/qadi_country/train_processed.tsv', '', '', 1, '', header=0))


dp = DataProcess('../data_processed_splited/qadi/', 'user_level', 'twitter', 'http://alt.qcri.org/resources/qadi/', 'qadi', {},{},0, 'corpus', 'original')
dp.save_file('test_processed.tsv', dp.preprocess('../../data_raw/qadi_country/test_processed.tsv', '', '', 1, '', header=0))


dp.standardize_labels('../data_processed_splited/qadi/test_processed.tsv', '../data_processed_splited/qadi/test_processed.tsv', ['country'])
dp.standardize_labels('../data_processed_splited/qadi/train_processed.tsv', '../data_processed_splited/qadi/train_processed.tsv', ['country'])


dp.save_features("../datasets_splited_features.tsv")


#Change Lybian region to Maghreb rather than Levant
folder = '../data_processed/qadi/'
files = os.listdir(folder)
for file in files:
    df = pd.read_csv(f'{folder}{file}', delimiter='\t', header=0)
    for index, row in df.iterrows():
        if row['dialect_country_id'] == 'ly':
            df.at[index,'dialect_region_id'] = 'maghreb'
    df.to_csv(f'{folder}{file}', sep='\t', index=False)
    
    
df[['dialect_country_id', 'dialect_region_id']].drop_duplicates()
from data_process import DataProcess

data_splits = pd.read_csv('../../data_raw/Gumar/data_splits.tsv', delimiter='\t', header=0, dtype={'DOC_ID': str})
data_splits.iloc[1231]['Splits']

train = data_splits.loc[data_splits['Splits'] == 'train']['DOC_ID'].unique()
dev = data_splits.loc[data_splits['Splits'] == 'dev']['DOC_ID'].unique()
test = data_splits.loc[data_splits['Splits'] == 'test']['DOC_ID'].unique()
NaN = data_splits.loc[data_splits['Splits'].isnull()]['DOC_ID'].unique()

files = os.listdir('../data_processed/gumar/')
for file in files:
    df = pd.read_csv('../data_processed/gumar/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'original'
    doc_id = file[10:15]
    if doc_id in train:
        df.to_csv('../data_processed_splited/gumar/train_'+file, sep='\t')
    elif doc_id in test:
        df.to_csv('../data_processed_splited/gumar/test_'+file, sep='\t')
    elif doc_id in dev:
        df.to_csv('../data_processed_splited/gumar/dev_'+file, sep='\t')
    elif doc_id in NaN:
        df.to_csv('../data_processed_splited/gumar/nan_'+file, sep='\t')


dp = DataProcess('../data_processed_splited/gumar/', 'document_level', 'forum_novel', 'https://camel.abudhabi.nyu.edu/gumar/?page=publications&lang=en', 'gumar', {},{},0, 'corpus', 'original')

dp.save_features('../datasets_splited_features.tsv')

df_dialect = pd.read_csv('../../data_raw/Gumar/dialect_info.tsv', delimiter='\t', header=0, dtype={'DOC_ID': str})
df_dialect


def preprocess_failures():
    folder = '../../data_raw/Gumar/data/'
    failures = ['Gumar_nvl_01145.txt', 'Gumar_nvl_00093.doc.txt', 'Gumar_nvl_00291.doc.txt', 'Gumar_nvl_01148.txt']
    for file in failures:
        df = pd.DataFrame(columns={'original_sentence'})
        with open(folder + file, 'r') as f:
            lines = f.read().splitlines()
        df['original_sentence'] = lines
        df = df.to_csv('../../data_raw/Gumar/data_labels/' + file[:-3] + 'tsv' , sep='\t', index=False)
        dp.save_file('nan_'+file[:-3] + 'tsv', dp.preprocess('../../data_raw/Gumar/data_labels/' + file[:-3] + 'tsv', '', '', '', '', header=0))


preprocess_failures()


def preprocess_gumar(): 
    folder = '../../data_raw/Gumar/data/'
    failures = ['Gumar_nvl_01145.txt', 'Gumar_nvl_00093.doc.txt', 'Gumar_nvl_00291.doc.txt', 'Gumar_nvl_01148.txt']
    files = [i for i in os.listdir(folder) if (i in failures) == False]
    for file in files:
        df = pd.DataFrame(columns={'original_sentence', 'dialect_country_id', 'dialect_region_id'})
        doc_id = file[10:15]
        try:
            dialect = df_dialect[df_dialect['DOC_ID'] == doc_id].iloc[0]['Dialect']
        except:
            failures.append(file)
            print("sad reacts")
            dialect = ''
        with open(folder + file, 'r') as f:
            lines = f.read().splitlines()
        df['original_sentence'] = lines
        df['dialect_country_id'] = dialect
        df = df.to_csv('../../data_raw/Gumar/data_labels/' + file[:-3] + 'tsv' , sep='\t', index=False)
    print('Files without dialect label ', failures)

preprocess_gumar()

folder = '../../data_raw/Gumar/data_labels/'
files = os.listdir(folder)
for file in files:  
    dp.save_file(file, dp.preprocess(folder+file, '', '', 1, '', header=0))

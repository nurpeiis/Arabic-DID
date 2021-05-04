from data_process import DataProcess

def preprocess(filename):
    df = pd.read_csv('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-2/{}.tsv'.format(filename), delimiter='\t')
    ar_letters = charsets.AR_LETTERS_CHARSET
    reg=re.compile('^[{}]+$'.format(ar_letters))
    for i in range(len(df.iloc[:, 5])):
        word = df.iloc[:, 5][i].split()
        line = ""
        for w in word:
            if reg.match(w):
                line += w + " "
        line = line[:-1]
        df.iloc[:, 5][i] = line
    df.to_csv('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-2/{}_preprocessed.tsv'.format(filename), sep='\t', index=False)
    return df


df_dev = preprocess('dev')
df_train = preprocess('train')
df_test = preprocess('test')


df_train = pd.read_csv('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-2/train_preprocessed.tsv', delimiter='\t')
df_dev = pd.read_csv('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-2/dev_preprocessed.tsv', delimiter='\t')
df_test = pd.read_csv('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-2/test_preprocessed.tsv', delimiter='\t')
cities = df_train.columns.values[3].replace('#4 Features ', '').split(',')


def more_preprocess(df_input, filename):
    df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_country_id'])
    def split_city(s):
        vals = [float(i) if i != '<NIL>'  else 0.0 for i in s.split(',')]
        index = np.where(vals == np.amax(vals))[0][0]
        return cities[index]
    df['dialect_city_id'] = df_input.iloc[:, 3].apply(split_city)
    df['original_sentence'] = df_input.iloc[:, 5]
    df['dialect_country_id'] = df_input.iloc[:, 4]
    df.to_csv('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-2/{}_processed.tsv'.format(filename), sep='\t', index=False)
    

more_preprocess(df_train, 'train')
more_preprocess(df_test, 'test')
more_preprocess(df_dev, 'dev')



dp = DataProcess('../data_processed_splited/madar_shared_task2/', 'user_level', 'twitter', 'https://camel.abudhabi.nyu.edu/madar-shared-task-2019/', 'madar_shared_task2', {},{},0, 'corpus', 'original')

files = os.listdir('../data_processed/madar_shared_task2/')
for file in files:
    df = pd.read_csv('../data_processed/madar_shared_task2/'+file, delimiter='\t', header=0, index_col=0)
    df['split_original_manual'] = 'original'
    df.to_csv('../data_processed_splited/madar_shared_task2/'+file, sep='\t')
    

dp.save_file('train_processed.tsv', dp.preprocess('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-2/train_processed.tsv', 1, '', 2, '', header=0))
dp.save_file('dev_processed.tsv', dp.preprocess('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-2/dev_processed.tsv', 1, '', 2, '', header=0))
dp.save_file('test_processed.tsv', dp.preprocess('../../data_raw/MADAR-SHARED-TASK/MADAR-Shared-Task-Subtask-2/test_processed.tsv', 1, '', 2, '', header=0))


dp.save_features('../datasets_splited_features.tsv')


#Change Tripoli to Lybian tripoli rather than Lebanon
folder = '../data_processed/madar_shared_task2/'
files = os.listdir(folder)
for file in files:
    df = pd.read_csv(f'{folder}{file}', delimiter='\t', header=0)
    for index, row in df.iterrows():
        if row['dialect_city_id'] == 'tripoli_west':
            row['dialect_city_id'] = 'tripoli'
            row['dialect_province_id'] = 'tripoli'
            row['dialect_country_id'] = 'ly'
            row['dialect_region_id'] = 'maghreb'
    df.to_csv(f'{folder}{file}', sep='\t', index=False)
    
    
#Change Ad Dahwan to Doha tripoli rather than Lebanon
folder = '../data_processed/madar_shared_task2/'
files = os.listdir(folder)
for file in files:
    df = pd.read_csv(f'{folder}{file}', delimiter='\t', header=0)
    for index, row in df.iterrows():
        if row['dialect_province_id'] == 'ad_dawhah':
            row['dialect_province_id'] = 'doha'
    df.to_csv(f'{folder}{file}', sep='\t', index=False)
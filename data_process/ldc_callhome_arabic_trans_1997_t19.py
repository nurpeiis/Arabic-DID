from data_process import DataProcess

def get_speakers_info():
    df_speaker_info = pd.read_csv('../../data_raw/ldc_callhome_arabic_trans_1997_t19/doc/spkrinfo.tbl', header=None)
    df_speaker_info.loc[df_speaker_info.iloc[:, 4] == 'Alexandria011203570was'].iloc[:, 4]
    df_speaker_info.at[20, 4] = 'Alexandria'
    df_speaker_info.loc[df_speaker_info.iloc[:, 4] == 'USA'].iloc[:, 4]
    df_speaker_info.at[5, 4] = ''
    df_speaker_info.at[22, 4] = ''
    df_speaker_info.loc[df_speaker_info.iloc[:, 4] == 'Kuwait'].iloc[:, 4]
    df_speaker_info.at[102, 4] = ''
    return df_speaker_info


def speakers_split(path_to_directory, filename):
    # Cleaning data and splitting between speaker A and B
    with open('{}/{}'.format(path_to_directory, filename), encoding='ISO-8859-6', errors="ignore") as f:
        try:
            lines_raw = f.read().splitlines()
        except:
            return False
        lines = [i for i in lines_raw if i] 
    ar_letters = charsets.AR_LETTERS_CHARSET
    reg=re.compile('^[{}]+$'.format(ar_letters))
    speakers = {'A': [], 'B': []}
    curr_speaker = 'A'
    for l in lines:
        word = l.split()
        line = ""
        for w in word:
            if w == 'A:' or w == 'B:' or reg.match(w):
                line += w + " "
        line = line[:-1]
        if line == 'A:':
            curr_speaker = 'A'
        elif line == 'B:':
            curr_speaker = 'B'
        #Get only lines that are purely in Arabic
        if re.match("[\(A-Za-z]", line) == None and line != '':
            speakers[curr_speaker].append(line)
    return speakers


def get_processed(speakers, city_a):
    df_processed = pd.DataFrame()
    df_processed['original_sentence'] = speakers['A']
    df_processed['dialect_city_id'] = city_a
    df_processed['dialect_country_id'] = 'Egypt'
    df_processed_b = pd.DataFrame()
    df_processed_b['original_sentence'] = speakers['B']
    df_processed_b['dialect_city_id'] = ''
    df_processed_b['dialect_country_id'] = 'Egypt'
    df_processed = df_processed.append(df_processed_b, ignore_index=True)
    return df_processed


def process_directory(path_to_directory):
    df = pd.DataFrame(columns={'original_sentence', 'dialect_city_id', 'dialect_country_id'})

    df_speakers = get_speakers_info()
    counter_bad = 0
    files = os.listdir(path_to_directory)
    for f in files:
        key = df_speakers.loc[df_speakers.iloc[:,0] == f[:-4]].iloc[:,4].keys()[0]
        city_a = df_speakers.loc[df_speakers.iloc[:,0] == f[:-4]].iloc[:,4][key]
        speakers = speakers_split(path_to_directory, f)
        if speakers:
            df = df.append(get_processed(speakers, city_a), ignore_index=True)
        else:
            counter_bad += 1
    print('Unable to process {} files in the following directory: {}'.format(counter_bad, path_to_directory))
    return df


df_train = process_directory('../../data_raw/ldc_callhome_arabic_trans_1997_t19/transcrp/train/script')
df_dev = process_directory('../../data_raw/ldc_callhome_arabic_trans_1997_t19/transcrp/devtest/script')
df_test = process_directory('../../data_raw/ldc_callhome_arabic_trans_1997_t19/transcrp/evaltest/script')


df_train.to_csv('../../data_raw/ldc_callhome_arabic_trans_1997_t19/train_processed.tsv', sep='\t', index=False)
df_test.to_csv('../../data_raw/ldc_callhome_arabic_trans_1997_t19/test_processed.tsv', sep='\t', index=False)
df_dev.to_csv('../../data_raw/ldc_callhome_arabic_trans_1997_t19/dev_processed.tsv', sep='\t', index=False)


dp = DataProcess('../data_processed_splited/ldc_callhome_arabic_trans_1997_t19/', 'user_level', 'speech_transcript', 'https://catalog.ldc.upenn.edu/LDC97T19', 'ldc_callhome_arabic_trans_1997_t19', {},{},1, 'corpus', 'original')


dp.save_file('dev.tsv', dp.preprocess('../../data_raw/ldc_callhome_arabic_trans_1997_t19/dev_processed.tsv', 0, '', 2, '', header=0))
dp.save_file('train.tsv', dp.preprocess('../../data_raw/ldc_callhome_arabic_trans_1997_t19/train_processed.tsv', 0, '', 2, '', header=0))
dp.save_file('test.tsv', dp.preprocess('../../data_raw/ldc_callhome_arabic_trans_1997_t19/test_processed.tsv', 0, '', 2, '', header=0))


df_train = pd.read_csv('../data_processed/ldc_callhome_arabic_trans_1997_t19/train.tsv', delimiter='\t', header=0)
df_train['split_original_manual'] = 'original'
df_test = pd.read_csv('../data_processed/ldc_callhome_arabic_trans_1997_t19/test.tsv', delimiter='\t', header=0)
df_test['split_original_manual'] = 'original'
df_dev = pd.read_csv('../data_processed/ldc_callhome_arabic_trans_1997_t19/dev.tsv', delimiter='\t', header=0)
df_dev['split_original_manual'] = 'original'
df_train.to_csv('../data_processed_splited/ldc_callhome_arabic_trans_1997_t19/train_processed.tsv', sep='\t', index=False)
df_test.to_csv('../data_processed_splited/ldc_callhome_arabic_trans_1997_t19/test_processed.tsv', sep='\t',index=False)
df_dev.to_csv('../data_processed_splited/ldc_callhome_arabic_trans_1997_t19/dev_processed.tsv', sep='\t',index=False)



dp.save_features('../datasets_splited_features.tsv')
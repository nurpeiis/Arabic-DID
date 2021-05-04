from data_process import DataProcess

def get_speakers_info():
    df= pd.read_csv('../../data_raw/ldc_arb_lev_cttr_2007_t01/docs/training_calls.tab', delimiter='\t') 
    df_speaker_info = pd.DataFrame()
    df_speaker_info['filename'] = df['File-ID'].apply(lambda x: x.replace('-', '_'))    
    df_speaker_info['A'] = df['A-Region']  
    df_speaker_info['B'] = df['B-Region']
    df_devtest= pd.read_csv('../../data_raw/ldc_arb_lev_cttr_2007_t01/docs/devtest_calls.tab', delimiter='\t') #in line 4 of the file delete extra tabs from Places to go
    df_speaker_info_devtest = pd.DataFrame()
    df_speaker_info_devtest['filename'] = df_devtest['File-ID'].apply(lambda x: x.replace('-', '_'))       
    df_speaker_info_devtest['A'] = df_devtest['A-Region']  
    df_speaker_info_devtest['B'] = df_devtest['B-Region']
    df_speaker_info = df_speaker_info.append(df_speaker_info_devtest, ignore_index=True)
    return df_speaker_info


def speakers_split(path_to_directory, filename):
    # Cleaning data and splitting between speaker A and B
    with open('{}/{}'.format(path_to_directory, filename)) as f:
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
        if 'A:' in line:
            curr_speaker = 'A'
            line = line.replace('A:','')
        elif 'B:' in line:
            curr_speaker = 'B'            
            line = line.replace('B:','')
        line = line.replace('tnfs', '')
        #Get only lines that are purely in Arabic
        line = line.replace('(', '')        
        line = line.replace(')', '')
        if re.match("[\(A-Za-z]", line) == None and line != '':
            speakers[curr_speaker].append(line)
    return speakers


def get_processed(speakers, country_a, country_b):
    df_processed = pd.DataFrame()
    df_processed['original_sentence'] = speakers['A']
    df_processed['dialect_country_id'] = country_a
    df_processed['dialect_region_id'] = 'Levantine'
    df_processed_b = pd.DataFrame()
    df_processed_b['original_sentence'] = speakers['B']
    df_processed_b['dialect_country_id'] = country_b
    df_processed_b['dialect_region_id'] = 'Levantine'

    df_processed = df_processed.append(df_processed_b, ignore_index=True)
    return df_processed


def process_directory(path_to_directory):
    df = pd.DataFrame(columns={'original_sentence', 'dialect_country_id', 'dialect_region_id'})

    df_speakers = get_speakers_info()
    counter_bad = 0
    files = os.listdir(path_to_directory)
    for f in files:
        try:
            country_a = df_speakers.loc[df_speakers["filename"] == f[:-4].replace('-', '_')].iloc[0]['A']
            country_b = df_speakers.loc[df_speakers["filename"] == f[:-4].replace('-', '_')].iloc[0]['B']
            speakers = speakers_split(path_to_directory, f)
            if speakers:
                df = df.append(get_processed(speakers, country_a, country_b), ignore_index=True)
            else:
                counter_bad += 1
        except:
            print("Bad", f)
            continue
    print('Unable to process {} files in the following directory: {}'.format(counter_bad, path_to_directory))
    return df


df_train = process_directory('../../data_raw/ldc_arb_lev_cttr_2007_t01/data/transc/train2c')
df_train.to_csv('../../data_raw/ldc_arb_lev_cttr_2007_t01/train_processed.tsv', sep='\t', index=False)


df_devtest = process_directory('../../data_raw/ldc_arb_lev_cttr_2007_t01/data/transc/devtest')
df_devtest.to_csv('../../data_raw/ldc_arb_lev_cttr_2007_t01/devtest_processed.tsv', sep='\t', index=False)


dp = DataProcess('../data_processed/ldc_arb_lev_cttr_2007_t01/', 'user_level', 'speech_transcript', 'https://catalog.ldc.upenn.edu/LDC2007T01', 'ldc_arb_lev_cttr_2007_t01', {},{},1, 'corpus', 'original')

dp.save_file('train_processed.tsv', dp.preprocess('../../data_raw/ldc_arb_lev_cttr_2007_t01/train_processed.tsv', '', '', 0, 2, header=0))
dp.save_file('devtest_processed.tsv', dp.preprocess('../../data_raw/ldc_arb_lev_cttr_2007_t01/devtest_processed.tsv', '', '', 0, 2, header=0))


df_train = pd.read_csv('../data_processed/ldc_arb_lev_cttr_2007_t01/train_processed.tsv', delimiter='\t', header=0)
df_train['split_original_manual'] = 'original'
df_test = pd.read_csv('../data_processed/ldc_arb_lev_cttr_2007_t01/devtest_processed.tsv', delimiter='\t', header=0)
df_test['split_original_manual'] = 'original'
df_train.to_csv('../data_processed_splited/ldc_arb_lev_cttr_2007_t01/train_processed.tsv', sep='\t', index=False)
df_test.to_csv('../data_processed_splited/ldc_arb_lev_cttr_2007_t01/devtest_processed.tsv', sep='\t',index=False)


dp.save_features('../datasets_splited_features.tsv')
from data_process import DataProcess

def get_speakers_info():
    df= pd.read_csv('../../data_raw/ldc_arb_iraq_cttr_2006_t16/docs/training.tab', delimiter='\t', header=None) 
    df_speaker_info = pd.DataFrame()
    df_speaker_info['filename'] = df.iloc[:,0]   
    df_speaker_info['A'] = df.iloc[:,4].apply(lambda x: '' if x == '(na)' else x)  
    df_speaker_info['B'] = df.iloc[:,9].apply(lambda x: '' if x == '(na)' else x)    
    df_devtest= pd.read_csv('../../data_raw/ldc_arb_iraq_cttr_2006_t16/docs/dev_test.tab', delimiter='\t', header=None) #in line 4 of the file delete extra tabs from Places to go
    df_speaker_info_devtest = pd.DataFrame()
    df_speaker_info_devtest['filename'] = df_devtest.iloc[:,0]     
    df_speaker_info_devtest['A'] = df_devtest.iloc[:,4].apply(lambda x: '' if x == '(na)' else x)     
    df_speaker_info_devtest['B'] = df_devtest.iloc[:,9].apply(lambda x: '' if x == '(na)' else x)    
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
        line = line.replace('free', '')
        line = line.replace('roster', '')
        #print(line)
        #Get only lines that are purely in Arabic
        line = line.replace('(', '')        
        line = line.replace(')', '')
        #print(line)
        if re.match("[\(A-Za-z]", line) == None and line != '':
            speakers[curr_speaker].append(line)
    return speakers


def get_processed(speakers, province_a, province_b):
    df_processed = pd.DataFrame()
    df_processed['original_sentence'] = speakers['A']
    df_processed['dialect_province_id'] = province_a
    df_processed['dialect_country_id'] = 'Iraq'
    df_processed_b = pd.DataFrame()
    df_processed_b['original_sentence'] = speakers['B']
    df_processed_b['dialect_province_id'] = province_b
    df_processed_b['dialect_country_id'] = 'Iraq'

    df_processed = df_processed.append(df_processed_b, ignore_index=True)
    return df_processed


def process_directory(path_to_directory):
    df = pd.DataFrame(columns={'original_sentence', 'dialect_country_id', 'dialect_province_id'})

    df_speakers = get_speakers_info()
    counter_bad = 0
    files = os.listdir(path_to_directory)
    for f in files:
        try:
            province_a = df_speakers.loc[df_speakers["filename"] == f[:-4]].iloc[0]['A']
            province_b = df_speakers.loc[df_speakers["filename"] == f[:-4]].iloc[0]['B']
            speakers = speakers_split(path_to_directory, f)
            if speakers:
                df = df.append(get_processed(speakers, province_a, province_b), ignore_index=True)
            else:
                counter_bad += 1
        except:
            print("Bad", f)
            continue
    print('Unable to process {} files in the following directory: {}'.format(counter_bad, path_to_directory))
    return df


df_train = process_directory('../../data_raw/ldc_arb_iraq_cttr_2006_t16/data/transc/train2c')
df_train.to_csv('../../data_raw/ldc_arb_iraq_cttr_2006_t16/train_processed.tsv', sep='\t', index=False)

df_devtest = process_directory('../../data_raw/ldc_arb_iraq_cttr_2006_t16/data/transc/devtest')
df_devtest.to_csv('../../data_raw/ldc_arb_iraq_cttr_2006_t16/devtest_processed.tsv', sep='\t', index=False)

dp = DataProcess('../data_processed/ldc_arb_iraq_cttr_2006_t16/', 'user_level', 'speech_transcript', 'https://catalog.ldc.upenn.edu/LDC2006T16', 'ldc_arb_iraq_cttr_2006_t16', {},{},1, 'corpus')

dp.save_file('train_processed.tsv', dp.preprocess('../../data_raw/ldc_arb_iraq_cttr_2006_t16/train_processed.tsv', '', 2, 0, '', header=0))
dp.save_file('devtest_processed.tsv', dp.preprocess('../../data_raw/ldc_arb_iraq_cttr_2006_t16/devtest_processed.tsv', '', 2, 0, '', header=0))

dp.save_features('../datasets_features.tsv')
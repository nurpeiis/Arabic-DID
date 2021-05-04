from data_process import DataProcess

def speakers_split(path_to_directory, filename):
    # Cleaning data and splitting between speaker A and B
    with open('{}/{}'.format(path_to_directory, filename), encoding='ISO-8859-6', errors='ignore') as f:
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


def get_processed(speakers):
    df_processed = pd.DataFrame()
    df_processed['original_sentence'] = speakers['A']
    df_processed['dialect_country_id'] = 'eg'
    df_processed['dialect_region_id'] = 'nile_basin'

    df_processed_b = pd.DataFrame()
    df_processed_b['original_sentence'] = speakers['B']
    df_processed_b['dialect_country_id'] = 'eg'
    df_processed_b['dialect_region_id'] = 'nile_basin'


    df_processed = df_processed.append(df_processed_b, ignore_index=True)
    return df_processed


def process_directory(path_to_directory):
    df = pd.DataFrame(columns={'original_sentence', 'dialect_country_id', 'dialect_region_id'})
    counter_bad = 0
    files = os.listdir(path_to_directory)
    for f in files:
        if (f[-3:] == "scr"):
            speakers = speakers_split(path_to_directory, f)
            if speakers:
                df = df.append(get_processed(speakers), ignore_index=True)
            else:
                counter_bad += 1
    print('Unable to process {} files in the following directory: {}'.format(counter_bad, path_to_directory))
    return df


df = process_directory('../../data_raw/ldc_callhome_ara_transcr_suppl_2002_t38/transcr')


df.to_csv('../../data_raw/ldc_callhome_ara_transcr_suppl_2002_t38/processed.tsv', sep='\t', index=False)


dp = DataProcess('../data_processed_splited/ldc_callhome_ara_transcr_suppl_2002_t38/', 'user_level', 'speech_transcript', 'https://catalog.ldc.upenn.edu/LDC2002T38', 'ldc_callhome_ara_transcr_suppl_2002_t38', {},{},1, 'corpus', 'manual')


dp.save_file('processed.tsv', dp.preprocess('../../data_raw/ldc_callhome_ara_transcr_suppl_2002_t38/processed.tsv', '', '', 2, 0, header=0))


dp.save_features('../datasets_splited_features.tsv')


files = ['processed.tsv']
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.10, 0.10)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)
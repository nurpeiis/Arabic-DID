from data_process import DataProcess

def get_speakers_info():
    with open('../../data_raw/ldc_la_cts_ann_2007_t04/docs/speaker_info_new.txt') as f: #removed description
        lines = f.read().splitlines()
    df_speaker_info = pd.DataFrame(columns={"filename", "A", "B"})
    for l in lines:
        try:
            filename = l.split()[0][:-1]
            A = l.split()[1].split(':')[1] if l.split()[1].split(':')[1] != 'Lev' else ''
            B = l.split()[1].split(':')[5] if l.split()[1].split(':')[5] != 'Lev' else ''
            file_dict = {"filename": filename, 'A': A, 'B': B}
            df_speaker_info = df_speaker_info.append(file_dict, ignore_index=True)
        except:
            continue
    return df_speaker_info
#get_speakers_info()['B'].unique()


def speakers_split(path_to_directory, filename):
    # Cleaning data and splitting between speaker A and B
    with open('{}/{}'.format(path_to_directory, filename)) as f:
        try:
            lines_raw = f.read().splitlines()
        except:
            return False
        lines = [i for i in lines_raw if i] 
    ar_letters = charsets.AR_LETTERS_CHARSET
    reg=re.compile('^[{}\^]+$'.format(ar_letters))
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
        #print(line)
        #Get only lines that are purely in Arabic
        line = line.replace('(', '')        
        line = line.replace(')', '')
        #print(line)
        if re.match("[\(A-Za-z]", line) == None and line != '':
            speakers[curr_speaker].append(line)
    return speakers
#speakers_split('../../data_raw/ldc_la_cts_ann_2007_t04/data/annotation', 'fla_0001.txt')


def get_processed(speakers, country_a, country_b):
    df_processed = pd.DataFrame()
    df_processed['original_sentence'] = speakers['A']
    df_processed['dialect_country_id'] = country_a
    df_processed['dialect_region_id'] = 'Levantine' if (country_a != 'IRQ' and country_a != 'YEM') else ''
    df_processed_b = pd.DataFrame()
    df_processed_b['original_sentence'] = speakers['B']
    df_processed_b['dialect_country_id'] = country_b
    df_processed_b['dialect_region_id'] = 'Levantine'  if (country_b != 'IRQ' and country_b != 'YEM') else ''

    df_processed = df_processed.append(df_processed_b, ignore_index=True)
    return df_processed


def process_directory(path_to_directory):
    df = pd.DataFrame(columns={'original_sentence', 'dialect_country_id', 'dialect_region_id'})

    df_speakers = get_speakers_info()
    counter_bad = 0
    files = os.listdir(path_to_directory)
    for f in files:
        try:
            country_a = df_speakers.loc[df_speakers["filename"] == f].iloc[0]['A']
            country_b = df_speakers.loc[df_speakers["filename"] == f].iloc[0]['B']
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


df = process_directory('../../data_raw/ldc_la_cts_ann_2007_t04/data/annotation')
df.to_csv('../../data_raw/ldc_la_cts_ann_2007_t04/processed.tsv', sep='\t', index=False)


dp = DataProcess('../data_processed_splited/ldc_la_cts_ann_2007_t04/', 'user_level', 'speech_transcript', 'https://catalog.ldc.upenn.edu/LDC2007T04', 'ldc_la_cts_ann_2007_t04', {},{},2, 'corpus', 'manual')


df = pd.read_csv('../data_processed/ldc_la_cts_ann_2007_t04/processed.tsv', delimiter='\t', header=0, index_col=0)
df['split_original_manual'] = 'manual'
df.to_csv('../data_processed_splited/ldc_la_cts_ann_2007_t04/processed.tsv', sep='\t')
files = ['processed.tsv']
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.10, 0.10)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)
    
    
dp.save_file('processed.tsv', dp.preprocess('../../data_raw/ldc_la_cts_ann_2007_t04/processed.tsv', '', '', 1, 0, header=0))

dp.save_features('../datasets_splited_features.tsv')
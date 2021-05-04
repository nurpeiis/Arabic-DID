from data_process import DataProcess

#Use Nasser's splits
def get_sentences(filename):
    sentences = []
    trns = tsl.Transliterator(
                chmap.CharMapper.builtin_mapper('bw2ar'))
    with open('../../data_raw/lev_nasser/'+filename, 'r') as f:
        lines = f.read().splitlines()
        for i in range(len(lines)):
            if 'SENTENCE_ID' in lines[i]:
                if 'SENTENCE_ID' in lines[i+1]:
                    sentence = lines[i+1].replace(';;; SENTENCE_ID ', '')
                    sentence = trns.transliterate(sentence)
                    sentences.append(sentence)
    df = pd.DataFrame(columns = ['original_sentence', 'dialect_country_id', 'dialect_region_id'])
    df['original_sentence'] = sentences
    df['dialect_country_id'] = 'ps'
    df['dialect_region_id'] = 'levant'
    return df


files = [file for file in os.listdir('../../data_raw/lev_nasser/') if file[-6:] =='magold']
for file in files:
    get_sentences(file).to_csv('../../data_raw/lev_nasser/'+file+'.tsv', sep='\t', index=False)


dp = DataProcess('../data_processed_splited/curras/', 'manual_search', 'web_mixed', 'https://link.springer.com/article/10.1007/s10579-016-9370-7#Tab1', 'curras', {},{},0, 'corpus', 'original')

for file in files:
    dp.save_file(file+'.tsv', dp.preprocess('../../data_raw/lev_nasser/'+file+'.tsv', '', '', 1, 2, header=0))

dp.save_features('../datasets_splited_features.tsv')

with open('../../data_raw/Curras-Arabic-NYH/Curras.all.txt') as file:
    lines = [i for i in file.read().splitlines() if i != '']
final = []
ar_letters = charsets.AR_LETTERS_CHARSET
reg=re.compile('^[{}]+$'.format(ar_letters))
for l in lines:
    word = l.split()
    line = ""
    for w in word:
        if reg.match(w):
            line += w + " "
    line = line[:-1]
    if line != '' and not ('on' in line):
        final.append(line)
    
df = pd.DataFrame(final)
df.to_csv('../../data_raw/Curras-Arabic-NYH/processed.tsv', sep='t', index=False, header=None)
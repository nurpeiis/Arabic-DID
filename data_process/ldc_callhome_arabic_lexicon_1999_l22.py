from data_process import DataProcess

#Step 0: Get only unique lexems and save them into the words.tsv
df = pd.read_csv('../../data_raw/ldc_callhome_arabic_lexicon_1999_l22/ar_lex.v07', encoding='iso-8859-6', delimiter='\t', header=None)
df = pd.DataFrame(df.iloc[:,1].unique())
df.to_csv('../../data_raw/ldc_callhome_arabic_lexicon_1999_l22/words.tsv', sep='t', header=None, index=False)

#Step 1: Create an object of DataProcess class in order to preclean and preprocess the original dataset

dp = DataProcess('../data_processed/ldc_callhome_arabic_lexicon_1999_l22/', 'word_level', 'speech_transcript', 'https://catalog.ldc.upenn.edu/LDC99L22', 'ldc_callhome_arabic_lexicon_1999_l22', {},{},0, 'lexicon')


dp.save_file('lexicon.tsv', dp.preprocess('../../data_raw/ldc_callhome_arabic_lexicon_1999_l22/words.tsv', '', '', 'Egypt', '', header=None))
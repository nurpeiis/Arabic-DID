from data_process import DataProcess
import pandas as pd
files = ['EGY', 'GULF', 'LEV', 'NorthAfrica']
folder_name = '../../data_raw/ArabicMultiDialectTextCorpora/'
for f in files:
    with open(folder_name + f) as r:
        lines = r.read().splitlines()

    # Remove lines that contain NEW_PAGE
    lines = lines[lines.index('NEW_PAGE')+1:]
    print("Before", len(lines))
    lines = list(filter(lambda a: a != 'NEW_PAGE', lines))
    print("After", len(lines))
    df = pd.DataFrame(lines)
    df.to_csv(folder_name + f + '_cleaned.tsv',
              sep='\t', index=False, header=None)

dp = DataProcess('../data_processed/arabic_multi_dialect_text_corpora/', 'word_level', 'web_mixed', 'http://almeman.weebly.com/arabic-multi-dialect-text-corpora.html',
                 'arabic_multi_dialect_text_corpora', {}, {}, 0, 'corpus', 'manual')
# Map dialects based on file names
dialect_map = {'EGY': ['eg', 'nile_basin'], 'GULF': [
    '', 'gulf'], 'LEV': ['', 'levant'], 'NorthAfrica': ['', 'maghreb']}
for f in files:
    dp.save_file(f + '_cleaned.tsv', dp.preprocess(folder_name + f +
                 '_cleaned.tsv', '', '', dialect_map[f][0], dialect_map[f][1], header=None))
# Manually split the data
for file in files:
    df_train, df_dev, df_test = dp.split(file + '_cleaned.tsv', 0.8, 0.1, 0.1)
    dp.save_file('train_'+file+'.tsv', df_train)
    dp.save_file('dev_'+file+'.tsv', df_dev)
    dp.save_file('test_'+file+'.tsv', df_test)

dp.save_features('../datasets_splited_features.tsv')

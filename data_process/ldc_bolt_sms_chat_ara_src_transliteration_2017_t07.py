from data_process import DataProcess
import xml.etree.ElementTree as et 

train_set = set()
test_set = set()
dev_set = set()
with open('../test.txt', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        test_set.add(line)
with open('../train.txt', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        train_set.add(line)
with open('../dev.txt', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        dev_set.add(line)
        


#Step 0: From xml to csv
files = os.listdir('../../data_raw/ldc_bolt_sms_chat_ara_src_transliteration_2017_t07/data/transliteration')

cols = ["original_sentence"]
df_train = pd.DataFrame(columns=cols)
df_test = pd.DataFrame(columns=cols)
df_dev = pd.DataFrame(columns=cols)



for f in files:
    xtree = et.parse('../../data_raw/ldc_bolt_sms_chat_ara_src_transliteration_2017_t07/data/transliteration/' + f)
    xroot = xtree.getroot()
    rows = []
    for node in xroot:
        if node.find("corrected_transliteration") != None:
            rows.append({"original_sentence": node.find("corrected_transliteration").text})
    if f in train_set:
        df_train = df_train.append(rows, ignore_index=True)
    elif f in test_set:
        df_test = df_test.append(rows, ignore_index=True)
    elif f in dev_set:
        df_dev = df_dev.append(rows, ignore_index=True)



df_dev.to_csv('../../data_raw/ldc_bolt_sms_chat_ara_src_transliteration_2017_t07/dev_processed.tsv', sep='\t', index=False)
df_test.to_csv('../../data_raw/ldc_bolt_sms_chat_ara_src_transliteration_2017_t07/test_processed.tsv', sep='\t', index=False)
df_train.to_csv('../../data_raw/ldc_bolt_sms_chat_ara_src_transliteration_2017_t07/train_processed.tsv', sep='\t', index=False)

# Create an object of DataProcess class in order to preclean and preprocess the original dataset
dp = DataProcess('../data_processed_splited/ldc_bolt_sms_chat_ara_src_transliteration_2017_t07/', 'user_level', 'sms', 'https://catalog.ldc.upenn.edu/LDC2017T07', 'ldc_bolt_sms_chat_ara_src_transliteration_2017_t07', {},{},0, 'corpus', 'original')

dp.save_file('dev_processed.tsv', dp.preprocess('../../data_raw/ldc_bolt_sms_chat_ara_src_transliteration_2017_t07/dev_processed.tsv', '', '', 'eg', 'nile_basin', header=0))
dp.save_file('test_processed.tsv', dp.preprocess('../../data_raw/ldc_bolt_sms_chat_ara_src_transliteration_2017_t07/test_processed.tsv', '', '', 'eg', 'nile_basin', header=0))
dp.save_file('train_processed.tsv', dp.preprocess('../../data_raw/ldc_bolt_sms_chat_ara_src_transliteration_2017_t07/train_processed.tsv', '', '', 'eg', 'nile_basin', header=0))

dp.save_features('../datasets_splited_features.tsv')
from data_process import DataProcess
import json

files = os.listdir('../../data_raw/qadi_country/dataset')
for file in files:
    os.system('twarc hydrate ../../data_raw/qadi_country/dataset/' + file + '> ../../data_raw/qadi_country/train_json/' + file[:-3] + 'json')
    

# Step 1: Preprocess json trainset
files = [i for i in os.listdir('../../data_raw/qadi_country/train_json') if i[-4:] == 'json']
df = pd.DataFrame(columns={'original_sentence', 'dialect_country_id'})
for file in files:
    dialect = file[15:-5]
    with open('../../data_raw/qadi_country/train_json/' + file) as f:
        lines = f.read().splitlines()
    final = []
    sentences = []
    for l in lines:
        final.append(json.loads(l))
        sentences.append(json.loads(l)['full_text'])
    df_temp = pd.DataFrame(columns={'original_sentence', 'dialect_country_id'})
    df_temp['original_sentence'] = sentences
    df_temp['dialect_country_id'] = dialect
    df = df.append(df_temp, ignore_index=True)
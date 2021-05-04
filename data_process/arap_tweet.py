from data_process import DataProcess

country_map = {'ae': ['ae', 'gulf'],
               'algeria': ['dz', 'maghreb'],
               'ba': ['bh', 'gulf'],
               'bh': ['bh', 'gulf'],
               'bahrain': ['bh', 'gulf'],
               'dz': ['dz', 'maghreb'],
               'djibouti': ['dj', 'gulf_aden'],
               'eg': ['eg', 'nile_basin'],
               'egy': ['eg', 'nile_basin'],
               'egypt': ['eg', 'nile_basin'],
               'egyptian': ['eg', 'nile_basin'],
               'ga': ['gulf', 'gulf'],
               'iq': ['iq', 'iraq'],
               'irq': ['iq', 'iraq'],
               'iraq': ['iq', 'iraq'],
               'jo': ['jo', 'levant'],
               'jor': ['jo', 'levant'],
               'jordan': ['jo', 'levant'],
               'kw': ['kw', 'gulf'],
               'kuwait': ['kw', 'gulf'],
               'lb': ['lb', 'levant'],
               'leb': ['lb', 'levant'],
               'lev': ['levant', 'levant'],
               'ly': ['ly', 'levant'],
               'lebanees': ['lb', 'levant'],
               'lebanon': ['lb', 'levant'],
               'lebanon syria': ['lb, sy', 'levant'],
               'libya': ['ly', 'maghreb'],
               'ma': ['ma', 'maghreb'],
               'mar': ['ma', 'maghreb'],
               'mixed': ['mixed', 'mixed'],
               'mor': ['ma', 'maghreb'],
               'msa': ['msa', 'msa'],
               'msa (translated)': ['msa', 'msa'],
               'mauritania': ['mr', 'maghreb'],
               'morocco': ['ma', 'maghreb'],
               'morroco': ['ma', 'maghreb'],
               'om': ['om', 'gulf'],
               'oman': ['om', 'gulf'],
               'pa': ['ps', 'levant'],
               'pal': ['ps', 'levant'],
               'pl': ['ps', 'levant'],
               'palestine': ['ps', 'levant'],
               'palestinejordan': ['ps,jo', 'levant'],
               'palestinian': ['ps', 'levant'],
               'qa': ['qa', 'gulf'],
               'qatar': ['qa', 'gulf'],
               'sa': ['sa', 'gulf'],
               'sd': ['sd', 'nile_basin'],
               'sud': ['sd', 'nile_basin'],
               'sy': ['sy', 'levant'],
               'syr': ['sy', 'levant'],
               'saudi': ['sa', 'gulf'],
               'saudi arabia': ['sa', 'gulf'],
               'saudi_arabia': ['sa', 'gulf'],
               'somalia': ['so', 'nile_basin'],
               'sudan': ['sd', 'nile_basin'],
               'syria': ['sy', 'levant'],
               'tn': ['tn', 'maghreb'],
               'tun': ['tn', 'maghreb', 'maghreb'],
               'tunisia': ['tn', 'maghreb'],
               'uae': ['ae', 'gulf'],
               'united_arab_emirates': ['ae', 'gulf'],
               'ye': ['ye', 'gulf_aden'],
               'yem': ['ye', 'gulf_aden'],
               'yemen': ['ye', 'gulf_aden'],
               'jordinian': ['jo', 'levant'],
               'msa': ['msa', 'msa'],
               'nan': [],
               '': [],
               'syrian': ['sy', 'levant']}

def get_processed(folder):
    files = [(i) for i in os.listdir('../../data_raw/ArapTweet_Set_of_102/' +folder ) if i[-3:] == 'xml']
    cols = ['original_sentence', 'dialect_country_id', 'dialect_region_id']
    df = pd.DataFrame(columns=cols)
    for file in files:
        xtree = et.parse('../../data_raw/ArapTweet_Set_of_102/{}/{}'.format(folder,file))
        xroot = xtree.getroot()
        rows = []
        for node in xroot.find("documents"):
            l = node.text
            rows.append({"original_sentence": l})
        df = df.append(rows, ignore_index=True)
    correct_labels = country_map[folder.lower()]
    df['dialect_country_id'] = correct_labels[0]
    df['dialect_region_id'] = correct_labels[1]


    return df

folders = [(i) for i in os.listdir('../../data_raw/ArapTweet_Set_of_102') if i!=".DS_Store" and i!='Corpus-Stats.txt' and i != 'processed' and i != 'processed.tsv']
cols = ['original_sentence', 'dialect_country_id', 'dialect_region_id']
for folder in folders:
    print(folder)
    df_preprocessed = get_processed(folder)
    df_preprocessed.to_csv('../../data_raw/ArapTweet_Set_of_102/processed/{}.tsv'.format(folder), sep='\t', index=False)
df_preprocessed



dp = DataProcess('../data_processed_splited/arap_tweet/', 'user_level', 'twitter', 'https://www.aclweb.org/anthology/L18-1111/', 'arap_tweet', {},{},0, 'corpus', 'manual')

files = [i for i in os.listdir('../../data_raw/ArapTweet_Set_of_102/processed/') if i!=".DS_Store" and i!='Corpus-Stats.txt' and i != 'processed' and i != 'processed.tsv']
for file in files:
    dp.save_file(file, dp.preprocess('../../data_raw/ArapTweet_Set_of_102/processed/{}'.format(file), '', '', 1, 2, header=0))

    #Arap tweet does not have split in the paper, so for each dialect do 80/10/10 manual split
for file in files:
    df_train, df_dev, df_test = dp.split(file, 0.8, 0.1, 0.1)
    dp.save_file('train_'+file, df_train)    
    dp.save_file('dev_'+file, df_dev)
    dp.save_file('test_'+file, df_test)

dp.save_features('../datasets_splited_features.tsv')
from data_process import DataProcess

folder = '../../data_raw/arb_gw_5/data/'
sub_folders = [i for i in os.listdir(folder) if ('DS_Store' in i) == False]
for sub_folder in sub_folders:
    # unzip all files
    files = [i for i in os.listdir(f'{folder}{sub_folder}') if ('DS_Store' in i) == False and '.gz' in i]
    for file in files:
        with gzip.open(f'{folder}{sub_folder}/{file}', 'rb') as f_in:
            with open(f'{folder}{sub_folder}/{file[:-3]}', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    print(f'{sub_folder} done')
    os.system(f'rm {folder}{sub_folders[0]}/*.gz')

files = [i for i in os.listdir(f'{folder}{sub_folders[0]}') if ('DS_Store' in i) == False]

folder = '../../data_raw/arb_gw_5/data/'
sub_folders = [i for i in os.listdir(folder) if ('DS_Store' in i) == False and ('final' in i) == False]

for sub_folder in sub_folders:
    # unzip all files
    files = [i for i in os.listdir(f'{folder}{sub_folder}') if ('DS_Store' in i) == False]
    print(sub_folder)
    for file in files:
        with open(f'{folder}{sub_folder}/{file}') as f:
            lines = f.read().splitlines()
        sentences = dict()
        counter = 0
        for i in range(len(lines)):
            if lines[i] == '<HEADLINE>':
                counter += 1
                sentences[counter] = []
                sentences[counter].append(lines[i+1])
            if lines[i] == '<P>':
                i = i+1
                while lines[i] != '</P>':
                    if (counter in sentences) == False:
                        sentences[counter] = []
                    sentences[counter].append(lines[i])
                    i = i + 1
            
        for key in sentences.keys():
            with open(f'{folder}final/{sub_folder}_{key}', 'w') as f:
                for item in sentences[key]:
                    f.write("%s\n" % item)


folder = '../../data_raw/arb_gw_5/data/'
sub_folders = [i for i in os.listdir(folder) if ('DS_Store' in i) == False and ('final' in i) == False]


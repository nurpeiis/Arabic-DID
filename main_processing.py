import os
import pandas as pd

# Create 9 dataframes for storing the data from 29 datasets
dev_city_df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id','dataset_name'])
dev_country_df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id','dataset_name'])
dev_region_df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id','dataset_name'])
test_city_df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id','dataset_name'])
test_country_df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id','dataset_name'])
test_region_df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id','dataset_name'])
train_city_df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id','dataset_name'])
train_country_df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id','dataset_name'])
train_region_df = pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id','dataset_name'])


cwd = os.getcwd()
print("cwd: ", cwd)
datasets = os.listdir(cwd)

# This is a path to the directory with all 29 datasets.
datasets_path = '/Users/alemshaimardanov/PycharmProjects/NLP/arabic_did/data_processed_second'

# Save the list of datasets in 'unprocessed_datasets'
unprocessed_datasets = os.listdir(datasets_path)
print(unprocessed_datasets)
i = 1
unprocessed_datasets_length = len(unprocessed_datasets)
# Loop through all 29 datasets
for ud in unprocessed_datasets:
    # Make sure you are working with correct folders that do not start with a dot.
    if ud[0] != '.':
        print(i, "/", unprocessed_datasets_length - 2, ": ", ud, '\n')
        curDatasetPath = datasets_path + '/' + ud
        print(curDatasetPath)
        i += 1

        # Process current dataset:
        curDataset_files = os.listdir(curDatasetPath)

        # Get the name of the dataset from info_data.txt file's first line
        curDataset_info_data_path = curDatasetPath + '/info_data.txt'
        curDataset_info_data_file = open(curDataset_info_data_path, "r")
        # dataset_name is located after the whitespace on the first line
        curDataset_NAME = curDataset_info_data_file.readline().split()[1]
        curDataset_info_data_file.close()

        # Loop through test_,dev_,train_ files of the current dataset
        for curDataset_file in curDataset_files:
            print("Current DATASET: ", ud, ", Current FILE: ", curDataset_file)
            curFile_path = curDatasetPath + '/' + curDataset_file

            # Initialize check variables
            isTestPresent = False
            isTestCity_present = False
            isTestCountry_present = False
            isTestRegion_present = False

            isTrainPresent = False
            isTrainCity_present = False
            isTrainCountry_present = False
            isTrainRegion_present = False

            isDevPresent = False
            isDevCity_present = False
            isDevCountry_present = False
            isDevRegion_present = False

            # Check if file's name contains 'test' in it:
            if 'test' in curDataset_file:
                isTestPresent = True
                # Open current test file
                curTestFile = open(curFile_path, "r")
                # Move to the next line after the header
                curTestFile.readline()

                while True:
                    curLine = curTestFile.readline()
                    if not curLine:
                        break
                    list_of_strings = curLine.split(sep='\t')
                    list_of_strings_length = len(list_of_strings)

                    # Now decide which dataframe is suitable for the current row
                    # If city_id is not empty, then put the current row into dev_city_df
                    if list_of_strings_length > 1 and list_of_strings[1] != '':
                        isTestCity_present = True
                        test_city_df = test_city_df.append(
                            {'original_sentence': list_of_strings[0], 'dialect_city_id': list_of_strings[1],
                             'dialect_province_id': list_of_strings[2], 'dialect_country_id': list_of_strings[3],
                             'dialect_region_id': list_of_strings[4].rstrip("\n"), 'dataset_name': curDataset_NAME},
                            ignore_index=True, sort=False)

                    # If country_id is not empty, then put the current row into dev_country_df
                    elif list_of_strings_length > 3 and list_of_strings[3] != '':
                        isTestCountry_present = True
                        test_country_df = test_country_df.append(
                            {'original_sentence': list_of_strings[0], 'dialect_city_id': list_of_strings[1],
                             'dialect_province_id': list_of_strings[2], 'dialect_country_id': list_of_strings[3],
                             'dialect_region_id': list_of_strings[4].rstrip("\n"), 'dataset_name': curDataset_NAME},
                            ignore_index=True, sort=False)

                    # If region_id is not empty, then put the current row into dev_region_df
                    elif list_of_strings_length > 4 and list_of_strings[4] != '':
                        isTestRegion_present = True
                        test_region_df = test_region_df.append(
                            {'original_sentence': list_of_strings[0], 'dialect_city_id': list_of_strings[1],
                             'dialect_province_id': list_of_strings[2], 'dialect_country_id': list_of_strings[3],
                             'dialect_region_id': list_of_strings[4].rstrip("\n"), 'dataset_name': curDataset_NAME},
                            ignore_index=True, sort=False)

                # Close the file
                curTestFile.close()

            # Check if the dataset has 'dev' in its name
            elif 'dev' in curDataset_file:
                isDevPresent = True
                # Open current test file
                curDevFile = open(curFile_path, "r")
                # Move to the next line after the header
                curDevFile.readline()

                while True:
                    curLine = curDevFile.readline()
                    if not curLine:
                        break
                    list_of_strings = curLine.split(sep='\t')
                    list_of_strings_length = len(list_of_strings)

                    # Now decide which dataframe is suitable for the current row
                    # If city_id is not empty, then put the current row into dev_city_df
                    if list_of_strings_length > 1 and list_of_strings[1] != '':
                        isDevCity_present = True
                        dev_city_df = dev_city_df.append(
                            {'original_sentence': list_of_strings[0], 'dialect_city_id': list_of_strings[1],
                             'dialect_province_id': list_of_strings[2], 'dialect_country_id': list_of_strings[3],
                             'dialect_region_id': list_of_strings[4].rstrip("\n"), 'dataset_name': curDataset_NAME},
                            ignore_index=True, sort=False)

                    # If country_id is not empty, then put the current row into dev_country_df
                    elif list_of_strings_length > 3 and list_of_strings[3] != '':
                        isDevCountry_present = True
                        dev_country_df = dev_country_df.append(
                            {'original_sentence': list_of_strings[0], 'dialect_city_id': list_of_strings[1],
                             'dialect_province_id': list_of_strings[2], 'dialect_country_id': list_of_strings[3],
                             'dialect_region_id': list_of_strings[4].rstrip("\n"), 'dataset_name': curDataset_NAME},
                            ignore_index=True, sort=False)

                    # If region_id is not empty, then put the current row into dev_region_df
                    elif list_of_strings_length > 4 and list_of_strings[4] != '':
                        isDevRegion_present = True
                        dev_region_df = dev_region_df.append(
                            {'original_sentence': list_of_strings[0], 'dialect_city_id': list_of_strings[1],
                             'dialect_province_id': list_of_strings[2], 'dialect_country_id': list_of_strings[3],
                             'dialect_region_id': list_of_strings[4].rstrip("\n"), 'dataset_name': curDataset_NAME},
                            ignore_index=True, sort=False)

                # Close the file
                curDevFile.close()

            elif 'train' in curDataset_file:
                isTrainPresent = True
                # Open current test file
                curTrainFile = open(curFile_path, "r")
                # Move to the next line after the header
                curTrainFile.readline()

                while True:
                    curLine = curTrainFile.readline()
                    if not curLine:
                        break
                    list_of_strings = curLine.split(sep='\t')
                    list_of_strings_length = len(list_of_strings)

                    #print("List_of_Strings: ", list_of_strings)
                    # Now decide which dataframe is suitable for the current row
                    # If city_id is not empty, then put the current row into dev_city_df
                    if list_of_strings_length > 1 and list_of_strings[1] != '':
                        isTrainCity_present = True
                        train_city_df = train_city_df.append(
                            {'original_sentence': list_of_strings[0], 'dialect_city_id': list_of_strings[1],
                             'dialect_province_id': list_of_strings[2], 'dialect_country_id': list_of_strings[3],
                             'dialect_region_id': list_of_strings[4].rstrip("\n"), 'dataset_name': curDataset_NAME},
                            ignore_index=True, sort=False)

                    # If country_id is not empty, then put the current row into dev_country_df
                    elif list_of_strings_length > 3 and list_of_strings[3] != '':
                        isTrainCountry_present = True
                        train_country_df = train_country_df.append(
                            {'original_sentence': list_of_strings[0], 'dialect_city_id': list_of_strings[1],
                             'dialect_province_id': list_of_strings[2], 'dialect_country_id': list_of_strings[3],
                             'dialect_region_id': list_of_strings[4].rstrip("\n"), 'dataset_name': curDataset_NAME},
                            ignore_index=True, sort=False)

                    # If region_id is not empty, then put the current row into dev_region_df
                    elif list_of_strings_length > 4 and list_of_strings[4] != '':
                        isTrainRegion_present = True
                        train_region_df = train_region_df.append(
                            {'original_sentence': list_of_strings[0], 'dialect_city_id': list_of_strings[1],
                             'dialect_province_id': list_of_strings[2], 'dialect_country_id': list_of_strings[3],
                             'dialect_region_id': list_of_strings[4].rstrip("\n"), 'dataset_name': curDataset_NAME},
                            ignore_index=True, sort=False)

                # Close the file
                curTrainFile.close()

            # --------------------------------------------------------------------------------------------
            # Save dataframes into csv files

            # if test file exists, then append newly created dataframe to it
            if isTestPresent:
                if isTestCity_present:
                    if os.path.isfile('test_city_df.csv'):
                        test_city_df.to_csv('test_city_df.csv', index=False, mode='a', header=False)
                    else:
                        test_city_df.to_csv('test_city_df.csv', index=False)

                if isTestRegion_present:
                    if os.path.isfile('test_region_df.csv'):
                        test_region_df.to_csv('test_region_df.csv', index=False, mode='a', header=False)
                    else:
                        test_region_df.to_csv('test_region_df.csv', index=False)

                if isTestCountry_present:
                    if os.path.isfile('test_country_df.csv'):
                        test_country_df.to_csv('test_country_df.csv', index=False, mode='a', header=False)
                    else:
                        test_country_df.to_csv('test_country_df.csv', index=False)

            # Check if dev csv files exist
            if isDevPresent:
                if isDevCity_present:
                    if os.path.isfile('dev_city_df.csv'):
                        dev_city_df.to_csv('dev_city_df.csv', index=False, mode='a', header=False)
                    else:
                        dev_city_df.to_csv('dev_city_df.csv', index=False)

                if isDevRegion_present:
                    if os.path.isfile('dev_region_df.csv'):
                        dev_region_df.to_csv('dev_region_df.csv', index=False, mode='a', header=False)
                    else:
                        dev_region_df.to_csv('dev_region_df.csv', index=False)

                if isDevCountry_present:
                    if os.path.isfile('dev_country_df.csv'):
                        dev_country_df.to_csv('dev_country_df.csv', index=False, mode='a', header=False)
                    else:
                        dev_country_df.to_csv('dev_country_df.csv', index=False)

            # Check if train csv files exist
            if isTrainPresent:
                if isTrainCity_present:
                    if os.path.isfile('train_city_df.csv'):
                        train_city_df.to_csv('train_city_df.csv', index=False, mode='a', header=False)
                    else:
                        train_city_df.to_csv('train_city_df.csv', index=False)

                if isTrainRegion_present:
                    if os.path.isfile('train_region_df.csv'):
                        train_region_df.to_csv('train_region_df.csv', index=False, mode='a', header=False)
                    else:
                        train_region_df.to_csv('train_region_df.csv', index=False)

                if isTrainCountry_present:
                    if os.path.isfile('train_country_df.csv'):
                        train_country_df.to_csv('train_country_df.csv', index=False, mode='a', header=False)
                    else:
                        train_country_df.to_csv('train_country_df.csv', index=False)


print('++++++++++++++++++++')

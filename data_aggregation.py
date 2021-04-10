# TODO: collect data based on the name and all
import pandas as pd
import os
import time


def get_df():
    """
    Get dataframe with all required columns
    Returns:
      df: DataFrame with all required columns
    """
    return pd.DataFrame(columns=['original_sentence', 'dialect_city_id', 'dialect_province_id', 'dialect_country_id', 'dialect_region_id', 'dataset_name'])


def process_single_file(filename, dataset_name, df_city, df_country, df_region):
    """Processes a single file and put sentences per respective row

    Args:
      filename: name of the file to get the DataFrame
      dataset_name: the name of the dataset
      df_city: DataFrame with sentences at city level 
      df_country: DataFrame with sentences at country level 
      df_region: DataFrame with sentences at region level 
    Returns:
      df_city: DataFrame with sentences at city level 
      df_country: DataFrame with sentences at country level 
      df_region: DataFrame with sentences at region level 
    """
    df = pd.read_csv(filename, sep='\t', header=0)
    df['dataset_name'] = dataset_name
    df_city_tmp = df[df.dialect_city_id.notnull()]
    df_country_tmp = df[df.dialect_country_id.notnull()]
    df_region_tmp = df[df.dialect_region_id.notnull()]
    if df_city_tmp.shape[0]:
        df_city = df_city.append(df_city_tmp, ignore_index=True)
    if df_country_tmp.shape[0]:
        df_country = df_country.append(
            df_country_tmp, ignore_index=True)
    if df_region_tmp.shape[0]:
        df_region = df_region.append(
            df_region_tmp, ignore_index=True)
    return df_city, df_country, df_region


def get_children(directory):

    no_include_list = ['.DS_Store', 'misc']
    children = [i for i in os.listdir(directory) if (
        i in no_include_list) == False]
    return children


def aggregate_data(processed_folder, to_folder):
    """Aggregate data at each level of the hierarchy
    Args:
      processed_folder: folder containing all the processed data
      to_folder: folder to put aggregated dataframes
    """
    folders = get_children(processed_folder)
    df_city_train = get_df()
    df_country_train = get_df()
    df_region_train = get_df()
    df_city_test = get_df()
    df_country_test = get_df()
    df_region_test = get_df()
    df_city_dev = get_df()
    df_country_dev = get_df()
    df_region_dev = get_df()
    for folder in folders:
        files = get_children(f'{processed_folder}/{folder}')
        start_time = time.time()
        dataset_name = folder
        print(f'Starting {dataset_name}')
        # print(folder, files)
        for file in files:
            filename = f'{processed_folder}/{folder}/{file}'
            if 'train' in file:
                df_city_train, df_country_train, df_region_train = process_single_file(
                    filename, dataset_name, df_city_train, df_country_train, df_region_train)
            elif 'test' in file:
                df_city_test, df_country_test, df_region_test = process_single_file(
                    filename, dataset_name, df_city_test, df_country_test, df_region_test)
            elif 'dev' in file:
                df_city_dev, df_country_dev, df_region_dev = process_single_file(
                    filename, dataset_name, df_city_dev, df_country_dev, df_region_dev)

        end_time = time.time()
        print(
            f'Finished processing {folder} in {int(end_time - start_time)}s')
    start_time = time.time()
    print('Starting to save to files')
    df_city_train.to_csv(f'{to_folder}/city_train.tsv', sep='\t', index=False)
    df_country_train.to_csv(
        f'{to_folder}/country_train.tsv', sep='\t', index=False)
    df_region_train.to_csv(
        f'{to_folder}/region_train.tsv', sep='\t', index=False)
    df_city_test.to_csv(f'{to_folder}/city_test.tsv', sep='\t', index=False)
    df_country_test.to_csv(
        f'{to_folder}/country_test.tsv', sep='\t', index=False)
    df_region_test.to_csv(
        f'{to_folder}/region_test.tsv', sep='\t', index=False)
    df_city_dev.to_csv(f'{to_folder}/city_dev.tsv', sep='\t', index=False)
    df_country_dev.to_csv(
        f'{to_folder}/country_dev.tsv', sep='\t', index=False)
    df_region_dev.to_csv(f'{to_folder}/region_dev.tsv', sep='\t', index=False)
    print(
        f'Finished saving to files in {int(end_time - start_time)}s')


def check_labels(aggregated_data_folder):
    df_dev = pd.read_csv(
        f'{aggregated_data_folder}/city_dev.tsv', sep='\t', header=0)


def filter_aggregated_data(aggregated_data_folder):
    # TODO: collect city, country, region that is in the label space
    pass

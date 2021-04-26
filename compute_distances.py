from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd


def compute_list_distances(list_long_lat):
    distances = [0]*len(list_long_lat)
    for i in range(len(list_long_lat)):
        distances[i] = []
        for j in range(len(list_long_lat)):
            distances[i].append(compute_distance(
                list_long_lat[i], list_long_lat[j]))

    return distances


def compute_dict_distances(dict_long_lat):
    distances = {}
    keys = list(dict_long_lat.keys())
    for i in range(len(keys)):
        distances[keys[i]] = {}
        for j in range(len(keys)):
            distances[keys[i]][keys[j]] = compute_distance(
                dict_long_lat[keys[i]], dict_long_lat[keys[j]])

    return distances


def get_long_lat(name, geolocator):
    location = geolocator.geocode(name)
    print(name, location)
    return (location.latitude, location.longitude)


def compute_distance(loc_a, loc_b):
    return geodesic(loc_a, loc_b).kilometers


def get_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        list_cities = [i.split(',')[0] for i in lines if ('msa' in i) == False]
        return list_cities


def get_dictionary_locations(filename='labels/final_label_space.tsv'):
    df = pd.read_csv(filename, sep='\t', header=0)
    map_locator_name = {}
    for index, row in df.iterrows():
        label = ''
        if row['dialect_city_id'] != 'msa' and row['dialect_city_id'] != 'jerusalem' and row['dialect_city_id'] != 'al_suwayda':
            label = f'{row["dialect_city_id"].replace("_", " ")} {row["dialect_country_id"]}'
        elif row['dialect_city_id'] == 'jerusalem':
            label = 'jerusalem'
        elif row['dialect_city_id'] == 'al_suwayda':
            label = 'as suwayda sy'
        if label != '':
            map_locator_name[label] = [row['dialect_city_id'],
                                       row['dialect_country_id'], row['dialect_region_id']]
    return map_locator_name


def save_distances_file(distances, filename):
    """
    data = map(list, zip(*distances.keys())) + [distances.values()]
    df = pd.DataFrame(zip(*data)).set_index([0, 1])[2].unstack()
    df = df.combine_first(df.T).fillna(0)
    """
    df = pd.DataFrame.from_dict(distances)
    df.to_csv(filename, index=False)


geolocator = Nominatim(user_agent="a")

map_locator_name = get_dictionary_locations()
dict_long_latitude = {}
for location in map_locator_name.keys():
    actual_location = ' '.join(map_locator_name[location])
    dict_long_latitude[actual_location] = get_long_lat(location, geolocator)

distances = compute_dict_distances(dict_long_latitude)
save_distances_file(distances, 'labels/distances_hdid.tsv')

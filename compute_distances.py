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


def get_list_from_label_space(filename='labels/label_space.tsv'):
    df = pd.read_csv(filename, sep='\t', header=0)
    cities = []
    for index, row in df.iterrows():
        if row['dialect_city_id'] != 'msa' and row['dialect_city_id'] != 'jerusalem' and row['dialect_city_id'] != 'al_suwayda':
            label = f'{row["dialect_city_id"].replace("_", " ")} {row["dialect_country_id"]}'
            cities.append(label)
    cities.append('jerusalem')
    cities.append('as suwayda sy')

    return cities


def save_distances_file(distances, list_names, filename):
    with open(filename, 'w') as f:
        heading = [i for i in list_names]
        heading.insert(0, 'separator')
        f.write('\t'.join(heading))
        f.write('\n')
        for i in range(len(list_names)):
            # print(i, distances[i])
            new_column = [str(name) for name in distances[i]]
            new_column.insert(0, list_names[i])
            f.write('\t'.join(new_column))
            f.write('\n')


geolocator = Nominatim(user_agent="a")

list_cities = get_list_from_label_space()
list_long_lat = [get_long_lat(i, geolocator) for i in list_cities]
distances = compute_list_distances(list_long_lat)
save_distances_file(distances, list_cities, 'labels/distances_hdid.tsv')

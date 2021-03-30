from geopy.geocoders import Nominatim
from geopy.distance import geodesic

def compute_list_distances(list_long_lat):
    distances = [0]*len(list_long_lat)
    for i in range(len(list_long_lat)):
        distances[i] = []
        for j in range(len(list_long_lat)):
            distances[i].append(compute_distance(list_long_lat[i], list_long_lat[j]))

    return  distances

def get_long_lat(name, geolocator):
    location  = geolocator.geocode(name)
    return (location.latitude, location.longitude)

def compute_distance(loc_a, loc_b):
    return geodesic(loc_a, loc_b).kilometers


def get_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        list_cities = [i.split(',')[0] for i in lines if ('msa' in  i) == False] 
        return list_cities

def save_distances_file(distances, list_names, filename):
    with open(filename, 'w') as f:
        heading = [i for i in list_names]
        heading.insert(0, 'separator')
        f.write('\t'.join(heading))
        f.write('\n')
        for i in range(len(list_names)):
            #print(i, distances[i])
            new_column = [str(name) for name in distances[i]]
            new_column.insert(0, list_names[i])
            f.write('\t'.join(new_column))
            f.write('\n')

geolocator = Nominatim(user_agent="a")
list_cities =  get_list('madar_labels.txt')
list_long_lat = [get_long_lat(i, geolocator) for i in list_cities]
distances  = compute_list_distances(list_long_lat)
save_distances_file(distances, list_cities, 'distances_madar.tsv')

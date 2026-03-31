#Helper functions

from geopy.distance import geodesic

# Your coordinates (London)
#USER_POS = (51.5074, -0.1278)

def add_distances(cities, user_pos): # Add user_pos here
    for city in cities:
        city_pos = (city['lat'], city['lon'])
        # Calculate distance from the provided user_pos
        city['dist'] = geodesic(user_pos, city_pos).km
    return cities

def prepare_data(cities, user_pos):
    cities_with_dist = add_distances(cities, user_pos)
    final_data = normalize_cities(cities_with_dist)
    return final_data


def normalize_cities(cities):
    """
    This function creates a balance between the categories population,
    distance, and weather. A population in the millions would skew the data
    so this functions balances them. 
    """
    pops = [c['pop'] for c in cities]
    temps = [c['temp'] for c in cities]
    dists = [c['dist'] for c in cities]

    # Get min/max once to save processing power
    min_p, max_p = min(pops), max(pops)
    min_t, max_t = min(temps), max(temps)
    min_d, max_d = min(dists), max(dists)

    for city in cities:
        city['pop_norm'] = normalize(city['pop'], min_p, max_p)
        city['temp_norm'] = normalize(city['temp'], min_t, max_t)
        city['dist_norm'] = normalize(city['dist'], min_d, max_d)
    
    return cities



def normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)



def get_path_distance(route, user_pos):
    # route is a list of city dictionaries
    total = 0
    current_pos = user_pos
    
    for city in route:
        next_pos = (city['lat'], city['lon'])
        total += geodesic(current_pos, next_pos).km
        current_pos = next_pos
        
    # Return to home
    total += geodesic(current_pos, user_pos).km
    return total
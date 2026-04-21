#Helper functions

from geopy.distance import geodesic

# Your coordinates (London)
#USER_POS = (51.5074, -0.1278)

def add_distances(cities, user_pos):
    """
    Adds a 'dist' field (km) to each city based on geodesic distance from the user.

    Parameters:
        cities (list[dict]): City dictionaries each containing 'lat' and 'lon'.
        user_pos (tuple[float, float]): User's (latitude, longitude).

    Returns:
        list[dict]: Same city list with 'dist' (float, km) added to each entry.
    """
    for city in cities:
        city_pos = (city['lat'], city['lon'])
        # Calculate distance from the provided user_pos
        city['dist'] = geodesic(user_pos, city_pos).km
    return cities

def prepare_data(cities, user_pos):
    """
    Full pipeline: adds distances then normalizes all city attributes.

    Parameters:
        cities (list[dict]): Raw city data from cities.json.
        user_pos (tuple[float, float]): User's (latitude, longitude).

    Returns:
        list[dict]: Processed city list with 'dist', 'dist_norm', 'temp_norm', 'pop_norm' fields.
    """
    cities_with_dist = add_distances(cities, user_pos)
    final_data = normalize_cities(cities_with_dist)
    return final_data


def normalize_cities(cities):
    """
    Min-max normalizes pop, temp, and dist across all cities to a [0, 1] range.
    Prevents large-magnitude fields (e.g. population in millions) from dominating scores.

    Parameters:
        cities (list[dict]): City dictionaries each containing 'pop', 'temp', 'dist'.

    Returns:
        list[dict]: Same list with 'pop_norm', 'temp_norm', 'dist_norm' (float, 0–1) added.
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
    """
    Min-max normalizes a single value to the range [0, 1].

    Parameters:
        value (float): The raw value to normalize.
        min_val (float): Minimum value in the dataset.
        max_val (float): Maximum value in the dataset.

    Returns:
        float: Normalized value between 0.0 and 1.0. Returns 0 if min == max.
    """
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)



def get_path_distance(route, user_pos):
    """
    Calculates total round-trip distance for a route starting and ending at user's home.

    Parameters:
        route (list[dict]): Ordered list of city dicts each containing 'lat' and 'lon'.
        user_pos (tuple[float, float]): User's (latitude, longitude).

    Returns:
        float: Total geodesic distance in km (home → city1 → ... → cityN → home).
    """
    total = 0
    current_pos = user_pos
    
    for city in route:
        next_pos = (city['lat'], city['lon'])
        total += geodesic(current_pos, next_pos).km
        current_pos = next_pos
        
    # Return to home
    total += geodesic(current_pos, user_pos).km
    return total
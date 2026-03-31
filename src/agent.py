class TravelAgent:
    def __init__(self, cities):
        self.cities = cities

    def score_city(self, city, w_dist, w_temp, w_pop):
        """Calculates the utility score for a single city dictionary."""
        # Calculate how close the city is to the user's ideal (1.0 is a perfect match)
        d_m = 1 - abs(city['dist_norm'] - w_dist)
        t_m = 1 - abs(city['temp_norm'] - w_temp)
        p_m = 1 - abs(city['pop_norm'] - w_pop)
        
        # Multiplicative utility with a tiny 'buffer' to prevent total zeros
        return (d_m + 0.01) * (t_m + 0.01) * (p_m + 0.01)

    def evaluate(self, w_dist, w_temp, w_pop):
        results = []
        for city in self.cities:
            score = self.score_city(city, w_dist, w_temp, w_pop)
            results.append({"name": city['name'], "score": score})
        
        # Sort by highest score first
        return sorted(results, key=lambda x: x['score'], reverse=True)
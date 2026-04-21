class TravelAgent:
    def __init__(self, cities):
        """
        Initializes the TravelAgent with a list of processed city data.

        Parameters:
            cities (list[dict]): City dictionaries with normalized fields
                                 (dist_norm, temp_norm, pop_norm).
        """
        self.cities = cities

    def score_city(self, city, w_dist, w_temp, w_pop):
        """
        Calculates a multiplicative utility score for a single city.

        Parameters:
            city (dict): City dictionary containing dist_norm, temp_norm, pop_norm.
            w_dist (float): User's distance preference (0=close, 1=far).
            w_temp (float): User's temperature preference (0=cold, 1=hot).
            w_pop (float): User's population preference (0=small, 1=big).

        Returns:
            float: Utility score — higher means better match.
        """
        # Calculate how close the city is to the user's ideal (1.0 is a perfect match)
        d_m = 1 - abs(city['dist_norm'] - w_dist)
        t_m = 1 - abs(city['temp_norm'] - w_temp)
        p_m = 1 - abs(city['pop_norm'] - w_pop)
        
        # Multiplicative utility with a tiny 'buffer' to prevent total zeros
        return (d_m + 0.01) * (t_m + 0.01) * (p_m + 0.01)

    def evaluate(self, w_dist, w_temp, w_pop):
        """
        Scores all cities and returns them sorted best-to-worst.

        Parameters:
            w_dist (float): User's distance preference (0=close, 1=far).
            w_temp (float): User's temperature preference (0=cold, 1=hot).
            w_pop (float): User's population preference (0=small, 1=big).

        Returns:
            list[dict]: List of {'name': str, 'score': float} sorted by score descending.
        """
        results = []
        for city in self.cities:
            score = self.score_city(city, w_dist, w_temp, w_pop)
            results.append({"name": city['name'], "score": score})
        
        # Sort by highest score first
        return sorted(results, key=lambda x: x['score'], reverse=True)
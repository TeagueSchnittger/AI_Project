import os
from src.nn_scorer import load_model, predict

MODEL_PATH = "data/scorer_model.pkl"

class TravelAgent:
    def __init__(self, cities):
        """
        Initializes the TravelAgent with city data and loads the NN scorer if available.

        Parameters:
            cities (list[dict]): City dictionaries with normalized fields
                                 (dist_norm, temp_norm, pop_norm).
        """
        self.cities = cities
        self.nn_model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    def score_city(self, city, w_dist, w_temp, w_pop):
        """
        Calculates a multiplicative utility score for a single city (rule-based baseline).

        Parameters:
            city   (dict):  City dictionary containing dist_norm, temp_norm, pop_norm.
            w_dist (float): User's distance preference (0=close, 1=far).
            w_temp (float): User's temperature preference (0=cold, 1=hot).
            w_pop  (float): User's population preference (0=small, 1=big).

        Returns:
            float: Utility score — higher means better match.
        """
        d_m = 1 - abs(city['dist_norm'] - w_dist)
        t_m = 1 - abs(city['temp_norm'] - w_temp)
        p_m = 1 - abs(city['pop_norm'] - w_pop)
        return (d_m + 0.01) * (t_m + 0.01) * (p_m + 0.01)

    def score_city_nn(self, city, w_dist, w_temp, w_pop):
        """
        Scores a city using the trained neural network scorer.

        Parameters:
            city   (dict):  City dictionary containing dist_norm, temp_norm, pop_norm.
            w_dist (float): User's distance preference (0=close, 1=far).
            w_temp (float): User's temperature preference (0=cold, 1=hot).
            w_pop  (float): User's population preference (0=small, 1=big).

        Returns:
            float | None: NN predicted score in [0, 1], or None if model not loaded.
        """
        if self.nn_model is None:
            return None
        return predict(self.nn_model, city['dist_norm'], city['temp_norm'], city['pop_norm'],
                       w_dist, w_temp, w_pop)

    def evaluate(self, w_dist, w_temp, w_pop):
        """
        Scores all cities with both the utility function and the neural network,
        returning them sorted best-to-worst by NN score (falls back to utility if
        no model is loaded).

        Parameters:
            w_dist (float): User's distance preference (0=close, 1=far).
            w_temp (float): User's temperature preference (0=cold, 1=hot).
            w_pop  (float): User's population preference (0=small, 1=big).

        Returns:
            list[dict]: List of {'name': str, 'score': float, 'nn_score': float | None}
                        sorted by best match descending.
        """
        results = []
        for city in self.cities:
            utility = self.score_city(city, w_dist, w_temp, w_pop)
            nn = self.score_city_nn(city, w_dist, w_temp, w_pop)
            results.append({
                "name": city['name'],
                "score": utility,
                "nn_score": nn,
            })

        sort_key = "nn_score" if self.nn_model is not None else "score"
        return sorted(results, key=lambda x: x[sort_key] or 0, reverse=True)

import random
from src.utils import get_path_distance

class GeneticPlanner:
    def __init__(self, cities, user_pos, agent):
        self.cities = cities
        self.user_pos = user_pos
        self.agent = agent

    def get_fitness(self, route, w_dist, w_temp, w_pop):
        total_utility = 0
        for city in route:
            total_utility += self.agent.score_city(city, w_dist, w_temp, w_pop)
        
        dist = get_path_distance(route, self.user_pos)
        # Higher penalty for distance (dividing by 500 instead of 1000)
        return total_utility / (max(dist, 1) / 500)

    def evolve(self, top_recommendations, w_dist, w_temp, w_pop, generations=50, pop_size=20):
        # Ensure we are working with at least 5 cities to choose from
        pool = top_recommendations if len(top_recommendations) >= 5 else self.cities[:10]
        
        # 1. INITIAL POPULATION: Start with the absolute best 3 as a baseline
        population = [pool[:3]] 
        
        # Fill the rest with unique samples from the 'pool' of top cities
        while len(population) < pop_size:
            route = random.sample(pool, 3)
            if route not in population:
                population.append(route)
        
        for _ in range(generations):
            # 2. SELECTION: Sort by fitness
            population = sorted(population, key=lambda r: self.get_fitness(r, w_dist, w_temp, w_pop), reverse=True)
            
            # 3. ELITISM: Keep the top 5 routes exactly as they are
            new_population = population[:5]
            
            # 4. BREEDING: Create children from the top routes
            while len(new_population) < pop_size:
                parent1 = population[random.randint(0, 4)]
                parent2 = population[random.randint(0, 4)]
                
                # Take 2 from P1, and find 1 unique from P2
                child = parent1[:2]
                for gene in parent2:
                    if gene not in child:
                        child.append(gene)
                        break
                
                # Final safety check for duplicates/length
                if len(child) < 3:
                    for city in pool:
                        if city not in child:
                            child.append(city)
                            break
                
                new_population.append(child)
            
            population = new_population
            
        return population[0] # Return the absolute best
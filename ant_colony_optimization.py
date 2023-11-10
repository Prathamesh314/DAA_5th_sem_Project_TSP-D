import numpy as np

class AntColony:
    def __init__(self, distances, n_ants, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones_like(distances) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self, n_iterations):
        best_path = None
        all_time_best_distance = float('inf')

        for i in range(n_iterations):
            ants = self.generate_ants()
            self.spread_pheromone(ants)
            self.pheromone * self.decay

            current_best_path, current_best_distance = self.get_best_path(ants)

            if current_best_distance < all_time_best_distance:
                best_path = current_best_path
                all_time_best_distance = current_best_distance
            # print(f"Best path {i+1}: {best_path}")
        return best_path, all_time_best_distance

    def generate_ants(self):
        ants = []
        for ant in range(self.n_ants):
            start = np.random.randint(len(self.distances))
            ant_path = self.generate_ant_path(start)
            ants.append((ant_path, self.calculate_path_distance(ant_path)))
        return ants

    def generate_ant_path(self, start):
        ant_path = []
        unvisited = set(self.all_inds)
        unvisited.remove(start)
        ant_path.append(start)

        while unvisited:
            next_city = self.choose_next_city(ant_path[-1], unvisited)
            ant_path.append(next_city)
            unvisited.remove(next_city)

        return ant_path

    def choose_next_city(self, current_city, unvisited):
        pheromone_values = self.pheromone[current_city, list(unvisited)]
        heuristic_values = 1 / (self.distances[current_city, list(unvisited)] + 1e-10)
        probabilities = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)
        probabilities /= probabilities.sum()
        next_city = np.random.choice(list(unvisited), p=probabilities)
        return next_city

    def spread_pheromone(self, ants):
        pheromone_change = np.zeros_like(self.pheromone)
        for ant_path, distance in ants:
            for i in range(len(ant_path) - 1):
                pheromone_change[ant_path[i], ant_path[i + 1]] += 1 / distance
                pheromone_change[ant_path[i + 1], ant_path[i]] += 1 / distance
        self.pheromone = (1 - self.decay) * self.pheromone + pheromone_change

    def get_best_path(self, ants):
        best_path = None
        best_distance = float('inf')
        for ant_path, distance in ants:
            if distance < best_distance:
                best_path = ant_path
                best_distance = distance
        return best_path, best_distance

    def calculate_path_distance(self, path):
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.distances[path[i]][path[i + 1]]
        return total_distance


distances = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
])

n_ants = 5
decay = 0.95
alpha = 1
beta = 2
iterations = 100

aco = AntColony(distances, n_ants, decay, alpha, beta)

best_path, best_distance = aco.run(iterations)

print("Best Path:", best_path)
print("Best Distance:", best_distance)

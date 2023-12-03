import numpy as np

population = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
truck = [i + 1 for i in range(len(population)) if population[i] == 0]
drone = [i + 1 for i in range(len(population)) if population[i] == 1]
transportation_array = [1, 3, 3, 1, 3, 2, 4, 2, 3, 1]
depot_point = 0
tsp_set = set(range(len(population)))
Ts = 10  # Speed of Truck
Ds = 20  # Speed of Drone
alpha = 1  # GA-AS constant
beta = 1  # GA-AS constant
decay = 0.95  # GA-AS constant
# Created distance matrix for 10 customers
ditances_matrix = np.array([
    [0, 29, 20, 21, 16, 31, 100, 12, 4, 31],
    [29, 0, 15, 29, 28, 40, 72, 21, 29, 41],
    [20, 15, 0, 15, 14, 25, 81, 9, 23, 27],
    [21, 29, 15, 0, 4, 12, 92, 12, 25, 13],
    [16, 28, 14, 4, 0, 16, 94, 9, 20, 16],
    [31, 40, 25, 12, 16, 0, 95, 24, 36, 3],
    [100, 72, 81, 92, 94, 95, 0, 90, 101, 99],
    [12, 21, 9, 12, 9, 24, 90, 0, 15, 25],
    [4, 29, 23, 25, 20, 36, 101, 15, 0, 35],
    [31, 41, 27, 13, 16, 3, 99, 25, 35, 0]
])


class AntColony:
    def __init__(self, distances, n_ants, decay, alpha, beta, speed) -> None:
        self.distances = distances
        self.pheromone = np.ones_like(distances) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.speed = speed
        self.tcost = self.distances * (1 / self.speed)

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
        return best_path, all_time_best_distance

    def generate_ants(self):
        ants = []
        for ant in range(self.n_ants):
            start = 0
            ant_path = self.generate_ant_path(start)
            ants.append((ant_path, self.calculate_path_distance(ant_path)))
        return ants

    def generate_ant_path(self, start):
        ant_path = []
        unvisited = set(self.all_inds)
        unvisited.remove(start)
        ant_path.append(start)
        while unvisited:
            new_unvisited = []
            if transportation_array[ant_path[-1]] == 1:
                new_unvisited = list(
                    filter(lambda x: transportation_array[x] == 1 or transportation_array[x] == 3, unvisited))
            elif transportation_array[ant_path[-1]] == 2:
                new_unvisited = list(filter(lambda x: transportation_array[x] == 3, unvisited))
            elif transportation_array[ant_path[-1]] == 3:
                new_unvisited = list(filter(lambda x: transportation_array[x] != 1, unvisited))
            else:
                new_unvisited = list(
                    filter(lambda x: transportation_array[x] == 3 or transportation_array[x] == 2, unvisited))

            next_city = None
            if len(new_unvisited):
                next_city = self.choose_next_city(ant_path[-1], new_unvisited)
            else:
                next_city = self.choose_next_city(ant_path[-1], unvisited)
            ant_path.append(next_city)
            unvisited.remove(next_city)
        ant_path.append(start)
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
            total_distance += self.tcost[path[i]][path[i + 1]]
        return total_distance


def fitness_measurement(path, Truck_Cost, Drone_Cost):
    total_cost = 0
    for i in range(len(path) - 1):
        if transportation_array[path[i + 1]] == 3 or transportation_array[path[i + 1]] == 1:
            total_cost += Truck_Cost[path[i]][path[i + 1]]
        elif transportation_array[path[i + 1]] == 4:
            total_cost += (Drone_Cost[path[i]][path[i + 1]] * 2)
        else:
            total_cost += max(Truck_Cost[path[i]][path[i + 1]], Drone_Cost[path[i]][path[i + 1]])
    return 1 / total_cost


def Type_Matrix(path):
    arr = []
    for i in path:
        arr.append(transportation_array[i])
    return arr


def Total_cost(path, Tcost, Dcost):
    total_cost = 0
    for i in range(len(path) - 1):
        if transportation_array[path[i + 1]] == 3 or transportation_array[path[i + 1]] == 1:
            total_cost += Tcost[path[i]][path[i + 1]]
        elif transportation_array[path[i + 1]] == 4:
            total_cost += (Dcost[path[i]][path[i + 1]]) * 2
        else:
            total_cost += max(Tcost[path[i]][path[i + 1]], Dcost[path[i]][path[i + 1]])
    return total_cost


Truck = AntColony(ditances_matrix, 5, decay, alpha, beta, Ts)
Drone = AntColony(ditances_matrix, 5, decay, alpha, beta, Ds)
best_truck_path, best_truck_time = Truck.run(100)
best_drone_path, best_drone_time = Drone.run(100)
fitness_truck = fitness_measurement(best_truck_path, Truck.tcost, Drone.tcost)
fitness_drone = fitness_measurement(best_drone_path, Truck.tcost, Drone.tcost)
truck_type = Type_Matrix(best_truck_path)
drone_type = Type_Matrix(best_drone_path)
TotalCost = Total_cost(best_truck_path, Truck.tcost, Drone.tcost)
Totalcost1 = Total_cost(best_drone_path, Truck.tcost, Drone.tcost)

print(f"Best Truck distance: {best_truck_time} s")
print(f"Best Truck Path: {best_truck_path}")
print(f"Truck Fitness: {fitness_truck}")
print(f"Truck Type Matrix: {truck_type}")
print(f"Total cost if we take Truck path: {TotalCost}")
print()
print(f"Best Drone distance: {best_drone_time} s")
print(f"Best Drone path: {best_drone_path}")
print(f"Drone Fitness: {fitness_drone}")
print(f"Drone Type Matrix: {drone_type}")
print(f"Total cost if we take drone path: {Totalcost1}")
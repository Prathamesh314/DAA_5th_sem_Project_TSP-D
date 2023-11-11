# Genetic Algorithm - Ant Colony Algorithm

#  Initialize Population
# 2: iter = 0
# 3: for (iter < maxiter)
# 4:  gpop = 0;
# 5:  Generate truckphmtrx and dronephmtrx
# 6:  for (gpop = 0, gpop < maximum number of population, gpop++)
# 7:   Select the chromosome (gpop)
# 8:   Apply AS algorithm (gpop, truckphmtrx and dronephmtrx)
# 9:   Generate the route and fitness of each chromosome
# 10:     end
# 11:     for (until the max number of crossover)
# 12:      Select the parents P1 and P2
# 13:      Generate offspring individual O from P1 and P2
# 14:      Replace unwilling chromosome with offspring O
# 15:     end
# 16:     for (until the max number of mutation)
# 17:      Select the random allele on the chromosome
# 18:      Mutate the selected allele
# 19:     end
# 20:     update truckphmtrx and dronephmtrx
# 21: end

# We need to initialize population

# print(f"Truck: {truck}")
# print(f"Drone: {drone}")

# There are 4 types of trasportation
# 1. Truck alone
# 2. Drone alone
# 3. Truck and Drone together
# 4. Drone return :- i -> j -> i, it launches from i to j and then comeback from j to i

# Type 1 and 3 for Truck
# Type 2 and 4 for Drone

#                             Types of selection
# --------------------------------------------------------------------------------
#                                                                                |
# If node i is type 1, then node j may only be one of: type 1 or type 3.         |
# If node i is type 2, then node j may only be type 3.                           |
# If node i is type 3, then node j may only be one of: type 3, type 2 or type 4. |
# If node i is type 4, then node j may only be one of: type3 or type 2.          |
#                                                                                |
# --------------------------------------------------------------------------------

# Determine nodes where truck is delievering the order
# 2->9->8->1->4->3

# Determine nodes where drone is delievering the order
# 6->5->7

# Determine transportation type array for above mentioned 4 types
# 3->3->3->4->2->1->3->3->3->2->3

# We need to calculate solution array, in this case
# 0->8->2->6->5->3->9->1->4->7->0

# Aim :- Our aim is to minimize time of delievery, we are given list of customers which are to be delievered either by truck or dron

# We are given with a graph G, V = C U {r} where C in depot and r is set of customers to be served by either drone or a truck

# W[i,j] represents time to travel between pair of nodes (i,j) 
# p is the ratio of truck's and drone's travel time / unit distance

import numpy as np

population = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
truck = [i+1 for i in range(len(population)) if population[i]==0]
drone = [i+1 for i in range(len(population)) if population[i]==1]
transportation_array = [1, 3, 3, 1, 3, 2, 4, 2, 3, 1]
depot_point = 0
tsp_set = set(range(len(population)))

Ts = 10 # Speed of Truck
Ds = 20 # Speed of Drone

alpha = 1 # GA-AS constant
beta = 1 # GA-AS constant
decay = 0.95  # GA-AS constant


# Created distance matrix for 10 customers
distance_matrix = np.array([
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

def tsp(i, start, s):
    if len(s) == 0:
        return [start], distance_matrix[i][start]
    ans = 1e9
    path = []
    for j in s:
        s.remove(j)
        rpath, cost = tsp(j, start, s)
        s.add(j)
        cost += distance_matrix[i][j]
        rpath.append(j)
        if cost < ans:
            ans = cost
            path = rpath
    return path, ans


class AntColony:
    def __init__(self,distances, n_ants, decay, alpha, beta, speed) -> None:
        self.distances = distances
        self.pheromone = np.ones_like(distances) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.speed = speed
        self.tcost = self.distances * (1/self.speed)
    
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
                new_unvisited = list(filter(lambda x:transportation_array[x]==1 or transportation_array[x]==3,unvisited))
            elif transportation_array[ant_path[-1]] == 2:
                new_unvisited = list(filter(lambda x:transportation_array[x]==3, unvisited))
            elif transportation_array[ant_path[-1]] == 3:
                new_unvisited = list(filter(lambda x: transportation_array[x]!=1,unvisited))
            else:
                new_unvisited = list(filter(lambda x:transportation_array[x]==3 or transportation_array[x]==2, unvisited))
            
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

def fitness_measurement(path,Truck_Cost, Drone_Cost):
    total_cost = 0
    for i in range(len(path) - 1):
        if transportation_array[path[i+1]] == 3 or transportation_array[path[i+1]] == 1:
            total_cost += Truck_Cost[path[i]][path[i+1]]
        elif transportation_array[path[i+1]] == 4:
            total_cost += (Drone_Cost[path[i]][path[i+1]]*2)
        else:
            total_cost += max(Truck_Cost[path[i]][path[i+1]], Drone_Cost[path[i]][path[i+1]])
    return  1/total_cost

def Type_Matrix(path):
    arr = []
    for i in path:
        arr.append(transportation_array[i])
    return arr

def Total_cost(path, Tcost, Dcost):
    total_cost = 0
    for i in range(len(path)-1):
        if transportation_array[path[i+1]] == 3 or transportation_array[path[i+1]] == 1:
            total_cost += Tcost[path[i]][path[i+1]]
        elif transportation_array[path[i+1]] == 4:
            total_cost += (Dcost[path[i]][path[i+1]])*2
        else:
            total_cost += max(Tcost[path[i]][path[i+1]], Dcost[path[i]][path[i+1]])
    return total_cost

Truck = AntColony(distance_matrix, 5, decay, alpha, beta, Ts)
Drone = AntColony(distance_matrix, 5, decay, alpha, beta, Ds)

best_truck_path, best_truck_time = Truck.run(100)
best_drone_path, best_drone_time = Drone.run(100)

fitness_truck = fitness_measurement(best_truck_path, Truck.tcost, Drone.tcost)
fitness_drone = fitness_measurement(best_drone_path, Truck.tcost, Drone.tcost)

truck_type = Type_Matrix(best_truck_path)
drone_type = Type_Matrix(best_drone_path)

TotalCost = Total_cost(best_truck_path, Truck.tcost, Drone.tcost)
Totalcost1 = Total_cost(best_drone_path, Truck.tcost, Drone.tcost)

print(f"Best Truck time: {best_truck_time} s")
print(f"Best Truck Path: {best_truck_path}")
print(f"Truck Fitness: {fitness_truck}")
print(f"Truck Type Matrix: {truck_type}")
print(f"Total cost if we take Truck path: {TotalCost}")
print()
print(f"Best Drone time: {best_drone_time} s")
print(f"Best Drone path: {best_drone_path}")
print(f"Drone Fitness: {fitness_drone}")
print(f"Drone Type Matrix: {drone_type}")
print(f"Total cost if we take drone path: {Totalcost1}")
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
beta = 5  # GA-AS constant
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

# List of Directed edges for transppoprtation
edges = [
    (1,4),
    (4,0),
    (4,7),
    (7,0),
    (0,8),
    (8,2),
    (2,6),
    (6,2),
    (2,5),
    (2,3),
    (3,9),
    (5,9),
    (9,1),
]

adj = {}

TDarray = {}
DDarray = {}


# Initializing Visitied array for Truck and Drone
Drone_visited = np.zeros(len(population))
Truck_visited = np.zeros(len(population))
Drone_visited[depot_point] = Truck_visited[depot_point] = 1  # Started point [depot] is 0 so marking it as 1

# Creating Adjacency list for given graph
for i in edges:
    a = i[0]
    b = i[1]
    if population[a] == 0 and population[b]==0: # Path is travelled only by truck
        if a in TDarray:
            TDarray[a].append(b)
        else:
            l = [b]
            TDarray[a] = l
    else:  # Path is travelled only by Drone
        if a in DDarray:
            DDarray[a].append(b)
        else:
            l = [b]
            DDarray[a] = l


# Step 1. Initially 1/n is assigned to each Ptij [Truck] and Pdij [Drone]
Pt = np.ones_like(ditances_matrix)/len(ditances_matrix)  # Truck pheromone matrix, initialized with 1/n
Pd = np.ones_like(ditances_matrix)/len(ditances_matrix)  # Drone pheromone matrix, initialized with 1/n


def find_best_path(num_of_iters):
    best_path = None
    all_time_best_distance = float('inf')
    for i in range(num_of_iters):
        path, cost = generate_ant_path(0)
        ants = [(path, cost)]
        spread_pheromone(ants)
        Pt * decay
        # current_path = ants[0]
        # current_cost = ants[1]
        if cost < all_time_best_distance:
            best_path = path
            all_time_best_distance = cost
        # print(ants)
    return best_path, all_time_best_distance


def spread_pheromone(ants):
    global Pt
    pheromone_change = np.zeros_like(Pt)
    for ant_path, distance in ants:
        for i in range(len(ant_path) - 1):
            pheromone_change[ant_path[i], ant_path[i + 1]] += 1 / distance
            pheromone_change[ant_path[i + 1], ant_path[i]] += 1 / distance
    Pt = (1 - decay) * Pt + pheromone_change


# Function to choose next city with probability function
def choose_next_city(start, unvisited):
    pheromone_values = Pt[start, list(unvisited)]
    heuristic_values = 1 / (ditances_matrix[start, list(unvisited)] + 1e-10)
    probabilities = ( pheromone_values ** alpha ) * ( heuristic_values ** beta )
    probabilities /= probabilities.sum()
    next_city = np.random.choice(list(unvisited), p=probabilities)
    return next_city

# Main function to generate ant path from starting point 0
def generate_ant_path(start):

    '''
    Generate an ant path with given start position
    '''
    ant_path = []
    unvisited = set(range(len(population)))
    unvisited.remove(start)
    ant_path.append(start)
    while unvisited:
        next_city = choose_next_city(ant_path[-1], unvisited)
        ant_path.append(next_city)
        unvisited.remove(next_city)
    ant_path.append(start)
    cost = calculate_path_distance(ant_path)
        
    return ant_path, cost

def calculate_path_distance( path):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += ditances_matrix[path[i]][path[i + 1]]
    return total_distance

def tsp(i, start, s):

    if len(s) == 0:
        return ditances_matrix[i][start], [start]

    ans = 1e9
    path = []

    for j in s:
        s.remove(j)
        cost, rpath = tsp(j,start,s)
        s.add(j)
        cost += ditances_matrix[i][j]
        rpath.append(j)
        if cost < ans:
            ans = cost
            path = rpath
    return ans, path


path, distance = find_best_path(100)
print(f"Best Path: {path}")
print(f"Best distance: {distance}")
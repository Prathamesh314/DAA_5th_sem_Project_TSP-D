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
transporation_array = [1, 3, 3, 1, 3, 2, 4, 2, 3, 1]
depot_point = 0

Ts = 10 # Speed of Truck
Ds = 20 # Speed of Drone

alpha = 1 # GA-AS constant
beta = 5  # GA-AS constant

ditances_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 10, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 5, 6, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
    [10, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
    [9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 0, 0, 0, 0, 0]
])

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

Drone_visited = np.zeros(len(population))
Truck_visited = np.zeros(len(population))
Drone_visited[depot_point] = Truck_visited[depot_point] = 1


for i in edges:
    a = i[0]
    b = i[1]
    if population[a] == 0 and population[b]==0:
        if a in TDarray:
            TDarray[a].append(b)
        else:
            l = [b]
            TDarray[a] = l
    else:
        if a in DDarray:
            DDarray[a].append(b)
        else:
            l = [b]
            DDarray[a] = l

# print(f"TDarray: {TDarray}")
# print(f"DDarray: {DDarray}")
# print(f"Drone visited: {Drone_visited}")
# print(f"Truck visited: {Truck_visited}")

Pt = np.ones_like(ditances_matrix)/len(ditances_matrix)  # Truck pheromone matrix, initialized with 1/n
Pd = np.ones_like(ditances_matrix)/len(ditances_matrix)  # Drone pheromone matrix, initialized with 1/n

# print(f"Truck pheromone matrix:\n{Pt}")

# print(set(range(len(population))))

def choose_next_city(start, unvisited):
    pheromone_values = Pt[start, list(unvisited)]
    heuristic_values = 1 / (ditances_matrix[start, list(unvisited)] + 1e-10)
    probabilities = ( pheromone_values ** alpha ) * ( heuristic_values ** beta )
    probabilities /= probabilities.sum()
    next_city = np.random.choice(list(unvisited), p=probabilities)
    return next_city

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
    return ant_path

path = generate_ant_path(0)
print(f"Ant path starting from 0: {path}")
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

cities = list(range(10))
population = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
truck = [i for i in range(len(population)) if population[i] == 0]
drone = [i for i in range(len(population)) if population[i] == 1]
transportation_array = [1, 3, 3, 1, 3, 2, 4, 2, 3, 1]
depot_point = 0
tsp_set = set(range(len(population)))

l = [0] * (len(population) - len(truck))
truck.extend(l)
l = [0] * (len(population) - len(drone))
drone.extend(l)

Ts = 10  # Speed of Truck
Ds = 20  # Speed of Drone

alpha = 1  # GA-AS constant
beta = 1  # GA-AS constant
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

maxitr = 10

trckphmtrix = np.ones_like(distance_matrix) / len(distance_matrix)
drnephmtrix = np.ones_like(distance_matrix) / len(distance_matrix)

Tcost = distance_matrix * (1 / Ts)
Dcost = distance_matrix * (1 / Ds)

crossovers = 100
maximum_number_of_mutation = 10
mutation_rate = 0.1


def fitness_measurement(path):
    if len(path) == 0:
        return 10**5
    global Tcost, Dcost
    total_cost = 0
    for i in range(len(path) - 1):
        if i in truck:
            total_cost += Tcost[path[i]][path[i + 1]]
        else:
            total_cost += Dcost[path[i]][path[i + 1]]
    return 1 / total_cost


def choose_next_city(current_city, unvisited):
    global trckphmtrix, drnephmtrix

    pheromone_values = np.zeros_like(distance_matrix)
    for j in unvisited:
        if j in truck:
            p_tij = trckphmtrix[current_city][j]
            d_ij = distance_matrix[current_city][j]
            pheromone_values[current_city][j] = (p_tij ** alpha) * (d_ij ** beta)
        else:
            d_tij = drnephmtrix[current_city][j]
            d_ij = distance_matrix[current_city][j]
            pheromone_values[current_city][j] = (d_tij ** alpha) * (d_ij ** beta)
    # for i in range(len(pheromone_values)):
    #     for j in range(len(pheromone_values[0])):
    #         if transportation_array[current_city] != 1:
    #             j1 = random.choice(truck)
    #             j2 = random.choice(drone)
    #             n = random.choice([0, 1])
    #             if not n:
    #                 p_tij = trckphmtrix[current_city][j1]
    #                 d_ij = distance_matrix[current_city][j1]
    #                 pheromone_values[i][_] = (p_tij ** alpha) * (d_ij ** beta)
    #             else:
    #                 d_tij = drnephmtrix[current_city][j2]
    #                 d_ij = distance_matrix[current_city][j2]
    #                 pheromone_values[i][_] = (d_tij ** alpha) * (d_ij ** beta)
    #         else:
    #             tj = random.choice(truck)
    #             p_tij = trckphmtrix[current_city][tj]
    #             d_ijt = distance_matrix[current_city][tj]
    #             pheromone_values[i][_] = (p_tij ** alpha) * (d_ijt ** beta)


            # if j in truck:
            #     p_tij = trckphmtrix[i][j]
            #     d_ij = distance_matrix[i][j]
            #     pheromone_values[i][j] = (p_tij ** alpha) * (d_ij ** beta)
            # else:
            #     d_tij = drnephmtrix[i][j]
            #     d_ij = distance_matrix[i][j]
            #     pheromone_values[i][j] = (d_tij ** alpha) * (d_ij ** beta)

    arr_sum = 0
    for i in range(len(pheromone_values)):
        for j in range(len(pheromone_values[0])):
            arr_sum += pheromone_values[i][j]
    # if arr_sum == 0:
    #
    #     print(current_city, arr_sum)
    probabilities = pheromone_values * (1 / (arr_sum if arr_sum > 0 else 10**5))
    # print(pheromone_values)
    r = random.random()
    next_city = None
    cummulative_sum = 0
    for i in range(len(pheromone_values)):
        for j in range(len(pheromone_values[0])):
            cummulative_sum += probabilities[i][j]
            if cummulative_sum >= r and j != current_city and j in unvisited:
                next_city = j
                break
    # print(next_city)
    return next_city


def AS_Algo(start):
    ant_path = []
    unvisited = set(range(len(population)))
    unvisited.remove(start)
    ant_path.append(start)

    while len(unvisited):
        new_unvisited = []
        # If current_city is of type 2, 3 or 4 then possible candidate for next_city is truck or dron
        if transportation_array[ant_path[-1]] != 1:
            new_unvisited = unvisited
        else:
            # else possible candidate for next_city is only truck
            for j in unvisited:
                if j in truck:
                    new_unvisited.append(j)
        # if transportation_array[ant_path[-1]] == 1:
        #     new_unvisited = list(
        #         filter(lambda x: transportation_array[x] == 1 or transportation_array[x] == 3, unvisited))
        # elif transportation_array[ant_path[-1]] == 2:
        #     new_unvisited = list(filter(lambda x: transportation_array[x] == 3, unvisited))
        # elif transportation_array[ant_path[-1]] == 3:
        #     new_unvisited = list(filter(lambda x: transportation_array[x] != 1, unvisited))
        # else:
        #     new_unvisited = list(
        #         filter(lambda x: transportation_array[x] == 3 or transportation_array[x] == 2, unvisited))
        next_city = choose_next_city(ant_path[-1], new_unvisited)
        if next_city is None:
            return [], 10**5
        ant_path.append(next_city)
        unvisited.remove(next_city)

    # Update pheromone matrices
    # for i in range(len(ant_path) - 1):
    #     if ant_path[i] in truck and ant_path[i + 1] in truck:
    #         trckphmtrix[ant_path[i]][ant_path[i + 1]] *= (1 - decay)
    #     elif ant_path[i] in drone and ant_path[i + 1] in drone:
    #         drnephmtrix[ant_path[i]][ant_path[i + 1]] *= (1 - decay)

    return ant_path, fitness_measurement(ant_path)


def mate(parent1, parent2):
    child_chromosome = []
    if len(parent1) == 0 or len(parent2) == 0:
        return child_chromosome
    for i in range(len(parent1)):
        prob = random.random()
        gp1 = parent1[i]
        gp2 = parent2[i]
        if prob < 0.45:
            child_chromosome.append(gp1)
        elif prob < 0.90:
            child_chromosome.append(gp2)
        else:
            child_chromosome.append(random.randint(0, 9))
    return child_chromosome

#
# def mutation(chromosome):
#     for i in range(maximum_number_of_mutation):
#         index = random.randint(0, len(chromosome) - 1)
#         chromosome[index][i] = random.randint(0, 9)
#     return chromosome


all_paths = []
for i in range(maxitr):
    gpop = 0
    while gpop < len(population):
        random.shuffle(cities)
        start = cities[gpop]
        path, fitness = AS_Algo(start)
        all_paths.append((path, fitness))
        gpop += 1

    all_paths.sort(key=lambda x: x[1])
    new_generation = []
    s = (10 * crossovers) // 100
    new_generation.extend(all_paths[:s])
    s = 100 - s
    for i in range(s):
        parent1 = random.choice(all_paths)
        parent2 = random.choice(all_paths)
        child = mate(parent1[0], parent2[0])
        new_generation.append((child, fitness_measurement(child)))

    # Mutation
    # for i in range(maximum_number_of_mutation):
    #     mutated_child = mutation(random.choice(new_generation))
    #     new_generation.append(mutated_child)

    all_paths = new_generation

    trckphmtrix = trckphmtrix * decay
    drnephmtrix = drnephmtrix * decay


def plot_graph(path):
    adjacency_matrix = np.array([
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

    G = nx.DiGraph()

    city_names = {0: 'City 0', 1: 'City 1', 2: 'City 2', 3: 'City 3', 4: 'City 4',
                  5: 'City 5', 6: 'City 6', 7: 'City 7', 8: 'City 8', 9: 'City 9'}

    num_cities = len(cities)
    for i in range(num_cities - 1):
        G.add_edge(city_names[cities[i]], city_names[cities[i + 1]], weight=adjacency_matrix[cities[i]][cities[i + 1]])

    tsp_edges = [(city_names[cities[i]], city_names[cities[i + 1]]) for i in range(len(cities) - 1)]

    theta = np.linspace(0, 2 * np.pi, len(cities), endpoint=False)
    pos = {city_names[c]: (np.cos(angle), np.sin(angle)) for c, angle in zip(cities, theta)}

    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8, connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_edges(G, pos, edgelist=tsp_edges, edge_color='red', width=2, arrows=True)

    total_cost = sum(adjacency_matrix[cities[i], cities[i + 1]] for i in range(num_cities - 1))
    plt.text(0, -0.5, f'Total Cost: {total_cost}', fontsize=12, ha='center')

    plt.show()


all_paths.sort(key=lambda x: x[1])
print(all_paths[0])
ans, cost = all_paths[0]
ans.append(ans[0])
print(ans)
print(cost)

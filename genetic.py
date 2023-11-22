
import numpy as np

import random 

POPULATION_SIZE = 100

GENES = [0,1,2,3,4,5,6,7,8,9,10]
 
TARGET = [0, 8, 7, 2, 1, 6, 5, 9, 3, 4, 0]

old_population = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]

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

class Genetic(object):
    def __init__(self, chromosome) -> None:
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()
    
    @classmethod
    def mutate_genes(self):
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_gnome(self):
        global TARGET
        gnome_len = len(TARGET)
        return [self.mutate_genes() for _ in range(gnome_len)]

    def mate(self, par2):
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            prob =  random.random()
            if prob < 0.45:
                child_chromosome.append(gp1)
            elif prob < 0.90:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(self.mutate_genes())
        return Genetic(child_chromosome)
    
    def cal_fitness(self):
        global TARGET
        fitness = 0
        for gs, gt in zip(self.chromosome, TARGET):
            if gs != gt: fitness += 1
        return fitness


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


generation = 1
found = False
population = []

for _ in range(POPULATION_SIZE):
    gnome = Genetic.create_gnome()
    population.append(Genetic(gnome))

while not found:
    population = sorted(population, key=lambda x:x.fitness)
    if population[0].fitness <= 0:
        found = True
        break
    new_generation = []
    s = int((10*POPULATION_SIZE)/100)
    new_generation.extend(population[:s])
    s = int((90*POPULATION_SIZE)/100)
    for _ in range(s):
        parent1 = random.choice(population[:50])
        parent2 = random.choice(population[:50])
        child = parent1.mate(parent2)
        new_generation.append(child)
    population = new_generation
    print("Generation: {}\tPath: {}\tFitness: {}".format(generation, 
		(population[0].chromosome), 
		population[0].fitness)) 
    generation += 1

print("Generation: {}\tPath: {}\tFitness: {}".format(generation, 
		(population[0].chromosome), 
		population[0].fitness)) 

start = 0
tsp_set = set(range(len(old_population)))
tsp_set.remove(start)
path, cost = tsp(start, start, tsp_set)
tsp_set.add(start)

path.append(start)
print(f"Path: {path}")
print(f"Cost: {cost}")
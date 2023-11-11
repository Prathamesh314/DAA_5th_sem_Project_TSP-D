import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

cities = [0, 8, 7, 2, 1, 6, 5, 9, 3, 4, 0]

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

theta = np.linspace(0, 2*np.pi, len(cities), endpoint=False)
pos = {city_names[c]: (np.cos(angle), np.sin(angle)) for c, angle in zip(cities, theta)}

nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8, connectionstyle='arc3,rad=0.1')

nx.draw_networkx_edges(G, pos, edgelist=tsp_edges, edge_color='red', width=2, arrows=True)

total_cost = sum(adjacency_matrix[cities[i], cities[i + 1]] for i in range(num_cities - 1))
plt.text(0, -0.5, f'Total Cost: {total_cost}', fontsize=12, ha='center')

plt.show()

# =======================================================
# Write a Python function that takes a NetworkX graph as 
# input and returns the number of nodes in the graph.
# =======================================================

import networkx as nx
from networkx.classes.function import path_weight
import os

# Loading in txt file
curr_dir = os.path.dirname(__file__) # get the current directory of this file
edges_fil = curr_dir + "/" + "hw8.txt"

file = open(edges_fil)

g = nx.DiGraph()

# Create graph
edges = []

for line in file.readlines():
    node1, node2, weight = line.split(",")
    weight = float(weight)
    edges.append((node1, node2, weight))

print(edges)
g.add_weighted_edges_from(edges)

# print all nodes
print("\n All nodes in the graph: ", g.nodes) 


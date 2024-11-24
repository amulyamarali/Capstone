from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import networkx as nx
import matplotlib.pyplot as plt
import ast

# Assuming LinearWeightNorm is defined in functional module
from functional import LinearWeightNorm

# *************** DATA SET RELATED CODE *************** #
# Define a small set of triples for the dummy knowledge graph
triples = []
file_name = "../data/final_triple.txt"

with open(file_name, 'r') as file:
    # Read the contents of the file
    file_content = file.read()

    # Convert the string representation of the list back to an actual list
    new_data = ast.literal_eval(file_content)

    # Append the new data to the existing list
    if isinstance(new_data, list):
        triples.extend(new_data)
    else:
        print("The data in the file is not a list.")

# Generate a KG representation using NetworkX
# Create a directed graph
G = nx.DiGraph()

# Add triples to the graph
for head, relation, tail in triples:
    G.add_edge(head, tail, label=relation)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=100, font_size=3,
        node_color='lightblue', font_weight='bold', font_color='darkred',
        bbox=dict(facecolor='lightblue', edgecolor='none', boxstyle='round,pad=0.3'))
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=3)
plt.show()
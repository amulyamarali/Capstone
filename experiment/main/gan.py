import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import networkx as nx
import matplotlib.pyplot as plt
import ast

from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
# Assuming LinearWeightNorm is defined in functional module
from functional import LinearWeightNorm





# *************** DATA SET RELATED CODE *************** # 
# Define a small set of triples for the dummy knowledge graph
triples = []
file_name = "../data/triples.txt"

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
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10,
        node_color='lightblue', font_weight='bold', font_color='darkred')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

# Convert the list of triples to a NumPy array
triples_array = np.array(triples)
# print("Triples Array:\n", triples_array)

# Create a TriplesFactory from the triples
triples_factory = TriplesFactory.from_labeled_triples(triples_array)
# print("Triples Factory:\n", triples_factory)

# Function to safely split triples and handle errors


def safe_split(triples_factory, ratios):
    try:
        training, testing = triples_factory.split(ratios)
        return training, testing
    except ValueError as e:
        print("Error during split:", e)
        return None, None


# Attempt to split the triples into training and testing sets
training, testing = safe_split(triples_factory, [0.8, 0.2])
if training is None:
    print("Failed to split triples. Exiting.")
    exit(1)


# ******************** Define the RESCAL model and train it using the pipeline *************************
# result = pipeline(
#     model='RESCAL',
#     training=training,
#     testing=testing,
#     training_kwargs=dict(num_epochs=100, batch_size=2),
#     optimizer_kwargs=dict(lr=0.01),
# )

# # save the RESCAL embedding model
# torch.save(result.model, 'rescal_model.pth')

# # Extract the entity and relation embeddings
# entity_embeddings = result.model.entity_representations[0](
#     indices=None).cpu().detach().numpy()
# relation_embeddings = result.model.relation_representations[0](
#     indices=None).cpu().detach().numpy()


# For saved RESCAL model 
result = torch.load("../models/rescal_model.pth")

entity_embeddings = result.entity_representations[0](
    indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](
    indices=None).cpu().detach().numpy()



# Create a simple knowledge graph
G = nx.Graph()

entities = list(triples_factory.entity_to_id.keys())
entity_to_id = {entity: idx for idx, entity in enumerate(entities)}

# Generate the nodes list
nodes = [(entity_to_id[entity], {
        'feature': entity_embeddings[entity_to_id[entity]].tolist()}) for entity in entities]

# Generate the edges list
edges = [(entity_to_id[head], entity_to_id[tail]) for head, _, tail in triples]

# Print nodes and edges
# print("Nodes:\n", nodes, len(nodes))
# print("Edges:\n", edges, len(edges))

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Extract node features
node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])

# print("Node Features:")
# print(node_features)
nx.draw(G, with_labels=True)
plt.show()


# *************** SPARSITY DETECTION CODE *************** #
def gini_index(array):
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    print("array:", array)
    print("index:", index)
    print("shape:", array.shape[0])
    n = array.shape[0]
    return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)


# Calculate the degree of each node
degrees = np.array([G.degree(n) for n in G.nodes])

# degrees is no.of incoming and outgoing edges for each node

print("degrees:", degrees)

# Normalize node degrees
max_total_degree = np.max(degrees)
normalized_degrees = degrees / max_total_degree

# Calculate the Gini index
gini = gini_index(normalized_degrees)
print("Gini Index:", gini)

# Threshold for Gini index to determine sparsity
gini_threshold = 0.6  # This threshold can be adjusted

is_incomplete = gini > gini_threshold
print("Is the Knowledge Graph incomplete?", is_incomplete)

# Identify sparse nodes based on a threshold
threshold = 0.6 
sparse_nodes_indices = np.where(normalized_degrees < threshold)[0]
sparse_nodes = [n for n in sparse_nodes_indices]

print("\nSparse nodes:", sparse_nodes)



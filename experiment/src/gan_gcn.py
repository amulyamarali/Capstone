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
# nx.draw(G, pos, with_labels=True, node_size=2000, font_size=3,
#         node_color='lightblue', font_weight='bold', font_color='darkred')
# edge_labels = nx.get_edge_attributes(G, 'label')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# plt.show()

# Convert the list of triples to a NumPy array
triples_array = np.array(triples)

# Create a TriplesFactory from the triples
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

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

# ******************** Using RESCAL Model for embeddings *************************

# For saved RESCAL model 
result = torch.load("../models/conve_model.pth")

entity_embeddings = result.entity_representations[0](
    indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](
    indices=None).cpu().detach().numpy()

# Create a simple knowledge graph
G = nx.Graph()

entities = list(triples_factory.entity_to_id.keys())
entity_to_id = {entity: idx for idx, entity in enumerate(entities)}

# print("entity_to_id:", entity_to_id)

# Generate the nodes list
nodes = [(entity_to_id[entity], {
        'feature': entity_embeddings[entity_to_id[entity]].tolist()}) for entity in entities]

# Generate the edges list
edges = [(entity_to_id[head], entity_to_id[tail]) for head, _, tail in triples]

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Extract node features
node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])


# *************** SPARSITY DETECTION CODE *************** #
def calculate_density(G):
    densities = {}
    for node in G.nodes:
        degree = G.degree(node)
        if G.number_of_nodes() > 1:
            density = degree / (G.number_of_nodes() - 1)
        else:
            density = 0
        densities[node] = density
    return densities

densities = calculate_density(G)

# Sort nodes by density in descending order
sorted_nodes = sorted(densities, key=densities.get, reverse=True)

# Get the top seven densely populated nodes
top_seven_dense_nodes = sorted_nodes[:7]

# Determine the least densely populated node among the top seven
least_dense_of_top_seven = min(top_seven_dense_nodes, key=densities.get)
least_dense_of_top_seven_density = densities[least_dense_of_top_seven]

print("\nTop seven densely populated nodes:", top_seven_dense_nodes)
print("\nNode densities of top seven:", {node: densities[node] for node in top_seven_dense_nodes})
print("\nLeast dense node among top seven densely populated nodes:", least_dense_of_top_seven, "with density:", least_dense_of_top_seven_density)

# Highlight the top seven densely populated nodes in yellow
node_colors = ['yellow' if node in top_seven_dense_nodes else 'lightblue' for node in G.nodes]
node_sizes = [1000 if node in top_seven_dense_nodes else 100 for node in G.nodes]

# Reposition nodes to ensure top seven densely populated nodes are not overlapped
pos = nx.spring_layout(G)
for i, node in enumerate(top_seven_dense_nodes):
    pos[node] = (pos[node][0], pos[node][1] + 0.1 * i)  # Adjust position to avoid overlap

nx.draw(G, pos, with_labels=True, node_size=node_sizes, font_size=5,
        node_color=node_colors, font_weight='bold', font_color='darkred')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

# *************** FIND SPARSE NODE AND ITS NEIGHBORHOOD EMBEDDINGS *************** #
# Find the first node with density less than 0.1
sparse_node = None
for node in densities:
    if densities[node] < 0.1:
        sparse_node = node
        break

if sparse_node is not None:
    print(f"\nFirst node with density less than 0.1: {sparse_node} with density {densities[sparse_node]}")
    neighbors = list(G.neighbors(sparse_node))
    print(f"Neighbors of node {sparse_node}:", neighbors)
    neighbor_embeddings = [entity_embeddings[neighbor] for neighbor in neighbors]
    
    print(f"\nNeighborhood embeddings of node {sparse_node}:")
    for i, embedding in enumerate(neighbor_embeddings):
        print(f"Neighbor {neighbors[i]}: {embedding}")
else:
    print("No node found with density less than 0.1.")


# *************** GAN MODEL DEFINITIONS *************** #
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# *************** HYPERPARAMETERS *************** #
input_dim = len(neighbor_embeddings[0])
output_dim = input_dim
lr = 0.001
num_epochs = 1000
batch_size = len(neighbor_embeddings)

# Instantiate the models
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# Convert neighborhood embeddings to tensor
real_data = torch.tensor(entity_embeddings, dtype=torch.float)

# *************** TRAINING THE GAN *************** #
for epoch in range(num_epochs):
    # Train discriminator
    optimizer_d.zero_grad()

    # Real data
    real_data = torch.tensor(entity_embeddings, dtype=torch.float)
    real_labels = torch.ones(real_data.size(0), 1)  # Match size with real_data
    outputs = discriminator(real_data)
    
    # Ensure the output and label sizes match
    if outputs.size() != real_labels.size():
        raise ValueError(f"Size mismatch: outputs size {outputs.size()} does not match real_labels size {real_labels.size()}")

    d_loss_real = criterion(outputs, real_labels)

    # Fake data
    noise = torch.randn(real_data.size(0), input_dim)  # Use the same batch size
    fake_data = generator(noise)
    fake_labels = torch.zeros(fake_data.size(0), 1)  # Match size with fake_data
    
    outputs = discriminator(fake_data.detach())
    
    # Ensure the output and label sizes match
    if outputs.size() != fake_labels.size():
        raise ValueError(f"Size mismatch: outputs size {outputs.size()} does not match fake_labels size {fake_labels.size()}")

    d_loss_fake = criterion(outputs, fake_labels)

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_d.step()

    # Train generator
    optimizer_g.zero_grad()

    noise = torch.randn(real_data.size(0), input_dim)  # Use the same batch size
    fake_data = generator(noise)
    
    outputs = discriminator(fake_data)
    
    # Ensure the output and label sizes match
    if outputs.size() != real_labels.size():
        raise ValueError(f"Size mismatch: outputs size {outputs.size()} does not match real_labels size {real_labels.size()}")

    g_loss = criterion(outputs, real_labels)

    g_loss.backward()
    optimizer_g.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")


# Convert neighborhood embeddings to tensor
neighbor_data = torch.tensor(neighbor_embeddings, dtype=torch.float)

# Generate a new node embedding
with torch.no_grad():
    noise = torch.randn(1, input_dim)
    new_node_embedding = generator(neighbor_data).numpy().flatten()

# print("\nGenerated embedding for the new node:")
# print(new_node_embedding)



# *************** FIND NEAREST EXISTING NODE (cosine similarity) *************** #
from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity between new node embedding and existing node embeddings
existing_embeddings = np.array([data['feature'] for _, data in G.nodes(data=True)])
cos_similarities = cosine_similarity([new_node_embedding], existing_embeddings).flatten()

# Find the index of the nearest existing node
nearest_idx = np.argmax(cos_similarities)
nearest_node = list(G.nodes)[nearest_idx]

# Decode the real-world name of the nearest existing node
# Here we assume that node names are stored in the 'real_name' attribute in the graph
# Adjust this according to your actual attribute names or data

sparse_node = entities[sparse_node]
print(f"\nSparse node: {sparse_node}")

real_name_of_nearest_node = entities[nearest_node]  # Adjust this line if you have a mapping

print(f"\nNearest existing node to the generated node is {nearest_node} with cosine similarity: {cos_similarities[nearest_idx]}")

# Assuming node names are directly mapped to their IDs
print(f"Real-world name of the nearest node: {real_name_of_nearest_node}")



# *************** PREDICT RELATION BETWEEN SPARSE NODE AND NEW NODE (cosine similarity) *************** #

# Get the list of relations from the triples factory
relations = list(triples_factory.relation_to_id.keys())

print("\nRelations:", relations)
relation_to_id = {relation: idx for idx, relation in enumerate(relations)}


# Get the embedding of the sparse node
sparse_node_embedding = entity_embeddings[entity_to_id[sparse_node]]
# print("\nSparse node embedding:", sparse_node_embedding)


# Function to calculate relation scores between two nodes
def calculate_relation_scores(head_embedding, tail_embedding, relation_embeddings):
    scores = []
    for relation_embedding in relation_embeddings:
        score = np.dot(head_embedding + relation_embedding, tail_embedding)
        scores.append(score)
    return scores

# Calculate relation scores between the sparse node and the new node
relation_scores = calculate_relation_scores(sparse_node_embedding, new_node_embedding, relation_embeddings)

# Find the relation with the highest score
best_relation_idx = np.argmax(relation_scores)
best_relation = relations[best_relation_idx]

print("\nRelation scores:", relation_scores)

print(f"\nPredicted relation between the new node and the sparse node: {best_relation}")
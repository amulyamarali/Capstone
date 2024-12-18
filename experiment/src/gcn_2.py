import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

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

# ******************** Define the RESCAL model and train it using the pipeline *************************

# For saved RESCAL model
result = torch.load("../new_models/rotate_model.pth", map_location='cuda' if torch.cuda.is_available() else 'cpu')

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

# Print nodes and edges
# print("Nodes:\n", nodes, len(nodes))
# print("Edges:\n", edges, len(edges))

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Extract node features
node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])

# print("Node Features:")
# print(node_features)
# nx.draw(G, with_labels=True)
# plt.show()

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
print("\nNode densities of top seven:", {
      node: densities[node] for node in top_seven_dense_nodes})
print("\nLeast dense node among top seven densely populated nodes:",
      least_dense_of_top_seven, "with density:", least_dense_of_top_seven_density)

# Highlight the top seven densely populated nodes in yellow
node_colors = [
    'yellow' if node in top_seven_dense_nodes else 'lightblue' for node in G.nodes]
node_sizes = [
    1000 if node in top_seven_dense_nodes else 100 for node in G.nodes]

# Reposition nodes to ensure top seven densely populated nodes are not overlapped
pos = nx.spring_layout(G)
for i, node in enumerate(top_seven_dense_nodes):
    # Adjust position to avoid overlap
    pos[node] = (pos[node][0], pos[node][1] + 0.1 * i)

nx.draw(G, pos, with_labels=True, node_size=node_sizes, font_size=5,
        node_color=node_colors, font_weight='bold', font_color='darkred')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

# *************** FIND SPARSE NODE AND ITS NEIGHBORHOOD EMBEDDINGS *************** #
# Find the first node with density less than 0.1
# sparse_node = least_dense_of_top_seven
print("Densities:", densities)
print("TYPE OF DENSITIES:", type(densities))    

# exit(0)
sparse_node = None
for node in densities:
    if densities[node] < 0.01:
        print("TYPE THIS IS ",type(node))
        sparse_node = node
        break

# # print dimensions of entity_embeddings
# print("\nSparse node:", entity_embeddings[sparse_node].shape)
# print("least_dense_of_top_seven:", entity_embeddings[least_dense_of_top_seven].shape)

# exit(0)

if sparse_node is not None:
    print(
        f"\nFirst node with density less than 0.1: {sparse_node} with density {densities[sparse_node]}")
    neighbors = list(G.neighbors(sparse_node))
    print(f"Neighbors of node {sparse_node}:", neighbors)
    neighbor_embeddings = [entity_embeddings[neighbor]
                        for neighbor in neighbors]

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
        raise ValueError(
            f"Size mismatch: outputs size {outputs.size()} does not match real_labels size {real_labels.size()}")

    d_loss_real = criterion(outputs, real_labels)

    # Fake data
    # Use the same batch size
    noise = torch.randn(real_data.size(0), input_dim)
    fake_data = generator(noise)
    # Match size with fake_data
    fake_labels = torch.zeros(fake_data.size(0), 1)

    outputs = discriminator(fake_data.detach())

    # Ensure the output and label sizes match
    if outputs.size() != fake_labels.size():
        raise ValueError(
            f"Size mismatch: outputs size {outputs.size()} does not match fake_labels size {fake_labels.size()}")

    d_loss_fake = criterion(outputs, fake_labels)

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_d.step()

    # Train generator
    optimizer_g.zero_grad()

    # Use the same batch size
    noise = torch.randn(real_data.size(0), input_dim)
    fake_data = generator(noise)

    outputs = discriminator(fake_data)

    # Ensure the output and label sizes match
    if outputs.size() != real_labels.size():
        raise ValueError(
            f"Size mismatch: outputs size {outputs.size()} does not match real_labels size {real_labels.size()}")

    g_loss = criterion(outputs, real_labels)

    g_loss.backward()
    optimizer_g.step()

    # if epoch % 10 == 0:
    #     print(
    #         f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")


# Convert neighborhood embeddings to tensor
real_data = torch.tensor(neighbor_embeddings, dtype=torch.float)

# Generate a new node embedding
with torch.no_grad():
    noise = torch.randn(1, input_dim)
    new_node_embedding = generator(real_data).numpy().flatten()

print("\nGenerated embedding for the new node:")
print(new_node_embedding.shape)


# *************** FIND NEAREST EXISTING NODE (cosine similarity) *************** #
# Convert neighborhood embeddings and new node embedding to real values by using their magnitudes
real_neighbor_embeddings = np.abs(np.array(neighbor_embeddings))
real_new_node_embedding = np.abs(np.array(new_node_embedding))

# Compute cosine similarity between new node embedding and existing node embeddings
existing_embeddings = np.array([data['feature']
                            for _, data in G.nodes(data=True)])
real_existing_embeddings = np.abs(existing_embeddings)

cos_similarities = cosine_similarity(
    [real_new_node_embedding], real_existing_embeddings).flatten()

# Find the index of the nearest existing node
nearest_idx = np.argmax(cos_similarities)
nearest_node = list(G.nodes)[nearest_idx]

sparse_node = entities[sparse_node]
print(f"\nSparse node: {sparse_node}")

# Decode the real-world name of the nearest existing node
# Assuming node names are directly mapped to their IDs
print(
    f"\nNearest existing node to the generated node is {nearest_node} with cosine similarity: {cos_similarities[nearest_idx]}")

# Adjust this line if you have a mapping
real_name_of_nearest_node = entities[nearest_node]
print(f"Real-world name of the nearest node: {real_name_of_nearest_node}")



# **************************** GCN for Relation Prediction ****************************** #

# Define a simple GCN model for relation prediction
class GCNRelationPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNRelationPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, output_dim)  # Learnable layer for scoring

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def score(self, node_rep1, node_rep2):
        # Use a learnable layer for scoring
        combined_rep = node_rep1 * node_rep2  # Element-wise multiplication
        scores = self.fc(combined_rep)  # Linear transformation
        return F.softmax(scores, dim=0)  # Apply softmax to normalize score

# Create a directed graph
G = nx.DiGraph()
for head, relation, tail in triples:
    G.add_edge(head, tail, label=relation)

triples_array = np.array(triples)
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

# Prepare the dataset
entities = list(triples_factory.entity_to_id.keys())
entity_to_id = {entity: idx for idx, entity in enumerate(entities)}

node_features = torch.tensor(entity_embeddings, dtype=torch.float)

# Generate edges and labels
head_indices = [entity_to_id[head] for head, _, _ in triples]
tail_indices = [entity_to_id[tail] for _, _, tail in triples]
relation_indices = [triples_factory.relation_to_id[relation]
                    for _, relation, _ in triples]
edge_index = torch.tensor([head_indices, tail_indices], dtype=torch.long)
edge_labels = torch.tensor(relation_indices, dtype=torch.long)

# Create PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index)

# Initialize GCN model
input_dim = node_features.shape[1]
hidden_dim = 128
output_dim = len(triples_factory.relation_to_id)  # Number of relations
model = GCNRelationPredictor(input_dim, hidden_dim, output_dim)

# Set up optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Print the number of edges
print(f"Number of edges in the graph: {data.edge_index.shape[1]}")
print(f"Number of edge labels: {edge_labels.size(0)}")

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Ensure edge_labels and predicted_logits are aligned
    edge_labels = edge_labels.to(out.device)
    predicted_logits = out[data.edge_index[0]]

    # Check if the sizes match
    if predicted_logits.size(0) != edge_labels.size(0):
        # Align the sizes by truncating or padding
        min_size = min(predicted_logits.size(0), edge_labels.size(0))
        predicted_logits = predicted_logits[:min_size]
        edge_labels = edge_labels[:min_size]

    # Calculate loss
    loss = criterion(predicted_logits, edge_labels)

    loss.backward()
    optimizer.step()

    # if (epoch + 1) % 20 == 0:
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Add new node and sparse node to the graph
new_node_id = len(entities)  # New node index
sparse_node_id = entity_to_id[sparse_node]  # Existing sparse node index

# Add new node embedding to node_features
new_node_embedding_tensor = torch.tensor(
    new_node_embedding, dtype=torch.float).unsqueeze(0)
node_features = torch.cat([node_features, new_node_embedding_tensor], dim=0)

# Update edge_index to include new edges
# Add edges from the new node to the sparse node and vice versa
new_edges = torch.tensor([[new_node_id, sparse_node_id], [
    sparse_node_id, new_node_id]], dtype=torch.long)
edge_index = torch.cat([data.edge_index, new_edges], dim=1)

# Update data object
data = Data(x=node_features, edge_index=edge_index)

sparse_node_embedding_tensor = torch.tensor(
    entity_embeddings[sparse_node_id], dtype=torch.float).unsqueeze(0)

# Get predictions for the edge between the new node and the sparse node
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    # Get representations for new node and sparse node
    new_node_rep = out[new_node_id]
    sparse_node_rep = out[sparse_node_id]

    # Predict the relation for the edge new_node -> sparse_node
    # Using the learnable scoring function
    relation_scores = model.score(new_node_rep, sparse_node_rep)

    # Find the predicted relation
    predicted_relation = relation_scores.argmax().item()

    # Get the score for the predicted relation
    predicted_relation_score = relation_scores[predicted_relation].item()

    print(
        f"Predicted relation type to connect new node '{real_name_of_nearest_node}' and sparse node '{sparse_node}': {list(triples_factory.relation_to_id.keys())[predicted_relation]}")
    print(f"Score for predicted relation: {predicted_relation_score:.4f}")
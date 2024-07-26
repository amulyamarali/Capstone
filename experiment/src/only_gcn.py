import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from pykeen.triples import TriplesFactory
import networkx as nx
import matplotlib.pyplot as plt
import ast

# Define a simple GCN model for relation prediction


class GCNRelationPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNRelationPredictor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Load and prepare the dataset
triples = []
file_name = "../data/final_fbk_triple.txt"

with open(file_name, 'r') as file:
    file_content = file.read()
    new_data = ast.literal_eval(file_content)
    if isinstance(new_data, list):
        triples.extend(new_data)
    else:
        print("The data in the file is not a list.")

# Create a directed graph
G = nx.DiGraph()
for head, relation, tail in triples:
    G.add_edge(head, tail, label=relation)

triples_array = np.array(triples)
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

# Prepare the dataset
entities = list(triples_factory.entity_to_id.keys())
entity_to_id = {entity: idx for idx, entity in enumerate(entities)}

# Generate node features
# Replace with actual embeddings if available
# For saved RESCAL model
result = torch.load("../models/rotate_fbk_model.pth",
                    map_location='cuda' if torch.cuda.is_available() else 'cpu')

entity_embeddings = result.entity_representations[0](
    indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](
    indices=None).cpu().detach().numpy()
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

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Calculate loss
    # Ensure edge_labels is on the same device as out
    edge_labels = edge_labels.to(out.device)
    # Adjusted to match the output and edge_attr shapes
    loss = criterion(out[data.edge_index[0]], edge_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predict the relation type using GCN
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)

    # Check if output dimensions are correct
    print(f"Model output shape: {out.shape}")

    # Example prediction: Predict the relation for the edges in the dataset
    pred = out[data.edge_index[0]]
    # Get the indices of the max logits
    predicted_relations = pred.argmax(dim=1)

    # Display predictions
    for i, (head, tail) in enumerate(zip(head_indices, tail_indices)):
        predicted_relation = predicted_relations[i].item()
        print(
            f"Head: {entities[head]}, Tail: {entities[tail]}, Predicted Relation: {list(triples_factory.relation_to_id.keys())[predicted_relation]}")
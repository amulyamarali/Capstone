import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define a synthetic knowledge graph (example edges)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Example edges
G = nx.DiGraph(edges)

# Convert the NetworkX graph to a PyTorch Geometric Data object
data = Data()

# Add node features
num_nodes = G.number_of_nodes()
data.num_nodes = num_nodes
data.edge_index = torch.tensor(list(G.edges)).t().contiguous()  # Convert edges to tensor

# Generate random node features for demonstration
feature_dim = 16
data.x = torch.randn(num_nodes, feature_dim)

# Print graph and data attributes for debugging
print("NetworkX Graph Nodes:", G.nodes)
print("NetworkX Graph Edges:", G.edges)
print("Converted Data Object:")
print("  Edge Index:", data.edge_index)
print("  Node Features:", data.x)

# Define the GNN model (GCN)
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Initialize GNN model
model = GCN(input_dim=feature_dim, hidden_dim=32, output_dim=feature_dim)  # Adjust output_dim

# Set model to training mode
model.train()

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out, data.x)
    loss.backward()
    optimizer.step()

# Extract node embeddings
with torch.no_grad():
    model.eval()
    embeddings = model(data.x, data.edge_index).cpu().numpy()

# Visualize or further process node embeddings as needed
print("Node Embeddings:\n", embeddings)

# Example of using node embeddings for similarity calculation
similarities = cosine_similarity(embeddings, embeddings)
print("Cosine Similarity Matrix:\n", similarities)

# Example of generating a new node using GNN
# new_node_features = model(torch.randn(0, feature_dim), data.edge_index).cpu().numpy().flatten()
# print("Generated Node Features:\n", new_node_features)

# Continue with your code to update the graph or perform other tasks

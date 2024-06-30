
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# Load the triples data
triples = [
    ('A', 'likes', 'B'),
    ('B', 'friends', 'C'),
    ('A', 'likes', 'C'),
    ('C', 'likes', 'D'),
    ('D', 'likes', 'E'),
    ('E', 'friends', 'F'),
]

# Convert the list of triples to a NumPy array
triples_array = np.array(triples)
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

result = torch.load("rescal_model.pth",
                    map_location=torch.device('cpu'))

# Extract the entity and relation embeddings
entity_embeddings = result.entity_representations[0](
    indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](
    indices=None).cpu().detach().numpy()

entities = list(triples_factory.entity_to_id.keys())
relations = list(triples_factory.relation_to_id.keys())

# Create the PyTorch Geometric Data object
entity_to_id = triples_factory.entity_to_id
relation_to_id = triples_factory.relation_to_id
edges = [(entity_to_id[head], entity_to_id[tail]) for head, _, tail in triples]

print(entity_to_id)
print(relation_to_id)
print("edges:", edges)

# Generate the nodes list
nodes = [(entity_to_id[entity], {
    'feature': entity_embeddings[entity_to_id[entity]].tolist()}) for entity in entities]

# Generate the edges list
edges = [(entity_to_id[head], entity_to_id[tail]) for head, _, tail in triples]

# Print nodes and edges
print("Nodes:\n", nodes, len(nodes))
print("Edges:\n", edges, len(edges))

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)



# GCN Model Definition

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Random initial features
node_features = torch.randn((len(entities), 50), dtype=torch.float)

print("node_features: ", node_features)

data = Data(x=node_features, edge_index=edge_index)

# Model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(data.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# Create edge_label attribute
data.edge_label = torch.ones(data.edge_index.size(1), dtype=torch.float)

# Training function

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)

    # Perform negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1), method='sparse')

    # Concatenate positive and negative edge indices
    edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        torch.ones(data.edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0).to(device)

    # Model output and loss calculation
    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss

# Test function


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)

    # Perform negative sampling
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1), method='sparse')

    # Concatenate positive and negative edge indices
    edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        torch.ones(data.edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0).to(device)

    out = model.decode(z, edge_label_index).view(-1).sigmoid()
    return roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())


# Train/Test Loop
best_val_auc = final_test_auc = 0
for epoch in range(1, 101):
    loss = train()
    val_auc = test(data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = val_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')


# Define the Generator class
class Generator(nn.Module):
    def __init__(self, embedding_dim):
        super(Generator, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            # output same dimension as embedding_dim
            nn.Linear(128, embedding_dim)
        )

    def forward(self, noise):
        return self.fc(noise)


##################################################
embedding_dim = entity_embeddings.shape[1]
# generator = torch.load("generator_model.pth")
generator = torch.load("generator_model.pth")

# Add a new node and find the most likely link
with torch.no_grad():
    model.eval()

    # Encode the node features
    z = model.encode(data.x, data.edge_index)

    # Generate a new node feature
    new_z = torch.randn(1, embedding_dim).to(device)
    generated_feature = generator(new_z).detach().cpu().numpy().flatten()
    new_node_feature = torch.tensor(generated_feature, dtype=torch.float, device=device).unsqueeze(0)

    # Extend node features
    extended_node_features = torch.cat([data.x, new_node_feature], dim=0)

    # Create edge_label_index for the new node with all existing nodes
    new_node_index = torch.tensor([data.num_nodes], dtype=torch.long, device=device)
    existing_node_indices = torch.arange(data.num_nodes, dtype=torch.long, device=device)
    new_edge_label_index = torch.stack([new_node_index.repeat(existing_node_indices.size(0)), existing_node_indices], dim=0)

    # Encode the extended node features
    extended_z = model.encode(extended_node_features, data.edge_index)

    # Decode the edges between the new node and existing nodes
    new_edge_predictions = model.decode(extended_z, new_edge_label_index).sigmoid()

    print("New Edge Predictions:")
    print(new_edge_predictions)

    # Find the node with the highest likelihood for linking
    most_likely_link_index = new_edge_predictions.argmax().item()
    most_likely_link_node = existing_node_indices[most_likely_link_index].item()

    print(f"The new node is most likely to link with node {most_likely_link_node}.")

# Visualize the updated graph
G = nx.Graph()
edges = torch.cat([data.edge_index, new_edge_label_index[:,most_likely_link_index].unsqueeze(1)], dim=1).cpu().numpy()
G.add_edges_from(edges.T)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold')
plt.show()

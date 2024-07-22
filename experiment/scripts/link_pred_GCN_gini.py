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
import ast


# ********************** METRICS ********************** #
from sklearn.metrics import precision_score, recall_score, f1_score

@torch.no_grad()
def compute_metrics(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)

    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1), method='sparse'
    ).to(device)

    edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        torch.ones(data.edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0).to(device)

    out = model.decode(z, edge_label_index).view(-1).sigmoid()
    predictions = (out > 0.5).float()

    precision = precision_score(edge_label.cpu().numpy(), predictions.cpu().numpy())
    recall = recall_score(edge_label.cpu().numpy(), predictions.cpu().numpy())
    f1 = f1_score(edge_label.cpu().numpy(), predictions.cpu().numpy())

    return precision, recall, f1
# ***************************************************** #

triples = []
file_name = "../data/triples.txt"

with open(file_name, 'r') as file:
    file_content = file.read()
    new_data = ast.literal_eval(file_content)
    if isinstance(new_data, list):
        triples.extend(new_data)
    else:
        print("The data in the file is not a list.")

triples_array = np.array(triples)
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

# result = torch.load("experiment/models/rescal_model.pth", map_location=torch.device('cuda'))
result = torch.load("../models/rescal_model.pth",
                    map_location='cuda' if torch.cuda.is_available() else 'cpu')

entity_embeddings = result.entity_representations[0](indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](indices=None).cpu().detach().numpy()

entities = list(triples_factory.entity_to_id.keys())
relations = list(triples_factory.relation_to_id.keys())

entity_to_id = triples_factory.entity_to_id
relation_to_id = triples_factory.relation_to_id

nodes = [(entity_to_id[entity], {'feature': entity_embeddings[entity_to_id[entity]].tolist()}) for entity in entities]
edges = [(entity_to_id[head], entity_to_id[tail]) for head, _, tail in triples]

###############################################

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
print("Nodes:\n", nodes, len(nodes))
print("Edges:\n", edges, len(edges))

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Extract node features
node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])

print("Node Features:")
print(node_features)
nx.draw(G, with_labels=True)
plt.show()


########################################################

def gini_index(array):
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)


# Calculate the degree of each node
degrees = np.array([G.degree(n) for n in G.nodes])

# Normalize node degrees
max_total_degree = np.max(degrees)
normalized_degrees = degrees / max_total_degree

# Calculate the Gini index
gini = gini_index(degrees)
print("Gini Index:", gini)

# Threshold for Gini index to determine sparsity
gini_threshold = 0.3  # This threshold can be adjusted

is_incomplete = gini > gini_threshold
print("Is the Knowledge Graph incomplete?", is_incomplete)

# Identify sparse nodes based on a threshold
threshold = 0.5 
sparse_nodes_indices = np.where(normalized_degrees < threshold)[0]
sparse_nodes = [n for n in sparse_nodes_indices]

print("\nSparse nodes:", sparse_nodes)
print(len(sparse_nodes))

################################################

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

node_feature_list = np.array([entity_embeddings[entity_to_id[entity]] for entity in entities])
node_features = torch.tensor(node_feature_list, dtype=torch.float)


# Create a mapping from original indices to new sparse node indices
sparse_node_indices_map = {
    sparse_node: idx for idx, sparse_node in enumerate(sparse_nodes)}

print("Sparse Node Indices Map:\n", sparse_node_indices_map)

# Find edges where both source and target nodes are in sparse_node_indices
sparse_edges = [
    (sparse_node_indices_map[source], sparse_node_indices_map[target])
    for source, target in edges
    if source in sparse_node_indices_map and target in sparse_node_indices_map
]

# Convert the list of sparse edges to a tensor
sparse_edge_index = torch.tensor(
    sparse_edges, dtype=torch.long).t().contiguous()
# print("sparse edge index:", sparse_edge_index)

# Create data object for sparse node embeddings
sparse_node_list = np.array(
    [nodes[sparse_node][1]['feature'] for sparse_node in sparse_nodes])
sparse_node_embeddings = torch.tensor(sparse_node_list, dtype=torch.float)

#######################################################


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        print("edge_label_index: ", edge_label_index)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = Data(x=node_features, edge_index=edge_index)
data = data.to(device)  # Ensure data is on the same device
data.edge_label = torch.ones(
    data.edge_index.size(1), dtype=torch.float).to(device)

model = Net(data.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

data_1 = Data(x=sparse_node_embeddings, edge_index=sparse_edge_index)
data_1 = data_1.to(device)
data_1.edge_label = torch.ones(
    data_1.edge_index.size(1), dtype=torch.float).to(device)

#######################################################

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)

    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1), method='sparse'
    ).to(device)

    edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        torch.ones(data.edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0).to(device)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)

    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1), method='sparse'
    ).to(device)

    edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        torch.ones(data.edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0).to(device)

    out = model.decode(z, edge_label_index).view(-1).sigmoid()
    return roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())

best_val_auc = final_test_auc = 0
for epoch in range(1, 101):
    loss = train()
    val_auc = test(data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = val_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')

class Generator(nn.Module):
    def __init__(self, embedding_dim):
        super(Generator, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, noise):
        return self.fc(noise)

# Load the generator model with the correct embedding dimension
embedding_dim = entity_embeddings.shape[1]
generator = torch.load("../models/generator_model.pth").to(device)

# Function to get key from value
def get_key_from_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None

# Metric computation
precision, recall, f1 = compute_metrics(data)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

with torch.no_grad():
    model.eval()
    z = model.encode(data_1.x, data_1.edge_index)

    sparse_node_embeddings = np.array(
        [nodes[sparse_node][1]['feature'] for sparse_node in sparse_nodes])

    # Calculate the mean of the sparse node embeddings
    mean_sparse_node_embedding = np.mean(sparse_node_embeddings, axis=0)
    new_z = torch.tensor(
        mean_sparse_node_embedding, dtype=torch.float32).unsqueeze(0).to(device)

    print("Shape of new_z before passing to generator:", new_z.shape)
    
    generated_feature = generator(new_z).detach().cpu().numpy().flatten()
    new_node_feature = torch.tensor(generated_feature, dtype=torch.float, device=device).unsqueeze(0)

    new_node = (len(nodes), {'feature': generated_feature})

    extended_node_features = torch.cat([data.x, new_node_feature], dim=0)

    new_node_index = torch.tensor([data.num_nodes], dtype=torch.long, device=device)
    existing_node_indices = torch.arange(data.num_nodes, dtype=torch.long, device=device)
    new_edge_label_index = torch.stack([new_node_index.repeat(existing_node_indices.size(0)), existing_node_indices], dim=0)

    print("new node: ", new_node_index)

    # encode new node features to predict the tail entity suing only sparse node features
    extended_z = model.encode(extended_node_features, data_1.edge_index)

    new_edge_predictions = model.decode(extended_z, new_edge_label_index).sigmoid()

    print("New Edge Predictions:")
    print(new_edge_predictions)

    most_likely_link_index = new_edge_predictions.argmax().item()
    most_likely_link_node = existing_node_indices[most_likely_link_index].item()

    # get the original index of the node that gets connected with the new node from sparse nodes index map 
    original_node = get_key_from_value(sparse_node_indices_map, most_likely_link_node)

    print("node index before mapping: ", most_likely_link_node)
    print("original node index: ", original_node)

    print(f"The new node is most likely to link with node {original_node} with the score of {new_edge_predictions[most_likely_link_index]}.")

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
G.add_node(new_node[0], feature=generated_feature)
G.add_edge(new_node[0], original_node)
node_colors = ['red' if node == new_node[0]
                else 'lightblue' for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=node_colors,
            font_weight='bold', font_color='black')
plt.show()

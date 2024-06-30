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
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, add_self_loops

# Assuming LinearWeightNorm is defined in functional module
from functional import LinearWeightNorm

# Define a small set of triples for the dummy knowledge graph
triples = []
file_name = "triples.txt"

with open(file_name, 'r') as file:
    file_content = file.read()
    new_data = ast.literal_eval(file_content)
    if isinstance(new_data, list):
        triples.extend(new_data)
    else:
        print("The data in the file is not a list.")

# Generate a KG representation using NetworkX
G = nx.DiGraph()
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
print("Triples Array:\n", triples_array)

# Create a TriplesFactory from the triples
triples_factory = TriplesFactory.from_labeled_triples(triples_array)
print("Triples Factory:\n", triples_factory)

def safe_split(triples_factory, ratios):
    try:
        training, testing = triples_factory.split(ratios)
        return training, testing
    except ValueError as e:
        print("Error during split:", e)
        return None, None

training, testing = safe_split(triples_factory, [0.8, 0.2])
if training is None:
    print("Failed to split triples. Exiting.")
    exit(1)

result = torch.load("rescal_model.pth")
entity_embeddings = result.entity_representations[0](
    indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](
    indices=None).cpu().detach().numpy()

print("Entity Embeddings:\n", entity_embeddings)
print("Relation Embeddings:\n", relation_embeddings)

entities = list(triples_factory.entity_to_id.keys())
entity_to_id = {entity: idx for idx, entity in enumerate(entities)}

nodes = [(entity_to_id[entity], {
        'feature': entity_embeddings[entity_to_id[entity]].tolist()}) for entity in entities]

edges = [(entity_to_id[head], entity_to_id[tail]) for head, _, tail in triples]

print("Nodes:\n", nodes, len(nodes))
print("Edges:\n", edges, len(edges))

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])

print("Node Features:")
print(node_features)
nx.draw(G, with_labels=True)
plt.show()

def gini_index(array):
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)

degrees = np.array([G.degree(n) for n in G.nodes])
max_total_degree = np.max(degrees)
normalized_degrees = degrees / max_total_degree

gini = gini_index(degrees)
print("Gini Index:", gini)

gini_threshold = 0.3
is_incomplete = gini > gini_threshold
print("Is the Knowledge Graph incomplete?", is_incomplete)

threshold = 0.5
sparse_nodes_indices = np.where(normalized_degrees < threshold)[0]
sparse_nodes = [n for n in sparse_nodes_indices]

print("\nSparse nodes:", sparse_nodes)

# Define the GNN model for link prediction
class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Prepare data for PyTorch Geometric
edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
x = torch.tensor(node_features, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# Train-test split
data = train_test_split_edges(data)

# Initialize the model and optimizer
input_dim = node_features.shape[1]
hidden_dim = 16

model = GAE(GNNEncoder(input_dim, hidden_dim))
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        z = model.encode(data.x, data.train_pos_edge_index)
        auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}, AUC: {auc}, AP: {ap}')

# Define the generator and discriminator models
class OriginalDiscriminator(nn.Module):
    def __init__(self, input_dim=28 ** 2, output_dim=1):
        super(OriginalDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([
            LinearWeightNorm(input_dim, 500),
            LinearWeightNorm(500, 500),
            LinearWeightNorm(500, 250),
            LinearWeightNorm(250, 250),
            LinearWeightNorm(250, 250)
        ])
        self.final = LinearWeightNorm(250, output_dim, weight_scale=1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid layer

    def forward(self, x, feature=False, cuda=False, first=False):
        x = x.view(-1, self.input_dim)
        noise = torch.randn(x.size()) * 0.05 if self.training else torch.Tensor([0])
        if cuda:
            noise = noise.cuda()
        x = x + Variable(noise, requires_grad=False)
        if first:
            return self.layers[0](x)
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = F.elu(m(x))
            noise = torch.randn(x_f.size()) * 0.5 if self.training else torch.Tensor([0])
            if cuda:
                noise = noise.cuda()
            x = (x_f + Variable(noise, requires_grad=False))
        if feature:
            return x_f, self.sigmoid(self.final(x))  # Apply sigmoid here
        return self.sigmoid(self.final(x))  # Apply sigmoid here


class OriginalGenerator(nn.Module):
    def __init__(self, z_dim, output_dim=28 ** 2):
        super(OriginalGenerator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, 500, bias=False)
        self.bn1 = nn.BatchNorm1d(500, affine=False, eps=1e-6, momentum=0.5)
        self.fc2 = nn.Linear(500, 500, bias=False)
        self.bn2 = nn.BatchNorm1d(500, affine=False, eps=1e-6, momentum=0.5)
        self.fc3 = LinearWeightNorm(500, output_dim, weight_scale=1)
        self.bn1_b = Parameter(torch.zeros(500))
        self.bn2_b = Parameter(torch.zeros(500))
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, batch_size, cuda=False, seed=-1):
        x = Variable(torch.rand(batch_size, self.z_dim), requires_grad=False)
        if cuda:
            x = x.cuda()
        x = F.elu(self.bn1(self.fc1(x)) + self.bn1_b)
        x = F.elu(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.tanh(self.fc3(x))
        return x

class Generator(nn.Module):
    def __init__(self, embedding_dim, output_dim=28 ** 2):
        super(Generator, self).__init__()
        self.original_generator = OriginalGenerator(embedding_dim, output_dim=embedding_dim)

    def forward(self, x):
        return self.original_generator(x.size(0))

# Initialize generator
embedding_dim = entity_embeddings.shape[1]
generator = Generator(embedding_dim)

# Generate a new node if the graph is incomplete
if is_incomplete:
    generator.eval()
    sparse_node_embeddings = np.array(
        [nodes[sparse_node][1]['feature'] for sparse_node in sparse_nodes])
    mean_sparse_node_embedding = np.mean(sparse_node_embeddings, axis=0)
    mean_sparse_node_embedding_tensor = torch.tensor(
        mean_sparse_node_embedding, dtype=torch.float32).unsqueeze(0)

    generated_feature = generator(
        mean_sparse_node_embedding_tensor).detach().numpy().flatten()
    new_node = (len(nodes), {'feature': generated_feature})

    print("Generated Node Feature:")
    print(generated_feature)

    print("New Node:", new_node)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    new_node_idx = len(nodes)
    G.add_node(new_node_idx, feature=generated_feature)

    print("Updated Node Features and Graph:")
    node_features = np.array([data['feature']
                              for _, data in G.nodes(data=True)])
    print(node_features)


    # Predict links using the trained GNN model
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

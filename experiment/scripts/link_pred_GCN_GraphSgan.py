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
from torch.nn import functional as F
from torch.autograd import Variable

from torch.nn.parameter import Parameter

# Assuming LinearWeightNorm is defined in functional module
from functional import LinearWeightNorm

triples = []
file_name = "experiment\data\triples.txt"

with open(file_name, 'r') as file:
    file_content = file.read()
    new_data = ast.literal_eval(file_content)
    if isinstance(new_data, list):
        triples.extend(new_data)
    else:
        print("The data in the file is not a list.")

triples_array = np.array(triples)
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

result = torch.load("experiment\models\rescal_model.pth", map_location=torch.device('cuda'))

entity_embeddings = result.entity_representations[0](indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](indices=None).cpu().detach().numpy()

entities = list(triples_factory.entity_to_id.keys())
relations = list(triples_factory.relation_to_id.keys())

entity_to_id = triples_factory.entity_to_id
relation_to_id = triples_factory.relation_to_id

nodes = [(entity_to_id[entity], {'feature': entity_embeddings[entity_to_id[entity]].tolist()}) for entity in entities]
edges = [(entity_to_id[head], entity_to_id[tail]) for head, _, tail in triples]

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

node_feature_list = [entity_embeddings[entity_to_id[entity]] for entity in entities]
node_features = torch.tensor(node_feature_list, dtype=torch.float)

data = Data(x=node_features, edge_index=edge_index)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)  # Ensure data is on the same device

model = Net(data.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

data.edge_label = torch.ones(data.edge_index.size(1), dtype=torch.float).to(device)

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

# saving the model
torch.save(model, "experiment\models\gcn_model_1.pth")

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
        x = Variable(torch.rand(batch_size, self.z_dim), requires_grad=False, volatile=not self.training)
        if cuda:
            x = x.cuda()
        x = F.elu(self.bn1(self.fc1(x)) + self.bn1_b)
        x = F.elu(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.tanh(self.fc3(x))
        return x

class Generator(nn.Module):
    def __init__(self, embedding_dim):
        super(Generator, self).__init__()
        self.embedding_dim = embedding_dim
        self.original_generator = OriginalGenerator(embedding_dim, embedding_dim)

    def forward(self, noise):
        self.original_generator.eval()  # Ensure eval mode for batchnorm
        return self.original_generator(noise.size(0), noise.is_cuda)

embedding_dim = entity_embeddings.shape[1]
generator = torch.load("experiment\models\generator_model_2.pth").to(device)

with torch.no_grad():
    model.eval()
    z = model.encode(data.x, data.edge_index)

    new_z = torch.randn(1, embedding_dim).to(device)
    generated_feature = generator(new_z).detach().cpu().numpy().flatten()
    new_node_feature = torch.tensor(generated_feature, dtype=torch.float, device=device).unsqueeze(0)

    extended_node_features = torch.cat([data.x, new_node_feature], dim=0)

    new_node_index = torch.tensor([data.num_nodes], dtype=torch.long, device=device)
    existing_node_indices = torch.arange(data.num_nodes, dtype=torch.long, device=device)
    new_edge_label_index = torch.stack([new_node_index.repeat(existing_node_indices.size(0)), existing_node_indices], dim=0)

    extended_z = model.encode(extended_node_features, data.edge_index)

    new_edge_predictions = model.decode(extended_z, new_edge_label_index).sigmoid()

    print("New Edge Predictions:")
    print(new_edge_predictions)

    most_likely_link_index = new_edge_predictions.argmax().item()
    most_likely_link_node = existing_node_indices[most_likely_link_index].item()

    print(f"The new node is most likely to link with node {most_likely_link_node} with the score of {new_edge_predictions[most_likely_link_index]}.")

G = nx.Graph()
edges = torch.cat([data.edge_index, new_edge_label_index[:,most_likely_link_index].unsqueeze(1)], dim=1).cpu().numpy()
G.add_edges_from(edges.T)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold')
plt.show()

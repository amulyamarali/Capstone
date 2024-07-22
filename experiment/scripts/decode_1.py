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

# Load triples from a file
triples = []
file_name = "../data/triples.txt"

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

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10,
        node_color='lightblue', font_weight='bold', font_color='darkred')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

# Convert the list of triples to a NumPy array
triples_array = np.array(triples)
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

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

# Load saved RESCAL model
result = torch.load("../models/rescal_model.pth")

entity_embeddings = result.entity_representations[0](indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](indices=None).cpu().detach().numpy()

entities = list(triples_factory.entity_to_id.keys())
entity_to_id = {entity: idx for idx, entity in enumerate(entities)}

# Print nodes and their corresponding indices
print("Nodes and their corresponding indices:")
for entity, idx in entity_to_id.items():
    print(f"Entity: {entity}, Index: {idx}")

nodes = [(entity_to_id[entity], {'feature': entity_embeddings[entity_to_id[entity]].tolist()}) for entity in entities]
edges = [(entity_to_id[head], entity_to_id[tail]) for head, _, tail in triples]

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])
degrees = np.array([G.degree(n) for n in G.nodes])
max_total_degree = np.max(degrees)
normalized_degrees = degrees / max_total_degree

def gini_index(array):
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)

gini = gini_index(degrees)
gini_threshold = 0.3
is_incomplete = gini > gini_threshold

threshold = 0.5
sparse_nodes_indices = np.where(normalized_degrees < threshold)[0]
sparse_nodes = [n for n in sparse_nodes_indices]

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
        self.sigmoid = nn.Sigmoid()

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
            return x_f, self.sigmoid(self.final(x))
        return self.sigmoid(self.final(x))

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
        return self.original_generator(noise.size(0), noise.is_cuda)

class Discriminator(nn.Module):
    def __init__(self, embedding_dim):
        super(Discriminator, self).__init__()
        self.embedding_dim = embedding_dim
        self.original_discriminator = OriginalDiscriminator(embedding_dim, 1)

    def forward(self, embeddings):
        return self.original_discriminator(embeddings)

embedding_dim = entity_embeddings.shape[1]
learning_rate = 0.0002
batch_size = 2
num_epochs = 300

generator = Generator(embedding_dim)
discriminator = Discriminator(embedding_dim)

adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    optimizer_D.zero_grad()

    real_samples = torch.tensor(entity_embeddings[np.random.randint(0, entity_embeddings.shape[0], batch_size)], dtype=torch.float)
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    noise = torch.randn(batch_size, embedding_dim)
    generated_samples = generator(noise)

    real_validity = discriminator(real_samples)
    fake_validity = discriminator(generated_samples)

    real_loss = adversarial_loss(real_validity, real_labels)
    fake_loss = adversarial_loss(fake_validity, fake_labels)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()

    noise = torch.randn(batch_size, embedding_dim)
    generated_samples = generator(noise)

    validity = discriminator(generated_samples)
    g_loss = adversarial_loss(validity, real_labels)

    g_loss.backward()
    optimizer_G.step()

    if epoch % 10 == 0:
        print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

torch.save(generator, '../models/generator_model_2.pth')
torch.save(discriminator, '../models/discriminator_model_2.pth')

if is_incomplete:
    generator.eval()

    sparse_node_embeddings = np.array([nodes[sparse_node][1]['feature'] for sparse_node in sparse_nodes])
    mean_sparse_node_embedding = np.mean(sparse_node_embeddings, axis=0)
    mean_sparse_node_embedding_tensor = torch.tensor(mean_sparse_node_embedding, dtype=torch.float32).unsqueeze(0)

    generated_feature = generator(mean_sparse_node_embedding_tensor).detach().numpy().flatten()
    new_node = (len(nodes), {'feature': generated_feature})

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    from sklearn.metrics.pairwise import cosine_similarity

    existing_node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])
    similarities = cosine_similarity([generated_feature], existing_node_features).flatten()
    most_similar_node_index = np.argmax(similarities)
    most_similar_node = list(G.nodes)[most_similar_node_index]

    print("Most Similar Node Index:", most_similar_node_index)
    print("Most Similar Node:", most_similar_node)
    print("Similarity Score:", similarities[most_similar_node_index])

    original_entity = entities[most_similar_node_index]
    print("Original Entity Label:", original_entity)

    G.add_node(new_node[0], feature=new_node[1]['feature'], label=original_entity)
    G.add_edge(new_node[0], most_similar_node)

    node_colors = []
    node_labels = {}
    for node in G.nodes():
        if node == new_node[0]:
            node_colors.append('red')
            node_labels[node] = original_entity
        else:
            node_colors.append('lightblue')
            node_labels[node] = entities[node]

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=2000, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

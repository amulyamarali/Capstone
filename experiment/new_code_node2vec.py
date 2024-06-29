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
from node2vec import Node2Vec

# Assuming LinearWeightNorm is defined in functional module
from functional import LinearWeightNorm

# Define a small set of triples for the dummy knowledge graph
triples = []
file_name = "triples.txt"

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

# Generate node embeddings using Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
node2vec_model = node2vec.fit(window=10, min_count=1)

# Extract the node embeddings
node_embeddings = {str(node): node2vec_model.wv[str(node)] for node in G.nodes()}



# Print node embeddings
# for node, embedding in node_embeddings.items():
#     print(f"Node: {node}, Embedding: {embedding}")
G = nx.Graph()

# Convert the list of triples to a NumPy array
triples_array = np.array(triples)
# Create a TriplesFactory from the triples
triples_factory = TriplesFactory.from_labeled_triples(triples_array)

entities = list(triples_factory.entity_to_id.keys())

entity_to_id = {entity: idx for idx, entity in enumerate(entities)}

# # Generate the nodes list
# nodes = [(entity_to_id[entity], {
#         'feature': node_embeddings[entity_to_id[entity]].tolist()}) for entity in entities]

# Generate the nodes list
nodes = [(entity_to_id[entity], {
        'feature': node_embeddings[str(entity)].tolist()}) for entity in entities]

# Generate the edges list
edges = [(entity_to_id[head], entity_to_id[tail]) for head, _, tail in triples]

# Print nodes and edges
print("Nodes:\n", nodes, len(nodes))
print("Edges:\n", edges, len(edges))

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Extract node features for the graph
node_features = np.array([embedding for embedding in node_embeddings.values()])

print("Node Features:")
print(node_features)
nx.draw(G, with_labels=True)
plt.show()

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

# Hyperparameters
embedding_dim = node_features.shape[1]
learning_rate = 0.0002
batch_size = 2
num_epochs = 300
print(embedding_dim)

# Initialize generator and discriminator
generator = Generator(embedding_dim)
discriminator = Discriminator(embedding_dim)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    # Train Discriminator
    optimizer_D.zero_grad()

    # Sample real data
    real_samples = torch.tensor(node_features[np.random.randint(
        0, node_features.shape[0], batch_size)], dtype=torch.float)
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # Generate fake data
    noise = torch.randn(batch_size, embedding_dim)
    generated_samples = generator(noise)

    # Discriminator loss
    real_validity = discriminator(real_samples)
    fake_validity = discriminator(generated_samples)

    real_loss = adversarial_loss(real_validity, real_labels)
    fake_loss = adversarial_loss(fake_validity, fake_labels)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()

    # Generate fake data
    noise = torch.randn(batch_size, embedding_dim)
    generated_samples = generator(noise)

    # Generator loss
    validity = discriminator(generated_samples)
    g_loss = adversarial_loss(validity, real_labels)

    g_loss.backward()
    optimizer_G.step()

    # Print the progress
    if epoch % 10 == 0:
        print(
            f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

# Save the Generator model
torch.save(generator, 'generator_model_node2vec.pth')

# Save the Discriminator model
torch.save(discriminator, 'discriminator_node2vec.pth')

# Generate a new node if the graph is incomplete
if is_incomplete:
    generator.eval()  # Set generator to evaluation mode

    # # Check if sparse nodes are present in node_embeddings
    # sparse_node_embeddings = []
    # for sparse_node in sparse_nodes:
    #     sparse_node_key = str(sparse_node)
    #     if sparse_node_key in node_embeddings:
    #         sparse_node_embeddings.append(node_embeddings[sparse_node_key])
    #     else:
    #         print(f"Node {sparse_node} not found in node_embeddings.")

    # if len(sparse_node_embeddings) == 0:
    #     print("No sparse nodes found in node_embeddings. Exiting.")
    #     exit(1)

    # sparse_node_embeddings = np.array(sparse_node_embeddings)

    sparse_node_embeddings = np.array(
        [nodes[sparse_node][1]['feature'] for sparse_node in sparse_nodes])

    # Calculate the mean of the sparse node embeddings
    mean_sparse_node_embedding = np.mean(sparse_node_embeddings, axis=0)
    mean_sparse_node_embedding_tensor = torch.tensor(
        mean_sparse_node_embedding, dtype=torch.float32).unsqueeze(0)

    generated_feature = generator(
        mean_sparse_node_embedding_tensor).detach().numpy().flatten()
    new_node = (len(G.nodes), {'feature': generated_feature})

    print("Generated Node Feature:")
    print(generated_feature)

    print("New Node:", new_node)


    from sklearn.metrics.pairwise import cosine_similarity

    existing_node_features = np.array(
        [data['feature'] for _, data in G.nodes(data=True)]
    )
    similarities = cosine_similarity(
        [generated_feature], existing_node_features).flatten()
    most_similar_node = np.argmax(similarities)

    print("most_similar node:", most_similar_node)
    print("similarity of most_similar_node: ", similarities[most_similar_node])

    # Add new node to the graph
    G.add_node(new_node[0], feature=new_node[1]['feature'])

    # Add new edge based on the most similar node
    G.add_edge(new_node[0], most_similar_node)

    # Retrieve the original representation (if available)
    # original_entity = entities[most_similar_node] # Uncomment if `entities` is defined
    # print("Original Entity Label:", original_entity) # Uncomment if `entities` is defined

    print("Updated Node Features and Graph:")
    node_features = np.array([data['feature']
                              for _, data in G.nodes(data=True)])
    print(node_features)

    nx.draw(G, with_labels=True)
    plt.show()




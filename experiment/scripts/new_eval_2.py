import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import networkx as nx
import matplotlib.pyplot as plt
import ast
from sklearn.metrics.pairwise import cosine_similarity

from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

# Define a small set of triples for the dummy knowledge graph
triples = []
file_name = "experiment\data\triples.txt"

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

# Convert the list of triples to a NumPy array
triples_array = np.array(triples)
print("Triples Array:\n", triples_array)

# Create a TriplesFactory from the triples
triples_factory = TriplesFactory.from_labeled_triples(triples_array)
print("Triples Factory:\n", triples_factory)

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

# For saved RESCAL model
result = torch.load("experiment\models\rescal_model.pth", map_location=torch.device('cpu'))

entity_embeddings = result.entity_representations[0](
    indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](
    indices=None).cpu().detach().numpy()

print("Entity Embeddings:\n", entity_embeddings)
print("Relation Embeddings:\n", relation_embeddings)


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

###############################################
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

################################################

# Threshold for Gini index to determine sparsity
gini_threshold = 0.3  # This threshold can be adjusted

is_incomplete = gini > gini_threshold
print("Is the Knowledge Graph incomplete?", is_incomplete)

# Identify sparse nodes based on a threshold
threshold = 0.5
sparse_nodes_indices = np.where(normalized_degrees < threshold)[0]
sparse_nodes = [n for n in sparse_nodes_indices]

print("\nSparse nodes:", sparse_nodes)


# Original Generator
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
        x = Variable(torch.rand(batch_size, self.z_dim),
                    requires_grad=False, volatile=not self.training)
        if cuda:
            x = x.cuda()
        x = F.elu(self.bn1(self.fc1(x)) + self.bn1_b)
        x = F.elu(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.tanh(self.fc3(x))
        return x

# New Generator using OriginalGenerator layers
class Generator(nn.Module):
    def __init__(self, embedding_dim):
        super(Generator, self).__init__()
        self.embedding_dim = embedding_dim
        self.original_generator = OriginalGenerator(
            embedding_dim, embedding_dim)

    def forward(self, noise):
        return self.original_generator(noise.size(0), noise.is_cuda)


embedding_dim = entity_embeddings.shape[1]
# generator = torch.load("generator_model.pth")
generator = torch.load("experiment\models\generator_model_2.pth")


#  Generate a new node if the graph is incomplete
if is_incomplete:
    
    generator.eval()  # Set generator to evaluation mode

    # z = torch.randn(1, embedding_dim)
    # print("z: ", z)
    # generated_feature = generator(z).detach().numpy().flatten()
    # new_node = (len(nodes), {'feature': generated_feature})

    # Use the embeddings of all sparse nodes as input
    sparse_node_embeddings = np.array(
        [nodes[sparse_node][1]['feature'] for sparse_node in sparse_nodes])

    # Calculate the mean of the sparse node embeddings
    mean_sparse_node_embedding = np.mean(sparse_node_embeddings, axis=0)
    mean_sparse_node_embedding_tensor = torch.tensor(
        mean_sparse_node_embedding, dtype=torch.float32).unsqueeze(0)

    generated_feature = generator(
        mean_sparse_node_embedding_tensor).detach().numpy().flatten()
    new_node = (len(nodes), {'feature': generated_feature})

    print("Generated Node Feature:")
    print(generated_feature)

    print("New Node:", new_node)

    # Create a NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    existing_node_features = np.array(
        [data['feature'] for _, data in G.nodes(data=True)])
    similarities = cosine_similarity(
        [generated_feature], existing_node_features).flatten()
    most_similar_node = np.argmax(similarities)

    # print("Similarities:", similarities)
    print("most_similar node:", most_similar_node)

    print("similarity of most_similar_node: ", similarities[most_similar_node])

    # print the features of existing_node and new_node
    print("Existing Node Feature:", G.nodes[most_similar_node]['feature'])

    # Add new node to the graph
    G.add_node(new_node[0], feature=new_node[1]['feature'])

    print("New Node Feature:", G.nodes[new_node[0]]['feature'])

    G.add_edge(new_node[0], most_similar_node)

    print("Updated Node Features and Graph:")
    node_features = np.array([data['feature']
                            for _, data in G.nodes(data=True)])
    print(node_features)

    nx.draw(G, with_labels=True)
    plt.show()

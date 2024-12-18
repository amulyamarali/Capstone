import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import networkx as nx
import matplotlib.pyplot as plt
import ast

# Define a small set of triples for the dummy knowledge graph
triples = []
file_name = "../data/triples.txt"

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

# # Define the RESCAL model and train it using the pipeline
# result = pipeline(
#     model='RESCAL',
#     training=training,
#     testing=testing,
#     training_kwargs=dict(num_epochs=100, batch_size=2),
#     optimizer_kwargs=dict(lr=0.01),
# )

# # save the RESCAL embedding model
# torch.save(result.model, 'rescal_model.pth')

# # Extract the entity and relation embeddings
# entity_embeddings = result.model.entity_representations[0](
#     indices=None).cpu().detach().numpy()
# relation_embeddings = result.model.relation_representations[0](
#     indices=None).cpu().detach().numpy()


# For saved RESCAL model 
result = torch.load("../models/complex_model.pth")

entity_embeddings = result.entity_representations[0](
    indices=None).cpu().detach().numpy()
relation_embeddings = result.relation_representations[0](
    indices=None).cpu().detach().numpy()

print("Entity Embeddings:\n", entity_embeddings)
print("Relation Embeddings:\n", relation_embeddings)

# Create a simple knowledge graph
G = nx.Graph()
# nodes = [(0, {'feature': [0.1, 0.2]}), 
#         (1, {'feature': [0.3, 0.4]}), 
#         (2, {'feature': [0.5, 0.6]}), 
#         (3, {'feature': [0.7, 0.8]}), 
#         (4, {'feature': [0.9, 1.0]}),
#         (5, {'feature': [1.1, 1.2]}),
#         (6, {'feature': [0.34688088, 0.42952386]})]
# edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3,6)]

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

# Define GAN components
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


class Discriminator(nn.Module):
    def __init__(self, embedding_dim):
        super(Discriminator, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),  # input is embedding_dim
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings):
        return self.fc(embeddings)


# Hyperparameters
embedding_dim = entity_embeddings.shape[1]
learning_rate = 0.0002
batch_size = 2
num_epochs = 200
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
    real_samples = torch.tensor(entity_embeddings[np.random.randint(
        0, entity_embeddings.shape[0], batch_size)], dtype=torch.float)
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
torch.save(generator, '../models/generator_model.pth')

# Save the Discriminator model
torch.save(discriminator, '../models/discriminator_model.pth')

# Generate a new node if the graph is incomplete
# is_incomplete = True  # Assuming the graph is incomplete for demonstration
if is_incomplete:
    z = torch.randn(1, embedding_dim)
    print("z: ", z)
    # z is the noise form which generator will generate the new node
    generated_feature = generator(z).detach().numpy().flatten()
    new_node = (len(nodes), {'feature': generated_feature})

    print("Generated Node Feature:")
    print(generated_feature)

    print("New Node:", new_node)

    # Create a NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Add new node to the graph
    G.add_node(new_node[0], feature=new_node[1]['feature'])

    # Optionally, connect the new node to an existing node (simple heuristic)
    existing_node = np.random.choice(G.nodes)
    G.add_edge(new_node[0], existing_node)

    # existing node to which new node is connected
    print("Existing Node:", existing_node)

    # print the features of existing_node and new_node
    print("Existing Node Feature:", G.nodes[existing_node]['feature'])
    print("New Node Feature:", G.nodes[new_node[0]]['feature'])

    print("Updated Node Features and Graph:")
    node_features = np.array([data['feature']
                            for _, data in G.nodes(data=True)])
    print(node_features)
    

    nx.draw(G, with_labels=True)
    plt.show()

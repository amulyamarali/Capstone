import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Create a simple knowledge graph
G = nx.Graph()
nodes = [(0, {'feature': [0.1, 0.2]}), 
         (1, {'feature': [0.3, 0.4]}), 
         (2, {'feature': [0.5, 0.6]}), 
         (3, {'feature': [0.7, 0.8]}), 
         (4, {'feature': [0.9, 1.0]}),
         (5, {'feature': [1.1, 1.2]}),
         (6, {'feature': [0.34688088, 0.42952386]})]
edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3,6)]
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

################################################

import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Dimensions
input_dim = 2  # Dimension of random noise vector
output_dim = 2  # Dimension of node features

# Instantiate models
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# Hyperparameters
lr = 0.0002
batch_size = 64
epochs = 10000

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Train the GAN
for epoch in range(epochs):
    # Train Discriminator
    optimizer_D.zero_grad()

    # Real samples
    real_samples = torch.tensor(node_features, dtype=torch.float32)
    real_labels = torch.ones(real_samples.size(0), 1)

    # Fake samples
    fake_samples = torch.randn(real_samples.size(0), input_dim)
    generated_samples = generator(fake_samples).detach()
    generated_labels = torch.zeros(real_samples.size(0), 1)

    # Calculate losses
    real_loss = adversarial_loss(discriminator(real_samples), real_labels)
    fake_loss = adversarial_loss(discriminator(generated_samples), generated_labels)
    d_loss = (real_loss + fake_loss) / 2

    # Backward pass and optimization
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()

    # Generate fake samples
    fake_samples = torch.randn(real_samples.size(0), input_dim)
    generated_samples = generator(fake_samples)

    # Labels for generated samples are real (1)
    valid_labels = torch.ones(real_samples.size(0), 1)

    # Calculate generator loss
    g_loss = adversarial_loss(discriminator(generated_samples), valid_labels)

    # Backward pass and optimization
    g_loss.backward()
    optimizer_G.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} / {epochs} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}")

# Generate a new node if the graph is incomplete
if is_incomplete:
    z = torch.randn(1, input_dim)
    generated_feature = generator(z).detach().numpy().flatten()
    new_node = (len(G.nodes), {'feature': generated_feature})

    print("Generated Node Feature:")
    print(generated_feature)

    # Add new node to the graph
    G.add_node(new_node[0], feature=new_node[1]['feature'])

    # Optionally, connect the new node to an existing node (simple heuristic)
    existing_node = np.random.choice(G.nodes)
    G.add_edge(new_node[0], existing_node)

    print("Updated Node Features and Graph:")
    node_features = np.array([data['feature'] for _, data in G.nodes(data=True)])
    print(node_features)

    nx.draw(G, with_labels=True)
    plt.show()



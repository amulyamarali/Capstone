import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define a simple generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, noise):
        return self.fc(noise)

# Initialize the generator
embedding_dim = 50  # Assume the generator output dimension matches node feature size
generator = Generator(input_dim=embedding_dim, output_dim=embedding_dim)

# Pre-existing node embeddings in the knowledge graph
# Assume we have some pre-trained embeddings for the nodes
node_embeddings = np.random.rand(10, embedding_dim)  # 10 existing nodes

# Generate a sample node feature using the generator
noise = torch.randn(1, embedding_dim)
generated_embedding = generator(noise).detach().numpy().flatten()

# Decode the generated embedding by finding the most similar existing node
def decode_embedding(generated_embedding, existing_embeddings):
    # Calculate cosine similarity between generated embedding and existing node embeddings
    similarities = cosine_similarity([generated_embedding], existing_embeddings)
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities[0][most_similar_index]

# Find the most similar node in the existing graph
most_similar_node, similarity_score = decode_embedding(generated_embedding, node_embeddings)

print(f"Most similar existing node index: {most_similar_node}")
print(f"Similarity score: {similarity_score:.4f}")

# Sample code to integrate this new node into a knowledge graph
# Assume a simple graph structure using NetworkX for demonstration

import networkx as nx

# Initialize a simple graph
G = nx.Graph()

# Add existing nodes (for context)
G.add_nodes_from([(i, {'embedding': node_embeddings[i]}) for i in range(node_embeddings.shape[0])])
G.add_edges_from([(0, 1), (2, 3), (4, 5)])  # Add some edges for illustration

# Add the generated node
new_node_id = max(G.nodes) + 1
G.add_node(new_node_id, embedding=generated_embedding)

# Add an edge between the generated node and its most similar existing node
G.add_edge(new_node_id, most_similar_node)

# Print the updated graph
print("Graph Nodes and Edges:")
for node_id, attr in G.nodes(data=True):
    print(f"Node {node_id}: Embedding {attr['embedding'][:3]}...")  # Print partial embeddings for brevity

print("Edges:")
print(G.edges())

# Visualize the graph (optional)
import matplotlib.pyplot as plt

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold')
plt.show()

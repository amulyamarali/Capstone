import torch
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Define a small set of triples for the dummy knowledge graph
triples = [
    ('Alice', 'likes', 'IceCream'),
    ('Bob', 'likes', 'IceCream'),
    ('Alice', 'knows', 'Bob'),
    ('Bob', 'knows', 'Charlie'),
    ('Charlie', 'likes', 'Pizza'),
    ('Charlie', 'dislikes', 'IceCream'),
]

# generate a KG representation using netwrokx
import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add triples to the graph
for head, relation, tail in triples:
    G.add_edge(head, tail, label=relation)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, node_color='lightblue', font_weight='bold', font_color='darkred')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

# Convert the list of triples to a NumPy array
triples_array = np.array(triples)
print("Triples Array:\n", triples_array)

# Create a TriplesFactory from the triples
triples_factory = TriplesFactory.from_labeled_triples(triples_array)
print("Triples Factory:\n", triples_factory)

# Split the triples into training, testing, and validation sets
training, testing, validation = triples_factory.split([0.8, 0.1, 0.1])

# Define the RESCAL model and train it using the pipeline
result = pipeline(
    model='RESCAL',
    training=training,
    testing=testing,
    validation=validation,
    training_kwargs=dict(num_epochs=100, batch_size=2),
    optimizer_kwargs=dict(lr=0.01),  # Specify the learning rate here
)

# Extract the entity and relation embeddings
entity_embeddings = result.model.entity_representations[0](indices=None).cpu().detach().numpy()
relation_embeddings = result.model.relation_representations[0](indices=None).cpu().detach().numpy()

# Example of how to use embeddings
print("Entity Embeddings:\n", entity_embeddings)
print("Relation Embeddings:\n", len(relation_embeddings))

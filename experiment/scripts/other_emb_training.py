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
file_name = "../data/final_fbk_triple.txt"

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
# result_rescal = pipeline(
#     model='RESCAL',
#     training=training,
#     testing=testing,
#     training_kwargs=dict(num_epochs=100, batch_size=2),
#     optimizer_kwargs=dict(lr=0.01),
# )

# # Save the RESCAL embedding model
# torch.save(result_rescal.model, '../models/rescal_fbk_model.pth')

# # Define the TransE model and train it using the pipeline
# result_transe = pipeline(
#     model='TransE',
#     training=training,
#     testing=testing,
#     training_kwargs=dict(num_epochs=100, batch_size=2),
#     optimizer_kwargs=dict(lr=0.01),
# )

# # Save the TransE embedding model
# torch.save(result_transe.model, '../models/transe_fbk_model.pth')

# # Define the DistMult model and train it using the pipeline
# result_distmult = pipeline(
#     model='DistMult',
#     training=training,
#     testing=testing,
#     training_kwargs=dict(num_epochs=100, batch_size=2),
#     optimizer_kwargs=dict(lr=0.01),
# )

# # Save the DistMult embedding model
# torch.save(result_distmult.model, '../models/distmult_fbk_model.pth')

# # Define the HolE model and train it using the pipeline
# result_hole = pipeline(
#     model='HolE',
#     training=training,
#     testing=testing,
#     training_kwargs=dict(num_epochs=100, batch_size=2),
#     optimizer_kwargs=dict(lr=0.01),
# )

# # Save the HolE embedding model
# torch.save(result_hole.model, '../models/hole_fbk_model.pth')

# # Define the ComplEx model and train it using the pipeline
# result_complex = pipeline(
#     model='ComplEx',
#     training=training,
#     testing=testing,
#     training_kwargs=dict(num_epochs=100, batch_size=2),
#     optimizer_kwargs=dict(lr=0.01),
# )

# # Save the ComplEx embedding model
# torch.save(result_complex.model, '../models/complex_fbk_model.pth')

# # Define the ConvE model and train it using the pipeline
# result_conve = pipeline(
#     model='ConvE',
#     training=training,
#     testing=testing,
#     training_kwargs=dict(num_epochs=100, batch_size=2),
#     optimizer_kwargs=dict(lr=0.01),
# )

# # Save the ConvE embedding model
# torch.save(result_conve.model, '../models/conve_fbk_model.pth')


# Train the RotatE model
result = pipeline(
    model='RotatE',
    training=training,
    testing=testing,
    training_kwargs=dict(num_epochs=200, batch_size=2),  # Adjust hyperparameters as needed
    optimizer_kwargs=dict(lr=0.01),  # Learning rate
)

# Save the RotatE model
torch.save(result.model, '../models/rotate2.pth')
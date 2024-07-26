import networkx as nx
import matplotlib.pyplot as plt
import ast

# Define the file path
file_name = "../data/final_fbk_triple.txt"

# Initialize a list to store triples
triples = []
ini = []

# Read and parse the file content
with open(file_name, 'r') as file:
    file_content = file.read()
    try:
        # Convert the string representation of the list back to an actual list
        new_data = ast.literal_eval(file_content)
        if isinstance(new_data, list):
            ini.extend(new_data)
        else:
            print("The data in the file is not a list.")
    except Exception as e:
        print(f"Error parsing file content: {e}")

# Create a directed graph
G = nx.DiGraph()

# Add triples to the graph
count = 0
for elem in ini:
    if len(elem) > 3:
        count += 1
        print(elem)
    else:
        triples.append(elem)

print(count)

for head, relation, tail in triples:
    if G.has_edge(head, tail):
        # If edge exists, append to existing relations
        G[head][tail]['label'].append(relation)
    else:
        # Otherwise, create a new edge
        G.add_edge(head, tail, label=[relation])

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducible layout

# Create the plot
plt.figure(figsize=(20, 20))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')

# Draw edges
nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20)

# Draw labels for nodes
nx.draw_networkx_labels(G, pos, font_size=10, font_color='darkred', font_weight='bold')

# Prepare edge labels
edge_labels = {}
for (u, v, data) in G.edges(data=True):
    # Join multiple labels if they exist
    if isinstance(data['label'], list):
        edge_labels[(u, v)] = ', '.join(data['label'])
    else:
        edge_labels[(u, v)] = data['label']

# Draw edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green', font_size=8)

# Remove axis and boundary
plt.axis('off')

# Set transparent background for the figure
plt.gcf().patch.set_facecolor('none')

# Show the plot
plt.title("Knowledge Graph")
plt.show()

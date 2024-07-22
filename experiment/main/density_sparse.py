import networkx as nx

# Create a graph (example)
G = nx.Graph()

# Add nodes and edges (example)
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2), (1, 3), (2, 4)])

# Calculate the number of nodes and edges
N = G.number_of_nodes()
E = G.number_of_edges()

# Calculate density
density = nx.density(G)

print(f"Number of nodes: {N}")
print(f"Number of edges: {E}")
print(f"Graph density: {density:.4f}")

# Determine if the graph is sparse
if density < 0.5:  # Example threshold for sparsity
    print("The knowledge graph is sparse.")
else:
    print("The knowledge graph is not sparse.")

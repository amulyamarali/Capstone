import numpy as np

# Define the set of triples
triples = [
    ('Alice', 'likes', 'IceCream'),
    ('Bob', 'likes', 'IceCream'),
    ('Alice', 'knows', 'Bob'),
    ('Bob', 'knows', 'Charlie'),
    ('Charlie', 'likes', 'Pizza'),
    ('Charlie', 'dislikes', 'IceCream'),
]

# Create a mapping from node names to indices
node_names = list(set([item for triple in triples for item in [triple[0], triple[2]]]))
node_to_index = {name: index for index, name in enumerate(node_names)}
index_to_node = {index: name for name, index in node_to_index.items()}

# Initialize an adjacency matrix
num_nodes = len(node_names)
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

# Populate the adjacency matrix based on triples
for triple in triples:
    subj, _, obj = triple
    subj_index = node_to_index[subj]
    obj_index = node_to_index[obj]
    adj_matrix[subj_index][obj_index] = 1

# Step 1: Compute node degrees
in_degrees = np.sum(adj_matrix, axis=0)  # In-degree
out_degrees = np.sum(adj_matrix, axis=1)  # Out-degree
# Calculate total degrees as a combination of in-degree and out-degree
total_degrees = in_degrees + out_degrees

# Step 2: Normalize node degrees
max_total_degree = np.max(total_degrees)
normalized_degrees = total_degrees / max_total_degree

# Step 3: Calculate Gini index for node degrees
def gini_index(arr):
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumulative_sums = np.cumsum(sorted_arr)
    gini = (np.sum((2 * np.arange(1, n + 1) - n - 1) * cumulative_sums)) / (n * np.sum(sorted_arr))
    return gini

gini_index_sparsity = gini_index(normalized_degrees)

# Identify sparse nodes based on a threshold
threshold = 0.5 
sparse_nodes_indices = np.where(normalized_degrees < threshold)[0]
sparse_nodes = [index_to_node[index] for index in sparse_nodes_indices]

print("Adjacency Matrix:")
print(adj_matrix)
print("\nGini Index Sparsity:", gini_index_sparsity)
print("\nSparse nodes:", sparse_nodes)

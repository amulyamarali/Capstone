import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt

def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def extract_nodes_and_relations(data_dict):

    # Extracting nodes and relations
    node_dict = {}
    triples = []

    sentence_len = 0
    # Extract nodes from 'ner' section
    for data in data_dict:
        for sentence_ner in data['ner']:
            for entity in sentence_ner:
                start_idx, end_idx, entity_type = entity
                start = start_idx - sentence_len
                end = end_idx - sentence_len
                node = data['sentences'][data['ner'].index(
                    sentence_ner)][start:end + 1]
                node = " ".join(node)
                key = (start_idx, end_idx)
                node_dict[key] = node
                # print(node)
            sentence_len += len(data['sentences'][data['ner'].index(
                    sentence_ner)])

        sentence_len = 0
    # Extract relations from 'relations' section
    for data in data_dict:
        for sentence_rel in data['relations']:
            for rel in sentence_rel:
                src_idx, src_end, dst_idx, dst_end, relation = rel
                src_node = node_dict.get((src_idx, src_end))
                dst_node = node_dict.get((dst_idx, dst_end))
                triples.append((src_node, dst_node, relation))

    nodes = set()
    for node in node_dict.values():
        nodes.add(node)

    print("Nodes:")
    # print(nodes, len(nodes))

    print("\nRelations:")
    print(triples, len(triples))

    return nodes, triples


file_path = 'train.jsonl'
data = read_json_file(file_path)

# print(data)

nodes, triples = extract_nodes_and_relations(data)

file_p = 'experiment\data\triples.txt'
with open(file_p, 'w') as f:
    f.write(str(triples))

# Create a mapping from node names to indices
node2idx = {node: idx for idx, node in enumerate(nodes)}

# Create a mapping from relation types to indices
relation2idx = {relation[2]: idx for idx, relation in enumerate(
    set(rel[2] for rel in triples))}

# # Create adjacency matrix
adj_matrix = np.zeros((len(nodes), len(nodes)))

for rel in triples:
    src, dst, _ = rel
    adj_matrix[node2idx[src], node2idx[dst]] = 1

print("Adjacency Matrix:")
# print(adj_matrix)


# Visualize the graph using networkx
G = nx.Graph()

# Add nodes
for node in nodes:
    G.add_node(node)

# Add edges
for rel in triples:
    src, dst, relation = rel
    G.add_edge(src, dst, label=relation)

# Draw the graph
pos = nx.spring_layout(G)  # positions for all nodes
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue",
        font_size=10, font_weight="bold", edge_color="gray")
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

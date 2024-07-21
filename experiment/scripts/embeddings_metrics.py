import torch
import csv

# Function to load triplets from a text file
def load_triples(file_path):
    triplets = []
    entities = set()
    relations = set()

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            head, relation, tail = row[0], row[2], row[1]
            triplets.append((head, relation, tail))
            entities.update([head, tail])
            relations.add(relation)

    return triplets, list(entities), list(relations)

# Function to convert true_triplets to indices
def convert_triplets_to_indices(true_triplets, entity_to_index, relation_to_index):
    return [(entity_to_index[head], relation_to_index[relation], entity_to_index[tail]) for head, relation, tail in true_triplets]

# Function to compute the score
def compute_score(entity_embeddings, relation_embeddings, head, relation, tail):
    head_emb = entity_embeddings[head]
    relation_emb = relation_embeddings[relation]
    tail_emb = entity_embeddings[tail]
    score = torch.dot(head_emb + relation_emb, tail_emb)
    return score.item()

# Function to evaluate embeddings
def evaluate_embeddings(entity_embeddings, relation_embeddings, true_triplets, all_entities):
    mean_rank = 0
    mrr = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0

    for head, relation, true_tail in true_triplets:
        scores = []
        for entity in all_entities:
            score = compute_score(entity_embeddings, relation_embeddings, head, relation, entity)
            scores.append((score, entity))

        scores.sort(reverse=True, key=lambda x: x[0])  # Assuming higher score means better ranking
        ranks = [entity for _, entity in scores]

        rank_of_true_tail = ranks.index(true_tail) + 1
        mean_rank += rank_of_true_tail
        mrr += 1 / rank_of_true_tail
        hits_at_1 += 1 if rank_of_true_tail == 1 else 0
        hits_at_3 += 1 if rank_of_true_tail <= 3 else 0
        hits_at_10 += 1 if rank_of_true_tail <= 10 else 0

    num_triplets = len(true_triplets)
    mean_rank /= num_triplets
    mrr /= num_triplets
    hits_at_1 /= num_triplets
    hits_at_3 /= num_triplets
    hits_at_10 /= num_triplets

    return mean_rank, mrr, hits_at_1, hits_at_3, hits_at_10


# Function to load and evaluate a model
def load_and_evaluate_model(file_path, triplets_file):
    model = torch.load(file_path, map_location=torch.device('cpu'))
    entity_embeddings = model.entity_representations[0]._embeddings.weight.detach().cpu()
    relation_embeddings = model.relation_representations[0]._embeddings.weight.detach().cpu()

    true_triplets, all_entities_list, all_relations_list = load_triples(triplets_file)

    entity_to_index = {entity: idx for idx, entity in enumerate(all_entities_list)}
    relation_to_index = {relation: idx for idx, relation in enumerate(all_relations_list)}

    true_triplets_indices = convert_triplets_to_indices(true_triplets, entity_to_index, relation_to_index)
    all_entities_indices = list(entity_to_index.values())

    return evaluate_embeddings(entity_embeddings, relation_embeddings, true_triplets_indices, all_entities_indices)

# List of model file paths
model_files = [
    "../models/complex_model.pth",
    "../models/conve_model.pth",
    "../models/distmult_model.pth",
    "../models/hole_model.pth",
    "../models/transe_model.pth"
]

# Triplets file path
triplets_file = "../data/triples_output.txt"

# Evaluate each model and print the results
for model_file in model_files:
    mean_rank, mrr, hits_at_1,hits_at_3, hits_at_10 = load_and_evaluate_model(model_file, triplets_file)
    print(f"Results for {model_file}:")
    print(f"Mean Rank: {mean_rank}")
    print(f"MRR: {mrr}")
    print(f"Hits@1: {hits_at_1}")
    print(f"Hits@3: {hits_at_3}")
    print(f"Hits@10: {hits_at_10}")
    print("\n")

# Function to compute the score for RESCAL
def compute_score_rescal(entity_embeddings, relation_tensor, head, relation, tail):
    head_emb = entity_embeddings[head]
    tail_emb = entity_embeddings[tail]
    relation_mat = relation_tensor[relation].view(50, 50)  # Adjust the shape if necessary
    score = torch.matmul(torch.matmul(head_emb, relation_mat), tail_emb)
    return score.item()

# Function to evaluate RESCAL embeddings
def evaluate_rescal_embeddings(entity_embeddings, relation_tensor, true_triplets, all_entities):
    mean_rank = 0
    mrr = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0

    for head, relation, true_tail in true_triplets:
        scores = []
        for entity in all_entities:
            score = compute_score_rescal(entity_embeddings, relation_tensor, head, relation, entity)
            scores.append((score, entity))

        scores.sort(reverse=True, key=lambda x: x[0])  # Assuming higher score means better ranking
        ranks = [entity for _, entity in scores]

        rank_of_true_tail = ranks.index(true_tail) + 1
        mean_rank += rank_of_true_tail
        mrr += 1 / rank_of_true_tail
        hits_at_1 += 1 if rank_of_true_tail == 1 else 0
        hits_at_3 += 1 if rank_of_true_tail <= 3 else 0
        hits_at_10 += 1 if rank_of_true_tail <= 10 else 0

    num_triplets = len(true_triplets)
    mean_rank /= num_triplets
    mrr /= num_triplets
    hits_at_1 /= num_triplets
    hits_at_3 /= num_triplets
    hits_at_10 /= num_triplets

    return mean_rank, mrr, hits_at_1, hits_at_3, hits_at_10

# Function to load and evaluate a RESCAL model
def load_and_evaluate_rescal(file_path, triplets_file):
    model = torch.load(file_path, map_location=torch.device('cpu'))
    entity_embeddings = model.entity_representations[0]._embeddings.weight.detach().cpu()

    # Extract relation tensor and adjust dimensions if necessary
    relation_tensor = model.relation_representations[0]._embeddings.weight.detach().cpu()

    true_triplets, all_entities_list, all_relations_list = load_triples(triplets_file)

    entity_to_index = {entity: idx for idx, entity in enumerate(all_entities_list)}
    relation_to_index = {relation: idx for idx, relation in enumerate(all_relations_list)}

    true_triplets_indices = convert_triplets_to_indices(true_triplets, entity_to_index, relation_to_index)
    all_entities_indices = list(entity_to_index.values())

    return evaluate_rescal_embeddings(entity_embeddings, relation_tensor, true_triplets_indices, all_entities_indices)

# List of RESCAL model file paths
rescal_model_files = [
    "../models/rescal_model.pth"
]


# Evaluate each RESCAL model and print the results
for model_file in rescal_model_files:
    mean_rank, mrr, hits_at_1,hits_at_3,hits_at_10 = load_and_evaluate_rescal(model_file, triplets_file)
    print(f"Results for {model_file}:")
    print(f"Mean Rank: {mean_rank}")
    print(f"MRR: {mrr}")
    print(f"Hits@1: {hits_at_1}")
    print(f"Hits@3: {hits_at_3}")
    print(f"Hits@10: {hits_at_10}")
    print("\n")
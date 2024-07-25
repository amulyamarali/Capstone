import ast

# Define file paths
input_file = '../data/all_triples.txt'
output_file = '../data/updated_triples.txt'

def swap_elements(triples):
    updated_triples = []
    for triple in triples:
        if len(triple) == 3:
            head, relation, tail = triple
            updated_triples.append((head, tail, relation))
    return updated_triples

def process_file(input_file, output_file):
    updated_triples = []
    
    with open(input_file, 'r') as file:
        for line in file:
            # Remove any leading/trailing whitespace and split by comma
            parts = line.strip().strip('[]').split(', ')
            
            # Make sure we have exactly 3 parts
            if len(parts) == 3:
                # Remove quotes around strings
                parts = [part.strip("'") for part in parts]
                updated_triples.append((parts[0], parts[2], parts[1]))
    
    with open(output_file, 'w') as file:
        for triple in updated_triples:
            # Write each triple in the same format as the input file
            triple = list(triple)
            file.write(f"{triple}\n")

    print(f"Updated triples have been written to {output_file}")

process_file(input_file, output_file)
input_file = '../data/fbk_original.txt'
output_file = '../data/fbk_triple.txt'

def split_and_expand_relations(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                head, relation, tail = parts
                relation_components = relation.split('/')
                for component in relation_components:
                    if component:  # Ensure the component is not empty
                        outfile.write(f"['{head}', '{component}', '{tail}']\n")

split_and_expand_relations(input_file, output_file)

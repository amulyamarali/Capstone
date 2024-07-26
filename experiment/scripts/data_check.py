import ast

file_name = "../data/final_fbk_triple.txt"

# Read and print the content of the file for debugging purposes
with open(file_name, 'r') as file:
    content = file.read()  # Read the entire content of the file

# Parse the file content into a list of triples
try:
    # Convert the string representation of the list into an actual list
    triples = ast.literal_eval(content)
    if isinstance(triples, list) and all(isinstance(t, tuple) and len(t) == 3 for t in triples):
        print("Successfully parsed triples:")
        for triple in triples:
            print(triple)
    else:
        print("Parsed data is not in the expected format.")
except Exception as e:
    print(f"Error parsing the content: {e}")

# Optional: If you need to further process the triples, you can do it here
# For example, saving the parsed triples to a new file or processing them for visualization

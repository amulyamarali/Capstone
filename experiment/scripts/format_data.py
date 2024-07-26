file_path = '../data/fbk_triple.txt'

# Initialize an empty list to store the tuples
tuple_list = []

# Read the content of fbk_triple.txt
with open(file_path, 'r') as file:
    for line in file:
        # Strip whitespace characters and remove enclosing brackets
        line = line.strip()[1:-1]
        # Split the line by comma, considering the possibility of commas inside quotes
        parts = line.split(', ')
        # Remove any enclosing quotes
        parts = [part.strip("'") for part in parts]
        # Convert the list to a tuple and append to the list
        tuple_list.append(tuple(parts))

# Print the list of tuples
print(tuple_list)

# Save the list of tuples to a file
output_file = '../data/final_fbk_triple.txt'
with open(output_file, 'w') as file:
    file.write(str(tuple_list))

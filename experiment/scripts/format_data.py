file_path = '../data/updated_triples.txt'

# Initialize an empty list to store the tuples
tuple_list = []

# Read the content of abcd.txt
with open(file_path, 'r') as file:
    for line in file:
        # Convert the string representation of the list to an actual list
        line_list = eval(line.strip())
        # Convert the list to a tuple and append to the list
        tuple_list.append(tuple(line_list))

# Print the list of tuples
print(tuple_list)

# save the list of tuples to a file
output_file = '../data/final_triple.txt'
with open(output_file, 'w') as file:
    file.write(str(tuple_list))

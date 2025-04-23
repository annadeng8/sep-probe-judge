import pickle

# Load the pickle file
with open('validation_generations3', 'rb') as f:
    data = pickle.load(f)

# Open a text file for writing
with open('validation_generations3.txt', 'w', encoding='utf-8') as f:
    # Write the exact same output as the original print statements
    f.write(f"{type(data)}\n")
    f.write(f"{len(data)}\n")
    f.write(f"{data.keys()}\n")
    
    for k, v in data.items():
        f.write(f"Key: {k}\n")
        f.write(f"Value: {v}\n\n")
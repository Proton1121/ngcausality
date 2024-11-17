import numpy as np

def load_dependencies(file_path):
    """
    Load dependencies from a file.
    
    :param file_path: Path to the text file containing dependencies.
    :return: List of tuples (cause, effect, value) representing the dependencies.
    """
    dependencies = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                cause, effect, value = parts
                dependencies.append((cause, effect, int(value)))
    return dependencies

def create_causal_matrix(dependencies):
    """
    Create a causal matrix from a list of dependencies.

    :param dependencies: List of tuples (cause, effect, value).
    :return: Causal matrix (causes in rows and effects in columns).
    """
    # Extract all unique items (causes and effects)
    unique_items = sorted(set(cause for cause, _, _ in dependencies) | set(effect for _, effect, _ in dependencies))

    # Create a mapping from item names to indices in the matrix
    item_index = {item: idx for idx, item in enumerate(unique_items)}

    # Initialize the causal matrix with zeros
    n = len(unique_items)
    matrix = np.zeros((n, n), dtype=int)

    # Populate the matrix with the given dependencies
    for cause, effect, value in dependencies:
        cause_idx = item_index[cause]
        effect_idx = item_index[effect]
        
        # Causal matrix: If cause -> effect, set matrix[cause_idx, effect_idx] = 1
        if value == 1:
            matrix[cause_idx, effect_idx] = 1

    return matrix, unique_items

def generate_causal_matrix(file_path):
    """
    Generate a causal matrix from the file and return it.

    :param file_path: Path to the input text file containing the dependencies.
    :return: Causal matrix (causes in rows and effects in columns).
    """
    # Load dependencies from the file
    dependencies = load_dependencies(file_path)
    
    # Create the causal matrix
    matrix, unique_items = create_causal_matrix(dependencies)
    
    return matrix, unique_items

def print_matrix(matrix, items):
    """
    Print the causal matrix with item names as headers.

    :param matrix: The causal matrix.
    :param items: The list of unique item names (causes/effects).
    """
    print("\t" + "\t".join(items))  # Print header row
    for idx, row in enumerate(matrix):
        print(f"{items[idx]}\t" + "\t".join(map(str, row)))

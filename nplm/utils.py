import json

def load_vocab(vocab_file):
    """
    Load the vocabulary from a JSON file and return it as a dictionary.
    
    Args:
    - vocab_file (str): Path to the vocabulary JSON file.

    Returns:
    - dict: A dictionary mapping words to their token IDs.
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    return vocab
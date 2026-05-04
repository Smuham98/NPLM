import os
import json
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, jsonl_dir, vocab, context_size=5):
        """
        Initialize the dataset.
        
        Args:
        - jsonl_dir (str): Directory containing the JSONL files.
        - vocab (dict): A dictionary containing the word-to-ID mapping.
        - context_size (int): The number of previous tokens used for predicting the next token.
        """
        self.context_size = context_size
        self.vocab = vocab
        
        # Load the JSONL files directly (without sharding)
        self.data = []
        for split in ["train", "test"]:
            jsonl_file = os.path.join(jsonl_dir, f"{split}_encoded.jsonl")  # Use the non-sharded file names
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    sentence = doc["text"] if isinstance(doc["text"], list) else doc["text"].split()
                    
                    # Debugging: Check for empty sentences
                    if not sentence:
                        print(f"Warning: Empty sentence at index {len(self.data)}. Skipping.")
                        continue  # Skip empty sentences
                    
                    self.data.append(sentence)

        # Convert words to token IDs
        self.data = [self.encode_sentence(sentence) for sentence in self.data]

    def encode_sentence(self, sentence):
        """
        Convert words to token IDs using the vocabulary.
        """
        return [self.vocab.get(word, self.vocab.get('<unk>')) for word in sentence]

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sequence of tokens and its target token for training.
        """
        sentence = self.data[idx]

        # Ensure the sentence length is sufficient for context size
        if len(sentence) < self.context_size:
            print(f"Warning: Sentence too short at index {idx}: {sentence}")
            # Use the last token as the context and target if the sentence is too short
            context = sentence
            target = sentence[-1]  # Target is the last token in this case
        else:
            # Create context and target
            context = sentence[:self.context_size]
            target = sentence[self.context_size]

        # Convert to tensor
        if context:  # Ensure context is not empty
            context_tensor = torch.tensor(context, dtype=torch.long)
        else:
            print(f"Error: Empty context at index {idx}.")
            context_tensor = torch.tensor([self.vocab.get('<unk>')] * self.context_size, dtype=torch.long)  # Default to <unk>

        target_tensor = torch.tensor(target, dtype=torch.long)

        return {"input_ids": context_tensor, "target_ids": target_tensor}
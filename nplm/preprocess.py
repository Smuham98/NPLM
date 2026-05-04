import argparse
import os
import json
from typing import List

def preprocess(input_dir: str, output_dir: str, shard_size: int, lowercase: bool):
    """
    Reads raw data, splits into documents, and saves them as JSONL shards.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Files to process
    train_file = os.path.join(input_dir, "train.txt")
    test_file = os.path.join(input_dir, "test.txt")

    # Process train and test files
    for split, file_path in [("train", train_file), ("test", test_file)]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if lowercase:
            lines = [line.lower() for line in lines]

        # Split the data into shards
        total_lines = len(lines)
        num_shards = (total_lines // shard_size) + 1
        print(f"[INFO] {split}: {total_lines} lines, {num_shards} shards")

        # Save the shards
        for i in range(num_shards):
            shard_lines = lines[i * shard_size : (i + 1) * shard_size]
            shard_filename = os.path.join(output_dir, f"{split}_shard_{i}.jsonl")
            with open(shard_filename, "w", encoding="utf-8") as shard_file:
                for line in shard_lines:
                    shard_file.write(json.dumps({"text": line.strip()}) + "\n")
            print(f"[INFO] Saved {shard_filename}")

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess raw data into JSONL shards.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw text files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save JSONL shards.")
    parser.add_argument("--shard_size", type=int, default=10000, help="Number of documents per shard.")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase the text.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    preprocess(args.input_dir, args.output_dir, args.shard_size, args.lowercase)
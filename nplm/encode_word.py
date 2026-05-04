import argparse
import os
import json
from tqdm import tqdm
from nplm.word_tokenizer import WordTokenizer

def encode_data(jsonl_dir: str, vocab_file: str, output_dir: str):
    """
    Encode the raw text data into token IDs using the provided vocabulary.
    """
    # Load vocabulary
    tokenizer = WordTokenizer.load(vocab_file)

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the shards and encode the documents
    for split in ["train", "test"]:
        input_file = os.path.join(jsonl_dir, f"{split}_shard_0.jsonl")
        output_file = os.path.join(output_dir, f"{split}_encoded.jsonl")

        print(f"[INFO] Processing {split} data...")

        with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
            for line in tqdm(infile, desc=f"Encoding {split} data", unit="doc"):
                doc = json.loads(line)
                tokens = tokenizer.tokenize(doc["text"])
                encoded_ids = tokenizer.encode_tokens(tokens)
                outfile.write(json.dumps({"text": encoded_ids}) + "\n")

        print(f"[INFO] Encoded data saved to {output_file}")

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Encode the dataset into token IDs.")
    parser.add_argument("--jsonl_dir", type=str, required=True, help="Directory containing JSONL files.")
    parser.add_argument("--vocab", type=str, required=True, help="Path to the vocabulary JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the encoded files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    encode_data(args.jsonl_dir, args.vocab, args.output_dir)
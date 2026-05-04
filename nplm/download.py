# nplm/download.py
import argparse
from datasets import load_dataset
import os

def download_dataset(dataset_name: str, out_dir: str):
    # Load dataset using Hugging Face's datasets library
    dataset = load_dataset(dataset_name)

    # Make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save the dataset to the output directory
    dataset.save_to_disk(out_dir)
    print(f"Dataset '{dataset_name}' downloaded and saved to {out_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Download a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to download.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the downloaded dataset.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    download_dataset(args.dataset, args.out_dir)
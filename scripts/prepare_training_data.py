#!/usr/bin/env python3
"""
Prepare training data for position encoding experiments.
Uses C4 dataset (English corpus) as specified in the paper.
"""

import os
import sys
from datasets import load_dataset
from tqdm import tqdm


def prepare_c4_subset(
    output_dir: str = "data/training",
    num_tokens: int = 10_000_000,
    max_seq_length: int = 1024,
    save_file: str = "train_texts.txt"
):
    """
    Prepare a subset of C4 dataset for training.

    Args:
        output_dir: Directory to save the data
        num_tokens: Approximate number of tokens to collect
        max_seq_length: Maximum sequence length
        save_file: Output file name
    """
    print("Loading C4 dataset (English)...")

    # Load C4 dataset - English only
    try:
        ds = load_dataset("c4", "en", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading c4: {e}")
        print("Trying alternative method...")
        ds = load_dataset("c4", "realnews", split="train", streaming=True)

    print(f"Dataset loaded. Collecting ~{num_tokens//1000}K tokens...")

    # Collect texts
    texts = []
    current_tokens = 0

    for i, example in enumerate(tqdm(ds, desc="Collecting texts")):
        if current_tokens >= num_tokens:
            break

        text = example.get("text", "")
        if text and len(text) > 100:  # Skip very short texts
            texts.append(text)
            current_tokens += len(text.split())

    print(f"Collected {len(texts)} text samples")
    print(f"Total tokens (approx): {current_tokens:,}")

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, save_file)

    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    print(f"Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return output_path


def prepare_smaller_subset(
    output_dir: str = "data/training",
    num_samples: int = 1000,
    save_file: str = "train_small.txt"
):
    """
    Prepare a smaller subset for quick testing.
    """
    print(f"Loading C4 dataset (first {num_samples} samples)...")

    try:
        ds = load_dataset("c4", "en", split="train[:1000]", trust_remote_code=True)
    except:
        ds = load_dataset("c4", "en", split="train", streaming=True)
        ds = list(ds.take(num_samples))

    texts = [ex["text"] for ex in ds if ex.get("text") and len(ex["text"]) > 100]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, save_file)

    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    print(f"Saved {len(texts)} samples to: {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/training")
    parser.add_argument("--num-tokens", type=int, default=10_000_000)
    parser.add_argument("--small", action="store_true", help="Use small subset (1000 samples)")
    args = parser.parse_args()

    if args.small:
        prepare_smaller_subset(args.output_dir)
    else:
        prepare_c4_subset(args.output_dir, args.num_tokens)

    print("\nDone! You can now use this data for training.")
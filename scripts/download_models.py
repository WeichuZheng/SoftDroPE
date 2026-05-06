#!/usr/bin/env python3
"""Download base models for the SoftDroPE project."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def download_gpt2(model_dir: str = "checkpoints"):
    """Download GPT-2 model and tokenizer."""
    os.makedirs(model_dir, exist_ok=True)

    print("Downloading GPT-2 small (124M)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Save model
    save_path = os.path.join(model_dir, "gpt2")
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"GPT-2 saved to {save_path}")
    return model, tokenizer


def download_gpt2_medium(model_dir: str = "checkpoints"):
    """Download GPT-2 medium (355M)."""
    os.makedirs(model_dir, exist_ok=True)

    print("Downloading GPT-2 medium (355M)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    save_path = os.path.join(model_dir, "gpt2-medium")
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"GPT-2 medium saved to {save_path}")
    return model, tokenizer


def load_gpt2(model_dir: str = "checkpoints/gpt2"):
    """Load GPT-2 model and tokenizer from local disk."""
    print(f"Loading GPT-2 from {model_dir}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"])
    parser.add_argument("--model-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    if args.model == "gpt2":
        download_gpt2(args.model_dir)
    elif args.model == "gpt2-medium":
        download_gpt2_medium(args.model_dir)

    print("Done!")
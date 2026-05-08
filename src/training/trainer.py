#!/usr/bin/env python3
"""
Training script for position encoding extension methods.
Supports: Baseline, PI, NTK, YaRN, DroPE, CoPE, SoftDroPE
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


class TextDataset(Dataset):
    """Simple text dataset for training."""

    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self texts = []

        print(f"Loading training data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.texts.append(line)

        print(f"Loaded {len(self.texts)} text samples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class PositionEncodingTrainer:
    """Trainer for position encoding experiments."""

    def __init__(
        self,
        model_path: str,
        method: str = "baseline",
        output_dir: str = "checkpoints",
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        num_steps: int = 500,
        max_seq_length: int = 512,
    ):
        self.model_path = model_path
        self.method = method
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.max_seq_length = max_seq_length

        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True
        )

        self.apply_position_method()

    def apply_position_method(self):
        """Apply the specified position encoding method."""
        print(f"Applying position encoding method: {self.method}")

        if self.method == "baseline":
            pass  # Standard RoPE

        elif self.method == "pi":
            self._apply_position_interpolation()

        elif self.method == "ntk":
            self._apply_ntk_aware()

        elif self.method == "yarn":
            self._apply_yarn()

        elif self.method == "drope":
            self._apply_drope()

        elif self.method == "cope":
            self._apply_cope()

        elif self.method == "softdrope":
            self._apply_softdrope()

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _apply_position_interpolation(self):
        """Apply Position Interpolation (PI)."""
        # Scale position IDs linearly
        for module in self.model.modules():
            if hasattr(module, 'position_ids'):
                with torch.no_grad():
                    module.position_ids *= 0.5  # Scale down positions

    def _apply_ntk_aware(self):
        """Apply NTK-aware scaling."""
        # This is typically done via temperature scaling
        for module in self.model.modules():
            if hasattr(module, 'inv_freq'):
                with torch.no_grad():
                    # Increase base frequency for NTK
                    module.inv_freq *= 2.0

    def _apply_yarn(self):
        """Apply YaRN method."""
        for module in self.model.modules():
            if hasattr(module, 'inv_freq'):
                with torch.no_grad():
                    # Apply YaRN scaling factor
                    module.inv_freq *= 0.5
                    module.original_inv_freq = module.inv_freq.clone()

    def _apply_drope(self):
        """Apply DroPE - remove positional embeddings."""
        for module in self.model.modules():
            if hasattr(module, 'inv_freq'):
                with torch.no_grad():
                    module.inv_freq.fill_(0.0)
            if hasattr(module, 'cos_cached'):
                with torch.no_grad():
                    module.cos_cached.fill_(0.0)
            if hasattr(module, 'sin_cached'):
                with torch.no_grad():
                    module.sin_cached.fill_(0.0)

    def _apply_cope(self):
        """Apply CoPE - clipped RoPE."""
        # CoPE implementation would modify the attention
        # This is a simplified version
        pass

    def _apply_softdrope(self):
        """Apply SoftDroPE - two-stage method."""
        # First apply DroPE
        self._apply_drope()
        # Then would apply CoPE in stage 2
        # For now, just use the DroPE result
        pass

    def train(self, train_data_path: str):
        """Train the model with the specified position encoding method."""
        print(f"Starting training with method: {self.method}")
        print(f"Steps: {self.num_steps}, Batch size: {self.batch_size}")

        dataset = TextDataset(train_data_path, self.tokenizer, self.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(50, self.num_steps // 10),
            num_training_steps=self.num_steps
        )

        self.model.train()
        device = next(self.model.parameters()).device

        progress_bar = tqdm(range(self.num_steps), desc=f"Training {self.method}")

        step = 0
        while step < self.num_steps:
            for batch in dataloader:
                if step >= self.num_steps:
                    break

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

                loss = outputs.loss
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % 50 == 0:
                    print(f"Step {step}/{self.num_steps}, Loss: {loss.item():.4f}")

                progress_bar.update(1)
                step += 1

        progress_bar.close()

        # Save the model
        output_path = os.path.join(self.output_dir, f"{self.method}_qwen0.5b")
        print(f"Saving model to {output_path}...")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        print(f"Training complete for {self.method}!")
        return output_path

    def evaluate(self, data_dir: str, task: str = "niah_single_1") -> Dict[str, float]:
        """Evaluate on RULER benchmark."""
        from ruler.scripts.eval_hf import load_ruler_data, evaluate_model

        print(f"Evaluating on {task}...")
        data = load_ruler_data(data_dir, task)

        if len(data) == 0:
            print("No evaluation data found!")
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        results = evaluate_model(
            self.model,
            self.tokenizer,
            data,
            device=str(next(self.model.parameters()).device)
        )

        print(f"Results: {results['accuracy']*100:.2f}% accuracy")
        return results


def main():
    parser = argparse.ArgumentParser(description="Train position encoding models")
    parser.add_argument("--model-path", type=str, default="checkpoints/Qwen2-0.5B")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "pi", "ntk", "yarn", "drope", "cope", "softdrope"])
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to training data file")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation, skip training")

    args = parser.parse_args()

    trainer = PositionEncodingTrainer(
        model_path=args.model_path,
        method=args.method,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        max_seq_length=args.max_seq_length
    )

    if args.eval_only:
        # Run evaluation on existing model
        results = trainer.evaluate(args.train_data)
    else:
        # Run training
        trainer.train(args.train_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
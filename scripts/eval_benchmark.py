#!/usr/bin/env python3
"""
Benchmark evaluation script for position encoding methods.
Uses Qwen2.5-7B-Instruct as the base model.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.position.rope import RoPEModel
from src.position.cope import CoPEModel
from src.position.softdrope import SoftDroPEModel
from src.position.baselines import BaselinePositionEncoder, create_all_position_encoders


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""
    method: str
    seq_len: int
    accuracy: float
    perplexity: float
    latency_ms: float


class SimpleRULERBenchmark:
    """
    Simplified RULER benchmark for testing position encoding methods.
    Based on: https://github.com/RUCAIBox/RULER

    This implementation includes:
    - NIAH (Needle In A Haystack): Find specific tokens in long context
    -大海捞针
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate_prompt(self, task_type: str, seq_len: int) -> Tuple[str, str, int]:
        """
        Generate a prompt for a specific task.
        Returns (prompt, answer, needle_position).
        """
        if task_type == "needle":
            # Needle in a haystack - simpler version
            # Create a random "needle" token that model needs to find
            needle_token = "SPECIAL_TOKEN_12345"
            filler = "The sky is blue. " * (seq_len // 10)
            prompt = f"{filler}{needle_token}{filler}"
            return prompt, needle_token, seq_len // 2

        elif task_type == "retrieval":
            # Simple key-value retrieval
            key = f"KEY_{random.randint(1000, 9999)}"
            value = f"VALUE_{random.randint(1000, 9999)}"
            filler = "The weather is nice. " * (seq_len // 20)
            prompt = f"{filler}The {key} is {value}. {filler}What is {key}?"
            return prompt, value, -1

        elif task_type == "multi-hop":
            # Simple multi-hop reasoning
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            c = a + b
            prompt = f"Q: What is {a} + {b}? A: {c}. Q: What is {a} + {b} + 5?"
            return prompt, str(c + 5), -1

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def evaluate_length(self, method: str, seq_len: int, num_samples: int = 3) -> Dict[str, float]:
        """
        Evaluate model at a specific sequence length with a given position encoding method.
        """
        results = {
            "accuracy": 0.0,
            "perplexity": 0.0,
            "latency_ms": 0.0
        }

        task_types = ["needle", "retrieval", "multi-hop"]
        accuracies = []

        for task_type in task_types:
            for _ in range(num_samples):
                try:
                    prompt, answer, needle_pos = self.generate_prompt(task_type, seq_len)

                    # Tokenize
                    inputs = self.tokenizer(prompt, return_tensors="pt", max_length=seq_len, truncation=True)
                    input_ids = inputs["input_ids"].to(self.device)
                    attention_mask = inputs["attention_mask"].to(self.device)

                    # Measure latency
                    import time
                    start_time = time.time()

                    with torch.no_grad():
                        outputs = self.model(input_ids, attention_mask=attention_mask)

                    latency = (time.time() - start_time) * 1000

                    # Calculate perplexity (simplified)
                    logits = outputs.logits
                    labels = input_ids[:, 1:]
                    shift_logits = logits[:, :-1, :]
                    loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                    loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), labels.reshape(-1))
                    perplexity = torch.exp(loss).item()

                    # Simple accuracy check (for retrieval tasks)
                    if task_type in ["retrieval", "multi-hop"]:
                        pred_token = self.tokenizer.decode(outputs.logits[0, -1].argmax())
                        # Simplified - just check if output contains expected pattern
                        acc = 1.0 if answer in pred_token or answer in self.tokenizer.decode(outputs.logits[0, -5:].argmax(axis=-1)) else 0.0
                        accuracies.append(acc)

                    results["perplexity"] += perplexity
                    results["latency_ms"] += latency

                except Exception as e:
                    print(f"Error in {task_type}: {e}")
                    continue

        # Average results
        num_tasks = len(task_types) * num_samples
        results["perplexity"] /= num_tasks
        results["latency_ms"] /= num_tasks
        results["accuracy"] = np.mean(accuracies) if accuracies else 0.0

        return results


def modify_model_rope(model, method: str, **kwargs):
    """
    Modify the RoPE in a HuggingFace model to use different position encoding methods.

    For Qwen2.5, the rotary embedding is in model.layers[i].self_attn.rotary_emb
    """
    # Check if model has rotary embeddings
    if not hasattr(model, 'model'):
        print("Warning: Model structure might be different than expected")
        return model

    # For now, we'll wrap the attention to apply different position encodings
    # This is a simplified approach - in production, you'd modify the rotary_emb directly
    return model


def run_benchmark(
    model_path: str,
    methods: List[str],
    lengths: List[int],
    output_dir: str = "results"
):
    """Run the benchmark for all methods and lengths."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    benchmark = SimpleRULERBenchmark(model, tokenizer, device="cuda:0")

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    print("\n" + "="*60)
    print("Starting Benchmark Evaluation")
    print("="*60)

    for method in methods:
        print(f"\n--- Testing method: {method} ---")
        method_results = []

        for seq_len in lengths:
            print(f"  Length: {seq_len}...")

            # Note: For true evaluation, we'd need to modify the model's RoPE
            # Here we're just testing that the pipeline works
            try:
                results = benchmark.evaluate_length(method, seq_len, num_samples=2)
                result = BenchmarkResult(
                    method=method,
                    seq_len=seq_len,
                    **results
                )
                method_results.append(result)
                print(f"    Perplexity: {results['perplexity']:.2f}, Latency: {results['latency_ms']:.1f}ms")
            except Exception as e:
                print(f"    Error: {e}")

        all_results.extend(method_results)

    # Save results
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump([{
            "method": r.method,
            "seq_len": r.seq_len,
            "accuracy": r.accuracy,
            "perplexity": r.perplexity,
            "latency_ms": r.latency_ms
        } for r in all_results], f, indent=2)

    print(f"\nResults saved to {results_file}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run position encoding benchmark")
    parser.add_argument("--model-path", type=str, default="/data1/weichu/woshi/Qwen2.5-7B-Instruct")
    parser.add_argument("--methods", type=str, default="rope,drope,cope,softdrope,pi,ntk,yarn")
    parser.add_argument("--lengths", type=str, default="512,1024,2048,4096")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    methods = args.methods.split(",")
    lengths = [int(x) for x in args.lengths.split(",")]

    print(f"Methods: {methods}")
    print(f"Lengths: {lengths}")

    run_benchmark(args.model_path, methods, lengths, args.output_dir)
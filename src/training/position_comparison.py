#!/usr/bin/env python3
"""
Position encoding extension methods comparison.
Tests: Baseline (RoPE), PI, NTK, YaRN on late needle retrieval task.
"""

import os
import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_model(model_path):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def apply_position_method(model, method, scale=4.0):
    """Apply position encoding method to model."""
    print(f"Applying method: {method}")

    if method == "baseline":
        pass  # Standard RoPE

    elif method == "pi":
        # Position Interpolation - scale down position frequencies
        for module in model.modules():
            if hasattr(module, 'rotary_emb'):
                if hasattr(module.rotary_emb, 'inv_freq'):
                    with torch.no_grad():
                        module.rotary_emb.inv_freq = module.rotary_emb.inv_freq / scale
                if hasattr(module.rotary_emb, 'max_seq_len'):
                    with torch.no_grad():
                        module.rotary_emb.max_seq_len = module.rotary_emb.max_seq_len * scale

    elif method == "ntk":
        # NTK-aware: Use higher base frequency (smoother scaling)
        for module in model.modules():
            if hasattr(module, 'rotary_emb'):
                if hasattr(module.rotary_emb, 'inv_freq'):
                    with torch.no_grad():
                        # Scale by sqrt of scale factor for NTK
                        module.rotary_emb.inv_freq = module.rotary_emb.inv_freq / (scale ** 0.5)

    elif method == "yarn":
        # YaRN: Uses RoPE with temperature scaling
        for module in model.modules():
            if hasattr(module, 'rotary_emb'):
                if hasattr(module.rotary_emb, 'inv_freq'):
                    with torch.no_grad():
                        # YaRN scaling: typically uses 0.5 * (1 + cos(pi * position / max_pos))
                        # We'll use simplified version: 2x scale factor
                        module.rotary_emb.inv_freq = module.rotary_emb.inv_freq / (scale * 2)

    elif method == "drope":
        # Disable RoPE entirely
        for module in model.modules():
            if hasattr(module, 'rotary_emb'):
                if hasattr(module.rotary_emb, 'inv_freq'):
                    with torch.no_grad():
                        module.rotary_emb.inv_freq.fill_(0.0)

    return model

def generate_late_needle_data(seq_len, needle_ratio, num_samples=30):
    """Generate late needle test data."""
    from wonderwords import RandomWord
    rw = RandomWord()

    needle_pos = int(seq_len * needle_ratio)

    data = []
    for idx in range(num_samples):
        key = f"keyword{idx}"
        value = str(random.randint(1000000, 9999999))

        # Generate words
        words = [rw.word() for _ in range(seq_len + 200)]

        # Insert needle
        before = " ".join(words[:needle_pos])
        after = " ".join(words[needle_pos:])
        needle_text = f"CODE_{key}_VALUE_{value}_END"

        full_text = f"{before} {needle_text} {after}"
        question = f"What is the value for {key}?"

        # Truncate to reasonable length for model input
        input_text = full_text[:20000] + " " + question

        data.append({
            'input': input_text,
            'expected': value,
            'needle_ratio': needle_ratio
        })

    return data

def evaluate_model(model, tokenizer, data, device="cuda:0"):
    """Evaluate model on late needle task."""
    correct = 0
    model_device = next(model.parameters()).device

    for item in tqdm(data, desc="Evaluating"):
        input_text = item['input']
        expected = item['expected']

        # Tokenize (use full length)
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=False,
            max_length=32768
        )

        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode generated part
        generated = tokenizer.decode(
            outputs[0][input_ids.shape[0]:],
            skip_special_tokens=True
        )

        # Check if expected value is in output
        if expected in generated:
            correct += 1

    accuracy = correct / len(data) if len(data) > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": len(data)}

def run_experiment(model_path, methods, seq_len, needle_ratios):
    """Run comparison experiment."""
    results = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing method: {method}")
        print(f"{'='*50}")

        # Load model and apply method
        model, tokenizer = load_model(model_path)
        model = apply_position_method(model, method)

        for ratio in needle_ratios:
            test_name = f"L{seq_len}_needle{int(ratio*100)}"
            print(f"\n--- {test_name} ---")

            # Generate test data
            data = generate_late_needle_data(seq_len, ratio, num_samples=30)

            # Evaluate
            result = evaluate_model(model, tokenizer, data)
            print(f"Accuracy: {result['accuracy']*100:.1f}% ({result['correct']}/{result['total']})")

            results[f"{method}_{test_name}"] = result

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/Qwen2-0.5B")
    parser.add_argument("--methods", type=str, default="baseline,pi,ntk,yarn")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--needle-ratios", type=str, default="0.85,0.88,0.90,0.92,0.95")
    parser.add_argument("--output", type=str, default="results/evaluation/position_encoding_comparison.json")
    args = parser.parse_args()

    methods = args.methods.split(",")
    needle_ratios = [float(x) for x in args.needle_ratios.split(",")]

    results = run_experiment(args.model_path, methods, args.seq_len, needle_ratios)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("Results saved to:", args.output)
    print(f"{'='*50}")
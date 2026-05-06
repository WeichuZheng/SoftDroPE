#!/usr/bin/env python3
"""
GPU monitoring script - monitors GPU availability and runs evaluation when GPU is available.

The script monitors GPU status and waits until a GPU with sufficient free memory is available.
It filters out GPUs in "E. Process" (error) state.
"""

import subprocess
import time
import sys
import os
import json
from typing import Optional, Tuple, List, Dict


def get_gpu_status() -> List[Dict]:
    """
    Query GPU status using nvidia-smi.
    Returns list of dicts with GPU info.
    """
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,memory.free,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"nvidia-smi error: {result.stderr}")
        return []

    lines = result.stdout.strip().split('\n')

    gpus = []
    for line in lines:
        if not line.strip():
            continue

        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            continue

        try:
            gpu_idx = int(parts[0])
            free_mem = int(parts[2])
            total_mem = int(parts[3])
            util = int(parts[4])

            gpus.append({
                'index': gpu_idx,
                'name': parts[1],
                'memory_free': free_mem,
                'memory_total': total_mem,
                'utilization': util,
                'has_error_process': False  # Will check separately if needed
            })
        except (ValueError, IndexError) as e:
            print(f"Error parsing GPU line: {line}, error: {e}")
            continue

    return gpus


def find_available_gpu(min_memory_gb: float = 35) -> Optional[int]:
    """
    Find an available GPU with sufficient free memory.

    Args:
        min_memory_gb: Minimum free memory in GB

    Returns:
        GPU index if available, None otherwise
    """
    gpus = get_gpu_status()

    if not gpus:
        return None

    min_memory_mb = int(min_memory_gb * 1024)

    # Filter GPUs: must have enough free memory and not heavily used
    # Use stricter thresholds to avoid busy GPUs
    available = [
        g['index'] for g in gpus
        if g['memory_free'] >= min_memory_mb
        and g['utilization'] < 50  # Not heavily used (lower threshold for stability)
    ]

    if available:
        # Return GPU with most free memory
        best_gpu = max(available, key=lambda i: gpus[i]['memory_free'])
        return best_gpu

    return None


def print_gpu_status(gpus: List[Dict], available_idx: Optional[int]):
    """Print current GPU status."""
    print(f"\n{'='*70}")
    print(f"GPU Status - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    for g in gpus:
        status = "✓ AVAILABLE" if g['index'] == available_idx else ("✗ BUSY/ERROR" if g['has_error_process'] or g['utilization'] >= 90 else "○ Available")
        print(f"GPU {g['index']}: {g['name']}")
        print(f"  Memory: {g['memory_free']/1024:.1f}GB / {g['memory_total']/1024:.1f}GB free")
        print(f"  Util: {g['utilization']}% | Status: {status}")

    print(f"{'='*70}")


def run_evaluation(gpu_idx: int, model_path: str, output_file: str):
    """
    Run the evaluation on the specified GPU.
    """
    cmd = [
        'micromamba', 'run', '-n', 'softdrope', 'python',
        'src/evaluation/ruler.py',
        '--model-path', model_path,
        '--lengths', '2048,4096,8192',
        '--tasks', 'needle,kv_retrieval',
        '--device', f'cuda:{gpu_idx}',
        '--output', output_file
    ]

    print(f"\nStarting evaluation on GPU {gpu_idx}...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd='/data2/weichu/CS7352/大作业')
    return result.returncode == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor GPU and run evaluation")
    parser.add_argument('--model-path', type=str, default='/data1/weichu/woshi/Qwen2.5-7B-Instruct')
    parser.add_argument('--output', type=str, default='results/ruler_benchmark.json')
    parser.add_argument('--min-memory-gb', type=float, default=35, help='Minimum free memory in GB')
    parser.add_argument('--check-interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--max-wait', type=int, default=3600, help='Maximum wait time in seconds')
    parser.add_argument('--run-once', action='store_true', help='Run once without waiting')
    args = parser.parse_args()

    print("GPU Monitor for RULER Benchmark")
    print(f"Model: {args.model_path}")
    print(f"Min memory required: {args.min_memory_gb}GB")
    print(f"Check interval: {args.check_interval}s")

    start_time = time.time()

    while True:
        gpu_idx = find_available_gpu(args.min_memory_gb)
        gpus = get_gpu_status()

        print_gpu_status(gpus, gpu_idx)

        if gpu_idx is not None:
            print(f"\n>>> GPU {gpu_idx} is available! Starting evaluation...")

            success = run_evaluation(gpu_idx, args.model_path, args.output)

            if success:
                print("\n✓ Evaluation completed successfully!")
                # Try to read and display results
                if os.path.exists(args.output):
                    with open(args.output) as f:
                        results = json.load(f)
                    print("\nResults:")
                    print(json.dumps(results, indent=2))
                return 0
            else:
                print("\n✗ Evaluation failed. Will retry...")

        if args.run_once:
            print("\nRun-once mode: exiting without waiting")
            return 0

        # Check max wait time
        elapsed = time.time() - start_time
        if args.max_wait > 0 and elapsed >= args.max_wait:
            print(f"\nMax wait time ({args.max_wait}s) reached. Exiting.")
            return 1

        # Wait before next check
        print(f"\nWaiting {args.check_interval}s before next check...")
        time.sleep(args.check_interval)


if __name__ == "__main__":
    sys.exit(main())
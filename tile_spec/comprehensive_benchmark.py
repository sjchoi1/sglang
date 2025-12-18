#!/usr/bin/env python3
"""
Comprehensive Tile-Spec Benchmark.

Compares AR, Eagle3, Eagle3+TileSpec across batch sizes with exact batch control.
Uses Engine restart between configurations for isolated measurements.

Usage:
    python tile_spec/comprehensive_benchmark.py --batch-sizes 1 2 4 8 16 32 --num-prompts 80
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_num_steps: int = 5
    speculative_eagle_topk: int = 8
    speculative_num_draft_tokens: int = 64
    tile_spec: bool = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    batch_size: int
    num_prompts: int
    total_time_s: float
    total_output_tokens: int
    throughput_tok_s: float
    acceptance_length: float
    num_verify_calls: int


def get_mtbench_prompts(num_prompts: int = 80) -> List[str]:
    """Load MT-bench prompts (first turn only for simplicity)."""
    mtbench_path = Path(__file__).parent.parent / "benchmark" / "mtbench" / "question.jsonl"

    if not mtbench_path.exists():
        # Fall back to simple prompts
        print(f"MT-bench not found at {mtbench_path}, using synthetic prompts")
        return [
            f"Explain the concept of {topic} in simple terms."
            for topic in [
                "machine learning", "quantum computing", "climate change",
                "blockchain", "artificial intelligence", "neural networks",
                "data science", "cybersecurity", "cloud computing", "IoT"
            ] * (num_prompts // 10 + 1)
        ][:num_prompts]

    prompts = []
    with open(mtbench_path, "r") as f:
        for line in f:
            if len(prompts) >= num_prompts:
                break
            obj = json.loads(line)
            # Use first turn only
            if "turns" in obj:
                prompts.append(obj["turns"][0])
            elif "prompt" in obj:
                prompts.append(obj["prompt"][0] if isinstance(obj["prompt"], list) else obj["prompt"])

    print(f"Loaded {len(prompts)} MT-bench prompts")
    return prompts


def run_benchmark(
    config: BenchmarkConfig,
    prompts: List[str],
    batch_size: int,
    model_path: str,
    max_new_tokens: int = 256,
    warmup_prompts: int = 5,
) -> BenchmarkResult:
    """Run benchmark with specific configuration and batch size."""
    import sglang as sgl

    # Build engine kwargs
    engine_kwargs = {
        "model_path": model_path,
        "dtype": "float16",
        "max_running_requests": batch_size,  # Control batch size
    }

    if config.speculative_algorithm:
        engine_kwargs["speculative_algorithm"] = config.speculative_algorithm
        engine_kwargs["speculative_draft_model_path"] = config.speculative_draft_model_path
        engine_kwargs["speculative_num_steps"] = config.speculative_num_steps
        engine_kwargs["speculative_eagle_topk"] = config.speculative_eagle_topk
        engine_kwargs["speculative_num_draft_tokens"] = config.speculative_num_draft_tokens

    if config.tile_spec:
        engine_kwargs["tile_spec"] = True

    print(f"\n{'='*60}")
    print(f"Config: {config.name}, Batch Size: {batch_size}")
    print(f"{'='*60}")

    # Create engine
    engine = sgl.Engine(**engine_kwargs)

    # Warmup (also triggers profiling for tile-spec)
    print(f"Warmup with {warmup_prompts} prompts...")
    warmup_batch = [prompts[i % len(prompts)] for i in range(warmup_prompts)]
    engine.generate(warmup_batch, sampling_params={"max_new_tokens": 32})

    # If tile-spec, do additional warmup to complete profiling
    if config.tile_spec:
        print("Additional warmup for tile-spec profiling (100 requests)...")
        for i in range(0, 100, batch_size):
            chunk_size = min(batch_size, 100 - i)
            chunk = [prompts[j % len(prompts)] for j in range(i, i + chunk_size)]
            engine.generate(chunk, sampling_params={"max_new_tokens": 32})

    # Select prompts for this batch size (cycle if needed)
    num_prompts = len(prompts)
    batch_prompts = [prompts[i % num_prompts] for i in range(batch_size)]

    # Run benchmark - send batch_size prompts as a single batch
    print(f"Running batch of {batch_size} prompts...")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Generate as a batch (list of prompts)
    results = engine.generate(
        batch_prompts,
        sampling_params={"max_new_tokens": max_new_tokens}
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    # Handle results (can be a single dict if batch=1, or list if batch>1)
    if not isinstance(results, list):
        results = [results]

    # Collect metrics
    total_output_tokens = 0
    num_verify_calls = 0
    total_accept_length = 0.0
    num_with_accept_length = 0

    for result in results:
        # Get actual completion tokens from meta_info
        meta = result.get("meta_info", {})
        tokens = meta.get("completion_tokens", 0)
        if tokens > 0:
            total_output_tokens += tokens
        else:
            # Fallback: estimate from text
            output_text = result.get("text", "")
            total_output_tokens += len(output_text.split())

        # Get spec_verify_ct if available
        if "spec_verify_ct" in meta and meta["spec_verify_ct"] > 0:
            num_verify_calls += meta["spec_verify_ct"]

        # Get pre-calculated acceptance length if available
        if "spec_accept_length" in meta and meta["spec_accept_length"] > 0:
            total_accept_length += meta["spec_accept_length"]
            num_with_accept_length += 1

    # Calculate metrics
    throughput = total_output_tokens / elapsed if elapsed > 0 else 0

    # Use pre-calculated acceptance length if available, otherwise calculate
    if num_with_accept_length > 0:
        acceptance_length = total_accept_length / num_with_accept_length
    elif num_verify_calls > 0:
        acceptance_length = total_output_tokens / num_verify_calls
    else:
        acceptance_length = 1.0  # AR baseline

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Output tokens: {total_output_tokens}")
    print(f"  Throughput: {throughput:.2f} tok/s")
    print(f"  Verify calls: {num_verify_calls}")
    print(f"  Acceptance length: {acceptance_length:.2f}")

    # Cleanup
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)  # Allow GPU to settle

    return BenchmarkResult(
        config_name=config.name,
        batch_size=batch_size,
        num_prompts=batch_size,
        total_time_s=elapsed,
        total_output_tokens=total_output_tokens,
        throughput_tok_s=throughput,
        acceptance_length=acceptance_length,
        num_verify_calls=num_verify_calls,
    )


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Tile-Spec Benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Target model path",
    )
    parser.add_argument(
        "--draft-model-path",
        type=str,
        default="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",
        help="Draft model path for EAGLE3",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=80,
        help="Number of prompts to load from MT-bench",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens per request",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="tile_spec/benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["AR", "Eagle3", "Eagle3+TileSpec"],
        choices=["AR", "Eagle3", "Eagle3+TileSpec"],
        help="Configurations to benchmark",
    )
    args = parser.parse_args()

    # Define configurations
    all_configs = {
        "AR": BenchmarkConfig(name="AR"),
        "Eagle3": BenchmarkConfig(
            name="Eagle3",
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path=args.draft_model_path,
        ),
        "Eagle3+TileSpec": BenchmarkConfig(
            name="Eagle3+TileSpec",
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path=args.draft_model_path,
            tile_spec=True,
        ),
    }

    configs = [all_configs[name] for name in args.configs]

    # Load prompts
    prompts = get_mtbench_prompts(args.num_prompts)

    # Run benchmarks
    all_results: List[BenchmarkResult] = []

    for config in configs:
        for batch_size in args.batch_sizes:
            try:
                result = run_benchmark(
                    config=config,
                    prompts=prompts,
                    batch_size=batch_size,
                    model_path=args.model_path,
                    max_new_tokens=args.max_new_tokens,
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error running {config.name} with batch_size={batch_size}: {e}")
                continue

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "metadata": {
            "model_path": args.model_path,
            "draft_model_path": args.draft_model_path,
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": args.num_prompts,
        },
        "results": [
            {
                "config": r.config_name,
                "batch_size": r.batch_size,
                "num_prompts": r.num_prompts,
                "total_time_s": r.total_time_s,
                "total_output_tokens": r.total_output_tokens,
                "throughput_tok_s": r.throughput_tok_s,
                "acceptance_length": r.acceptance_length,
                "num_verify_calls": r.num_verify_calls,
            }
            for r in all_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Config':<20} {'Batch':<8} {'Throughput':<15} {'Accept Len':<12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r.config_name:<20} {r.batch_size:<8} {r.throughput_tok_s:<15.2f} {r.acceptance_length:<12.2f}")


if __name__ == "__main__":
    main()

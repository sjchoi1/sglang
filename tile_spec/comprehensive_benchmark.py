#!/usr/bin/env python3
"""
Tile-Spec Benchmark.

Compares AR, Eagle3, Eagle3+TileSpec with batch size cap.
Methodology: Run entire dataset with max_running_requests as batch cap.
Natural EOS, temp=0 for deterministic comparison.

Usage:
    python tile_spec/comprehensive_benchmark.py --batch-sizes 1 2 4 8 16 32 64
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class Config:
    """Benchmark configuration."""
    name: str
    speculative_algorithm: Optional[str] = None
    draft_model_path: Optional[str] = None
    tile_spec: bool = False


@dataclass
class Result:
    """Benchmark result."""
    config: str
    batch_size: int
    num_prompts: int
    total_tokens: int
    time_s: float
    throughput: float
    acceptance_length: float
    speedup: float = 1.0


def load_prompts(path: Path, limit: int) -> List[str]:
    """Load prompts from MT-bench or similar JSONL."""
    if not path.exists():
        print(f"Dataset not found: {path}")
        # Fallback synthetic prompts
        topics = ["AI", "physics", "history", "coding", "math", "biology"]
        return [f"Explain {t} in detail." for t in topics * (limit // 6 + 1)][:limit]

    prompts = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            # Handle MT-bench format
            if "turns" in obj:
                prompts.append(obj["turns"][0])
            elif "prompt" in obj:
                p = obj["prompt"]
                prompts.append(p[0] if isinstance(p, list) else p)
            if len(prompts) >= limit:
                break

    print(f"Loaded {len(prompts)} prompts from {path.name}")
    return prompts


def run(
    config: Config,
    prompts: List[str],
    batch_size: int,
    model_path: str,
    draft_path: str,
) -> Result:
    """Run benchmark: entire dataset with batch cap."""
    import sglang as sgl

    # Engine config
    kwargs = {
        "model_path": model_path,
        "dtype": "float16",
        "max_running_requests": batch_size,
    }

    if config.speculative_algorithm:
        kwargs.update({
            "speculative_algorithm": config.speculative_algorithm,
            "speculative_draft_model_path": config.draft_model_path or draft_path,
            "speculative_num_steps": 5,
            "speculative_eagle_topk": 8,
            "speculative_num_draft_tokens": 64,
        })

    if config.tile_spec:
        kwargs["tile_spec"] = True

    print(f"\n[{config.name}] batch_cap={batch_size}, prompts={len(prompts)}")

    engine = sgl.Engine(**kwargs)

    # Warmup
    engine.generate(prompts[:5], sampling_params={"temperature": 0, "max_new_tokens": 64})

    # Extra warmup for tile-spec profiling
    if config.tile_spec:
        for i in range(0, 100, max(batch_size, 10)):
            engine.generate(
                prompts[:min(batch_size, len(prompts))],
                sampling_params={"temperature": 0, "max_new_tokens": 64}
            )

    # Benchmark: run entire dataset
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    results = engine.generate(
        prompts,
        sampling_params={"temperature": 0}  # No max_new_tokens cap - natural EOS
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Collect metrics
    if not isinstance(results, list):
        results = [results]

    total_tokens = 0
    total_accept_len = 0.0
    n_spec = 0

    for r in results:
        meta = r.get("meta_info", {})
        total_tokens += meta.get("completion_tokens", len(r.get("text", "").split()))

        if meta.get("spec_accept_length", 0) > 0:
            total_accept_len += meta["spec_accept_length"]
            n_spec += 1

    throughput = total_tokens / elapsed
    accept_len = total_accept_len / n_spec if n_spec > 0 else 1.0

    print(f"  {total_tokens} tokens in {elapsed:.1f}s = {throughput:.1f} tok/s, τ={accept_len:.2f}")

    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return Result(
        config=config.name,
        batch_size=batch_size,
        num_prompts=len(prompts),
        total_tokens=total_tokens,
        time_s=elapsed,
        throughput=throughput,
        acceptance_length=accept_len,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft", default="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--dataset", default="benchmark/mtbench/question.jsonl")
    parser.add_argument("--num-prompts", type=int, default=80)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--output", default="tile_spec/results.json")
    parser.add_argument("--configs", nargs="+", default=["AR", "Eagle3", "Eagle3+TileSpec"])
    args = parser.parse_args()

    # Configs
    configs = {
        "AR": Config("AR"),
        "Eagle3": Config("Eagle3", "EAGLE3", args.draft),
        "Eagle3+TileSpec": Config("Eagle3+TileSpec", "EAGLE3", args.draft, tile_spec=True),
    }

    prompts = load_prompts(Path(args.dataset), args.num_prompts)
    results: List[Result] = []

    # Run benchmarks
    for batch_size in args.batch_sizes:
        ar_throughput = None

        for name in args.configs:
            cfg = configs[name]
            try:
                r = run(cfg, prompts, batch_size, args.model, args.draft)

                # Calculate speedup vs AR
                if name == "AR":
                    ar_throughput = r.throughput
                    r.speedup = 1.0
                elif ar_throughput:
                    r.speedup = r.throughput / ar_throughput

                results.append(r)
            except Exception as e:
                print(f"  ERROR: {e}")

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "draft": args.draft,
            "results": [asdict(r) for r in results]
        }, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Config':<20} {'Batch':<8} {'Throughput':<12} {'τ':<8} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.config:<20} {r.batch_size:<8} {r.throughput:<12.1f} {r.acceptance_length:<8.2f} {r.speedup:<10.2f}x")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

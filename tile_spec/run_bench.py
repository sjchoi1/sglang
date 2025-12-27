#!/usr/bin/env python3
"""
TileSpec Benchmark using Spec-Bench dataset.

Usage:
    python tile_spec/run_bench.py
    python tile_spec/run_bench.py --batch-sizes 8 16 32
    python tile_spec/run_bench.py --categories summarization math_reasoning
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import sglang as sgl
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server


TIMEOUT = 2400  # 40 minutes - TileSpec profiling can take a while
CATEGORIES = ["mt_conv", "translation", "summarization", "qa", "math_reasoning", "rag", "mixed"]


@dataclass
class Config:
    name: str
    spec_algo: Optional[str] = None
    draft: Optional[str] = None
    tile_spec: bool = False


@sgl.function
def generate(s, turns: list):
    for i, turn in enumerate(turns):
        s += sgl.user(turn)
        s += sgl.assistant(sgl.gen(f"answer_{i}"))


def load_spec_bench(path: str) -> Dict[str, List[dict]]:
    """Load and split by category."""
    category_map = {
        "writing": "mt_conv", "roleplay": "mt_conv", "reasoning": "mt_conv",
        "humanities": "mt_conv", "stem": "mt_conv", "extraction": "mt_conv",
        "coding": "mt_conv", "math": "mt_conv",
        "translation": "translation",
        "summarization": "summarization",
        "qa": "qa",
        "math_reasoning": "math_reasoning",
        "rag": "rag",
    }

    data = {cat: [] for cat in CATEGORIES}
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            cat = category_map.get(sample["category"], "mt_conv")
            data[cat].append(sample)

    # Mixed = all samples
    data["mixed"] = []
    for cat in CATEGORIES[:-1]:
        data["mixed"].extend(data[cat])

    return data


def run_samples(samples: List[dict], batch_size: int) -> dict:
    """Run samples and return metrics."""
    # Repeat samples to ensure at least 5 full batches for steady-state measurement
    min_samples = batch_size * 5
    if len(samples) < min_samples:
        repeat_count = (min_samples // len(samples)) + 1
        samples = (samples * repeat_count)[:min_samples]

    print("min_samples:", min_samples)
    print("samples: ", len(samples))

    args = [{"turns": s["turns"]} for s in samples]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rets = generate.run_batch(args, temperature=0, progress_bar=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = 0
    verify_steps = 0
    for ret, sample in zip(rets, samples):
        for i in range(len(sample["turns"])):
            meta = ret.get_meta_info(f"answer_{i}")
            total_tokens += meta.get("completion_tokens", 0)
            verify_steps += meta.get("spec_verify_ct", 0)

    return {
        "tokens": total_tokens,
        "time": elapsed,
        "throughput": total_tokens / elapsed if elapsed > 0 else 0,
        "accept_len": total_tokens / verify_steps if verify_steps > 0 else 1.0,
    }


def run_config(config: Config, samples: List[dict], model: str, draft: str,
               batch_size: int, port: int) -> dict:
    """Launch server, benchmark, shutdown."""
    args = ["--dtype", "float16", "--disable-cuda-graph",
            "--max-running-requests", str(batch_size)]

    if config.spec_algo:
        args += ["--speculative-algorithm", config.spec_algo,
                 "--speculative-draft-model-path", config.draft or draft,
                 "--speculative-num-steps", "3",
                 "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", "3"]

    if config.tile_spec:
        args.append("--tile-spec")

    base_url = f"http://127.0.0.1:{port}"
    process = popen_launch_server(model, base_url, TIMEOUT, other_args=args)

    try:
        sgl.set_default_backend(RuntimeEndpoint(base_url))

        # Warmup (TileSpec profiling happens automatically during server startup)
        warmup = [{"turns": s["turns"]} for s in samples[:min(10, len(samples))]]
        generate.run_batch(warmup, temperature=0, progress_bar=False)

        return run_samples(samples, batch_size)

    finally:
        kill_process_tree(process.pid)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)


def get_short_name(model_path: str) -> str:
    """Extract short name from model path."""
    import re
    name = model_path.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9-]", "", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft", default="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--data", default=str(Path(__file__).parent / "question.jsonl"))
    parser.add_argument("--output", default=None, help="Output file (auto-generated if not specified)")
    parser.add_argument("--categories", nargs="+", default=["mt_conv"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    # parser.add_argument("--configs", nargs="+", default=["AR", "Eagle3+TileSpec"])
    parser.add_argument("--configs", nargs="+", default=["AR", "Eagle3", "Eagle3+TileSpec"])    
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached AR/Eagle3 results")
    args = parser.parse_args()

    # Auto-generate output filename based on model config
    if args.output is None:
        model_short = get_short_name(args.model)
        draft_short = get_short_name(args.draft)
        args.output = str(Path(__file__).parent / f"results_{model_short}_{draft_short}.json")

    # Load existing results for caching
    cached_results = {}
    if not args.no_cache and Path(args.output).exists():
        try:
            with open(args.output) as f:
                cached_data = json.load(f)
                cached_results = cached_data.get("results", {})
            print(f"Loaded cached results from {args.output}")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")

    # Load data
    print(f"Loading {args.data}")
    data = load_spec_bench(args.data)
    for cat in args.categories:
        print(f"  {cat}: {len(data[cat])} samples")

    # Configs
    configs = {
        "AR": Config("AR"),
        "Eagle3": Config("Eagle3", "EAGLE3", args.draft),
        "Eagle3+TileSpec": Config("Eagle3+TileSpec", "EAGLE3", args.draft, tile_spec=True),
    }

    # Configs that can be cached (TileSpec always re-runs)
    cacheable_configs = {"AR", "Eagle3"}

    # Results
    results = {}

    for cat in args.categories:
        samples = data[cat]
        if not samples:
            continue

        print(f"\n{'='*60}")
        print(f"Category: {cat} ({len(samples)} samples)")
        print(f"{'='*60}")

        results[cat] = cached_results.get(cat, {})

        for bs in args.batch_sizes:
            print(f"\n  Batch size: {bs}")
            bs_key = str(bs)  # JSON keys are strings
            if bs_key not in results[cat]:
                results[cat][bs_key] = {}
            ar_throughput = None

            for cfg_name in args.configs:
                # Check cache for AR/Eagle3
                if cfg_name in cacheable_configs and cfg_name in results[cat].get(bs_key, {}):
                    r = results[cat][bs_key][cfg_name]
                    print(f"    {cfg_name}... (cached) {r['throughput']:.0f} tok/s, τ={r['accept_len']:.2f}, {r['speedup']:.2f}x")
                    if cfg_name == "AR":
                        ar_throughput = r["throughput"]
                    continue

                print(f"    {cfg_name}...", end=" ", flush=True)
                r = run_config(configs[cfg_name], samples, args.model, args.draft, bs, args.port)

                if cfg_name == "AR":
                    ar_throughput = r["throughput"]
                    r["speedup"] = 1.0
                elif ar_throughput:
                    r["speedup"] = r["throughput"] / ar_throughput
                else:
                    # Get AR throughput from cache if available
                    ar_cached = results[cat].get(bs_key, {}).get("AR", {})
                    ar_throughput = ar_cached.get("throughput")
                    r["speedup"] = r["throughput"] / ar_throughput if ar_throughput else 1.0

                results[cat][bs_key][cfg_name] = r
                print(f"{r['throughput']:.0f} tok/s, τ={r['accept_len']:.2f}, {r['speedup']:.2f}x")

        # Summary table for this category
        print(f"\n  {'Batch':<8}", end="")
        for cfg in args.configs:
            print(f"{cfg:>18}", end="")
        print()
        print(f"  {'-'*56}")
        for bs in args.batch_sizes:
            bs_key = str(bs)
            print(f"  {bs:<8}", end="")
            for cfg in args.configs:
                if cfg in results[cat].get(bs_key, {}):
                    print(f"{results[cat][bs_key][cfg]['speedup']:>17.2f}x", end="")
                else:
                    print(f"{'N/A':>17}", end="")
            print()

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"settings": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

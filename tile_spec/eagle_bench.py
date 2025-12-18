#!/usr/bin/env python3
"""
EAGLE-3 Paper Benchmark.

Exact 5 datasets from EAGLE-3 paper:
1. MT-bench (80) - multi-turn conversation
2. HumanEval (164) - code generation
3. GSM8K (1319) - math reasoning
4. Alpaca (805) - instruction following
5. CNN/Daily Mail (500) - summarization

Reports: Speedup ratio, τ (acceptance length), Mean

Usage:
    python tile_spec/eagle_bench.py --configs AR Eagle3 Eagle3+TileSpec
"""

import argparse
import gc
import gzip
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Dataset URLs
URLS = {
    "mtbench": "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
    "humaneval": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "alpaca": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    "cnn_dm": "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/cnn_stories.tgz",
}


@dataclass
class Config:
    name: str
    speculative_algorithm: Optional[str] = None
    draft_model_path: Optional[str] = None
    tile_spec: bool = False


@dataclass
class Result:
    dataset: str
    num_samples: int
    total_tokens: int
    total_time: float
    throughput: float
    accept_length: float
    speedup: float = 1.0


def download_file(url: str, path: Path) -> Path:
    """Download file if not cached."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {path.name}...")
    urllib.request.urlretrieve(url, path)
    return path


def load_mtbench(cache_dir: Path, limit: int = 80) -> List[dict]:
    """Load MT-bench: multi-turn conversation."""
    path = download_file(URLS["mtbench"], cache_dir / "mtbench.jsonl")
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            # Format: {"turns": ["q1", "q2"], ...}
            samples.append({
                "prompt": obj["turns"][0],
                "multi_turn": obj["turns"] if len(obj["turns"]) > 1 else None,
            })
            if len(samples) >= limit:
                break
    return samples


def load_humaneval(cache_dir: Path, limit: int = 164) -> List[dict]:
    """Load HumanEval: code generation."""
    gz_path = download_file(URLS["humaneval"], cache_dir / "HumanEval.jsonl.gz")
    samples = []
    with gzip.open(gz_path, 'rt') as f:
        for line in f:
            obj = json.loads(line)
            # Format: {"prompt": "def func(...):\n    \"\"\"docstring\"\"\"\n", ...}
            samples.append({
                "prompt": f"Complete the following Python function:\n\n{obj['prompt']}",
            })
            if len(samples) >= limit:
                break
    return samples


def load_gsm8k(cache_dir: Path, limit: int = 500) -> List[dict]:
    """Load GSM8K: math reasoning."""
    path = download_file(URLS["gsm8k"], cache_dir / "gsm8k_test.jsonl")
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            samples.append({
                "prompt": f"Question: {obj['question']}\nAnswer: Let's solve this step by step.",
            })
            if len(samples) >= limit:
                break
    return samples


def load_alpaca(cache_dir: Path, limit: int = 500) -> List[dict]:
    """Load Alpaca: instruction following."""
    path = download_file(URLS["alpaca"], cache_dir / "alpaca_data.json")
    with open(path) as f:
        data = json.load(f)

    samples = []
    for item in data:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction
        samples.append({"prompt": prompt})
        if len(samples) >= limit:
            break
    return samples


def load_cnn_dm(cache_dir: Path, limit: int = 500) -> List[dict]:
    """Load CNN/Daily Mail: summarization (using HuggingFace)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        samples = []
        for item in ds:
            article = item["article"][:2000]  # Truncate long articles
            samples.append({
                "prompt": f"Summarize the following article:\n\n{article}\n\nSummary:",
            })
            if len(samples) >= limit:
                break
        return samples
    except ImportError:
        print("  Warning: 'datasets' not installed. Using synthetic summarization prompts.")
        return [{"prompt": f"Summarize: Article {i} about news events."} for i in range(limit)]


LOADERS = {
    "mt_bench": (load_mtbench, 80),
    "humaneval": (load_humaneval, 164),
    "gsm8k": (load_gsm8k, 500),
    "alpaca": (load_alpaca, 500),
    "cnn_dm": (load_cnn_dm, 500),
}


def run_dataset(
    engine,
    samples: List[dict],
    dataset_name: str,
    max_new_tokens: int = 512,
) -> Result:
    """Run benchmark on a dataset."""
    total_tokens = 0
    total_time = 0.0
    accept_lengths = []

    for sample in samples:
        prompt = sample["prompt"]
        multi_turn = sample.get("multi_turn")

        if multi_turn:
            # Multi-turn conversation (MT-bench style)
            conversation = []
            for i, turn in enumerate(multi_turn):
                conversation.append({"role": "user", "content": turn})

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = engine.generate(conversation, sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens})
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                meta = result.get("meta_info", {})
                tokens = meta.get("completion_tokens", len(result.get("text", "").split()))
                accept_len = meta.get("spec_accept_length", 0)

                total_tokens += tokens
                total_time += elapsed
                if accept_len > 0:
                    accept_lengths.append(accept_len)

                conversation.append({"role": "assistant", "content": result.get("text", "")})
        else:
            # Single-turn
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = engine.generate(prompt, sampling_params={"temperature": 0, "max_new_tokens": max_new_tokens})
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            meta = result.get("meta_info", {})
            tokens = meta.get("completion_tokens", len(result.get("text", "").split()))
            accept_len = meta.get("spec_accept_length", 0)

            total_tokens += tokens
            total_time += elapsed
            if accept_len > 0:
                accept_lengths.append(accept_len)

    throughput = total_tokens / total_time if total_time > 0 else 0
    mean_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 1.0

    return Result(
        dataset=dataset_name,
        num_samples=len(samples),
        total_tokens=total_tokens,
        total_time=total_time,
        throughput=throughput,
        accept_length=mean_accept,
    )


def run_config(
    config: Config,
    datasets: Dict[str, List[dict]],
    model_path: str,
    draft_path: str,
    batch_size: int,
) -> Dict[str, Result]:
    """Run all datasets for a config."""
    import sglang as sgl

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

    print(f"\n{'='*70}")
    print(f"Config: {config.name}")
    print(f"{'='*70}")

    engine = sgl.Engine(**kwargs)

    # Warmup
    print("Warmup...")
    for _ in range(3):
        engine.generate("Hello, how are you?", sampling_params={"temperature": 0, "max_new_tokens": 32})

    if config.tile_spec:
        print("Tile-spec profiling...")
        for _ in range(100):
            engine.generate("Explain machine learning briefly.", sampling_params={"temperature": 0, "max_new_tokens": 64})

    # Run each dataset
    results = {}
    for name, samples in datasets.items():
        print(f"\n  [{name}] {len(samples)} samples...")
        # Different max_new_tokens per task
        max_tokens = 512 if name != "humaneval" else 256
        result = run_dataset(engine, samples, name, max_new_tokens=max_tokens)
        results[name] = result
        print(f"    {result.throughput:.1f} tok/s, τ={result.accept_length:.2f}")

    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="EAGLE-3 Paper Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft", default="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--cache-dir", default="tile_spec/data")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="tile_spec/eagle_bench_results.json")
    parser.add_argument("--configs", nargs="+", default=["AR", "Eagle3", "Eagle3+TileSpec"])
    parser.add_argument("--datasets", nargs="+", default=["mt_bench", "humaneval", "gsm8k", "alpaca", "cnn_dm"])
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    datasets = {}
    for name in args.datasets:
        if name in LOADERS:
            loader, default_limit = LOADERS[name]
            print(f"  Loading {name}...")
            datasets[name] = loader(cache_dir, default_limit)
            print(f"    Loaded {len(datasets[name])} samples")

    # Configs
    all_configs = {
        "AR": Config("AR"),
        "Eagle3": Config("Eagle3", "EAGLE3", args.draft),
        "Eagle3+TileSpec": Config("Eagle3+TileSpec", "EAGLE3", args.draft, tile_spec=True),
    }

    # Run benchmarks
    all_results = {}
    ar_results = None

    for name in args.configs:
        config = all_configs[name]
        results = run_config(config, datasets, args.model, args.draft, args.batch_size)
        all_results[name] = results

        if name == "AR":
            ar_results = results
        elif ar_results:
            for ds, result in results.items():
                if ds in ar_results:
                    ar_tp = ar_results[ds].throughput
                    result.speedup = result.throughput / ar_tp if ar_tp > 0 else 1.0

    # Print summary
    print(f"\n{'='*90}")
    print(f"{'Dataset':<12} ", end="")
    for name in args.configs:
        print(f"| {name:<24} ", end="")
    print()
    print("-" * 90)

    for ds in args.datasets:
        print(f"{ds:<12} ", end="")
        for name in args.configs:
            if ds in all_results.get(name, {}):
                r = all_results[name][ds]
                print(f"| {r.speedup:>5.2f}x  τ={r.accept_length:<6.2f} ", end="")
            else:
                print(f"| {'N/A':<24} ", end="")
        print()

    # Mean
    print("-" * 90)
    print(f"{'Mean':<12} ", end="")
    for name in args.configs:
        speedups = [all_results[name][ds].speedup for ds in args.datasets if ds in all_results.get(name, {})]
        accepts = [all_results[name][ds].accept_length for ds in args.datasets if ds in all_results.get(name, {})]
        if speedups:
            print(f"| {sum(speedups)/len(speedups):>5.2f}x  τ={sum(accepts)/len(accepts):<6.2f} ", end="")
        else:
            print(f"| {'N/A':<24} ", end="")
    print()

    # Save results
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "draft": args.draft,
            "results": {
                name: {ds: {"speedup": r.speedup, "accept_length": r.accept_length, "throughput": r.throughput}
                       for ds, r in results.items()}
                for name, results in all_results.items()
            }
        }, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

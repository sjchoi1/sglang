#!/usr/bin/env python3
"""
EAGLE-3 Paper Benchmark.

Exact 5 datasets from EAGLE-3 paper:
1. MT-bench (80) - multi-turn conversation
2. HumanEval (164) - code generation
3. GSM8K (500) - math reasoning
4. Alpaca (500) - instruction following
5. CNN/Daily Mail (500) - summarization

Methodology:
- temp=0 (deterministic)
- No max_new_tokens cap (natural EOS)
- ShareGPT for profiling warmup

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

# URLs
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
MTBENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


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


def download(url: str, path: Path) -> Path:
    """Download file if not cached."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {path.name}...")
    urllib.request.urlretrieve(url, path)
    return path


def load_sharegpt(cache_dir: Path, limit: int = 200) -> List[str]:
    """Load ShareGPT prompts for profiling."""
    path = download(SHAREGPT_URL, cache_dir / "sharegpt.json")
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for item in data:
        if "conversations" in item:
            for conv in item["conversations"]:
                if conv.get("from") == "human" and conv.get("value"):
                    text = conv["value"].strip()
                    if 50 < len(text) < 2000:
                        prompts.append(text)
                        if len(prompts) >= limit:
                            return prompts
    return prompts


def load_mtbench(cache_dir: Path, limit: int = 80) -> List[dict]:
    """MT-bench: multi-turn conversation."""
    path = download(MTBENCH_URL, cache_dir / "mtbench.jsonl")
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            samples.append({"turns": obj["turns"]})
            if len(samples) >= limit:
                break
    return samples


def load_humaneval(cache_dir: Path, limit: int = 164) -> List[dict]:
    """HumanEval: code generation."""
    path = download(HUMANEVAL_URL, cache_dir / "HumanEval.jsonl.gz")
    samples = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            obj = json.loads(line)
            samples.append({"prompt": f"Complete the following Python function:\n\n{obj['prompt']}"})
            if len(samples) >= limit:
                break
    return samples


def load_gsm8k(cache_dir: Path, limit: int = 500) -> List[dict]:
    """GSM8K: math reasoning."""
    path = download(GSM8K_URL, cache_dir / "gsm8k.jsonl")
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            samples.append({"prompt": f"Question: {obj['question']}\nAnswer: Let's solve step by step."})
            if len(samples) >= limit:
                break
    return samples


def load_alpaca(cache_dir: Path, limit: int = 500) -> List[dict]:
    """Alpaca: instruction following."""
    path = download(ALPACA_URL, cache_dir / "alpaca.json")
    with open(path) as f:
        data = json.load(f)
    samples = []
    for item in data:
        instruction = item["instruction"]
        inp = item.get("input", "")
        prompt = f"{instruction}\n\nInput: {inp}" if inp else instruction
        samples.append({"prompt": prompt})
        if len(samples) >= limit:
            break
    return samples


def load_cnn_dm(cache_dir: Path, limit: int = 500) -> List[dict]:
    """CNN/Daily Mail: summarization."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        samples = []
        for item in ds:
            article = item["article"][:2000]
            samples.append({"prompt": f"Summarize:\n\n{article}\n\nSummary:"})
            if len(samples) >= limit:
                break
        return samples
    except ImportError:
        print("    Warning: 'datasets' not installed, using synthetic prompts")
        return [{"prompt": f"Summarize this news article about topic {i}."} for i in range(limit)]


DATASETS = {
    "mt_bench": (load_mtbench, 80),
    "humaneval": (load_humaneval, 164),
    "gsm8k": (load_gsm8k, 500),
    "alpaca": (load_alpaca, 500),
    "cnn_dm": (load_cnn_dm, 500),
}


def run_dataset(engine, samples: List[dict], name: str) -> Result:
    """Run benchmark on dataset. No max_new_tokens - natural EOS."""
    total_tokens = 0
    total_time = 0.0
    accept_lengths = []

    for sample in samples:
        if "turns" in sample:
            # Multi-turn (MT-bench)
            conv = []
            for turn in sample["turns"]:
                conv.append({"role": "user", "content": turn})
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = engine.generate(conv, sampling_params={"temperature": 0})
                torch.cuda.synchronize()
                total_time += time.perf_counter() - t0

                meta = result.get("meta_info", {})
                total_tokens += meta.get("completion_tokens", len(result.get("text", "").split()))
                if meta.get("spec_accept_length", 0) > 0:
                    accept_lengths.append(meta["spec_accept_length"])
                conv.append({"role": "assistant", "content": result.get("text", "")})
        else:
            # Single turn
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = engine.generate(sample["prompt"], sampling_params={"temperature": 0})
            torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            meta = result.get("meta_info", {})
            total_tokens += meta.get("completion_tokens", len(result.get("text", "").split()))
            if meta.get("spec_accept_length", 0) > 0:
                accept_lengths.append(meta["spec_accept_length"])

    return Result(
        dataset=name,
        num_samples=len(samples),
        total_tokens=total_tokens,
        total_time=total_time,
        throughput=total_tokens / total_time if total_time > 0 else 0,
        accept_length=sum(accept_lengths) / len(accept_lengths) if accept_lengths else 1.0,
    )


def run_config(config: Config, datasets: Dict, model: str, draft: str, cache_dir: Path) -> Dict[str, Result]:
    """Run all datasets for a config."""
    import sglang as sgl

    kwargs = {"model_path": model, "dtype": "float16"}
    if config.speculative_algorithm:
        kwargs.update({
            "speculative_algorithm": config.speculative_algorithm,
            "speculative_draft_model_path": config.draft_model_path or draft,
            "speculative_num_steps": 5,
            "speculative_eagle_topk": 8,
            "speculative_num_draft_tokens": 64,
        })
    if config.tile_spec:
        kwargs["tile_spec"] = True

    print(f"\n{'='*70}\nConfig: {config.name}\n{'='*70}")
    engine = sgl.Engine(**kwargs)

    # Warmup with ShareGPT (also does profiling for tile-spec)
    sharegpt = load_sharegpt(cache_dir, limit=150 if config.tile_spec else 10)
    print(f"Warmup with {len(sharegpt)} ShareGPT prompts...")
    for prompt in sharegpt:
        engine.generate(prompt, sampling_params={"temperature": 0})

    # Run datasets
    results = {}
    for name, samples in datasets.items():
        print(f"\n  [{name}] {len(samples)} samples...")
        result = run_dataset(engine, samples, name)
        results[name] = result
        print(f"    {result.throughput:.1f} tok/s, τ={result.accept_length:.2f}")

    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft", default="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--cache-dir", default="tile_spec/data")
    parser.add_argument("--output", default="tile_spec/results.json")
    parser.add_argument("--configs", nargs="+", default=["AR", "Eagle3", "Eagle3+TileSpec"])
    parser.add_argument("--datasets", nargs="+", default=["mt_bench", "humaneval", "gsm8k", "alpaca", "cnn_dm"])
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    # Load datasets
    print("Loading datasets...")
    datasets = {}
    for name in args.datasets:
        if name in DATASETS:
            loader, limit = DATASETS[name]
            datasets[name] = loader(cache_dir, limit)
            print(f"  {name}: {len(datasets[name])} samples")

    # Configs
    all_configs = {
        "AR": Config("AR"),
        "Eagle3": Config("Eagle3", "EAGLE3", args.draft),
        "Eagle3+TileSpec": Config("Eagle3+TileSpec", "EAGLE3", args.draft, tile_spec=True),
    }

    # Run
    all_results = {}
    ar_results = None
    for name in args.configs:
        results = run_config(all_configs[name], datasets, args.model, args.draft, cache_dir)
        all_results[name] = results
        if name == "AR":
            ar_results = results
        elif ar_results:
            for ds, r in results.items():
                if ds in ar_results:
                    r.speedup = r.throughput / ar_results[ds].throughput if ar_results[ds].throughput > 0 else 1.0

    # Print summary
    print(f"\n{'='*90}")
    print(f"{'Dataset':<12} | ", end="")
    for name in args.configs:
        print(f"{name:<24} | ", end="")
    print()
    print("-" * 90)
    for ds in args.datasets:
        print(f"{ds:<12} | ", end="")
        for name in args.configs:
            r = all_results[name].get(ds)
            if r:
                print(f"{r.speedup:>5.2f}x  τ={r.accept_length:<6.2f} | ", end="")
            else:
                print(f"{'N/A':<24} | ", end="")
        print()
    print("-" * 90)
    print(f"{'Mean':<12} | ", end="")
    for name in args.configs:
        speedups = [all_results[name][ds].speedup for ds in args.datasets if ds in all_results[name]]
        accepts = [all_results[name][ds].accept_length for ds in args.datasets if ds in all_results[name]]
        if speedups:
            print(f"{sum(speedups)/len(speedups):>5.2f}x  τ={sum(accepts)/len(accepts):<6.2f} | ", end="")
    print()

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model, "draft": args.draft,
            "results": {n: {d: {"speedup": r.speedup, "accept_length": r.accept_length, "throughput": r.throughput}
                            for d, r in res.items()} for n, res in all_results.items()}
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

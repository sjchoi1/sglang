#!/usr/bin/env python3
"""
TileSpec Benchmark - Rigorous Evaluation for ACL Submission.

5 datasets for comprehensive evaluation:
1. MT-bench (80) - multi-turn conversation
2. HumanEval (164) - code generation
3. GSM8K (500) - math reasoning
4. Alpaca (500) - instruction following
5. CNN/Daily Mail (500) - summarization

Methodology (following Spec-Bench ACL 2024):
- Multiple runs (default 3) with mean ± std reporting
- temp=0 (deterministic, greedy decoding)
- FP16 precision
- No max_new_tokens cap (natural EOS)
- ShareGPT for profiling warmup
- Hardware/software info logged

Usage:
    python tile_spec/run_bench.py --configs AR Eagle3 Eagle3+TileSpec
    python tile_spec/run_bench.py --batch-sizes 1 4 16 64 --num-runs 3
    python tile_spec/run_bench.py --batch-sizes 1 64 128 --num-iters 30
"""

import argparse
import gc
import gzip
import json
import os
import platform
import statistics
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


def get_hardware_info() -> dict:
    """Collect hardware/software info for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return info

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
    accept_length: float  # τ: mean tokens accepted per step
    accept_rate: float    # n-α: proportion of drafts accepted
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


def cycle_samples(samples: List[dict], target_count: int) -> List[dict]:
    """Cycle or truncate samples to reach exactly target_count."""
    if len(samples) == target_count:
        return samples
    elif len(samples) > target_count:
        # Truncate
        return samples[:target_count]
    else:
        # Cycle up
        repeated = []
        while len(repeated) < target_count:
            repeated.extend(samples)
        return repeated[:target_count]


def format_chat(messages: List[dict]) -> str:
    """Format conversation as Llama-style chat string."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            parts.append(f"{content}")
    return " ".join(parts)


def run_dataset(engine, samples: List[dict], name: str) -> Result:
    """Run benchmark on dataset. No max_new_tokens - natural EOS."""
    total_tokens = 0
    total_time = 0.0
    accept_lengths = []  # τ: tokens per step
    accept_rates = []    # n-α: proportion accepted

    for sample in samples:
        if "turns" in sample:
            # Multi-turn (MT-bench) - format as chat string
            messages = []
            for turn in sample["turns"]:
                messages.append({"role": "user", "content": turn})
                prompt = format_chat(messages)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = engine.generate(prompt, sampling_params={"temperature": 0})
                torch.cuda.synchronize()
                total_time += time.perf_counter() - t0

                meta = result.get("meta_info", {})
                total_tokens += meta.get("completion_tokens", len(result.get("text", "").split()))
                if meta.get("spec_accept_length", 0) > 0:
                    accept_lengths.append(meta["spec_accept_length"])
                if meta.get("spec_accept_rate", 0) > 0:
                    accept_rates.append(meta["spec_accept_rate"])
                messages.append({"role": "assistant", "content": result.get("text", "")})
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
            if meta.get("spec_accept_rate", 0) > 0:
                accept_rates.append(meta["spec_accept_rate"])

    return Result(
        dataset=name,
        num_samples=len(samples),
        total_tokens=total_tokens,
        total_time=total_time,
        throughput=total_tokens / total_time if total_time > 0 else 0,
        accept_length=sum(accept_lengths) / len(accept_lengths) if accept_lengths else 1.0,
        accept_rate=sum(accept_rates) / len(accept_rates) if accept_rates else 0.0,
    )


def run_config(config: Config, datasets: Dict, model: str, draft: str, cache_dir: Path, batch_size: int = 1, min_samples: int = 0) -> Dict[str, Result]:
    """Run all datasets for a config."""
    import sglang as sgl

    kwargs = {"model_path": model, "dtype": "float16", "max_running_requests": batch_size}
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

    print(f"\n{'='*70}\nConfig: {config.name} (batch_size={batch_size})\n{'='*70}")
    engine = sgl.Engine(**kwargs)

    # Warmup with ShareGPT (also does profiling for tile-spec)
    sharegpt = load_sharegpt(cache_dir, limit=500)
    sharegpt_iter = iter(sharegpt)

    if config.tile_spec:
        print("Profiling with ShareGPT (waiting for tile_spec_ready)...")
        count = 0
        while not engine.tile_spec_ready():
            prompt = next(sharegpt_iter, None)
            if prompt is None:
                break
            engine.generate(prompt, sampling_params={"temperature": 0})
            count += 1
        print(f"  Profiling complete after {count} prompts")
    else:
        print("Warmup with 10 ShareGPT prompts...")
        for prompt in list(sharegpt_iter)[:10]:
            engine.generate(prompt, sampling_params={"temperature": 0})

    # Run datasets
    results = {}
    for name, samples in datasets.items():
        if min_samples > 0 and len(samples) != min_samples:
            samples = cycle_samples(samples, min_samples)
        print(f"\n  [{name}] {len(samples)} samples")
        result = run_dataset(engine, samples, name)
        results[name] = result
        print(f"    {result.throughput:.1f} tok/s, τ={result.accept_length:.2f}, α={result.accept_rate:.2f}")

    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="TileSpec Benchmark - Rigorous Evaluation")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft", default="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--cache-dir", default="tile_spec/data")
    parser.add_argument("--output", default="tile_spec/results.json")
    parser.add_argument("--configs", nargs="+", default=["AR", "Eagle3", "Eagle3+TileSpec"])
    parser.add_argument("--datasets", nargs="+", default=["mt_bench", "humaneval", "gsm8k", "alpaca", "cnn_dm"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for mean±std (default: 3)")
    parser.add_argument("--num-iters", type=int, default=30, help="Number of iterations per batch size (samples = batch_size * num_iters)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    # Collect hardware info
    hw_info = get_hardware_info()
    print("=" * 70)
    print("Hardware/Software Info:")
    for k, v in hw_info.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
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

    # Run all batch sizes with multiple runs
    # Structure: {batch_size: {config: {dataset: [Result, Result, ...]}}}}
    all_runs = {}
    for batch_size in args.batch_sizes:
        num_samples = batch_size * args.num_iters
        print(f"\n{'#'*90}")
        print(f"# Batch Size: {batch_size} | Samples: {num_samples} | Runs: {args.num_runs}")
        print(f"{'#'*90}")

        all_runs[batch_size] = {cfg: {ds: [] for ds in args.datasets} for cfg in args.configs}

        for run_idx in range(args.num_runs):
            print(f"\n--- Run {run_idx + 1}/{args.num_runs} ---")
            ar_results = None

            for cfg_name in args.configs:
                results = run_config(
                    all_configs[cfg_name], datasets, args.model, args.draft,
                    cache_dir, batch_size, num_samples
                )

                # Compute speedup relative to AR
                if cfg_name == "AR":
                    ar_results = results
                elif ar_results:
                    for ds, r in results.items():
                        if ds in ar_results and ar_results[ds].throughput > 0:
                            r.speedup = r.throughput / ar_results[ds].throughput

                # Store results
                for ds, r in results.items():
                    all_runs[batch_size][cfg_name][ds].append(r)

        # Print summary with mean ± std
        print(f"\n{'='*100}")
        print(f"Results: Batch Size {batch_size} (mean ± std over {args.num_runs} runs)")
        print(f"{'='*100}")
        print(f"{'Dataset':<12} | ", end="")
        for cfg_name in args.configs:
            print(f"{cfg_name:<28} | ", end="")
        print()
        print("-" * 100)

        for ds in args.datasets:
            print(f"{ds:<12} | ", end="")
            for cfg_name in args.configs:
                runs = all_runs[batch_size][cfg_name].get(ds, [])
                if runs:
                    speedups = [r.speedup for r in runs]
                    accepts = [r.accept_length for r in runs]
                    rates = [r.accept_rate for r in runs]
                    sp_mean = statistics.mean(speedups)
                    sp_std = statistics.stdev(speedups) if len(speedups) > 1 else 0
                    ac_mean = statistics.mean(accepts)
                    ar_mean = statistics.mean(rates) if rates else 0
                    print(f"{sp_mean:>5.2f}x±{sp_std:<4.2f} τ={ac_mean:<4.2f} α={ar_mean:<4.2f}| ", end="")
                else:
                    print(f"{'N/A':<32} | ", end="")
            print()

        print("-" * 110)
        print(f"{'Overall':<12} | ", end="")
        for cfg_name in args.configs:
            all_speedups = []
            all_accepts = []
            all_rates = []
            for ds in args.datasets:
                runs = all_runs[batch_size][cfg_name].get(ds, [])
                all_speedups.extend([r.speedup for r in runs])
                all_accepts.extend([r.accept_length for r in runs])
                all_rates.extend([r.accept_rate for r in runs])
            if all_speedups:
                sp_mean = statistics.mean(all_speedups)
                sp_std = statistics.stdev(all_speedups) if len(all_speedups) > 1 else 0
                ac_mean = statistics.mean(all_accepts)
                ar_mean = statistics.mean(all_rates) if all_rates else 0
                print(f"{sp_mean:>5.2f}x±{sp_std:<4.2f} τ={ac_mean:<4.2f} α={ar_mean:<4.2f}| ", end="")
        print()

    # Compute summary statistics
    summary = {}
    for bs in args.batch_sizes:
        summary[bs] = {}
        for cfg in args.configs:
            summary[bs][cfg] = {}
            for ds in args.datasets:
                runs = all_runs[bs][cfg].get(ds, [])
                if runs:
                    speedups = [r.speedup for r in runs]
                    accepts = [r.accept_length for r in runs]
                    rates = [r.accept_rate for r in runs]
                    throughputs = [r.throughput for r in runs]
                    summary[bs][cfg][ds] = {
                        "speedup_mean": round(statistics.mean(speedups), 3),
                        "speedup_std": round(statistics.stdev(speedups), 3) if len(speedups) > 1 else 0,
                        "accept_length_mean": round(statistics.mean(accepts), 3),  # τ
                        "accept_rate_mean": round(statistics.mean(rates), 3) if rates else 0,  # α
                        "throughput_mean": round(statistics.mean(throughputs), 1),
                        "throughput_std": round(statistics.stdev(throughputs), 1) if len(throughputs) > 1 else 0,
                        "num_runs": len(runs),
                    }

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    output_data = {
        "hardware_info": hw_info,
        "settings": {
            "model": args.model,
            "draft": args.draft,
            "num_runs": args.num_runs,
            "num_iters": args.num_iters,
            "batch_sizes": args.batch_sizes,
            "datasets": args.datasets,
            "configs": args.configs,
        },
        "results": summary,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

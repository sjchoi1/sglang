#!/usr/bin/env python3
"""
Tile-spec offline benchmark.

Runs ShareGPT prompts through sglang.Engine to:
1. Profile latency (first ~100 requests)
2. Benchmark throughput with tile-spec optimization

Usage:
    python tile_spec/benchmark.py
"""

import json
import time
import urllib.request
from pathlib import Path

import sglang as sgl

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def download_sharegpt(cache_dir: Path) -> Path:
    """Download ShareGPT dataset if not cached."""
    dataset_path = cache_dir / "sharegpt.json"
    if dataset_path.exists():
        print(f"Using cached ShareGPT: {dataset_path}")
        return dataset_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ShareGPT dataset...")
    urllib.request.urlretrieve(SHAREGPT_URL, dataset_path)
    print(f"Downloaded to {dataset_path}")
    return dataset_path


def load_prompts(dataset_path: Path, num_prompts: int = 200) -> list:
    """Load prompts from ShareGPT."""
    with open(dataset_path) as f:
        data = json.load(f)

    prompts = []
    for item in data:
        if "conversations" in item:
            for conv in item["conversations"]:
                if conv.get("from") == "human" and conv.get("value"):
                    text = conv["value"].strip()
                    if 50 < len(text) < 2000:
                        prompts.append(text)
                        if len(prompts) >= num_prompts:
                            return prompts
    return prompts


def main():
    cache_dir = Path(__file__).parent / "cache"

    # Load prompts
    dataset_path = download_sharegpt(cache_dir)
    prompts = load_prompts(dataset_path, num_prompts=200)
    print(f"Loaded {len(prompts)} prompts")

    # Create engine with tile-spec
    print("\nStarting engine with tile-spec...")
    engine = sgl.Engine(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",
        speculative_num_steps=5,
        speculative_eagle_topk=8,
        speculative_num_draft_tokens=32,
        dtype="float16",
        tile_spec=True,
    )

    # Run benchmark
    print(f"\nRunning {len(prompts)} prompts...")
    print("(First ~100 will trigger profiling, then tile-spec optimization kicks in)\n")

    total_tokens = 0
    start_time = time.perf_counter()

    for i, prompt in enumerate(prompts):
        output = engine.generate(prompt, sampling_params={"max_new_tokens": 100})
        total_tokens += len(output["text"].split())  # rough token count

        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  {i+1}/{len(prompts)} prompts, {elapsed:.1f}s elapsed")

    elapsed = time.perf_counter() - start_time

    print(f"\n=== Results ===")
    print(f"Prompts: {len(prompts)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {len(prompts)/elapsed:.2f} prompts/s")
    print(f"Approx tokens: {total_tokens} ({total_tokens/elapsed:.1f} tok/s)")

    engine.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Offline profiling for tile-spec.

Runs inference on ShareGPT prompts to collect latency data.
No HTTP server needed - uses SGLang Engine API directly.
"""

import argparse
import sglang as sgl
from sglang.srt.speculative.tile_spec.profiler import download_sharegpt, load_sharegpt_prompts
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft-model-path", default="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    # Download ShareGPT
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    dataset_path = download_sharegpt(cache_dir)
    prompts = load_sharegpt_prompts(dataset_path, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts from ShareGPT")

    # Create engine with tile-spec enabled
    engine = sgl.Engine(
        model_path=args.model_path,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=args.draft_model_path,
        speculative_num_steps=5,
        speculative_eagle_topk=8,
        speculative_num_draft_tokens=32,
        dtype="float16",
        tile_spec=True,
    )

    # Run inference - this triggers verify() which collects profiling data
    print(f"Running inference on {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        engine.generate(prompt, max_new_tokens=args.max_tokens)
        if (i + 1) % 50 == 0:
            print(f"  Completed {i + 1}/{len(prompts)} prompts")

    print("Profiling complete! Models saved to cache.")
    engine.shutdown()


if __name__ == "__main__":
    main()

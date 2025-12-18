#!/usr/bin/env python3
"""
Tile-spec warmup script.

Sends ShareGPT prompts to a running sglang server to trigger profiling
in the actual speculation verify() path.

Usage:
    # Start server with tile-spec enabled (in one terminal):
    ./tile_spec/run_tilespec.sh

    # Run warmup (in another terminal):
    python -m sglang.srt.speculative.tile_spec.warmup --port 30000

The profiling data will be collected during verify() calls and saved
to the cache automatically when enough samples are collected.
"""

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def download_sharegpt(cache_dir: Path) -> Path:
    """Download ShareGPT dataset if not cached."""
    dataset_path = cache_dir / "sharegpt.json"
    if dataset_path.exists():
        logger.info(f"Using cached ShareGPT: {dataset_path}")
        return dataset_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading ShareGPT dataset...")

    try:
        urllib.request.urlretrieve(SHAREGPT_URL, dataset_path)
        logger.info(f"Downloaded to {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to download ShareGPT: {e}")
        raise

    return dataset_path


def load_sharegpt_prompts(dataset_path: Path, num_prompts: int = 200) -> List[str]:
    """Load prompts from ShareGPT dataset."""
    with open(dataset_path) as f:
        data = json.load(f)

    prompts = []
    for item in data:
        if "conversations" in item:
            for conv in item["conversations"]:
                if conv.get("from") == "human" and conv.get("value"):
                    text = conv["value"].strip()
                    # Filter reasonable length prompts
                    if 50 < len(text) < 2000:
                        prompts.append(text)
                        if len(prompts) >= num_prompts:
                            return prompts
    return prompts


def send_request(host: str, port: int, prompt: str, max_tokens: int = 50) -> bool:
    """Send a generate request to the server."""
    url = f"http://{host}:{port}/generate"
    data = json.dumps({
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status == 200
    except Exception as e:
        logger.warning(f"Request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Tile-spec warmup with ShareGPT")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--num-prompts", type=int, default=150, help="Number of prompts to send")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens per request")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for ShareGPT")
    args = parser.parse_args()

    # Determine cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        # Use tile_spec/cache/ in project root
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "tile_spec").exists() and (current / "python").exists():
                cache_dir = current / "tile_spec" / "cache"
                break
            current = current.parent
        else:
            cache_dir = Path.home() / ".cache" / "sglang" / "tile_spec"

    # Download and load ShareGPT
    dataset_path = download_sharegpt(cache_dir)
    prompts = load_sharegpt_prompts(dataset_path, args.num_prompts)
    logger.info(f"Loaded {len(prompts)} prompts from ShareGPT")

    # Send requests
    logger.info(f"Sending warmup requests to http://{args.host}:{args.port}...")
    success_count = 0

    for i, prompt in enumerate(prompts):
        if send_request(args.host, args.port, prompt, args.max_tokens):
            success_count += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{len(prompts)} ({success_count} successful)")

    logger.info(f"\nWarmup complete: {success_count}/{len(prompts)} requests successful")
    logger.info("Tile-spec profiling data should now be saved to cache.")
    logger.info("Check server logs for profiling status.")


if __name__ == "__main__":
    main()

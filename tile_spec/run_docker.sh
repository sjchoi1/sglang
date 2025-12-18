#!/bin/bash
# TileSpec Docker Environment
# Usage: ./tile_spec/run_docker.sh [sglang_path]

SGLANG_PATH="${1:-/datadisk/sglang}"

echo "Starting TileSpec Docker container..."
echo "Mounting: $SGLANG_PATH -> /workspace/sglang"

docker run --gpus all -it --rm \
  -v "$SGLANG_PATH":/workspace/sglang \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/sglang \
  --shm-size=16g \
  lmsysorg/sglang:latest \
  bash -c "pip install -e 'python[dev]' && exec bash"

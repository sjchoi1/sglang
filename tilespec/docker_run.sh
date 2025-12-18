#!/bin/bash
# TileSpec Docker Environment
# Usage: ./tilespec/docker_run.sh [sglang_path]

SGLANG_PATH="${1:-/datadisk/sglang}"

echo "Starting TileSpec Docker container..."
echo "Mounting: $SGLANG_PATH -> /workspace/sglang"

docker run --gpus all -it --rm \
  -v "$SGLANG_PATH":/workspace/sglang \
  -w /workspace/sglang \
  --shm-size=16g \
  lmsysorg/sglang:latest \
  bash

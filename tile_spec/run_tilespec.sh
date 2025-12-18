#!/bin/bash
# Run sglang with tile-spec enabled

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

sglang serve --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 32 \
    --dtype float16 \
    --tile-spec

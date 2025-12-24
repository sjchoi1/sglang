# EAGLE-3 model pairs for dense models only (no MoE)
# Format: (base_model, eagle3_draft_model)

EAGLE3_MODEL_PAIRS = [
    # Vicuna
    ("lmsys/vicuna-13b-v1.3", "yuhuili/EAGLE3-Vicuna1.3-13B"),

    # Llama 3.x
    ("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
    ("meta-llama/Llama-3.3-70B-Instruct", "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"),

    # DeepSeek
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"),

    # Qwen3 dense
    ("Qwen/Qwen3-1.7B", "AngelSlim/Qwen3-1.7B_eagle3"),
    ("Qwen/Qwen3-4B", "AngelSlim/Qwen3-4B_eagle3"),
    ("Qwen/Qwen3-4B-Instruct-2507", "Zjcxy-SmartAI/Eagle3-Qwen3-4B-Instruct-2507-zh"),
    ("Qwen/Qwen3-8B", "Tengyunw/qwen3_8b_eagle3"),
    ("Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3"),
    ("Qwen/Qwen3-8B", "Zjcxy-SmartAI/Eagle3-Qwen3-8B-zh"),
    ("Qwen/Qwen3-14B", "AngelSlim/Qwen3-14B_eagle3"),
    ("Qwen/Qwen3-32B", "AngelSlim/Qwen3-32B_eagle3"),
    ("Qwen/Qwen3-32B", "Zjcxy-SmartAI/Eagle3-Qwen3-32B-zh"),

    # MiniCPM
    ("openbmb/MiniCPM4-8B", "linglingdan/Eagle3_for_MiniCPM4"),
]

# Excluded MoE models:
# - meta-llama/Llama-4-Scout-17B-16E-Instruct (16 experts)
# - meta-llama/Llama-4-Maverick-17B-128E-Instruct (128 experts)
# - Qwen/Qwen3-30B-A3B (MoE, 3B active)
# - Qwen/Qwen3-235B-A22B (MoE, 22B active)
# - allenai/OLMoE-1B-7B-0125-Instruct (MoE)
# - openai/gpt-oss-120b (MoE)

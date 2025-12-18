#!/usr/bin/env python3
"""
Spec-Bench evaluation for SGLang.

Matches Spec-Bench methodology exactly:
- 480 questions across 6 tasks
- Multi-turn conversations
- temp=0, max_new_tokens=1024
- Reports speedup, τ (acceptance length), Mean

Usage:
    python tile_spec/spec_bench.py --configs AR Eagle3 Eagle3+TileSpec
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Task definitions matching Spec-Bench
TASKS = {
    "mt_bench": ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"],
    "translation": ["translation"],
    "summarization": ["summarization"],
    "qa": ["qa"],
    "math_reasoning": ["math_reasoning"],
    "rag": ["rag"],
}


@dataclass
class Config:
    name: str
    speculative_algorithm: Optional[str] = None
    draft_model_path: Optional[str] = None
    tile_spec: bool = False


@dataclass
class TaskResult:
    task: str
    num_questions: int
    total_tokens: int
    total_time: float
    throughput: float
    mean_accept_len: float
    speedup: float = 1.0


def load_spec_bench(path: Path) -> List[dict]:
    """Load Spec-Bench questions."""
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def get_questions_by_task(questions: List[dict], task: str) -> List[dict]:
    """Filter questions by task."""
    categories = TASKS.get(task, [task])
    return [q for q in questions if q.get("category") in categories]


def run_task(
    engine,
    questions: List[dict],
    task: str,
) -> TaskResult:
    """Run evaluation on a single task."""
    total_tokens = 0
    total_time = 0.0
    accept_lengths = []

    for q in questions:
        turns = q.get("turns", [q.get("prompt", "")])
        conversation = []

        for turn in turns:
            # Build conversation
            conversation.append({"role": "user", "content": turn})

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            result = engine.generate(
                conversation,
                sampling_params={"temperature": 0, "max_new_tokens": 1024}
            )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # Extract metrics
            meta = result.get("meta_info", {})
            tokens = meta.get("completion_tokens", len(result.get("text", "").split()))
            accept_len = meta.get("spec_accept_length", 1.0)

            total_tokens += tokens
            total_time += elapsed
            if accept_len > 0:
                accept_lengths.append(accept_len)

            # Add assistant response to conversation for multi-turn
            conversation.append({"role": "assistant", "content": result.get("text", "")})

    throughput = total_tokens / total_time if total_time > 0 else 0
    mean_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 1.0

    return TaskResult(
        task=task,
        num_questions=len(questions),
        total_tokens=total_tokens,
        total_time=total_time,
        throughput=throughput,
        mean_accept_len=mean_accept,
    )


def run_config(
    config: Config,
    questions: List[dict],
    model_path: str,
    draft_path: str,
    batch_size: int,
) -> Dict[str, TaskResult]:
    """Run all tasks for a config."""
    import sglang as sgl

    # Engine config
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
        engine.generate("Hello", sampling_params={"temperature": 0, "max_new_tokens": 32})

    # Extra warmup for tile-spec profiling
    if config.tile_spec:
        print("Tile-spec profiling warmup...")
        for _ in range(100):
            engine.generate("Explain AI briefly.", sampling_params={"temperature": 0, "max_new_tokens": 64})

    # Run each task
    results = {}
    for task in TASKS.keys():
        task_questions = get_questions_by_task(questions, task)
        if not task_questions:
            continue

        print(f"\n  [{task}] {len(task_questions)} questions...")
        result = run_task(engine, task_questions, task)
        results[task] = result
        print(f"    {result.throughput:.1f} tok/s, τ={result.mean_accept_len:.2f}")

    # Overall
    print(f"\n  [overall] {len(questions)} questions...")
    overall = run_task(engine, questions, "overall")
    results["overall"] = overall
    print(f"    {overall.throughput:.1f} tok/s, τ={overall.mean_accept_len:.2f}")

    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Spec-Bench evaluation for SGLang")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft", default="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--dataset", default="Spec-Bench/data/spec_bench/question.jsonl")
    parser.add_argument("--batch-size", type=int, default=1, help="max_running_requests")
    parser.add_argument("--output", default="tile_spec/spec_bench_results.json")
    parser.add_argument("--configs", nargs="+", default=["AR", "Eagle3", "Eagle3+TileSpec"])
    args = parser.parse_args()

    # Configs
    all_configs = {
        "AR": Config("AR"),
        "Eagle3": Config("Eagle3", "EAGLE3", args.draft),
        "Eagle3+TileSpec": Config("Eagle3+TileSpec", "EAGLE3", args.draft, tile_spec=True),
    }

    # Load dataset
    questions = load_spec_bench(Path(args.dataset))
    print(f"Loaded {len(questions)} questions from Spec-Bench")

    # Run benchmarks
    all_results = {}
    ar_results = None

    for name in args.configs:
        config = all_configs[name]
        results = run_config(config, questions, args.model, args.draft, args.batch_size)
        all_results[name] = results

        # Calculate speedups
        if name == "AR":
            ar_results = results
        elif ar_results:
            for task, result in results.items():
                if task in ar_results:
                    ar_tp = ar_results[task].throughput
                    result.speedup = result.throughput / ar_tp if ar_tp > 0 else 1.0

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Task':<15} ", end="")
    for name in args.configs:
        print(f"| {name:<22} ", end="")
    print()
    print("-" * 90)

    tasks_to_show = list(TASKS.keys()) + ["overall"]
    for task in tasks_to_show:
        print(f"{task:<15} ", end="")
        for name in args.configs:
            if task in all_results.get(name, {}):
                r = all_results[name][task]
                print(f"| {r.speedup:>5.2f}x τ={r.mean_accept_len:<5.2f} ", end="")
            else:
                print(f"| {'N/A':<22} ", end="")
        print()

    # Calculate Mean (average of task speedups, excluding overall)
    print("-" * 90)
    print(f"{'Mean':<15} ", end="")
    for name in args.configs:
        task_speedups = [all_results[name][t].speedup for t in TASKS.keys() if t in all_results.get(name, {})]
        task_accepts = [all_results[name][t].mean_accept_len for t in TASKS.keys() if t in all_results.get(name, {})]
        if task_speedups:
            mean_speedup = sum(task_speedups) / len(task_speedups)
            mean_accept = sum(task_accepts) / len(task_accepts)
            print(f"| {mean_speedup:>5.2f}x τ={mean_accept:<5.2f} ", end="")
        else:
            print(f"| {'N/A':<22} ", end="")
    print()

    # Save results
    Path(args.output).parent.mkdir(exist_ok=True)
    output_data = {
        "model": args.model,
        "draft": args.draft,
        "batch_size": args.batch_size,
        "results": {
            name: {
                task: {
                    "num_questions": r.num_questions,
                    "total_tokens": r.total_tokens,
                    "total_time": r.total_time,
                    "throughput": r.throughput,
                    "mean_accept_len": r.mean_accept_len,
                    "speedup": r.speedup,
                }
                for task, r in results.items()
            }
            for name, results in all_results.items()
        }
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

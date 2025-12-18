"""
Benchmark Speculative Decoding: AR vs EAGLE3 vs EAGLE3 + TileSpec

Following EAGLE3 paper evaluation methodology:
- Tasks: MT-bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail
- Metrics: Speedup Ratio, Average Acceptance Length (τ), Output Throughput

Usage:
    # Basic usage with Llama-3.1-8B
    python tilespec/benchmark_spec_decoding.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --draft-model-path lmsys/sglang-EAGLE3-LLaMA-3.1-Instruct-8B \
        --tasks gsm8k mt-bench \
        --methods ar eagle3 tilespec

    # Quick test with GSM8K only
    python tilespec/benchmark_spec_decoding.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --draft-model-path lmsys/sglang-EAGLE3-LLaMA-3.1-Instruct-8B \
        --tasks gsm8k \
        --num-samples 50

Output format matches EAGLE3 paper Table 1:
| Task     | Method  | Speedup | τ (Acc. Len) | Throughput |
|----------|---------|---------|--------------|------------|
| GSM8K    | AR      | 1.00x   | 1.00         | 45.2 tok/s |
| GSM8K    | EAGLE3  | 2.85x   | 3.42         | 128.8 tok/s|
| GSM8K    | TileSpec| 3.12x   | 3.42         | 141.0 tok/s|
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import requests

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import numpy as np

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    task: str
    method: str
    num_samples: int
    total_output_tokens: int
    total_time_s: float
    num_verify_steps: int = 0  # For spec decoding

    @property
    def throughput(self) -> float:
        """Output throughput in tokens/second."""
        return self.total_output_tokens / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def acceptance_length(self) -> float:
        """Average acceptance length (τ)."""
        if self.num_verify_steps > 0:
            return self.total_output_tokens / self.num_verify_steps
        return 1.0  # AR baseline

    def speedup_vs(self, baseline: "BenchmarkResult") -> float:
        """Compute speedup ratio vs baseline."""
        if baseline.throughput > 0:
            return self.throughput / baseline.throughput
        return 1.0


@dataclass
class MethodConfig:
    """Configuration for a benchmark method."""
    name: str
    display_name: str
    server_args: List[str] = field(default_factory=list)


class SpecDecodingBenchmark:
    """Benchmark harness for speculative decoding methods."""

    BASE_URL = "http://127.0.0.1:30000"

    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        calibration_path: Optional[str] = None,
        latency_path: Optional[str] = None,
        tp_size: int = 1,
        trust_remote_code: bool = False,
    ):
        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.calibration_path = calibration_path
        self.latency_path = latency_path
        self.tp_size = tp_size
        self.trust_remote_code = trust_remote_code

        # Define methods
        self.methods = self._build_method_configs()

    def _build_method_configs(self) -> Dict[str, MethodConfig]:
        """Build method configurations."""

        methods = {}

        # AR baseline (no speculative decoding)
        methods["ar"] = MethodConfig(
            name="ar",
            display_name="AR (baseline)",
            server_args=[],
        )

        # EAGLE3 speculative decoding
        eagle3_args = [
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", self.draft_model_path,
            "--speculative-num-steps", "5",
            "--speculative-eagle-topk", "4",
            "--speculative-num-draft-tokens", "64",
        ]
        methods["eagle3"] = MethodConfig(
            name="eagle3",
            display_name="EAGLE3",
            server_args=eagle3_args,
        )

        # EAGLE3 + TileSpec
        tilespec_args = eagle3_args.copy()
        tilespec_args.extend(["--speculative-tile-aware"])
        if self.calibration_path:
            tilespec_args.extend(["--speculative-calibration-path", self.calibration_path])
        if self.latency_path:
            tilespec_args.extend(["--speculative-latency-path", self.latency_path])

        methods["tilespec"] = MethodConfig(
            name="tilespec",
            display_name="EAGLE3+TileSpec",
            server_args=tilespec_args,
        )

        return methods

    def _build_server_args(self, method: MethodConfig) -> List[str]:
        """Build complete server arguments."""
        args = [
            "--tp-size", str(self.tp_size),
            "--mem-fraction-static", "0.85",
        ]
        if self.trust_remote_code:
            args.append("--trust-remote-code")

        args.extend(method.server_args)
        return args

    def _launch_server(self, method: MethodConfig) -> any:
        """Launch SGLang server with given configuration."""
        server_args = self._build_server_args(method)
        print(f"\n{'='*60}")
        print(f"Launching server for {method.display_name}")
        print(f"Args: {' '.join(server_args)}")
        print(f"{'='*60}\n")

        process = popen_launch_server(
            self.model_path,
            self.BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )
        return process

    def _get_server_info(self) -> dict:
        """Get server info including spec decoding stats."""
        try:
            resp = requests.get(f"{self.BASE_URL}/get_server_info", timeout=10)
            return resp.json()
        except Exception as e:
            print(f"Warning: Failed to get server info: {e}")
            return {}

    def _flush_cache(self):
        """Flush server cache between runs."""
        try:
            requests.get(f"{self.BASE_URL}/flush_cache", timeout=10)
        except:
            pass

    def run_gsm8k(
        self,
        method: MethodConfig,
        num_samples: int = 200,
        num_shots: int = 5,
    ) -> BenchmarkResult:
        """Run GSM8K benchmark."""
        from sglang.test.few_shot_gsm8k import run_eval
        from types import SimpleNamespace

        self._flush_cache()

        args = SimpleNamespace(
            num_shots=num_shots,
            data_path=None,
            num_questions=num_samples,
            max_new_tokens=512,
            parallel=64,
            host="http://127.0.0.1",
            port=30000,
        )

        tic = time.perf_counter()
        metrics = run_eval(args)
        elapsed = time.perf_counter() - tic

        # Get spec decoding stats
        server_info = self._get_server_info()
        num_verify = 0
        if "internal_states" in server_info and server_info["internal_states"]:
            state = server_info["internal_states"][0]
            # Total verify steps approximated from output / acceptance length
            avg_accept = state.get("avg_spec_accept_length", 1.0)
            if avg_accept and avg_accept > 1:
                num_verify = int(metrics.get("total_tokens", num_samples * 256) / avg_accept)

        return BenchmarkResult(
            task="GSM8K",
            method=method.display_name,
            num_samples=num_samples,
            total_output_tokens=metrics.get("total_tokens", num_samples * 256),
            total_time_s=elapsed,
            num_verify_steps=num_verify,
        )

    def run_mt_bench(
        self,
        method: MethodConfig,
        num_samples: int = 80,
        question_file: str = "question.jsonl",
    ) -> BenchmarkResult:
        """Run MT-bench benchmark."""
        import sglang as sgl
        from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

        self._flush_cache()

        # Load questions
        questions = []
        mtbench_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark", "mtbench")
        qfile = os.path.join(mtbench_dir, question_file)

        if not os.path.exists(qfile):
            print(f"Warning: MT-bench question file not found at {qfile}")
            print("Downloading MT-bench questions...")
            # Use HuggingFace dataset as fallback
            try:
                from datasets import load_dataset
                ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
                questions = [{"prompt": [d["prompt"][0], d["prompt"][1] if len(d["prompt"]) > 1 else "Continue."]}
                           for d in list(ds)[:num_samples]]
            except Exception as e:
                print(f"Failed to load MT-bench: {e}")
                # Return dummy result
                return BenchmarkResult(
                    task="MT-bench",
                    method=method.display_name,
                    num_samples=0,
                    total_output_tokens=0,
                    total_time_s=1.0,
                )
        else:
            with open(qfile, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    questions.append(obj)
            questions = questions[:num_samples]

        # Setup backend
        backend = RuntimeEndpoint(self.BASE_URL)
        sgl.set_default_backend(backend)

        @sgl.function
        def answer_mt_bench(s, question_1, question_2):
            s += sgl.system("You are a helpful assistant.")
            s += sgl.user(question_1)
            s += sgl.assistant(sgl.gen("answer_1", max_tokens=1024))
            s += sgl.user(question_2)
            s += sgl.assistant(sgl.gen("answer_2", max_tokens=1024))

        # Prepare arguments
        arguments = []
        for q in questions:
            if "prompt" in q:
                q1, q2 = q["prompt"][0], q["prompt"][1] if len(q["prompt"]) > 1 else "Continue the discussion."
            elif "turns" in q:
                q1, q2 = q["turns"][0], q["turns"][1] if len(q["turns"]) > 1 else "Continue the discussion."
            else:
                continue
            arguments.append({"question_1": q1, "question_2": q2})

        # Run benchmark
        tic = time.perf_counter()
        rets = answer_mt_bench.run_batch(
            arguments,
            temperature=0,
            num_threads=min(32, len(arguments)),
            progress_bar=True,
        )
        elapsed = time.perf_counter() - tic

        # Count tokens
        total_tokens = 0
        total_verify = 0
        for ret in rets:
            meta1 = ret.get_meta_info("answer_1")
            meta2 = ret.get_meta_info("answer_2")
            total_tokens += meta1.get("completion_tokens", 0) + meta2.get("completion_tokens", 0)
            total_verify += meta1.get("spec_verify_ct", 0) + meta2.get("spec_verify_ct", 0)

        return BenchmarkResult(
            task="MT-bench",
            method=method.display_name,
            num_samples=len(arguments),
            total_output_tokens=total_tokens,
            total_time_s=elapsed,
            num_verify_steps=total_verify if total_verify > 0 else 0,
        )

    def run_humaneval(
        self,
        method: MethodConfig,
        num_samples: int = 164,  # Full HumanEval has 164 problems
    ) -> BenchmarkResult:
        """Run HumanEval benchmark."""
        import sglang as sgl
        from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

        self._flush_cache()

        try:
            from human_eval.data import read_problems
            problems = list(read_problems().values())[:num_samples]
        except ImportError:
            print("Warning: human_eval not installed. Skipping HumanEval.")
            return BenchmarkResult(
                task="HumanEval",
                method=method.display_name,
                num_samples=0,
                total_output_tokens=0,
                total_time_s=1.0,
            )

        backend = RuntimeEndpoint(self.BASE_URL)
        sgl.set_default_backend(backend)

        @sgl.function
        def solve_humaneval(s, prompt):
            s += sgl.user(
                "Complete the following Python function. "
                "Only output the function body without explanation.\n\n" + prompt
            )
            s += sgl.assistant(sgl.gen("code", max_tokens=512, stop=["```", "\ndef ", "\nclass "]))

        arguments = [{"prompt": p["prompt"]} for p in problems]

        tic = time.perf_counter()
        rets = solve_humaneval.run_batch(
            arguments,
            temperature=0,
            num_threads=min(32, len(arguments)),
            progress_bar=True,
        )
        elapsed = time.perf_counter() - tic

        total_tokens = sum(ret.get_meta_info("code").get("completion_tokens", 0) for ret in rets)
        total_verify = sum(ret.get_meta_info("code").get("spec_verify_ct", 0) for ret in rets)

        return BenchmarkResult(
            task="HumanEval",
            method=method.display_name,
            num_samples=len(problems),
            total_output_tokens=total_tokens,
            total_time_s=elapsed,
            num_verify_steps=total_verify if total_verify > 0 else 0,
        )

    def run_alpaca(
        self,
        method: MethodConfig,
        num_samples: int = 200,
    ) -> BenchmarkResult:
        """Run Alpaca instruction-following benchmark."""
        import sglang as sgl
        from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

        self._flush_cache()

        try:
            from datasets import load_dataset
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            samples = list(ds)[:num_samples]
        except Exception as e:
            print(f"Warning: Failed to load Alpaca dataset: {e}")
            return BenchmarkResult(
                task="Alpaca",
                method=method.display_name,
                num_samples=0,
                total_output_tokens=0,
                total_time_s=1.0,
            )

        backend = RuntimeEndpoint(self.BASE_URL)
        sgl.set_default_backend(backend)

        @sgl.function
        def follow_instruction(s, instruction, input_text):
            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}"
            else:
                prompt = instruction
            s += sgl.user(prompt)
            s += sgl.assistant(sgl.gen("response", max_tokens=512))

        arguments = [{"instruction": s["instruction"], "input_text": s.get("input", "")} for s in samples]

        tic = time.perf_counter()
        rets = follow_instruction.run_batch(
            arguments,
            temperature=0,
            num_threads=min(64, len(arguments)),
            progress_bar=True,
        )
        elapsed = time.perf_counter() - tic

        total_tokens = sum(ret.get_meta_info("response").get("completion_tokens", 0) for ret in rets)
        total_verify = sum(ret.get_meta_info("response").get("spec_verify_ct", 0) for ret in rets)

        return BenchmarkResult(
            task="Alpaca",
            method=method.display_name,
            num_samples=len(samples),
            total_output_tokens=total_tokens,
            total_time_s=elapsed,
            num_verify_steps=total_verify if total_verify > 0 else 0,
        )

    def run_cnn_dm(
        self,
        method: MethodConfig,
        num_samples: int = 200,
    ) -> BenchmarkResult:
        """Run CNN/DailyMail summarization benchmark."""
        import sglang as sgl
        from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

        self._flush_cache()

        try:
            from datasets import load_dataset
            ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
            samples = list(ds)[:num_samples]
        except Exception as e:
            print(f"Warning: Failed to load CNN/DailyMail dataset: {e}")
            return BenchmarkResult(
                task="CNN/DM",
                method=method.display_name,
                num_samples=0,
                total_output_tokens=0,
                total_time_s=1.0,
            )

        backend = RuntimeEndpoint(self.BASE_URL)
        sgl.set_default_backend(backend)

        @sgl.function
        def summarize(s, article):
            # Truncate long articles
            article = article[:4000]
            s += sgl.user(f"Summarize the following article in 2-3 sentences:\n\n{article}")
            s += sgl.assistant(sgl.gen("summary", max_tokens=256))

        arguments = [{"article": s["article"]} for s in samples]

        tic = time.perf_counter()
        rets = summarize.run_batch(
            arguments,
            temperature=0,
            num_threads=min(64, len(arguments)),
            progress_bar=True,
        )
        elapsed = time.perf_counter() - tic

        total_tokens = sum(ret.get_meta_info("summary").get("completion_tokens", 0) for ret in rets)
        total_verify = sum(ret.get_meta_info("summary").get("spec_verify_ct", 0) for ret in rets)

        return BenchmarkResult(
            task="CNN/DM",
            method=method.display_name,
            num_samples=len(samples),
            total_output_tokens=total_tokens,
            total_time_s=elapsed,
            num_verify_steps=total_verify if total_verify > 0 else 0,
        )

    def run_task(
        self,
        task: str,
        method: MethodConfig,
        num_samples: int,
    ) -> BenchmarkResult:
        """Run a specific task."""
        task_lower = task.lower().replace("-", "").replace("_", "")

        if task_lower in ["gsm8k", "gsm"]:
            return self.run_gsm8k(method, num_samples)
        elif task_lower in ["mtbench", "mt"]:
            return self.run_mt_bench(method, num_samples)
        elif task_lower in ["humaneval", "human", "he"]:
            return self.run_humaneval(method, num_samples)
        elif task_lower in ["alpaca"]:
            return self.run_alpaca(method, num_samples)
        elif task_lower in ["cnndm", "cnn", "cnndailymail"]:
            return self.run_cnn_dm(method, num_samples)
        else:
            raise ValueError(f"Unknown task: {task}")

    def run_benchmark(
        self,
        tasks: List[str],
        method_names: List[str],
        num_samples: int = 200,
    ) -> List[BenchmarkResult]:
        """Run full benchmark suite."""
        all_results = []

        for method_name in method_names:
            if method_name not in self.methods:
                print(f"Warning: Unknown method '{method_name}', skipping")
                continue

            method = self.methods[method_name]
            process = None

            try:
                # Launch server
                process = self._launch_server(method)
                time.sleep(5)  # Give server time to stabilize

                # Run each task
                for task in tasks:
                    print(f"\n>>> Running {task} with {method.display_name}...")
                    try:
                        result = self.run_task(task, method, num_samples)
                        all_results.append(result)
                        print(f"    Completed: {result.total_output_tokens} tokens in {result.total_time_s:.1f}s")
                        print(f"    Throughput: {result.throughput:.1f} tok/s, τ: {result.acceptance_length:.2f}")
                    except Exception as e:
                        print(f"    Error running {task}: {e}")
                        import traceback
                        traceback.print_exc()

            finally:
                if process:
                    print(f"\nShutting down server for {method.display_name}...")
                    kill_process_tree(process.pid)
                    time.sleep(3)

        return all_results


def format_results_table(results: List[BenchmarkResult], baseline_method: str = "AR (baseline)") -> str:
    """Format results as a table similar to EAGLE3 paper."""

    # Group by task
    tasks = {}
    for r in results:
        if r.task not in tasks:
            tasks[r.task] = {}
        tasks[r.task][r.method] = r

    # Build table
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("Benchmark Results: AR vs EAGLE3 vs EAGLE3+TileSpec")
    lines.append("=" * 80)
    lines.append("")

    header = f"{'Task':<12} {'Method':<18} {'Speedup':>10} {'τ (Acc.Len)':>12} {'Throughput':>14}"
    lines.append(header)
    lines.append("-" * 80)

    for task_name, task_results in tasks.items():
        # Find baseline
        baseline = task_results.get(baseline_method)

        for method_name, result in task_results.items():
            if result.num_samples == 0:
                continue

            speedup = result.speedup_vs(baseline) if baseline else 1.0
            tau = result.acceptance_length
            throughput = result.throughput

            speedup_str = f"{speedup:.2f}x"
            tau_str = f"{tau:.2f}"
            throughput_str = f"{throughput:.1f} tok/s"

            lines.append(f"{task_name:<12} {method_name:<18} {speedup_str:>10} {tau_str:>12} {throughput_str:>14}")

        lines.append("-" * 80)

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding methods")

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to target model")
    parser.add_argument("--draft-model-path", type=str, required=True,
                       help="Path to EAGLE draft model")
    parser.add_argument("--tp-size", type=int, default=1,
                       help="Tensor parallelism size")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code")

    # TileSpec configuration
    parser.add_argument("--calibration-path", type=str, default=None,
                       help="Path to calibration.npz for TileSpec")
    parser.add_argument("--latency-path", type=str, default=None,
                       help="Path to latency_model.npz for TileSpec")

    # Benchmark configuration
    parser.add_argument("--tasks", type=str, nargs="+",
                       default=["gsm8k"],
                       choices=["gsm8k", "mt-bench", "humaneval", "alpaca", "cnn-dm"],
                       help="Tasks to benchmark")
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["ar", "eagle3", "tilespec"],
                       choices=["ar", "eagle3", "tilespec"],
                       help="Methods to benchmark")
    parser.add_argument("--num-samples", type=int, default=200,
                       help="Number of samples per task")

    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")

    args = parser.parse_args()

    # Set default paths for TileSpec models
    tilespec_dir = os.path.dirname(os.path.abspath(__file__))
    if args.calibration_path is None:
        args.calibration_path = os.path.join(tilespec_dir, "calibration.npz")
    if args.latency_path is None:
        args.latency_path = os.path.join(tilespec_dir, "latency_model.npz")

    # Check TileSpec files exist if tilespec method requested
    if "tilespec" in args.methods:
        if not os.path.exists(args.calibration_path):
            print(f"Warning: calibration file not found at {args.calibration_path}")
            print("TileSpec may not work correctly. Run calibration first.")
        if not os.path.exists(args.latency_path):
            print(f"Warning: latency model not found at {args.latency_path}")
            print("TileSpec may not work correctly. Run latency profiling first.")

    # Create benchmark
    benchmark = SpecDecodingBenchmark(
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        calibration_path=args.calibration_path,
        latency_path=args.latency_path,
        tp_size=args.tp_size,
        trust_remote_code=args.trust_remote_code,
    )

    print("\n" + "=" * 80)
    print("TileSpec Benchmark: Speculative Decoding Comparison")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Draft: {args.draft_model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Methods: {args.methods}")
    print(f"Samples per task: {args.num_samples}")
    print("=" * 80 + "\n")

    # Run benchmark
    results = benchmark.run_benchmark(
        tasks=args.tasks,
        method_names=args.methods,
        num_samples=args.num_samples,
    )

    # Print results table
    table = format_results_table(results)
    print(table)

    # Save results
    if args.output:
        output_data = {
            "config": {
                "model_path": args.model_path,
                "draft_model_path": args.draft_model_path,
                "tasks": args.tasks,
                "methods": args.methods,
                "num_samples": args.num_samples,
            },
            "results": [
                {
                    "task": r.task,
                    "method": r.method,
                    "num_samples": r.num_samples,
                    "total_output_tokens": r.total_output_tokens,
                    "total_time_s": r.total_time_s,
                    "throughput": r.throughput,
                    "acceptance_length": r.acceptance_length,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

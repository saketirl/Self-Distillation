"""
Standalone evaluation script for continual learning experiments.

Evaluates a model checkpoint on all tasks and reports:
- Per-task accuracy
- Forgetting metrics (if baseline provided)
- Detailed error analysis

Usage:
    # Evaluate a checkpoint on all tasks
    python evaluate.py --model_path outputs/sdft_run/task_2_spider --output_file results.json

    # Compare to baseline
    python evaluate.py --model_path outputs/sdft_run/task_2_spider --baseline_path Qwen/Qwen2.5-1.5B-Instruct

    # Evaluate specific tasks only
    python evaluate.py --model_path outputs/sdft_run/task_2_spider --tasks tooluse spider
"""

import argparse
import json
import os
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data_loaders import load_task_dataset, load_all_eval_datasets, TASK_ORDER


class ExecutionTimeout(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timing out code execution."""
    def handler(signum, frame):
        raise ExecutionTimeout(f"Execution timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    # Try to find code in markdown blocks
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    # Try to find a function definition
    func_match = re.search(r"(def\s+\w+.*?)(?=\n\S|\Z)", response, re.DOTALL)
    if func_match:
        return func_match.group(1).strip()

    # Return the whole response as a fallback
    return response.strip()


def execute_code_with_tests(code: str, test_cases: List[str], timeout_sec: int = 5) -> bool:
    """Execute generated code against test cases. Returns True if all pass."""
    namespace = {}

    # Execute the generated code
    try:
        with timeout(timeout_sec):
            exec(code, namespace)
    except:
        return False

    # Run test cases
    for test in test_cases:
        try:
            with timeout(timeout_sec):
                exec(test, namespace)
        except:
            return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on continual learning tasks")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint or HuggingFace model name")
    parser.add_argument("--baseline_path", type=str, default=None,
                        help="Path to baseline model for forgetting comparison")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--tasks", type=str, nargs="+", default=TASK_ORDER,
                        help="Tasks to evaluate on")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max samples per task for evaluation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for generation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results including samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer from path."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Print device info
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    return model, tokenizer


def evaluate_task(
    model,
    tokenizer,
    eval_dataset,
    task_name: str,
    max_samples: int = 200,
    max_new_tokens: int = 256,
    verbose: bool = False
) -> Dict:
    """Evaluate model on a single task."""

    device = next(model.parameters()).device
    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    results = {
        "task": task_name,
        "total": len(eval_subset),
        "correct": 0,
        "samples": []
    }

    for idx, example in enumerate(tqdm(eval_subset, desc=f"Evaluating {task_name}")):
        prompt = example["prompt"]
        expected = example["response"]

        # Format as chat
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Task-specific evaluation
        is_correct = False

        if task_name == "spider":
            # Normalize SQL for comparison
            gen_normalized = generated.strip().lower().replace(";", "").strip()
            exp_normalized = expected.strip().lower().replace(";", "").strip()
            is_correct = gen_normalized == exp_normalized

        elif task_name == "json_extraction":
            # Check if valid JSON and matches
            try:
                # Try to extract JSON from response
                gen_text = generated.strip()
                # Handle cases where model adds explanation
                if "{" in gen_text:
                    start = gen_text.index("{")
                    end = gen_text.rindex("}") + 1
                    gen_text = gen_text[start:end]

                gen_json = json.loads(gen_text)
                exp_json = json.loads(expected.strip())
                is_correct = gen_json == exp_json
            except:
                is_correct = False

        elif task_name == "tooluse":
            # Check if the key tool name is mentioned
            # Extract tool name from expected response (e.g., "I need to use the X tool")
            exp_lower = expected.lower()
            gen_lower = generated.lower()

            # Look for tool name patterns
            if "tool" in exp_lower:
                # Try to extract tool name
                tool_match = re.search(r'use the (\w+) tool', exp_lower)
                if tool_match:
                    tool_name = tool_match.group(1)
                    is_correct = tool_name in gen_lower
                else:
                    # Fallback: check if key words match
                    is_correct = any(word in gen_lower for word in exp_lower.split() if len(word) > 4)
            else:
                is_correct = expected.lower().strip() in gen_lower

        elif task_name == "mbpp":
            # Execution-based evaluation for MBPP
            code = extract_code(generated)
            test_list = example.get("test_list", [])
            if test_list:
                is_correct = execute_code_with_tests(code, test_list, timeout_sec=5)
            else:
                # Fallback to exact match if no tests
                is_correct = code.strip() == expected.strip()

        if is_correct:
            results["correct"] += 1

        # Store sample for verbose output
        if verbose or (not is_correct and len(results["samples"]) < 5):
            results["samples"].append({
                "idx": idx,
                "expected": expected[:200],
                "generated": generated[:200],
                "correct": is_correct
            })

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("Continual Learning Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Tasks: {args.tasks}")
    print(f"Max samples per task: {args.max_samples}")
    print()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Load eval datasets
    print("Loading evaluation datasets...")
    all_eval_datasets = load_all_eval_datasets(seed=args.seed)

    # Evaluate on each task
    all_results = {
        "model_path": args.model_path,
        "timestamp": datetime.now().isoformat(),
        "tasks": {}
    }

    for task_name in args.tasks:
        if task_name not in all_eval_datasets:
            print(f"Warning: No eval dataset for task {task_name}, skipping")
            continue

        eval_dataset = all_eval_datasets[task_name]
        task_results = evaluate_task(
            model, tokenizer, eval_dataset, task_name,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose
        )

        all_results["tasks"][task_name] = task_results

        print(f"\n{task_name.upper()}: {task_results['accuracy']:.2%} ({task_results['correct']}/{task_results['total']})")

        if args.verbose and task_results["samples"]:
            print("  Sample errors:")
            for sample in task_results["samples"][:3]:
                if not sample["correct"]:
                    print(f"    Expected: {sample['expected'][:80]}...")
                    print(f"    Got:      {sample['generated'][:80]}...")
                    print()

    # Compare to baseline if provided
    if args.baseline_path:
        print("\n" + "=" * 60)
        print("Baseline Comparison")
        print("=" * 60)

        baseline_model, baseline_tokenizer = load_model_and_tokenizer(args.baseline_path)

        all_results["baseline"] = {"model_path": args.baseline_path, "tasks": {}}

        for task_name in args.tasks:
            if task_name not in all_eval_datasets:
                continue

            eval_dataset = all_eval_datasets[task_name]
            baseline_results = evaluate_task(
                baseline_model, baseline_tokenizer, eval_dataset, task_name,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                verbose=False
            )

            all_results["baseline"]["tasks"][task_name] = baseline_results

            model_acc = all_results["tasks"][task_name]["accuracy"]
            base_acc = baseline_results["accuracy"]
            diff = model_acc - base_acc

            print(f"{task_name}: {base_acc:.2%} (baseline) -> {model_acc:.2%} (model) [{diff:+.2%}]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_correct = sum(r["correct"] for r in all_results["tasks"].values())
    total_samples = sum(r["total"] for r in all_results["tasks"].values())
    overall_acc = total_correct / total_samples if total_samples > 0 else 0

    print(f"Overall accuracy: {overall_acc:.2%} ({total_correct}/{total_samples})")
    print()
    for task_name, task_results in all_results["tasks"].items():
        print(f"  {task_name}: {task_results['accuracy']:.2%}")

    # Save results
    if args.output_file:
        # Remove samples from saved results to reduce file size
        save_results = {
            "model_path": all_results["model_path"],
            "timestamp": all_results["timestamp"],
            "tasks": {
                k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                for k, v in all_results["tasks"].items()
            }
        }
        if "baseline" in all_results:
            save_results["baseline"] = {
                "model_path": all_results["baseline"]["model_path"],
                "tasks": {
                    k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                    for k, v in all_results["baseline"]["tasks"].items()
                }
            }

        with open(args.output_file, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

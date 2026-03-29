"""
Evaluate Qwen 2.5 3B baseline performance on MBPP (Mostly Basic Python Problems).

Usage:
    python scripts/eval_mbpp_baseline.py
    python scripts/eval_mbpp_baseline.py --model Qwen/Qwen2.5-3B-Instruct --max_samples 100
    python scripts/eval_mbpp_baseline.py --num_samples 5  # For pass@5
"""

import argparse
import json
import re
import signal
import sys
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on MBPP")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (None = all 500 test samples)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens for code generation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per problem for pass@k",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout in seconds for code execution",
    )
    return parser.parse_args()


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timing out code execution."""
    def handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds}s")

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


def execute_code_with_tests(code: str, test_cases: List[str], timeout_sec: int = 5) -> Dict:
    """
    Execute generated code against test cases.

    Returns dict with:
        - passed: bool
        - error: Optional[str]
        - tests_passed: int
        - tests_total: int
    """
    result = {
        "passed": False,
        "error": None,
        "tests_passed": 0,
        "tests_total": len(test_cases),
    }

    # Create isolated namespace for execution
    namespace = {}

    # First, try to execute the generated code
    try:
        with timeout(timeout_sec):
            exec(code, namespace)
    except TimeoutError as e:
        result["error"] = str(e)
        return result
    except Exception as e:
        result["error"] = f"Code execution error: {type(e).__name__}: {str(e)[:100]}"
        return result

    # Now run test cases
    for test in test_cases:
        try:
            with timeout(timeout_sec):
                exec(test, namespace)
            result["tests_passed"] += 1
        except TimeoutError:
            result["error"] = f"Test timed out: {test[:50]}..."
            break
        except AssertionError:
            result["error"] = f"Assertion failed: {test[:50]}..."
            break
        except Exception as e:
            result["error"] = f"Test error: {type(e).__name__}: {str(e)[:50]}"
            break

    result["passed"] = result["tests_passed"] == result["tests_total"]
    return result


def format_mbpp_prompt(problem: Dict) -> str:
    """Format MBPP problem as a prompt for the model."""
    text = problem["text"]
    test_list = problem["test_list"]

    # Show 1-2 test cases as examples
    example_tests = test_list[:2]
    test_examples = "\n".join(example_tests)

    prompt = f"""Write a Python function to solve the following problem.

Problem: {text}

Example test cases:
{test_examples}

Write only the Python function, no explanations."""

    return prompt


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    return model, tokenizer


def evaluate_mbpp(
    model,
    tokenizer,
    dataset,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 512,
    num_samples: int = 1,
    temperature: float = 0.0,
    timeout_sec: int = 5,
    verbose: bool = False,
) -> Dict:
    """Evaluate model on MBPP dataset."""

    device = next(model.parameters()).device

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    results = {
        "total": len(dataset),
        "passed": 0,
        "failed": 0,
        "errors": [],
        "samples": [],
    }

    do_sample = temperature > 0

    for idx, problem in enumerate(tqdm(dataset, desc="Evaluating MBPP")):
        prompt_text = format_mbpp_prompt(problem)

        # Format as chat
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Generate possibly multiple samples
        problem_passed = False
        generations = []

        for sample_idx in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Extract code and test it
            code = extract_code(generated)
            test_result = execute_code_with_tests(
                code, problem["test_list"], timeout_sec
            )

            generations.append({
                "code": code,
                "result": test_result,
            })

            if test_result["passed"]:
                problem_passed = True
                if num_samples == 1:
                    break  # For pass@1, stop after first success

        if problem_passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
            # Store error info
            results["errors"].append({
                "task_id": problem["task_id"],
                "error": generations[0]["result"]["error"] if generations else "No generation",
            })

        # Store sample for verbose output
        if verbose or (not problem_passed and len(results["samples"]) < 10):
            results["samples"].append({
                "task_id": problem["task_id"],
                "prompt": problem["text"][:200],
                "generated_code": generations[0]["code"][:300] if generations else "",
                "passed": problem_passed,
                "error": generations[0]["result"]["error"] if generations else None,
            })

    results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("MBPP Baseline Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Num samples per problem: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print()

    # Load MBPP dataset
    print("Loading MBPP dataset...")
    dataset = load_dataset("mbpp", split="test")
    print(f"MBPP test set: {len(dataset)} problems")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Evaluate
    results = evaluate_mbpp(
        model,
        tokenizer,
        dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        timeout_sec=args.timeout,
        verbose=args.verbose,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Pass@{args.num_samples}: {results['pass_rate']:.2%} ({results['passed']}/{results['total']})")
    print()

    if args.verbose and results["samples"]:
        print("Sample failures:")
        for sample in results["samples"][:5]:
            if not sample["passed"]:
                print(f"\n  Task {sample['task_id']}:")
                print(f"    Problem: {sample['prompt'][:80]}...")
                print(f"    Error: {sample['error']}")
                print(f"    Code: {sample['generated_code'][:100]}...")

    # Save results
    if args.output_file:
        save_data = {
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_samples": args.max_samples,
                "num_samples": args.num_samples,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
            },
            "results": {
                "total": results["total"],
                "passed": results["passed"],
                "pass_rate": results["pass_rate"],
            },
            "errors": results["errors"][:20],  # Save first 20 errors
        }

        with open(args.output_file, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

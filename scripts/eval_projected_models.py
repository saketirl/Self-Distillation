#!/usr/bin/env python3
"""
Evaluate all projected gradient models on tooluse dataset.

Finds all models in outputs/projected_proj_* folders and evaluates them.

Usage:
    python scripts/eval_projected_models.py
    python scripts/eval_projected_models.py --output_file results.txt
    python scripts/eval_projected_models.py --max_samples 50
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loaders import load_task_dataset


def evaluate_tooluse(model, tokenizer, eval_dataset, device, max_samples=50):
    """Evaluate model on tooluse task."""
    model.eval()
    correct = 0
    total = 0
    results = []

    eval_subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    with torch.no_grad():
        for example in tqdm(eval_subset, desc="Evaluating", leave=False):
            prompt = example["prompt"]
            expected = example["response"]

            # Format as chat
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Check if correct tool is mentioned
            tool_match = re.search(r'use the (\w+)', expected, re.IGNORECASE)
            is_correct = False
            if tool_match:
                func_name = tool_match.group(1).lower()
                is_correct = func_name in generated.lower()

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "expected": expected,
                "generated": generated[:200],  # Truncate for readability
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def find_projected_models(base_dir: str = "outputs") -> list[Path]:
    """Find all model directories matching outputs/projected_proj_*"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # Find all directories starting with "projected_proj_"
    model_dirs = sorted(base_path.glob("projected_proj_*"))

    # Filter to only include directories that have model files
    valid_dirs = []
    for d in model_dirs:
        if d.is_dir():
            # Check for model files (config.json or model.safetensors)
            if (d / "config.json").exists() or (d / "model.safetensors").exists():
                valid_dirs.append(d)

    return valid_dirs


def extract_config_from_name(model_name: str) -> dict:
    """Extract LR and resync from model name like 'projected_proj_lr1e-5_resync100'"""
    config = {"lr": None, "resync": None}

    lr_match = re.search(r'lr([\d.e-]+)', model_name)
    if lr_match:
        config["lr"] = lr_match.group(1)

    resync_match = re.search(r'resync(\d+)', model_name)
    if resync_match:
        config["resync"] = int(resync_match.group(1))

    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate projected models on tooluse")
    parser.add_argument("--base_dir", type=str, default="outputs",
                        help="Base directory containing model folders")
    parser.add_argument("--output_file", type=str, default="projected_eval_results.txt",
                        help="Output file for results")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Max samples to evaluate per model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Find all models
    print(f"Searching for models in {args.base_dir}/projected_proj_*...")
    model_dirs = find_projected_models(args.base_dir)

    if not model_dirs:
        print("No models found!")
        return

    print(f"Found {len(model_dirs)} models:")
    for d in model_dirs:
        print(f"  - {d.name}")

    # Load evaluation dataset
    print(f"\nLoading tooluse evaluation dataset...")
    _, eval_dataset = load_task_dataset("tooluse", seed=args.seed)
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Results storage
    all_results = []

    # Open output file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(args.output_file, "w") as f:
        f.write(f"Projected Gradient Models Evaluation - Tooluse\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Max samples: {args.max_samples}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Models evaluated: {len(model_dirs)}\n")
        f.write(f"{'=' * 60}\n\n")

        # Evaluate each model
        for model_dir in model_dirs:
            model_name = model_dir.name
            config = extract_config_from_name(model_name)

            print(f"\n{'=' * 60}")
            print(f"Evaluating: {model_name}")
            print(f"  LR: {config['lr']}, Resync: {config['resync']}")

            f.write(f"Model: {model_name}\n")
            f.write(f"  LR: {config['lr']}, Resync: {config['resync']}\n")

            try:
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Evaluate
                results = evaluate_tooluse(
                    model, tokenizer, eval_dataset,
                    device=args.device,
                    max_samples=args.max_samples,
                )

                accuracy = results["accuracy"]
                correct = results["correct"]
                total = results["total"]

                print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
                f.write(f"  Accuracy: {accuracy:.2%} ({correct}/{total})\n")

                all_results.append({
                    "model": model_name,
                    "lr": config["lr"],
                    "resync": config["resync"],
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                })

                # Clean up
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR: {e}")
                f.write(f"  ERROR: {e}\n")
                all_results.append({
                    "model": model_name,
                    "lr": config["lr"],
                    "resync": config["resync"],
                    "accuracy": None,
                    "error": str(e),
                })

            f.write("\n")

        # Summary
        f.write(f"\n{'=' * 60}\n")
        f.write("SUMMARY\n")
        f.write(f"{'=' * 60}\n\n")

        # Sort by accuracy
        valid_results = [r for r in all_results if r.get("accuracy") is not None]
        valid_results.sort(key=lambda x: x["accuracy"], reverse=True)

        f.write(f"{'Model':<40} {'LR':<10} {'Resync':<10} {'Accuracy':<10}\n")
        f.write(f"{'-' * 70}\n")

        for r in valid_results:
            f.write(f"{r['model']:<40} {r['lr']:<10} {str(r['resync']):<10} {r['accuracy']:.2%}\n")

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"\n{'Model':<40} {'LR':<10} {'Resync':<10} {'Accuracy':<10}")
        print(f"{'-' * 70}")
        for r in valid_results:
            print(f"{r['model']:<40} {r['lr']:<10} {str(r['resync']):<10} {r['accuracy']:.2%}")

        # Best model
        if valid_results:
            best = valid_results[0]
            f.write(f"\nBest model: {best['model']} with {best['accuracy']:.2%} accuracy\n")
            print(f"\nBest model: {best['model']} with {best['accuracy']:.2%} accuracy")

    # Also save as JSON
    json_file = args.output_file.replace(".txt", ".json")
    with open(json_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "max_samples": args.max_samples,
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {args.output_file}")
    print(f"JSON saved to: {json_file}")


if __name__ == "__main__":
    main()

"""
Sequential training script for continual learning experiments.

Trains on 3 tasks in sequence:
1. Tool Selection (tooluse)
2. JSON Extraction
3. Text-to-SQL (Spider)

Supports both SFT and SDFT training methods.

Usage:
    # SDFT (default)
    python train_sequential.py --method sdft --output_dir outputs/sdft_run

    # SFT baseline
    python train_sequential.py --method sft --output_dir outputs/sft_run
"""

import argparse
import gc
import json
import os
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from data_loaders import load_task_dataset, load_all_eval_datasets, TASK_ORDER


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Continual Learning Training")

    # Method selection
    parser.add_argument("--method", type=str, choices=["sdft", "sft"], default="sdft",
                        help="Training method: sdft or sft")

    # Experiment tracking
    parser.add_argument("--exp_id", type=str, default=None,
                        help="Experiment ID (shared across tasks). Auto-generated if not provided.")

    # Model settings
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model name or path")

    # Training settings
    parser.add_argument("--output_dir", type=str, default="outputs/sequential_run",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs per task")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_samples_per_task", type=int, default=None,
                        help="Max samples per task (for debugging)")

    # SDFT specific settings
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01,
                        help="EMA alpha for reference model (SDFT only)")
    parser.add_argument("--sync_ref_model", action="store_true", default=True,
                        help="Whether to sync reference model (SDFT only)")

    # vLLM settings
    parser.add_argument("--use_vllm", action="store_true", default=True,
                        help="Use vLLM for generation (SDFT only)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                        help="vLLM GPU memory utilization")

    # Other settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tasks", type=str, nargs="+", default=TASK_ORDER,
                        help="Tasks to train on (in order)")
    parser.add_argument("--resume_from_task", type=int, default=0,
                        help="Resume from task index (0-indexed)")

    return parser.parse_args()


def evaluate_all_tasks(model, tokenizer, eval_datasets, device, max_samples=100):
    """Evaluate model on all tasks and return metrics."""
    model.eval()
    results = {}

    print(f"\n[Eval] Starting evaluation on device: {device}")
    print(f"[Eval] Tasks to evaluate: {list(eval_datasets.keys())}")

    for task_name, eval_ds in eval_datasets.items():
        correct = 0
        total = 0

        # Sample subset for evaluation
        eval_subset = eval_ds.select(range(min(max_samples, len(eval_ds))))
        print(f"[Eval] {task_name}: evaluating {len(eval_subset)} samples...")

        for idx, example in enumerate(eval_subset):
            if idx > 0 and idx % 20 == 0:
                print(f"[Eval] {task_name}: {idx}/{len(eval_subset)} samples processed...")

            prompt = example["prompt"]
            expected = example["response"]

            # Format as chat
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Task-specific evaluation
            is_correct = False

            if task_name == "spider":
                # Normalize SQL for comparison
                gen_normalized = generated.strip().lower().replace(";", "")
                exp_normalized = expected.strip().lower().replace(";", "")
                is_correct = gen_normalized == exp_normalized

            elif task_name == "json_extraction":
                # Check if valid JSON and matches
                try:
                    gen_json = json.loads(generated.strip())
                    exp_json = json.loads(expected.strip())
                    is_correct = gen_json == exp_json
                except:
                    pass

            elif task_name == "tooluse":
                # Strict matching - require exact function name
                import re
                gen_lower = generated.lower()

                # Extract function name from expected (e.g., "use the getRandomAxolotlImage tool")
                tool_match = re.search(r'use the (\w+)', expected, re.IGNORECASE)

                if tool_match:
                    func_name = tool_match.group(1).lower()
                    # Require exact function name match
                    is_correct = func_name in gen_lower
                else:
                    is_correct = False

            elif task_name == "gsm8k":
                # Extract numerical answer from generated text
                import re
                expected_num = expected.strip()

                # Try to find a number in the generated text
                # Look for patterns like "72", "the answer is 72", "= 72"
                numbers = re.findall(r'[-+]?\d*\.?\d+', generated)

                if numbers:
                    # Check if expected answer is among the numbers found
                    # Often the last number is the final answer
                    is_correct = expected_num in numbers or (numbers and numbers[-1] == expected_num)
                else:
                    is_correct = False

            if is_correct:
                correct += 1

            # Debug: print first 2 samples per task
            if idx < 2:
                print(f"[Eval Debug] {task_name} sample {idx}:")
                print(f"  Expected: {expected[:100]}...")
                print(f"  Generated: {generated[:100]}...")
                print(f"  Correct: {is_correct}")

            total += 1

        accuracy = correct / total if total > 0 else 0
        results[task_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        print(f"[Eval] {task_name}: {accuracy:.2%} ({correct}/{total})")

    print(f"[Eval] Evaluation complete.")
    model.train()
    return results


def train_sdft_on_task(model, ref_model, tokenizer, train_dataset, task_name, args, task_output_dir):
    """Train using SDFT (Self-Distillation Fine-Tuning)."""
    print(f"\n{'='*60}")
    print(f"SDFT Training on: {task_name}")
    print(f"{'='*60}")

    # Calculate save_steps to save at midpoint and end (2 checkpoints per task)
    # SDFT has ~2x steps due to on-policy generation
    estimated_steps = (len(train_dataset) // args.gradient_accumulation_steps) * 2
    save_steps = max(estimated_steps // 2, 1)
    print(f"[SDFT] Estimated steps: {estimated_steps}, saving at step {save_steps} (midpoint) and {estimated_steps} (end)")

    config = DistilConfig(
        seed=args.seed,
        output_dir=task_output_dir,

        # vLLM settings
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=False,  # Disabled for sequential training (can't reinitialize)
        vllm_importance_sampling_correction=True,

        # Training settings
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=1024,
        max_completion_length=512,
        num_train_epochs=args.num_train_epochs,
        save_steps=save_steps,
        save_total_limit=2,  # Keep midpoint and final checkpoints
        save_only_model=True,  # Skip optimizer state to reduce checkpoint size
        max_grad_norm=1,

        # SDFT specific
        sync_ref_model=args.sync_ref_model,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,

        # Logging
        report_to="wandb",
        run_name=f"exp{args.exp_id}_sdft_task{TASK_ORDER.index(task_name)+1}_{task_name}",
        log_completions=False,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save the model
    trainer.save_model(task_output_dir)

    # Cleanup trainer and vLLM to free GPU memory before next task
    if hasattr(trainer, 'llm') and trainer.llm is not None:
        # Try to properly shutdown vLLM engine
        try:
            if hasattr(trainer.llm, 'llm_engine'):
                if hasattr(trainer.llm.llm_engine, 'shutdown'):
                    trainer.llm.llm_engine.shutdown()
                del trainer.llm.llm_engine
            del trainer.llm
        except Exception as e:
            print(f"[SDFT] Warning during vLLM cleanup: {e}")
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[SDFT] Cleaned up trainer and freed GPU memory")

    return model, ref_model


def train_sft_on_task(model, tokenizer, train_dataset, task_name, args, task_output_dir):
    """Train using standard SFT (Supervised Fine-Tuning)."""
    print(f"\n{'='*60}")
    print(f"SFT Training on: {task_name}")
    print(f"{'='*60}")

    # Format dataset for SFT with truncation to 1024 tokens
    def format_for_sft(example):
        prompt = example["prompt"]
        response = example["response"]

        # Format as chat
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        messages.append({"role": "assistant", "content": response})

        text = tokenizer.apply_chat_template(messages, tokenize=False)

        # Truncate to 1024 tokens
        tokens = tokenizer.encode(text, truncation=True, max_length=1024)
        text = tokenizer.decode(tokens, skip_special_tokens=False)

        return {"text": text}

    sft_dataset = train_dataset.map(format_for_sft, remove_columns=train_dataset.column_names)

    # Calculate save_steps to save at midpoint and end (2 checkpoints per task)
    steps_per_epoch = len(sft_dataset) // args.gradient_accumulation_steps
    estimated_steps = steps_per_epoch * args.num_train_epochs
    save_steps = max(estimated_steps // 2, 1)
    print(f"[SFT] Estimated steps: {estimated_steps}, saving at step {save_steps} (midpoint) and {estimated_steps} (end)")

    # Note: max_seq_length may vary by TRL version. Using truncation in tokenizer instead.
    config = SFTConfig(
        output_dir=task_output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=save_steps,
        save_total_limit=2,  # Keep midpoint and final checkpoints
        save_only_model=True,  # Skip optimizer state to reduce checkpoint size
        max_grad_norm=1,
        report_to="wandb",
        run_name=f"exp{args.exp_id}_sft_task{TASK_ORDER.index(task_name)+1}_{task_name}",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save the model
    trainer.save_model(task_output_dir)

    # Cleanup trainer to free GPU memory before next task
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[SFT] Cleaned up trainer and freed GPU memory")

    return model


def main():
    args = parse_args()

    # Generate experiment ID if not provided
    if args.exp_id is None:
        args.exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'#'*60}")
    print(f"# Sequential Continual Learning Experiment")
    print(f"# Experiment ID: {args.exp_id}")
    print(f"# Method: {args.method.upper()}")
    print(f"# Model: {args.model_name}")
    print(f"# Tasks: {args.tasks}")
    print(f"{'#'*60}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on device: {next(model.parameters()).device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For SDFT, also load reference model
    ref_model = None
    if args.method == "sdft":
        print("Loading reference model for SDFT...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Reference model loaded on device: {next(ref_model.parameters()).device}")

    # Load all evaluation datasets
    print("Loading evaluation datasets...")
    eval_datasets = load_all_eval_datasets(seed=args.seed)

    # Results tracking
    all_results = {
        "method": args.method,
        "model": args.model_name,
        "tasks": args.tasks,
        "results_per_stage": []
    }

    # Evaluate before training (baseline)
    print("\nEvaluating baseline (before training)...")
    device = next(model.parameters()).device
    baseline_results = evaluate_all_tasks(model, tokenizer, eval_datasets, device)
    all_results["baseline"] = baseline_results
    print(f"Baseline results: {json.dumps(baseline_results, indent=2)}")

    # Sequential training loop
    for task_idx, task_name in enumerate(args.tasks):
        if task_idx < args.resume_from_task:
            print(f"Skipping task {task_idx}: {task_name} (resuming from {args.resume_from_task})")
            continue

        print(f"\n{'='*60}")
        print(f"TASK {task_idx + 1}/{len(args.tasks)}: {task_name}")
        print(f"{'='*60}")

        # Load task dataset
        train_dataset, _ = load_task_dataset(
            task_name,
            seed=args.seed,
            max_samples=args.max_samples_per_task
        )
        print(f"Loaded {len(train_dataset)} training examples")

        # Task output directory
        task_output_dir = str(output_dir / f"task_{task_idx}_{task_name}")

        # Train
        if args.method == "sdft":
            model, ref_model = train_sdft_on_task(
                model, ref_model, tokenizer, train_dataset, task_name, args, task_output_dir
            )
        else:
            model = train_sft_on_task(
                model, tokenizer, train_dataset, task_name, args, task_output_dir
            )

        # Evaluate on ALL tasks after this training stage
        print(f"\nEvaluating after task {task_idx + 1} ({task_name})...")
        stage_results = evaluate_all_tasks(model, tokenizer, eval_datasets, device)
        all_results["results_per_stage"].append({
            "task_trained": task_name,
            "task_idx": task_idx,
            "results": stage_results
        })
        print(f"Results after {task_name}: {json.dumps(stage_results, indent=2)}")

        # Calculate forgetting
        if task_idx > 0:
            print("\nForgetting analysis:")
            for prev_task in args.tasks[:task_idx]:
                if prev_task in baseline_results and prev_task in stage_results:
                    baseline_acc = baseline_results[prev_task]["accuracy"]
                    current_acc = stage_results[prev_task]["accuracy"]
                    # Compare to performance right after training on that task
                    prev_stage = all_results["results_per_stage"][args.tasks.index(prev_task)]
                    after_training_acc = prev_stage["results"][prev_task]["accuracy"]
                    forgetting = after_training_acc - current_acc
                    print(f"  {prev_task}: {after_training_acc:.2%} -> {current_acc:.2%} (forgetting: {forgetting:.2%})")

    # Save final results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Method: {args.method.upper()}")
    print(f"Final results saved to: {output_dir}")

    # Print forgetting summary
    if len(args.tasks) > 1:
        print("\nForgetting Summary:")
        final_results = all_results["results_per_stage"][-1]["results"]
        for task_idx, task_name in enumerate(args.tasks[:-1]):
            after_training = all_results["results_per_stage"][task_idx]["results"][task_name]["accuracy"]
            final_acc = final_results[task_name]["accuracy"]
            forgetting = after_training - final_acc
            print(f"  {task_name}: {after_training:.2%} -> {final_acc:.2%} (forgetting: {forgetting:.2%})")


if __name__ == "__main__":
    main()

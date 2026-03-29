"""
Train on a single task. Designed to be called as separate process per task.
This avoids GPU memory accumulation issues with vLLM.

Usage:
    python train_single_task.py --method sdft --task tooluse --model_name Qwen/Qwen2.5-3B-Instruct
    python train_single_task.py --method sdft --task gsm8k --model_name outputs/sdft/task_0_tooluse
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
    parser = argparse.ArgumentParser(description="Train on a single task")

    parser.add_argument("--method", type=str, choices=["sdft", "sft"], required=True)
    parser.add_argument("--task", type=str, required=True, help="Task to train on")
    parser.add_argument("--model_name", type=str, required=True, help="Model or checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_id", type=str, default=None)

    # Training settings
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)

    # SDFT settings
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def evaluate_all_tasks(model, tokenizer, eval_datasets, device, max_samples=10):
    """Evaluate model on all tasks."""
    model.eval()
    results = {}

    for task_name, eval_ds in eval_datasets.items():
        correct = 0
        total = 0
        eval_subset = eval_ds.select(range(min(max_samples, len(eval_ds))))

        for example in eval_subset:
            prompt = example["prompt"]
            expected = example["response"]

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
                gen_normalized = generated.strip().lower().replace(";", "")
                exp_normalized = expected.strip().lower().replace(";", "")
                is_correct = gen_normalized == exp_normalized
            elif task_name == "tooluse":
                import re
                tool_match = re.search(r'use the (\w+)', expected, re.IGNORECASE)
                if tool_match:
                    func_name = tool_match.group(1).lower()
                    is_correct = func_name in generated.lower()
            elif task_name == "gsm8k":
                import re
                expected_num = expected.strip()
                numbers = re.findall(r'[-+]?\d*\.?\d+', generated)
                if numbers:
                    is_correct = expected_num in numbers or (numbers and numbers[-1] == expected_num)

            if is_correct:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        results[task_name] = {"accuracy": accuracy, "correct": correct, "total": total}
        print(f"  {task_name}: {accuracy:.2%} ({correct}/{total})")

    model.train()
    return results


def train_sdft(model, ref_model, tokenizer, train_dataset, task_name, args):
    """Train using SDFT."""
    # Calculate checkpoints: midpoint and end
    estimated_steps = (len(train_dataset) // args.gradient_accumulation_steps) * 2
    save_steps = max(estimated_steps // 2, 1)
    print(f"[SDFT] Estimated steps: {estimated_steps}, saving at {save_steps} and end")

    config = DistilConfig(
        seed=args.seed,
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=False,
        vllm_importance_sampling_correction=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=1024,
        max_completion_length=512,
        num_train_epochs=args.num_train_epochs,
        save_steps=save_steps,
        save_total_limit=2,
        save_only_model=True,
        max_grad_norm=1,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        report_to="wandb",
        run_name=f"exp{args.exp_id}_sdft_{task_name}",
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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def train_sft(model, tokenizer, train_dataset, task_name, args):
    """Train using SFT."""
    # Format dataset
    def format_for_sft(example):
        prompt = example["prompt"]
        response = example["response"]
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        messages.append({"role": "assistant", "content": response})
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode(text, truncation=True, max_length=1024)
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        return {"text": text}

    sft_dataset = train_dataset.map(format_for_sft, remove_columns=train_dataset.column_names)

    # Calculate checkpoints
    estimated_steps = (len(sft_dataset) // args.gradient_accumulation_steps) * args.num_train_epochs
    save_steps = max(estimated_steps // 2, 1)
    print(f"[SFT] Estimated steps: {estimated_steps}, saving at {save_steps} and end")

    config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=save_steps,
        save_total_limit=2,
        save_only_model=True,
        max_grad_norm=1,
        report_to="wandb",
        run_name=f"exp{args.exp_id}_sft_{task_name}",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def main():
    args = parse_args()

    if args.exp_id is None:
        args.exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*60}")
    print(f"Training: {args.method.upper()} on {args.task}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load ref model for SDFT
    ref_model = None
    if args.method == "sdft":
        print("Loading reference model...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Load eval datasets
    print("Loading evaluation datasets...")
    eval_datasets = load_all_eval_datasets(seed=args.seed)

    # Evaluate before training
    device = next(model.parameters()).device
    print("\n[Eval] Before training:")
    before_results = evaluate_all_tasks(model, tokenizer, eval_datasets, device)

    # Load task dataset
    train_dataset, _ = load_task_dataset(args.task, seed=args.seed, max_samples=args.max_samples)
    print(f"\nTraining on {len(train_dataset)} examples")

    # Train
    if args.method == "sdft":
        train_sdft(model, ref_model, tokenizer, train_dataset, args.task, args)
    else:
        train_sft(model, tokenizer, train_dataset, args.task, args)

    # Reload model for evaluation (training might have moved things around)
    del model
    if ref_model is not None:
        del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.output_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Evaluate after training
    print("\n[Eval] After training:")
    after_results = evaluate_all_tasks(model, tokenizer, eval_datasets, device)

    # Save results
    results = {
        "task": args.task,
        "method": args.method,
        "model": args.model_name,
        "before": before_results,
        "after": after_results,
    }
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()

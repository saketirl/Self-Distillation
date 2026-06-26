"""Example: Analyze eigenspectrum of individual weight matrices in Qwen.

Uses real data from task eval sets (500 samples by default).
Default tasks: tooluse, gsm8k, mbpp  (Qwen2.5-3B experiments)
Qwen3-4B tasks: tooluse, spider, cot_math  — pass --data_tasks tooluse,spider,cot_math

Usage:
    python -m eigenspectrum.example_analyze \
        --model_path Qwen/Qwen3-4B \
        --layer 0 \
        --data_tasks tooluse,spider,cot_math

    # Explicit sample count
    python -m eigenspectrum.example_analyze \
        --model_path Qwen/Qwen3-4B \
        --layer 0 \
        --data_tasks tooluse,spider,cot_math \
        --num_data_samples 500

    # Use random tokens instead (legacy)
    python -m eigenspectrum.example_analyze \
        --model_path Qwen/Qwen2.5-3B-Instruct \
        --layer 0 \
        --use_random_data
"""

import argparse
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .spectrum_analyzer import SpectrumAnalyzer, quick_analyze


class DummyDataset(Dataset):
    """Simple dataset with random tokens for Hessian computation."""

    def __init__(self, tokenizer, num_samples=100, seq_len=128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len

        # Generate random token sequences
        vocab_size = tokenizer.vocab_size
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        return {"input_ids": input_ids, "labels": input_ids}


_LOADER_MAP = None

def _get_loader_map():
    global _LOADER_MAP
    if _LOADER_MAP is None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data_loaders import (
            load_tooluse_dataset,
            load_gsm8k_dataset,
            load_mbpp_dataset,
            load_spider_dataset,
            load_cot_math_dataset,
        )
        _LOADER_MAP = {
            "tooluse":  load_tooluse_dataset,
            "gsm8k":    load_gsm8k_dataset,
            "mbpp":     load_mbpp_dataset,
            "spider":   load_spider_dataset,
            "cot_math": load_cot_math_dataset,
        }
    return _LOADER_MAP


class CombinedTaskDataset(Dataset):
    """Dataset combining eval samples from one or more tasks for Hessian computation.

    Args:
        tasks: List of task names to load (e.g. ["tooluse", "spider", "cot_math"]).
               Defaults to ["tooluse", "gsm8k", "mbpp"] for backwards compatibility.
    """

    def __init__(self, tokenizer, num_samples=500, max_seq_len=512, seed=42,
                 tasks=None):
        if tasks is None:
            tasks = ["tooluse", "gsm8k", "mbpp"]

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        loader_map = _get_loader_map()
        for task in tasks:
            if task not in loader_map:
                raise ValueError(f"Unknown task '{task}'. Available: {list(loader_map)}")

        # Load eval split for each task
        per_task_eval = {}
        for task in tasks:
            _, eval_ds = loader_map[task](seed=seed)
            per_task_eval[task] = eval_ds

        # Distribute num_samples evenly across tasks, capped by available data
        quota = num_samples // len(tasks)
        counts = {t: min(quota, len(per_task_eval[t])) for t in tasks}
        # Give leftover samples to the first task
        leftover = num_samples - sum(counts.values())
        first = tasks[0]
        counts[first] = min(counts[first] + leftover, len(per_task_eval[first]))

        summary = " + ".join(f"{counts[t]} {t}" for t in tasks)
        print(f"Loading {summary} = {sum(counts.values())} total samples")

        all_examples = []
        for task in tasks:
            for i in range(counts[task]):
                all_examples.append((task, per_task_eval[task][i]))

        import random
        random.seed(seed)
        random.shuffle(all_examples)

        for task_name, example in all_examples:
            prompt = example["prompt"]
            response = example["response"]
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            messages = messages + [{"role": "assistant", "content": response}]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer.encode(text, truncation=True, max_length=max_seq_len)
            if len(tokens) > 0:
                self.samples.append(torch.tensor(tokens))

        print(f"Tokenized {len(self.samples)} samples for Hessian computation")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = self.samples[idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def collate_fn(batch):
    """Collate with padding."""
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]

    # Pad to max length in batch
    max_len = max(len(x) for x in input_ids)

    padded_input_ids = []
    padded_labels = []

    for inp, lab in zip(input_ids, labels):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            # Pad with 0 (will be masked)
            inp = torch.cat([inp, torch.zeros(pad_len, dtype=inp.dtype)])
            lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)])
        padded_input_ids.append(inp)
        padded_labels.append(lab)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--layer", type=int, default=0, help="Layer to analyze")
    parser.add_argument(
        "--param_pattern", type=str, default=None,
        help="Pattern to match param name (e.g., 'q_proj.weight')"
    )
    parser.add_argument("--lanczos_order", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=5, help="Number of Lanczos samples")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="spectrum_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--list_params", action="store_true", help="Just list params and exit")
    # Data options (real data is default)
    parser.add_argument("--use_random_data", action="store_true",
                        help="Use random tokens instead of real task data")
    parser.add_argument("--data_tasks", type=str, default="tooluse,gsm8k,mbpp",
                        help="Comma-separated task names for real data (e.g. tooluse,spider,cot_math)")
    parser.add_argument("--num_data_samples", type=int, default=500,
                        help="Number of real data samples to use (default: 500)")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Max sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Use bfloat16 instead of float32 (saves memory)")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing with reverse-mode HVP (saves memory, ~2x slower)")
    args = parser.parse_args()

    torch.cuda.empty_cache()

    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32
    print(f"Using dtype: {dtype}")

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use eager attention (not Flash/SDPA) because forward-mode autodiff for HVP
    # is not supported with Flash Attention
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()

    # Create dataloader (real data is default)
    if args.use_random_data:
        print("\nUsing random token data (default is real task data)")
        dataset = DummyDataset(tokenizer, num_samples=args.batch_size * args.max_batches)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    else:
        task_list = [t.strip() for t in args.data_tasks.split(",") if t.strip()]
        print(f"\nUsing real data from {task_list} test sets...")
        dataset = CombinedTaskDataset(
            tokenizer,
            num_samples=args.num_data_samples,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            tasks=task_list,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        # Adjust max_batches to use all data
        args.max_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
        print(f"Using {len(dataset)} samples in {args.max_batches} batches")

    # Initialize analyzer
    analyzer = SpectrumAnalyzer(
        model,
        dataloader,
        device=torch.device(args.device),
        dtype=dtype,
        max_batches=args.max_batches,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    if args.list_params:
        print(f"\nParameters in layer {args.layer}:")
        params = analyzer.list_params(layer_idx=args.layer)
        for name, shape, num in params:
            print(f"  {name}: {shape} ({num:,} params)")
        return

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Get params to analyze
    blocks = analyzer.get_layer_blocks(args.layer)
    print(f"\nLayer {args.layer} structure:")
    for block_type, param_names in blocks.items():
        print(f"  {block_type}: {len(param_names)} params")
        for p in param_names:
            print(f"    - {p}")

    # Filter by pattern if provided
    if args.param_pattern:
        all_params = sum(blocks.values(), [])
        params_to_analyze = [p for p in all_params if args.param_pattern in p]
    else:
        # Default: analyze largest weight matrices (skip biases/norms)
        params_to_analyze = []
        for p in sum(blocks.values(), []):
            param = dict(model.named_parameters())[p]
            if param.numel() > 100000:  # Skip small params
                params_to_analyze.append(p)

    print(f"\nAnalyzing {len(params_to_analyze)} parameters:")
    for p in params_to_analyze:
        print(f"  - {p}")

    results = {}
    for param_name in params_to_analyze:
        print(f"\n{'='*60}")
        result = analyzer.analyze_param(
            param_name,
            lanczos_order=args.lanczos_order,
            num_samples=args.num_samples,
            verbose=True,
        )
        results[param_name] = result

        # Plot density
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(result.grids, result.density)
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_title(f"{param_name}\nTop λ={result.top_eigenvalue:.2e}, Eff. Rank={result.effective_rank:.1f}")
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

        safe_name = param_name.replace(".", "_").replace("/", "_")
        png_path = f"{args.output_dir}/{safe_name}_density.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved PNG: {png_path}")

        # Save eigenspectrum data
        npz_path = f"{args.output_dir}/{safe_name}_spectrum.npz"
        np.savez(
            npz_path,
            param_name=param_name,
            param_shape=np.array(result.param_shape),
            num_params=result.num_params,
            eigenvalues=result.eigenvalues,
            weights=result.weights,
            density=result.density,
            grids=result.grids,
            effective_rank=result.effective_rank,
            top_eigenvalue=result.top_eigenvalue,
            trace_estimate=result.trace_estimate,
        )
        print(f"  Saved NPZ: {npz_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Parameter':<50} {'Top λ':>12} {'Eff. Rank':>12}")
    print("-"*76)
    for name, r in results.items():
        short_name = name.split(".")[-2] + "." + name.split(".")[-1]
        print(f"{short_name:<50} {r.top_eigenvalue:>12.2e} {r.effective_rank:>12.1f}")

    # Save summary JSON
    import json
    summary = {
        name: {
            "param_shape": list(r.param_shape),
            "num_params": r.num_params,
            "effective_rank": float(r.effective_rank),
            "top_eigenvalue": float(r.top_eigenvalue),
            "trace_estimate": float(r.trace_estimate),
        }
        for name, r in results.items()
    }
    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()

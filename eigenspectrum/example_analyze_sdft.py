"""Analyze eigenspectrum of SDFT-trained models using KL divergence loss.

Uses real data from tooluse/gsm8k/mbpp test sets and computes Hessian
with respect to KL divergence (matching SDFT training objective).

Usage:
    python -m eigenspectrum.example_analyze_sdft \
        --model_path /data/saket/continual/Self-Distillation/outputs/sdft_full_mbpp/task_0_tooluse/checkpoint-1011 \
        --ref_model_path /data/saket/continual/Self-Distillation/models/qwen2.5-3b-instruct \
        --layer 0
"""

import argparse
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .spectrum_analyzer import SpectrumAnalyzer


class CombinedTaskDataset(Dataset):
    """Dataset combining tooluse, gsm8k, and mbpp test samples for Hessian computation."""

    def __init__(self, tokenizer, num_samples=500, max_seq_len=512, seed=42):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        # Import data loaders from parent directory
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data_loaders import load_tooluse_dataset, load_gsm8k_dataset, load_mbpp_dataset

        # Load eval/test datasets
        _, tooluse_eval = load_tooluse_dataset(seed=seed)
        _, gsm8k_eval = load_gsm8k_dataset(seed=seed)
        _, mbpp_eval = load_mbpp_dataset(seed=seed)

        # Determine samples per task
        tooluse_count = min(len(tooluse_eval), num_samples // 3)
        remaining = num_samples - tooluse_count
        gsm8k_count = remaining // 2
        mbpp_count = remaining - gsm8k_count

        tooluse_count = min(tooluse_count, len(tooluse_eval))
        gsm8k_count = min(gsm8k_count, len(gsm8k_eval))
        mbpp_count = min(mbpp_count, len(mbpp_eval))

        print(f"Loading {tooluse_count} tooluse + {gsm8k_count} gsm8k + {mbpp_count} mbpp = {tooluse_count + gsm8k_count + mbpp_count} total samples")

        # Collect samples from each task
        all_examples = []
        for i in range(tooluse_count):
            all_examples.append(("tooluse", tooluse_eval[i]))
        for i in range(gsm8k_count):
            all_examples.append(("gsm8k", gsm8k_eval[i]))
        for i in range(mbpp_count):
            all_examples.append(("mbpp", mbpp_eval[i]))

        # Shuffle
        import random
        random.seed(seed)
        random.shuffle(all_examples)

        # Tokenize all samples
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

    max_len = max(len(x) for x in input_ids)

    padded_input_ids = []
    padded_labels = []

    for inp, lab in zip(input_ids, labels):
        pad_len = max_len - len(inp)
        if pad_len > 0:
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
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SDFT checkpoint to analyze")
    parser.add_argument("--ref_model_path", type=str, required=True,
                        help="Path to reference/teacher model")
    parser.add_argument("--layer", type=int, default=0, help="Layer to analyze")
    parser.add_argument("--param_pattern", type=str, default=None,
                        help="Pattern to match param name (e.g., 'q_proj.weight')")
    parser.add_argument("--lanczos_order", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=5, help="Number of Lanczos samples")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--list_params", action="store_true", help="Just list params and exit")
    parser.add_argument("--num_data_samples", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Use bfloat16 instead of float32 (saves memory, slightly less accurate)")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing with reverse-mode HVP (saves memory, ~2x slower)")
    args = parser.parse_args()

    # Set memory optimization
    torch.cuda.empty_cache()

    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float32
    print(f"Using dtype: {dtype}")

    print(f"Loading student model: {args.model_path}")
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

    # Check if ref model is same as model (e.g., base model analysis)
    # In that case, share weights to save memory
    if args.ref_model_path == args.model_path:
        print(f"Reference model same as student - sharing weights to save memory")
        ref_model = model
    else:
        print(f"Loading reference model: {args.ref_model_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_path,
            torch_dtype=dtype,
            device_map=args.device,
            attn_implementation="eager",
        )
        ref_model.eval()

    torch.cuda.empty_cache()

    # Create dataloader with real data
    print(f"\nUsing real data from tooluse/gsm8k/mbpp test sets...")
    dataset = CombinedTaskDataset(
        tokenizer,
        num_samples=args.num_data_samples,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    args.max_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    print(f"Using {len(dataset)} samples in {args.max_batches} batches")

    # Initialize analyzer with SDFT loss
    analyzer = SpectrumAnalyzer(
        model,
        dataloader,
        device=torch.device(args.device),
        dtype=dtype,
        max_batches=args.max_batches,
        ref_model=ref_model,
        loss_type="sdft",
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
        params_to_analyze = []
        for p in sum(blocks.values(), []):
            param = dict(model.named_parameters())[p]
            if param.numel() > 100000:
                params_to_analyze.append(p)

    print(f"\nAnalyzing {len(params_to_analyze)} parameters with SDFT loss:")
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
    print("SUMMARY (SDFT Loss)")
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

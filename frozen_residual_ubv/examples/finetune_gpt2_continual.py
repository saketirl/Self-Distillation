"""
Example: Continual fine-tuning of GPT-2 with rank-preserving UBV factorization.

This script demonstrates how to use the frozen_residual_ubv package for
continual learning without catastrophic forgetting. It fine-tunes GPT-2
on a sequence of simple tasks while preserving the effective rank of
weight matrices.

Usage:
    python finetune_gpt2_continual.py --task_id 0  # First task
    python finetune_gpt2_continual.py --task_id 1  # Second task (continual)

The key idea is that by factorizing W = U B V^T and optimizing on the
correct manifolds (Stiefel for U/V, S_{++} for B), we prevent spectral
collapse and maintain plasticity across task boundaries.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frozen_residual_ubv import (
    convert_model_to_factored,
    UBVOptimizer,
    FactoredLinear,
)


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration."""

    def __init__(self, texts, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []

        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def get_task_data(task_id):
    """Get training data for a specific task.

    Task 0: Simple arithmetic completions
    Task 1: Simple factual completions
    Task 2: Simple code completions
    """
    if task_id == 0:
        # Arithmetic task
        return [
            "2 + 2 = 4",
            "3 + 5 = 8",
            "10 - 4 = 6",
            "7 * 3 = 21",
            "15 / 3 = 5",
            "8 + 9 = 17",
            "12 - 7 = 5",
            "6 * 4 = 24",
            "20 / 5 = 4",
            "11 + 11 = 22",
        ] * 10  # Repeat for more training data

    elif task_id == 1:
        # Factual task
        return [
            "The capital of France is Paris.",
            "The capital of Japan is Tokyo.",
            "The capital of Germany is Berlin.",
            "Water freezes at 0 degrees Celsius.",
            "The sun is a star.",
            "Earth orbits the sun.",
            "Oxygen is essential for breathing.",
            "The moon orbits Earth.",
            "Light travels faster than sound.",
            "Plants need sunlight to grow.",
        ] * 10

    elif task_id == 2:
        # Simple code task
        return [
            "def add(a, b): return a + b",
            "def multiply(a, b): return a * b",
            "def square(x): return x * x",
            "def is_even(n): return n % 2 == 0",
            "def abs(x): return x if x >= 0 else -x",
            "def max(a, b): return a if a > b else b",
            "def min(a, b): return a if a < b else b",
            "def negate(x): return -x",
            "def double(x): return x * 2",
            "def half(x): return x / 2",
        ] * 10

    else:
        raise ValueError(f"Unknown task_id: {task_id}")


def compute_spectrum_stats(model):
    """Compute spectrum statistics for all FactoredLinear layers."""
    stats = []
    for name, module in model.named_modules():
        if isinstance(module, FactoredLinear):
            B = module.B.data.float()
            B_sym = (B + B.T) / 2
            eigenvalues = torch.linalg.eigvalsh(B_sym)

            stats.append({
                'name': name,
                'min_eigenvalue': eigenvalues.min().item(),
                'max_eigenvalue': eigenvalues.max().item(),
                'effective_rank': (eigenvalues.sum() / eigenvalues.max()).item(),
                'condition_number': (eigenvalues.max() / eigenvalues.min()).item(),
            })
    return stats


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Continual fine-tuning with UBV factorization")
    parser.add_argument('--task_id', type=int, default=0, help='Task ID (0, 1, or 2)')
    parser.add_argument('--r_star', type=int, default=16, help='Target rank for factorization')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr_U', type=float, default=1e-3, help='Learning rate for U (Stiefel)')
    parser.add_argument('--lr_B', type=float, default=5e-4, help='Learning rate for B (S++)')
    parser.add_argument('--lr_V', type=float, default=1e-3, help='Learning rate for V (Stiefel)')
    parser.add_argument('--lambda_min', type=float, default=1e-4, help='Spectrum floor')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Base model name')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load or create model
    checkpoint_path = Path(args.checkpoint_dir) / f"task_{args.task_id - 1}_model.pt"

    if args.task_id > 0 and checkpoint_path.exists():
        # Continual learning: load from previous task
        print(f"Loading model from {checkpoint_path}")
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        # First task or no checkpoint: load pretrained
        print(f"Loading pretrained model: {args.model_name}")
        model = GPT2LMHeadModel.from_pretrained(args.model_name)

    # Convert attention layers to factored form
    print(f"\nConverting model to factored form with r_star={args.r_star}...")
    # Only convert q_proj, k_proj, v_proj, out_proj (attention layers)
    converted_count = convert_model_to_factored(
        model,
        r_star=args.r_star,
        include_patterns=['c_attn', 'c_proj'],  # GPT-2 attention layers
        freeze_residual=True,
    )
    print(f"Converted {converted_count} layers to FactoredLinear")

    model = model.to(device)

    # Print initial spectrum statistics
    print("\nInitial spectrum statistics:")
    for stat in compute_spectrum_stats(model)[:3]:  # First 3 layers
        print(f"  {stat['name']}: effective_rank={stat['effective_rank']:.2f}, "
              f"condition={stat['condition_number']:.2f}")

    # Create optimizer
    optimizer = UBVOptimizer(
        model.parameters(),
        lr_U=args.lr_U,
        lr_B=args.lr_B,
        lr_V=args.lr_V,
        lambda_min=args.lambda_min,
    )

    # Prepare data
    train_texts = get_task_data(args.task_id)
    train_dataset = SimpleTextDataset(train_texts, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Also create eval dataloaders for all tasks (for measuring forgetting)
    eval_dataloaders = {}
    for tid in range(3):
        eval_texts = get_task_data(tid)[:10]  # First 10 samples for eval
        eval_dataset = SimpleTextDataset(eval_texts, tokenizer)
        eval_dataloaders[f"task_{tid}"] = DataLoader(eval_dataset, batch_size=args.batch_size)

    # Evaluate before training
    print(f"\n=== Task {args.task_id}: Before Training ===")
    for name, dataloader in eval_dataloaders.items():
        loss = evaluate(model, dataloader, device)
        print(f"  {name} loss: {loss:.4f}")

    # Training loop
    print(f"\n=== Task {args.task_id}: Training ===")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}")

        # Print spectrum statistics
        stats = compute_spectrum_stats(model)[:3]
        for stat in stats:
            print(f"  {stat['name']}: eff_rank={stat['effective_rank']:.2f}, "
                  f"min_eig={stat['min_eigenvalue']:.6f}")

    # Evaluate after training
    print(f"\n=== Task {args.task_id}: After Training ===")
    for name, dataloader in eval_dataloaders.items():
        loss = evaluate(model, dataloader, device)
        print(f"  {name} loss: {loss:.4f}")

    # Final spectrum statistics
    print("\nFinal spectrum statistics:")
    for stat in compute_spectrum_stats(model)[:3]:
        print(f"  {stat['name']}: effective_rank={stat['effective_rank']:.2f}, "
              f"condition={stat['condition_number']:.2f}, "
              f"min_eig={stat['min_eigenvalue']:.6f}")

    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"task_{args.task_id}_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

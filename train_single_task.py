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
import socket
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import TrainerCallback


def find_free_port() -> int:
    """Find a free port, using SLURM_JOB_ID if available to avoid conflicts."""
    import random

    # If running under SLURM, use job ID for deterministic port assignment
    slurm_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if slurm_job_id:
        # Use job ID to create a unique base port (range 30000-40000)
        base = 30000 + (int(slurm_job_id) % 10000)
    else:
        # Use high port range (49152-65535 are dynamic/private ports)
        base = random.randint(49152, 60000)

    for port in range(base, base + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue

    # Fallback: let OS assign a port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def setup_distributed_port():
    """Set up distributed environment variables with an available port."""
    # Destroy any existing process group to avoid conflicts
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Set MASTER_ADDR
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    # Find and set a free port
    port = find_free_port()
    os.environ["MASTER_PORT"] = str(port)

    # Set single-process distributed variables
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Disable NCCL P2P to avoid issues on some systems
    os.environ["NCCL_P2P_DISABLE"] = "1"

    # Use gloo backend for CPU-side coordination (more reliable than nccl for single GPU)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print(f"[Setup] Distributed env: MASTER_ADDR=127.0.0.1, MASTER_PORT={port}")


from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from data_loaders import load_task_dataset, load_all_eval_datasets, TASK_ORDER

# Add bw_riemannian_sgd to path
sys.path.insert(0, str(Path(__file__).parent / "bw_riemannian_sgd"))
from bw_riemannian_sgd import BWRiemannianSGD

# Import frozen_residual_ubv for UBV optimizer
from frozen_residual_ubv import (
    convert_model_to_factored,
    UBVOptimizer,
    create_rank_fn,
    get_factored_parameters,
    print_factored_summary,
    ProjectedGradientOptimizer,
    create_projected_optimizer,
)


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
    parser.add_argument("--frozen_teacher", action="store_true",
                        help="Keep teacher frozen (regular distillation, no EMA sync)")
    parser.add_argument("--no_vllm", action="store_true",
                        help="Disable vLLM, use HuggingFace generation instead (slower but no network issues)")

    # BWRiemannianSGD optimizer settings
    parser.add_argument("--use_bw_optimizer", action="store_true",
                        help="Use BWRiemannianSGD instead of AdamW")
    parser.add_argument("--bw_rank", type=str, default="auto",
                        help="Target rank for BWRiemannianSGD: int, 'auto' (maintain existing rank), or 'full'")
    parser.add_argument("--bw_energy_threshold", type=float, default=0.8,
                        help="Energy threshold for auto rank detection (0.0-1.0). Set to 0 to use tol-based rank.")
    parser.add_argument("--bw_momentum", type=float, default=0.0,
                        help="Momentum for BWRiemannianSGD")

    # UBV Riemannian optimizer settings (frozen residual factorization)
    parser.add_argument("--use_ubv_optimizer", action="store_true",
                        help="Use UBV factorization with Riemannian optimizer (rank-preserving)")
    parser.add_argument("--ubv_rank", type=str, default="auto",
                        help="Target rank for UBV: int, 'auto' (energy-based), or 'full'")
    parser.add_argument("--ubv_energy_threshold", type=float, default=0.8,
                        help="Energy threshold for auto rank detection (0.0-1.0). Set to 0 for tol-based.")
    parser.add_argument("--ubv_lr_U", type=float, default=None,
                        help="Learning rate for U (Stiefel). Default: same as --learning_rate")
    parser.add_argument("--ubv_lr_B", type=float, default=None,
                        help="Learning rate for B (S++). Default: learning_rate / 2")
    parser.add_argument("--ubv_lr_V", type=float, default=None,
                        help="Learning rate for V (Stiefel). Default: same as --learning_rate")
    parser.add_argument("--ubv_lambda_min", type=float, default=1e-4,
                        help="Spectrum floor for B (prevents spectral collapse)")
    parser.add_argument("--ubv_momentum", type=float, default=0.0,
                        help="Momentum for B updates")
    parser.add_argument("--ubv_include_patterns", type=str, nargs="*", default=None,
                        help="Regex patterns for layers to convert (default: attention layers)")
    parser.add_argument("--ubv_exclude_patterns", type=str, nargs="*", default=None,
                        help="Regex patterns for layers to exclude (default: embeddings, heads)")

    # Projected Gradient optimizer settings (no model surgery, UBV for gradient projection)
    parser.add_argument("--use_projected_optimizer", action="store_true",
                        help="Use ProjectedGradientOptimizer (no surgery, UBV for gradient projection)")
    parser.add_argument("--proj_energy_threshold", type=float, default=0.9,
                        help="Energy threshold for rank detection (0.0-1.0)")
    parser.add_argument("--proj_resync_every", type=int, default=100,
                        help="Re-sync UBV from W every N steps")
    parser.add_argument("--proj_no_resync", action="store_true",
                        help="Disable resync entirely (never re-factorize UBV)")
    parser.add_argument("--proj_admm_steps", type=int, default=10,
                        help="Number of ADMM iterations for manifold MUON")
    parser.add_argument("--proj_admm_rho", type=float, default=4.0,
                        help="ADMM penalty parameter for manifold MUON")
    parser.add_argument("--proj_grad_clip", type=float, default=100.0,
                        help="Gradient clipping for projected optimizer")

    # Muon optimizer settings
    parser.add_argument("--use_muon_optimizer", action="store_true",
                        help="Use Muon optimizer (Newton-Schulz orthogonalized momentum)")
    parser.add_argument("--muon_momentum", type=float, default=0.95,
                        help="Momentum for Muon optimizer")

    # Checkpoint settings
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--save_strategy", type=str, default="steps",
                        choices=["steps", "epoch", "no"],
                        help="When to save checkpoints (final model always saved)")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def evaluate_task_full(model, tokenizer, eval_dataset, task_name, device):
    """Evaluate model on full test set with detailed results.

    Works for any task by computing ROUGE-L based similarity.
    For tooluse, also checks if correct tool is mentioned.
    """
    import re
    from tqdm import tqdm

    model.eval()
    correct = 0
    total = 0
    detailed_results = []

    print(f"  Evaluating on {len(eval_dataset)} {task_name} samples...")

    for example in tqdm(eval_dataset, desc=f"{task_name} eval", leave=False):
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
        extra_info = {}

        if task_name == "tooluse":
            # Check if correct tool is mentioned
            tool_match = re.search(r'use the (\w+)', expected, re.IGNORECASE)
            if tool_match:
                func_name = tool_match.group(1).lower()
                is_correct = func_name in generated.lower()
                extra_info["tool_name"] = func_name
        elif task_name == "gsm8k":
            # Check if the final number matches
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            generated_nums = re.findall(r'-?\d+\.?\d*', generated)
            if expected_nums and generated_nums:
                is_correct = expected_nums[-1] == generated_nums[-1]
        else:
            # Generic: compute word overlap (simple ROUGE-like metric)
            expected_words = set(expected.lower().split())
            generated_words = set(generated.lower().split())
            if expected_words:
                overlap = len(expected_words & generated_words) / len(expected_words)
                is_correct = overlap > 0.3  # 30% word overlap threshold
                extra_info["word_overlap"] = overlap

        if is_correct:
            correct += 1
        total += 1

        detailed_results.append({
            "expected": expected[:500],
            "generated": generated[:500],  # Truncate for storage
            "correct": is_correct,
            **extra_info,
        })

    accuracy = correct / total if total > 0 else 0
    model.train()

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": detailed_results,
    }


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


def parse_bw_rank(rank_str):
    """Parse bw_rank argument: int, 'auto', or 'full' (None)."""
    if rank_str == "auto":
        return "auto"
    elif rank_str == "full" or rank_str == "none" or rank_str is None:
        return None
    else:
        try:
            return int(rank_str)
        except ValueError:
            raise ValueError(f"Invalid bw_rank: {rank_str}. Must be int, 'auto', or 'full'")


def create_bw_optimizer(model, args):
    """Create BWRiemannianSGD optimizer for 2D parameters, AdamW for others."""
    # Separate 2D (matrix) parameters from others
    matrix_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() == 2:
                matrix_params.append(param)
            else:
                other_params.append(param)

    rank = parse_bw_rank(args.bw_rank)

    # Energy threshold: 0 means use tol-based rank, otherwise use energy-based
    energy_threshold = args.bw_energy_threshold if args.bw_energy_threshold > 0 else None

    print(f"[BWRiemannianSGD] {len(matrix_params)} matrix params, {len(other_params)} other params")
    print(f"[BWRiemannianSGD] rank={rank}, energy_threshold={energy_threshold}")

    # Create optimizer with param groups
    # BWRiemannianSGD handles 2D params, falls back to vanilla SGD for others
    optimizer = BWRiemannianSGD(
        [{'params': matrix_params}, {'params': other_params}],
        lr=args.learning_rate,
        rank=rank,
        momentum=args.bw_momentum,
        energy_threshold=energy_threshold,
    )

    return optimizer


def parse_ubv_rank(rank_str: str):
    """Parse UBV rank argument."""
    if rank_str == "auto":
        return "auto"
    elif rank_str == "full":
        return "full"
    else:
        try:
            return int(rank_str)
        except ValueError:
            raise ValueError(f"Invalid ubv_rank: {rank_str}. Must be int, 'auto', or 'full'")


def setup_ubv_model(model, args):
    """
    Convert model to use UBV factorization with energy-based rank detection.

    Returns the converted model and a summary of detected ranks.
    """
    rank_spec = parse_ubv_rank(args.ubv_rank)

    # Energy threshold: 0 means use tol-based rank
    energy_threshold = args.ubv_energy_threshold if args.ubv_energy_threshold > 0 else None

    # Create rank function
    rank_fn = create_rank_fn(
        rank=rank_spec,
        energy_threshold=energy_threshold if energy_threshold else 0.8,
        min_rank=1,
    )

    # Build include/exclude patterns
    import re as re_module
    include_patterns = args.ubv_include_patterns
    exclude_patterns = args.ubv_exclude_patterns

    # Default: convert attention and MLP layers for HuggingFace transformers
    if include_patterns:
        # Convert list of patterns to regex
        include_fn = lambda name: any(re_module.match(p, name) for p in include_patterns)
    else:
        include_fn = None  # Use defaults

    if exclude_patterns:
        exclude_fn = lambda name: any(re_module.match(p, name) for p in exclude_patterns)
    else:
        exclude_fn = None  # Use defaults

    print(f"[UBV] Converting model with rank={rank_spec}, energy_threshold={energy_threshold}")

    # Track ranks for summary
    detected_ranks = {}

    def tracking_rank_fn(name, module):
        r = rank_fn(name, module)
        detected_ranks[name] = r
        return r

    # Convert model
    model = convert_model_to_factored(
        model,
        r_star=tracking_rank_fn,
        target_pattern=include_fn,
        exclude_pattern=exclude_fn,
        freeze_residual=True,
        verbose=True,
    )

    # Print rank summary
    if detected_ranks:
        ranks = list(detected_ranks.values())
        print(f"[UBV] Rank summary: min={min(ranks)}, max={max(ranks)}, mean={sum(ranks)/len(ranks):.1f}")
        print(f"[UBV] Total layers converted: {len(ranks)}")

    return model, detected_ranks


class CombinedUBVAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: UBVOptimizer for U/B/V params, AdamW for everything else.

    This ensures embeddings and other non-factored parameters use AdamW
    (with momentum and adaptive learning rates) instead of plain SGD.
    """

    def __init__(self, ubv_optimizer, adamw_optimizer):
        # We need to set defaults and param_groups for compatibility
        self.ubv_optimizer = ubv_optimizer
        self.adamw_optimizer = adamw_optimizer

        # Combine param_groups for LR scheduler compatibility
        # Use UBV's param_groups as the primary (scheduler will modify 'lr')
        self.param_groups = ubv_optimizer.param_groups + adamw_optimizer.param_groups
        self.defaults = ubv_optimizer.defaults.copy()
        self.state = {}

    def zero_grad(self, set_to_none=True):
        self.ubv_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step both optimizers
        self.ubv_optimizer.step()
        self.adamw_optimizer.step()

        return loss

    def state_dict(self):
        return {
            'ubv': self.ubv_optimizer.state_dict(),
            'adamw': self.adamw_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.ubv_optimizer.load_state_dict(state_dict['ubv'])
        self.adamw_optimizer.load_state_dict(state_dict['adamw'])


def create_ubv_optimizer(model, args):
    """
    Create combined optimizer: UBVOptimizer for U/B/V, AdamW for embeddings and biases.

    The optimizer applies different update rules:
    - U, V: Riemannian gradient descent on Stiefel manifold (QR retraction)
    - B: Riemannian gradient on S_{++} with eigenvalue clamping
    - bias, other (embeddings): AdamW
    """
    # Get learning rates (with defaults)
    lr_U = args.ubv_lr_U if args.ubv_lr_U is not None else args.learning_rate
    lr_B = args.ubv_lr_B if args.ubv_lr_B is not None else args.learning_rate / 2
    lr_V = args.ubv_lr_V if args.ubv_lr_V is not None else args.learning_rate

    # Get parameters grouped by role
    params_by_role = get_factored_parameters(model)

    n_U = len(params_by_role['U'])
    n_B = len(params_by_role['B'])
    n_V = len(params_by_role['V'])
    n_bias = len(params_by_role['bias'])
    n_other = len(params_by_role['other'])

    print(f"[UBVOptimizer] Parameters: U={n_U}, B={n_B}, V={n_V}, bias={n_bias}, other={n_other}")
    print(f"[UBVOptimizer] Learning rates: lr_U={lr_U}, lr_B={lr_B}, lr_V={lr_V}")
    print(f"[UBVOptimizer] lambda_min={args.ubv_lambda_min}, momentum={args.ubv_momentum}")

    # Collect UBV parameters (U, B, V only)
    ubv_params = params_by_role['U'] + params_by_role['B'] + params_by_role['V']

    # Collect AdamW parameters (bias and other, including embeddings)
    adamw_params = params_by_role['bias'] + params_by_role['other']

    # Create UBVOptimizer for factored parameters only
    ubv_optimizer = UBVOptimizer(
        ubv_params,
        lr_U=lr_U,
        lr_B=lr_B,
        lr_V=lr_V,
        lr_bias=args.learning_rate,  # Not used, but required
        lambda_min=args.ubv_lambda_min,
        momentum=args.ubv_momentum,
    )

    # Create AdamW for embeddings, biases, and other non-factored parameters
    if adamw_params:
        adamw_optimizer = torch.optim.AdamW(
            adamw_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        print(f"[UBVOptimizer] AdamW for {len(adamw_params)} non-factored params (embeddings, biases)")
    else:
        # Dummy optimizer if no AdamW params (shouldn't happen in practice)
        adamw_optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=args.learning_rate)
        print("[UBVOptimizer] No non-factored params for AdamW")

    # Combine into single optimizer interface
    combined = CombinedUBVAdamW(ubv_optimizer, adamw_optimizer)

    return combined


class ProjectedOptimizerMetricsCallback(TrainerCallback):
    """Callback to log ProjectedGradientOptimizer metrics to wandb."""

    def __init__(self, optimizer, log_every_n_steps: int = 10):
        self.optimizer = optimizer
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        """Log optimizer metrics after each step."""
        # Only log every N steps to avoid overhead
        if state.global_step % self.log_every_n_steps != 0:
            return control

        if hasattr(self.optimizer, 'get_detailed_metrics'):
            try:
                metrics = self.optimizer.get_detailed_metrics()
                if metrics:
                    # Try wandb first
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log(metrics, step=state.global_step)
                    except ImportError:
                        pass
            except Exception as e:
                # Silently ignore errors to not disrupt training
                pass
        return control


def create_projected_gradient_optimizer(model, args):
    """
    Create ProjectedGradientOptimizer for training without model surgery.

    This optimizer:
    - Keeps W as the actual parameter (no UBV factorization in model)
    - Uses UBV factors to project gradients onto constrained subspace
    - Applies manifold MUON for U/V, Riemannian SPD for B
    - Periodically re-syncs UBV from W
    - Uses energy-based rank detection (not fixed rank)
    - Maintains frozen residual R = W - U @ B @ V.T

    Key: B is initialized as FULL matrix (not diagonal) per Mishra et al. 2013.
    All factors (U, B, V) use the SAME learning rate for consistent optimization.
    """
    lr = args.learning_rate

    # Handle no-resync flag
    resync_every = 999999999 if args.proj_no_resync else args.proj_resync_every
    resync_str = "disabled" if args.proj_no_resync else str(resync_every)

    print(f"[ProjectedGradient] Creating optimizer with frozen residual...")
    print(f"[ProjectedGradient] lr={lr} (same for U, B, V)")
    print(f"[ProjectedGradient] energy_threshold={args.proj_energy_threshold}, resync_every={resync_str}")
    print(f"[ProjectedGradient] admm_steps={args.proj_admm_steps}, admm_rho={args.proj_admm_rho}")
    print(f"[ProjectedGradient] grad_clip={args.proj_grad_clip}")

    # Target attention and MLP layers (typical transformer patterns)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    optimizer = create_projected_optimizer(
        model,
        lr=lr,
        energy_threshold=args.proj_energy_threshold,
        resync_every=resync_every,
        admm_steps=args.proj_admm_steps,
        admm_rho=args.proj_admm_rho,
        target_modules=target_modules,
        grad_clip=args.proj_grad_clip,
    )

    # Count parameters
    n_projected = sum(1 for g in optimizer.param_groups for p in g['params'] if p.dim() == 2)
    n_other = sum(1 for g in optimizer.param_groups for p in g['params'] if p.dim() != 2)
    print(f"[ProjectedGradient] {n_projected} weight matrices, {n_other} other params")

    return optimizer


def create_muon_optimizer(model, args):
    """
    Create Muon optimizer (Newton-Schulz orthogonalized momentum).

    Muon applies Newton-Schulz iterations to orthogonalize momentum updates,
    which can help with optimization on matrices.
    """
    from muon import Muon

    lr = args.learning_rate
    momentum = args.muon_momentum

    print(f"[Muon] Creating optimizer...")
    print(f"[Muon] lr={lr}, momentum={momentum}")

    # Get all trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = Muon(
        params,
        lr=lr,
        momentum=momentum,
    )

    n_params = len(params)
    total_elements = sum(p.numel() for p in params)
    print(f"[Muon] {n_params} parameter tensors, {total_elements:,} total elements")

    return optimizer


def train_sdft(model, ref_model, tokenizer, train_dataset, task_name, args):
    """Train using SDFT (or DFT if frozen_teacher=True)."""
    estimated_steps = (len(train_dataset) // args.gradient_accumulation_steps) * 2

    # Determine if this is self-distillation or regular distillation
    sync_ref = not args.frozen_teacher
    method_name = "SDFT" if sync_ref else "DFT"

    print(f"[{method_name}] Estimated steps: {estimated_steps}, save_strategy={args.save_strategy}")
    if args.frozen_teacher:
        print(f"[{method_name}] Teacher frozen (no EMA sync)")
    if args.use_projected_optimizer:
        resync_str = "disabled" if args.proj_no_resync else str(args.proj_resync_every)
        print(f"[{method_name}] Using ProjectedGradientOptimizer (energy={args.proj_energy_threshold}, resync={resync_str})")
    elif args.use_muon_optimizer:
        print(f"[{method_name}] Using Muon optimizer (lr={args.learning_rate}, momentum={args.muon_momentum})")
    elif args.use_bw_optimizer:
        energy_str = f", energy={args.bw_energy_threshold}" if args.bw_energy_threshold > 0 else ", tol-based"
        print(f"[{method_name}] Using BWRiemannianSGD optimizer (rank={args.bw_rank}{energy_str})")
    elif args.use_ubv_optimizer:
        energy_str = f", energy={args.ubv_energy_threshold}" if args.ubv_energy_threshold > 0 else ", tol-based"
        print(f"[{method_name}] Using UBV Riemannian optimizer (rank={args.ubv_rank}{energy_str})")

    config = DistilConfig(
        seed=args.seed,
        output_dir=args.output_dir,
        use_vllm=not args.no_vllm,
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
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        max_grad_norm=1,
        sync_ref_model=sync_ref,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        report_to="wandb",
        run_name=f"exp{args.exp_id}_{method_name.lower()}_{task_name}",
        log_completions=False,
    )

    # Create custom optimizer if requested
    optimizers = (None, None)
    if args.use_projected_optimizer:
        optimizer = create_projected_gradient_optimizer(model, args)
        optimizers = (optimizer, None)  # Let trainer create scheduler
    elif args.use_muon_optimizer:
        optimizer = create_muon_optimizer(model, args)
        optimizers = (optimizer, None)  # Let trainer create scheduler
    elif args.use_ubv_optimizer:
        optimizer = create_ubv_optimizer(model, args)
        optimizers = (optimizer, None)  # Let trainer create scheduler
    elif args.use_bw_optimizer:
        optimizer = create_bw_optimizer(model, args)
        optimizers = (optimizer, None)  # Let trainer create scheduler

    trainer = DistilTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        optimizers=optimizers,
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

    estimated_steps = (len(sft_dataset) // args.gradient_accumulation_steps) * args.num_train_epochs
    print(f"[SFT] Estimated steps: {estimated_steps}, save_strategy={args.save_strategy}")

    if args.use_projected_optimizer:
        resync_str = "disabled" if args.proj_no_resync else str(args.proj_resync_every)
        print(f"[SFT] Using ProjectedGradientOptimizer (energy={args.proj_energy_threshold}, resync={resync_str})")
    elif args.use_muon_optimizer:
        print(f"[SFT] Using Muon optimizer (lr={args.learning_rate}, momentum={args.muon_momentum})")
    elif args.use_bw_optimizer:
        energy_str = f", energy={args.bw_energy_threshold}" if args.bw_energy_threshold > 0 else ", tol-based"
        print(f"[SFT] Using BWRiemannianSGD optimizer (rank={args.bw_rank}{energy_str})")
    elif args.use_ubv_optimizer:
        energy_str = f", energy={args.ubv_energy_threshold}" if args.ubv_energy_threshold > 0 else ", tol-based"
        print(f"[SFT] Using UBV Riemannian optimizer (rank={args.ubv_rank}{energy_str})")

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
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        max_grad_norm=1,
        report_to="wandb",
        run_name=f"exp{args.exp_id}_sft_{task_name}",
        seed=args.seed,
    )

    # Create custom optimizer if requested
    optimizers = (None, None)
    callbacks = []
    if args.use_projected_optimizer:
        optimizer = create_projected_gradient_optimizer(model, args)
        optimizers = (optimizer, None)  # Let trainer create scheduler
        callbacks.append(ProjectedOptimizerMetricsCallback(optimizer))
    elif args.use_muon_optimizer:
        optimizer = create_muon_optimizer(model, args)
        optimizers = (optimizer, None)  # Let trainer create scheduler
    elif args.use_ubv_optimizer:
        optimizer = create_ubv_optimizer(model, args)
        optimizers = (optimizer, None)  # Let trainer create scheduler
    elif args.use_bw_optimizer:
        optimizer = create_bw_optimizer(model, args)
        optimizers = (optimizer, None)  # Let trainer create scheduler

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        optimizers=optimizers,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def main():
    args = parse_args()

    # Set up distributed port before any vLLM initialization
    setup_distributed_port()

    if args.exp_id is None:
        args.exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Determine method name based on flags
    if args.method == "sdft":
        method_display = "DFT (frozen teacher)" if args.frozen_teacher else "SDFT"
    else:
        method_display = "SFT"

    print(f"\n{'='*60}")
    print(f"Training: {method_display} on {args.task}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    if args.use_projected_optimizer:
        resync_str = "disabled" if args.proj_no_resync else str(args.proj_resync_every)
        print(f"Optimizer: ProjectedGradient (energy={args.proj_energy_threshold}, resync={resync_str})")
    elif args.use_muon_optimizer:
        print(f"Optimizer: Muon (lr={args.learning_rate}, momentum={args.muon_momentum})")
    elif args.use_ubv_optimizer:
        energy_str = f", energy={args.ubv_energy_threshold}" if args.ubv_energy_threshold > 0 else ", tol-based"
        print(f"Optimizer: UBV Riemannian (rank={args.ubv_rank}{energy_str})")
    elif args.use_bw_optimizer:
        energy_str = f", energy={args.bw_energy_threshold}" if args.bw_energy_threshold > 0 else ", tol-based"
        print(f"Optimizer: BWRiemannianSGD (rank={args.bw_rank}{energy_str})")
    print(f"{'='*60}\n")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Determine device mapping strategy
    # UBV optimizer requires models on single device (no device_map='auto')
    # because the accelerator doesn't support it in distributed mode
    # ProjectedGradientOptimizer works with device_map='auto' (no surgery)
    use_device_map = not args.use_ubv_optimizer

    # Load model
    print(f"Loading model from: {args.model_name}")
    if use_device_map:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Convert model to UBV factorization if requested
    # This must be done BEFORE loading ref_model so both have same architecture
    if args.use_ubv_optimizer:
        print("\n[UBV] Converting model to factored form...")
        model, detected_ranks = setup_ubv_model(model, args)

    # Load ref model for SDFT
    ref_model = None
    if args.method == "sdft":
        print("Loading reference model...")
        if use_device_map:
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
            )
            ref_model = ref_model.cuda()

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

    # Full evaluation on training task's test set
    print(f"\n[Eval] Full {args.task} test set evaluation:")
    _, task_eval_dataset = load_task_dataset(args.task, seed=args.seed)
    task_full_results = evaluate_task_full(model, tokenizer, task_eval_dataset, args.task, device)
    print(f"  {args.task} full test: {task_full_results['accuracy']:.2%} ({task_full_results['correct']}/{task_full_results['total']})")

    # Log to wandb
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                f"eval/{args.task}_full_accuracy": task_full_results["accuracy"],
                f"eval/{args.task}_full_correct": task_full_results["correct"],
                f"eval/{args.task}_full_total": task_full_results["total"],
            })
            # Also log as summary for easy access
            wandb.run.summary[f"{args.task}_full_accuracy"] = task_full_results["accuracy"]
            wandb.run.summary[f"{args.task}_full_correct"] = task_full_results["correct"]
            wandb.run.summary[f"{args.task}_full_total"] = task_full_results["total"]
            print("  Logged to wandb")
    except Exception as e:
        print(f"  Warning: Could not log to wandb: {e}")

    # Save results
    results = {
        "task": args.task,
        "method": args.method,
        "model": args.model_name,
        "before": before_results,
        "after": after_results,
        f"{args.task}_full": {
            "accuracy": task_full_results["accuracy"],
            "correct": task_full_results["correct"],
            "total": task_full_results["total"],
        },
    }
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save detailed results separately
    with open(Path(args.output_dir) / f"{args.task}_eval_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model_path": args.output_dir,
            "task": args.task,
            "accuracy": task_full_results["accuracy"],
            "correct": task_full_results["correct"],
            "total": task_full_results["total"],
            "results": task_full_results["results"],
        }, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")
    print(f"Detailed {args.task} results saved to {args.output_dir}/{args.task}_eval_results.json")


if __name__ == "__main__":
    main()

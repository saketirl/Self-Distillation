"""
Train Qwen3-4B on new tasks using Distillation Fine-Tuning (DFT) with Projected Gradient.

Tasks:
- LiveCodeBench v6 (competitive programming)
- KernelBench Level 1 (CUDA kernel generation)
- MedQuad (medical QA)
- tooluse (simplified tool selection)

Usage:
    python train_qwen3_dft.py --task livecodebench --model_name Qwen/Qwen3-4B
    python train_qwen3_dft.py --task medquad --model_name outputs/qwen3_livecodebench
"""

import argparse
import gc
import json
import os
import re
import socket
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from data_loaders import load_task_dataset, TASK_ORDER_QWEN3
from evaluate_extended import evaluate_task, EVAL_FUNCTIONS

# Import frozen_residual_ubv for UBV optimizer
from frozen_residual_ubv import (
    ProjectedGradientOptimizer,
    create_projected_optimizer,
    FeaturePreservingMuon,
    CompositionPreservingMuon,
    SoftCompPreservingMuon,
)


def find_free_port() -> int:
    """Find a free port, using SLURM_JOB_ID if available to avoid conflicts."""
    import random

    slurm_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if slurm_job_id:
        base = 30000 + (int(slurm_job_id) % 10000)
    else:
        base = random.randint(49152, 60000)

    for port in range(base, base + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def setup_distributed_port():
    """Set up distributed environment variables with an available port."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    port = find_free_port()
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print(f"[Setup] Distributed env: MASTER_ADDR=127.0.0.1, MASTER_PORT={port}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-4B with DFT and Projected Gradient")

    parser.add_argument("--method", type=str, choices=["dft", "sft", "sdft"], default="dft",
                        help="Training method: dft (frozen teacher), sdft (EMA teacher), or sft")
    parser.add_argument("--task", type=str, required=True,
                        choices=["livecodebench", "kernelbench", "medquad", "tooluse",
                                 "mathbeyond", "polaris", "limo", "s1k", "dapo", "openthoughts10k",
                                 "spider", "cot_math"],
                        help="Task to train on")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        help="Model or checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_id", type=str, default=None)

    # Training settings (from proj_lr1em3_energy08.sbatch)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)

    # DFT settings
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--no_vllm", action="store_true",
                        help="Disable vLLM, use HuggingFace generation instead")

    # Projected Gradient optimizer settings (from proj_lr1em3_energy08.sbatch)
    parser.add_argument("--proj_energy_threshold", type=float, default=0.8,
                        help="Energy threshold for rank detection (0.0-1.0)")
    parser.add_argument("--proj_resync_every", type=int, default=100,
                        help="Re-sync UBV from W every N steps")
    parser.add_argument("--proj_no_resync", action="store_true",
                        help="Disable resync entirely")
    parser.add_argument("--proj_admm_steps", type=int, default=20,
                        help="Number of ADMM iterations for manifold MUON")
    parser.add_argument("--proj_admm_rho", type=float, default=4.0,
                        help="ADMM penalty parameter")
    parser.add_argument("--proj_grad_clip", type=float, default=1.0,
                        help="Gradient clipping for projected optimizer")

    # New Stiefel hyperparameters
    parser.add_argument("--stiefel_update", type=str, default="projected_qr",
                        choices=["projected_qr", "muon_admm"],
                        help="Stiefel update method")
    parser.add_argument("--stiefel_max_rms", type=float, default=0.03,
                        help="RMS threshold for U, V update clipping")
    parser.add_argument("--uv_lr_scale", type=float, default=0.3,
                        help="Learning rate scale for U, V")
    parser.add_argument("--b_lr_scale", type=float, default=0.3,
                        help="Learning rate scale for B")

    # Sequence length settings
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Max prompt tokens (left-truncated if exceeded)")
    parser.add_argument("--max_completion_length", type=int, default=2048,
                        help="Max completion tokens; 2048 default to accommodate Qwen3 <think> traces")

    # Evaluation settings
    parser.add_argument("--eval_max_samples", type=int, default=50,
                        help="Max samples for evaluation")
    parser.add_argument("--eval_all_tasks", action="store_true",
                        help="Evaluate on all tasks (for measuring forgetting)")
    parser.add_argument("--eval_tasks", type=str, default=None,
                        help="Comma-separated list of tasks to evaluate on, e.g. spider,tooluse,cot_math. "
                             "Overrides --eval_all_tasks task list.")
    parser.add_argument("--skip_before_eval", action="store_true",
                        help="Skip evaluation before training")

    # Checkpoint settings
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--save_strategy", type=str, default="steps",
                        choices=["steps", "epoch", "no"])
    parser.add_argument("--save_steps", type=float, default=0.25,
                        help="Checkpoint interval as fraction of epoch (e.g. 0.25 = 4x per epoch)")

    parser.add_argument("--adamw_only", action="store_true",
                        help="Use plain AdamW (no projected gradient optimizer); only applies to --method sft")

    # Muon optimizer settings
    parser.add_argument("--use_muon", action="store_true",
                        help="Use Muon for 2D weight matrices, AdamW for embeddings/biases/heads")
    parser.add_argument("--muon_lr", type=float, default=0.02,
                        help="Learning rate for Muon (2D weight matrices)")
    parser.add_argument("--muon_momentum", type=float, default=0.95,
                        help="Momentum for Muon")
    parser.add_argument("--muon_adamw_lr", type=float, default=3e-4,
                        help="AdamW learning rate for embeddings/biases/heads alongside Muon")

    # FeaturePreservingMuon optimizer settings
    parser.add_argument("--use_fp_muon", action="store_true",
                        help="Use FeaturePreservingMuon: projects gradient orthogonal to "
                             "the right singular directions capturing fp_energy_threshold "
                             "of each weight's initial spectral energy")
    parser.add_argument("--fp_energy_threshold", type=float, default=0.8,
                        help="Fraction of spectral energy to protect per weight matrix "
                             "(0.0–1.0). Higher = more directions frozen. "
                             "Matches proj_energy_threshold semantics.")
    parser.add_argument("--fp_momentum", type=float, default=0.0,
                        help="Momentum for FeaturePreservingMuon (0 disables)")
    parser.add_argument("--fp_adamw_lr", type=float, default=5e-5,
                        help="AdamW lr for 1-D params (biases, LayerNorm) alongside fp_muon")
    parser.add_argument("--fp_n_dual_iter", type=int, default=10,
                        help="Max dual ascent iterations to enforce U_r/V_r constraints "
                             "(0 = plain projection + single msign)")
    parser.add_argument("--fp_dual_tol", type=float, default=1e-4,
                        help="Early-stop dual ascent when both ‖A V_r‖ and ‖U_r^T A‖ "
                             "fall below this threshold")
    parser.add_argument("--fp_dual_step_size", type=float, default=0.01,
                        help="Step size for dual ascent Lambda updates (smaller = more stable)")
    parser.add_argument("--fp_vr_update_freq", type=int, default=0,
                        help="Recompute V_r from current weights every N steps (0 = never refresh)")
    parser.add_argument("--fp_residual_fallback_tol", type=float, default=0.1,
                        help="If ‖D V_r‖_F exceeds this after dual ascent, fall back to exact "
                             "G(I-VV^T) projection (default 0.1)")
    parser.add_argument("--fp_vr_bypass_alpha", type=float, default=0.0,
                        help="Fraction of the V_r-aligned gradient component allowed through "
                             "(0.0 = full projection, 1.0 = no projection / plain Muon). "
                             "Use 0.1-0.3 when the model fails to learn with full projection.")
    parser.add_argument("--fp_debug", action="store_true",
                        help="Print per-step constraint residuals for each FPMuon parameter")

    # CompositionPreservingMuon optimizer settings
    parser.add_argument("--use_comp_muon", action="store_true",
                        help="Use CompositionPreservingMuon: preserves singular directions of "
                             "M_QK = W_Q^T W_K and M_OV = W_O W_V for each attention head")
    parser.add_argument("--comp_head_dim", type=int, default=80,
                        help="Per-head dimension d_h (80 for Qwen3-4B); "
                             "actual d_h is always derived from weight shapes")
    parser.add_argument("--comp_num_kv_heads", type=int, default=8,
                        help="Number of KV heads H_kv (8 for Qwen3-4B GQA)")
    parser.add_argument("--comp_qk_energy_threshold", type=float, default=0.8,
                        help="Fraction of spectral energy of M_QK to protect per head. "
                             "k is chosen automatically as the smallest integer such that "
                             "sum(S[:k]^2)/sum(S^2) >= threshold. Default 0.8.")
    parser.add_argument("--comp_ov_energy_threshold", type=float, default=0.8,
                        help="Fraction of spectral energy of M_OV to protect per head. "
                             "Default 0.8.")
    parser.add_argument("--comp_no_preserve_qk", action="store_true",
                        help="Disable QK composition constraint entirely")
    parser.add_argument("--comp_no_preserve_ov", action="store_true",
                        help="Disable OV composition constraint entirely")
    parser.add_argument("--comp_adamw_lr", type=float, default=3e-4,
                        help="AdamW lr for non-attention params alongside CompMuon")
    parser.add_argument("--comp_residual_fallback_tol", type=float, default=float("inf"),
                        help="Legacy: if dual ascent residual exceeds this, fall back to plain "
                             "msign. Ignored when --comp_use_strict_fallback is set. Default inf.")
    parser.add_argument("--comp_dual_tol", type=float, default=1e-4,
                        help="Relative residual threshold to accept the dual-constrained update "
                             "(compared via constraint_residual). Default 1e-4.")
    parser.add_argument("--comp_fallback_tol", type=float, default=None,
                        help="Relative residual threshold to accept the strict composition "
                             "fallback. Default: 10 * comp_dual_tol.")
    parser.add_argument("--comp_no_strict_fallback", action="store_true",
                        help="Disable strict_composition_fallback; use legacy plain-msign fallback.")
    parser.add_argument("--comp_no_skip_if_fallback_fails", action="store_true",
                        help="When strict fallback residual > fallback_tol, apply it anyway "
                             "instead of zeroing the update.")
    parser.add_argument("--comp_fp_energy_threshold", type=float, default=0.8,
                        help="Fraction of spectral energy for V_r in FPMuon-style MLP update "
                             "inside CompMuon. Higher = more features preserved, more compute. "
                             "Default 0.8.")
    parser.add_argument("--comp_debug", action="store_true",
                        help="Print per-head residuals and fallback state for CompMuon")
    parser.add_argument("--comp_clip_gradient", action="store_true",
                        help="Scale each (D_A, D_B) pair by ||G_perp||/||G|| before applying. "
                             "Prevents msign from amplifying near-zero projected gradients "
                             "(which occur when the task gradient lies mostly in the blocked "
                             "U_k/V_k directions) into full-magnitude adversarial updates.")
    parser.add_argument("--comp_delta_norm_cap", type=float, default=None,
                        help="Cap ||D_A @ B + A @ D_B||_F to this value by rescaling (D_A, D_B). "
                             "Prevents large but directionally-feasible composed updates from "
                             "destabilizing training (the relative residual is direction-only). "
                             "Typical value: 1.0–10.0. None = disabled.")

    # SoftCompPreservingMuon + FeaturePreservingMuon (attn + MLP split)
    parser.add_argument("--use_soft_comp_muon", action="store_true",
                        help="Use SoftCompPreservingMuon for attention (QK/OV soft-core) "
                             "and FeaturePreservingMuon for MLP layers")
    parser.add_argument("--soft_comp_head_dim", type=int, default=80,
                        help="Per-head dimension d_h (80 for Qwen3-4B)")
    parser.add_argument("--soft_comp_num_kv_heads", type=int, default=8,
                        help="Number of KV heads (8 for Qwen3-4B GQA)")
    parser.add_argument("--soft_comp_qk_energy_threshold", type=float, default=0.8,
                        help="Fraction of QK spectral energy to preserve per head; "
                             "k = min{j : sum(S[:j]^2)/sum(S^2) >= threshold}")
    parser.add_argument("--soft_comp_ov_energy_threshold", type=float, default=0.8,
                        help="Fraction of OV spectral energy to preserve per head")
    parser.add_argument("--soft_comp_no_preserve_qk", action="store_true",
                        help="Disable QK core constraint")
    parser.add_argument("--soft_comp_no_preserve_ov", action="store_true",
                        help="Disable OV core constraint")
    parser.add_argument("--soft_comp_admm_steps", type=int, default=15,
                        help="Max ADMM inner iterations per optimizer step")
    parser.add_argument("--soft_comp_dual_lr", type=float, default=0.02,
                        help="Dual variable step size for ADMM update")
    parser.add_argument("--soft_comp_dual_tol", type=float, default=1e-4,
                        help="Stop ADMM when ||H||_F / sqrt(k^2) < dual_tol")
    parser.add_argument("--soft_comp_fp_energy_threshold", type=float, default=0.8,
                        help="Energy threshold for FPMuon on MLP layers")
    parser.add_argument("--soft_comp_fp_ur_alpha", type=float, default=0.0,
                        help="Protect both U_r and V_r in MLP FPMuon (0=V only, 1=full UV joint constraint)")
    parser.add_argument("--soft_comp_fp_fixed_subspaces", action="store_true",
                        help="Compute U_r/V_r from the loaded model weights once and freeze them "
                             "for the entire task (prevents intra-task V_r rotation)")
    parser.add_argument("--soft_comp_adamw_lr", type=float, default=3e-4,
                        help="AdamW lr for 1-D params / embed / lm_head")
    parser.add_argument("--soft_comp_debug", action="store_true",
                        help="Print per-head ADMM diagnostics each step")
    # Sylvester fallback controls (applied after ADMM when H_norm is still too large)
    parser.add_argument("--soft_comp_no_sylvester_fallback", action="store_true",
                        help="Disable Sylvester correction fallback (not recommended)")
    parser.add_argument("--soft_comp_fallback_tol", type=float, default=1e-2,
                        help="Trigger Sylvester correction when H_norm > this after ADMM")
    parser.add_argument("--soft_comp_hard_fallback_tol", type=float, default=0.5,
                        help="Zero updates when corrected H_norm still exceeds this (relative to H_scale)")
    parser.add_argument("--soft_comp_sylvester_damping", type=float, default=1e-4,
                        help="Regularisation added to P and Q before Kronecker solve")
    parser.add_argument("--soft_comp_sylvester_skip_tol", type=float, default=0.0,
                        help="Skip Sylvester solve when H_norm is already below this (0=always solve)")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class ProjectedOptimizerMetricsCallback(TrainerCallback):
    """Callback to log ProjectedGradientOptimizer metrics to wandb."""

    def __init__(self, optimizer, log_every_n_steps: int = 10):
        self.optimizer = optimizer
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return control

        if hasattr(self.optimizer, 'get_detailed_metrics'):
            try:
                metrics = self.optimizer.get_detailed_metrics()
                if metrics:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log(metrics, step=state.global_step)
                    except ImportError:
                        pass
            except Exception:
                pass
        return control


def create_projected_gradient_optimizer(model, args):
    """Create ProjectedGradientOptimizer with all hyperparameters."""
    lr = args.learning_rate
    resync_every = 999999999 if args.proj_no_resync else args.proj_resync_every
    resync_str = "disabled" if args.proj_no_resync else str(resync_every)

    print(f"[ProjectedGradient] Creating optimizer...")
    print(f"[ProjectedGradient] lr={lr}, uv_lr_scale={args.uv_lr_scale}, b_lr_scale={args.b_lr_scale}")
    print(f"[ProjectedGradient] energy_threshold={args.proj_energy_threshold}, resync_every={resync_str}")
    print(f"[ProjectedGradient] stiefel_update={args.stiefel_update}, stiefel_max_rms={args.stiefel_max_rms}")
    print(f"[ProjectedGradient] admm_steps={args.proj_admm_steps}, admm_rho={args.proj_admm_rho}")

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
        stiefel_update=args.stiefel_update,
        stiefel_max_rms=args.stiefel_max_rms,
        uv_lr_scale=args.uv_lr_scale,
        b_lr_scale=args.b_lr_scale,
    )

    n_projected = sum(1 for g in optimizer.param_groups for p in g['params'] if p.dim() == 2)
    n_other = sum(1 for g in optimizer.param_groups for p in g['params'] if p.dim() != 2)
    print(f"[ProjectedGradient] {n_projected} weight matrices, {n_other} other params")

    return optimizer


def create_muon_optimizer(model, args):
    """
    Create Muon optimizer for 2D weight matrices, AdamW for everything else.

    Per the Muon repo: hidden layer weights (ndim >= 2) go to Muon;
    embeddings, heads, biases, and 1D params go to AdamW.
    """
    from muon import Muon

    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    print(f"[Muon] {len(muon_params)} matrix params -> Muon (lr={args.muon_lr}, momentum={args.muon_momentum})")
    print(f"[Muon] {len(adamw_params)} scalar/1D params -> AdamW (lr={args.muon_adamw_lr})")

    muon = Muon(muon_params, lr=args.muon_lr, momentum=args.muon_momentum)
    adamw = torch.optim.AdamW(adamw_params, lr=args.muon_adamw_lr, betas=(0.9, 0.999), weight_decay=0.01)

    # Must inherit torch.optim.Optimizer so LambdaLR / cosine scheduler pass the
    # isinstance check.  We call super().__init__() with shallow-copied param
    # group dicts so the base-class setup runs cleanly without mutating the
    # sub-optimizers' internal dicts.  Immediately after, we replace param_groups
    # with the original shared references so scheduler LR updates propagate into
    # muon.step() / adamw.step().
    class CombinedMuonAdamW(torch.optim.Optimizer):
        def __init__(self, muon, adamw):
            self.muon = muon
            self.adamw = adamw
            dummy_groups = [dict(g) for g in (muon.param_groups + adamw.param_groups)]
            super().__init__(dummy_groups, {'lr': muon.param_groups[0]['lr']})
            # Replace with shared refs — LR scheduler modifies dicts in-place,
            # so updates here propagate directly into each sub-optimizer.
            self.param_groups = muon.param_groups + adamw.param_groups

        def zero_grad(self, set_to_none=True):
            self.muon.zero_grad(set_to_none=set_to_none)
            self.adamw.zero_grad(set_to_none=set_to_none)

        def step(self, closure=None):
            if not hasattr(self, '_step_count'):
                self._step_count = 0
            if self._step_count == 0:
                muon_lr = self.muon.param_groups[0]['lr']
                adamw_lr = self.adamw.param_groups[0]['lr']
                print(f"[Muon step=0] muon lr={muon_lr:.2e}, adamw lr={adamw_lr:.2e}", flush=True)
            self._step_count += 1
            self.muon.step()
            self.adamw.step()

        def state_dict(self):
            return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

        def load_state_dict(self, state_dict):
            self.muon.load_state_dict(state_dict["muon"])
            self.adamw.load_state_dict(state_dict["adamw"])

    return CombinedMuonAdamW(muon, adamw)


class FPMuonMetricsCallback(TrainerCallback):
    """Log FeaturePreservingMuon constraint residuals to stdout and wandb every N steps.

    Uses commit=False so metrics are buffered and WandbCallback's commit
    at the same step picks them up — avoids step-counter conflicts.
    """

    def __init__(self, fp_muon):
        self.fp_muon = fp_muon

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0:
            return control

        metrics = self.fp_muon.get_constraint_metrics()
        if not metrics:
            return control

        r_right    = metrics.get("fp_muon/residual_right_mean",  float("nan"))
        frac       = metrics.get("fp_muon/grad_frac_mean",       float("nan"))
        frac_in_vr = metrics.get("fp_muon/grad_frac_in_vr_mean", float("nan"))
        fallback   = metrics.get("fp_muon/fallback_rate",        float("nan"))
        msign_nan_in  = int(metrics.get("fp_muon/msign_nan_input",  0))
        msign_ns_div  = int(metrics.get("fp_muon/msign_ns_diverge", 0))
        msign_nan_out = int(metrics.get("fp_muon/msign_nan_output", 0))
        msign_total   = int(metrics.get("fp_muon/msign_total",      0))
        print(f"[FPMuon step={state.global_step}] "
              f"‖DV_r‖={r_right:.3e}  "
              f"grad_frac_surviving={frac:.3f}  "
              f"grad_frac_in_Vr={frac_in_vr:.3f}  "
              f"fallback_rate={fallback:.3f}  "
              f"msign(total={msign_total} nan_in={msign_nan_in} ns_div={msign_ns_div} nan_out={msign_nan_out})",
              flush=True)

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, commit=False)
        except Exception:
            pass
        return control


def create_fp_muon_optimizer(model, args):
    """
    Create FeaturePreservingMuon for 2-D weight matrices, AdamW for 1-D params.

    The effective LR matches the Stiefel UV update from ProjectedGradient:
        lr = base_lr * uv_lr_scale  (default: 5e-3 * 0.3 = 1.5e-3)
    Pass --learning_rate directly as the FP-Muon lr; the default in the sbatch
    scripts is already set to 1.5e-3.
    """
    # Exclude embedding and lm_head: [vocab, hidden] tensors are ~1.5 GiB in float32
    # and cause OOM during the Newton-Schulz steps inside msign.
    skip_names = {"embed_tokens.weight", "lm_head.weight"}
    named = list(model.named_parameters())
    matrix_params = [p for n, p in named
                     if p.requires_grad and p.ndim >= 2
                     and not any(s in n for s in skip_names)]
    scalar_params  = [p for n, p in named
                      if p.requires_grad
                      and (p.ndim < 2 or any(s in n for s in skip_names))]

    print(f"[FPMuon] {len(matrix_params)} matrix params -> FeaturePreservingMuon "
          f"(lr={args.learning_rate}, energy={args.fp_energy_threshold}, "
          f"bypass_alpha={args.fp_vr_bypass_alpha}, n_dual_iter={args.fp_n_dual_iter})")
    print(f"[FPMuon] {len(scalar_params)} scalar/1D params -> AdamW (lr={args.fp_adamw_lr}) "
          f"[includes embed_tokens, lm_head]")

    fp_muon = FeaturePreservingMuon(
        matrix_params,
        lr=args.learning_rate,
        energy_threshold=args.fp_energy_threshold,
        momentum=args.fp_momentum,
        nesterov=False,
        n_dual_iter=args.fp_n_dual_iter,
        dual_tol=args.fp_dual_tol,
        dual_step_size=args.fp_dual_step_size,
        vr_update_freq=args.fp_vr_update_freq,
        residual_fallback_tol=args.fp_residual_fallback_tol,
        vr_bypass_alpha=args.fp_vr_bypass_alpha,
        debug=args.fp_debug,
    )
    adamw = torch.optim.AdamW(scalar_params, lr=args.fp_adamw_lr,
                               betas=(0.9, 0.999), weight_decay=0.01)

    class CombinedFPMuonAdamW(torch.optim.Optimizer):
        def __init__(self, fp_muon, adamw):
            self.fp_muon = fp_muon
            self.adamw = adamw
            dummy_groups = [dict(g) for g in (fp_muon.param_groups + adamw.param_groups)]
            super().__init__(dummy_groups, {'lr': fp_muon.param_groups[0]['lr']})
            self.param_groups = fp_muon.param_groups + adamw.param_groups

        def zero_grad(self, set_to_none=True):
            self.fp_muon.zero_grad(set_to_none=set_to_none)
            self.adamw.zero_grad(set_to_none=set_to_none)

        def step(self, closure=None):
            if not hasattr(self, '_step_count'):
                self._step_count = 0
            self._step_count += 1

            # Diagnostic: on first 5 calls, report gradient status + current LR
            if self._step_count <= 5:
                n_with_grad = sum(
                    1 for g in self.fp_muon.param_groups
                    for p in g['params'] if p.grad is not None
                )
                n_total = sum(
                    len(g['params']) for g in self.fp_muon.param_groups
                )
                sample = next(
                    (p for g in self.fp_muon.param_groups for p in g['params'] if p.grad is not None),
                    None
                )
                grad_norm = sample.grad.float().norm().item() if sample is not None else float('nan')
                lr = self.fp_muon.param_groups[0]['lr']
                print(
                    f"[CombinedFPMuonAdamW step={self._step_count}] "
                    f"{n_with_grad}/{n_total} fp_muon params have grad  "
                    f"sample_grad_norm={grad_norm:.3e}  lr={lr:.3e}",
                    flush=True
                )

            self.fp_muon.step(closure)
            self.adamw.step(closure)

        def state_dict(self):
            return {"fp_muon": self.fp_muon.state_dict(), "adamw": self.adamw.state_dict()}

        def load_state_dict(self, state_dict):
            self.fp_muon.load_state_dict(state_dict["fp_muon"])
            self.adamw.load_state_dict(state_dict["adamw"])

    return CombinedFPMuonAdamW(fp_muon, adamw)


class CompMuonMetricsCallback(TrainerCallback):
    """Log CompositionPreservingMuon residuals to stdout and wandb every N steps.

    Uses commit=False so metrics are buffered and WandbCallback's commit
    at the same step picks them up — avoids step-counter conflicts.
    """

    def __init__(self, comp_muon):
        self.comp_muon = comp_muon

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0:
            return control
        metrics = self.comp_muon.get_constraint_metrics()
        if not metrics:
            return control
        comp_mean = metrics.get("comp_muon/comp_residual_mean", float("nan"))
        comp_max  = metrics.get("comp_muon/comp_residual_max",  float("nan"))
        skip_rate = metrics.get("comp_muon/skip_rate",          float("nan"))
        msign_nan_in  = int(metrics.get("comp_muon/msign_nan_input",  0))
        msign_ns_div  = int(metrics.get("comp_muon/msign_ns_diverge", 0))
        msign_nan_out = int(metrics.get("comp_muon/msign_nan_output", 0))
        msign_total   = int(metrics.get("comp_muon/msign_total",      0))
        print(f"[CompMuon step={state.global_step}] "
              f"comp_res={comp_mean:.3e}(max={comp_max:.3e}) skip={skip_rate:.3f}  "
              f"msign(total={msign_total} nan_in={msign_nan_in} ns_div={msign_ns_div} nan_out={msign_nan_out})",
              flush=True)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, commit=False)
        except Exception:
            pass
        return control


def create_comp_muon_optimizer(model, args):
    """
    Create a 3-way combined optimizer:
      - CompositionPreservingMuon for attention Q/K/V/O params only
        (strict algebraic projection preserves M_QK and M_OV singular directions)
      - FeaturePreservingMuon for all other 2-D params (MLP matrices)
        with vr_bypass_alpha=0 (full constraint) and n_dual_iter=0
        (warm-start projection only — no iterative dual ascent to diverge)
      - AdamW for 1-D params and embed/lm_head

    The inline MLP FPMuon path inside CompositionPreservingMuon is intentionally
    bypassed: it uses randomized SVD without QR reorthogonalization so V_r.T@V_r
    is not exactly I, causing the warm-start residual to be non-zero and dual
    ascent to diverge (residual ~0.6 → gradient explosion).  Using the actual
    FeaturePreservingMuon class fixes this via QR reorthogonalization + exact
    projection fallback at residual_fallback_tol=0.1.
    """
    skip_names = {"embed_tokens.weight", "lm_head.weight"}

    # Patterns that identify attention q/k/v/o weight matrices.
    _attn_pats = [
        re.compile(r".*\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"),
        re.compile(r".*\.layers\.(\d+)\.attention\.(q|k|v|o)_proj\.weight"),
    ]

    attn_named: list = []   # (name, param) for q/k/v/o attention weights → CompMuon
    mlp_params: list = []   # 2D non-attn, non-embed/lm_head → FPMuon
    scalar_params: list = []  # 1D + embed/lm_head → AdamW

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(s in name for s in skip_names) or p.ndim < 2:
            scalar_params.append(p)
        elif any(pat.match(name) for pat in _attn_pats):
            attn_named.append((name, p))
        else:
            mlp_params.append(p)

    print(f"[CompMuon] attn params={len(attn_named)} → CompMuon "
          f"(lr={args.learning_rate}, num_kv_heads={args.comp_num_kv_heads}, "
          f"qk_energy={args.comp_qk_energy_threshold}, ov_energy={args.comp_ov_energy_threshold})")
    print(f"[CompMuon] mlp params={len(mlp_params)} → FeaturePreservingMuon "
          f"(energy={args.comp_fp_energy_threshold}, vr_bypass_alpha=0.0, n_dual_iter=0, "
          f"residual_fallback_tol=0.1)")
    print(f"[CompMuon] scalar/embed params={len(scalar_params)} → AdamW (lr={args.comp_adamw_lr})")

    comp_muon = CompositionPreservingMuon(
        attn_named,
        lr=args.learning_rate,
        head_dim=args.comp_head_dim,
        num_kv_heads=args.comp_num_kv_heads,
        qk_energy_threshold=args.comp_qk_energy_threshold,
        ov_energy_threshold=args.comp_ov_energy_threshold,
        preserve_qk=not args.comp_no_preserve_qk,
        preserve_ov=not args.comp_no_preserve_ov,
        dual_tol=args.comp_dual_tol,
        fallback_tol=args.comp_fallback_tol,
        use_strict_fallback=not args.comp_no_strict_fallback,
        skip_if_fallback_fails=not args.comp_no_skip_if_fallback_fails,
        clip_comp_gradient=args.comp_clip_gradient,
        comp_delta_norm_cap=args.comp_delta_norm_cap,
        debug=args.comp_debug,
    )

    fp_muon = FeaturePreservingMuon(
        mlp_params,
        lr=args.learning_rate,
        energy_threshold=args.comp_fp_energy_threshold,
        vr_bypass_alpha=0.0,       # full projection: D @ V_r = 0
        n_dual_iter=0,             # warm-start projection only (no iterative dual ascent)
        residual_fallback_tol=0.1,
        ns_steps=8,
        eps=1e-7,
        scale_to_proj_norm=True,   # scale D by ‖G_perp‖/‖G‖ — same fix as comp_clip_gradient
        debug=args.comp_debug,
    )

    adamw = torch.optim.AdamW(scalar_params, lr=args.comp_adamw_lr,
                               betas=(0.9, 0.999), weight_decay=0.01)

    class CombinedCompMuonFPMuonAdamW(torch.optim.Optimizer):
        def __init__(self, comp_muon, fp_muon, adamw):
            self.comp_muon = comp_muon
            self.fp_muon = fp_muon
            self.adamw = adamw
            all_groups = comp_muon.param_groups + fp_muon.param_groups + adamw.param_groups
            dummy_groups = [dict(g) for g in all_groups]
            super().__init__(dummy_groups, {"lr": comp_muon.param_groups[0]["lr"]})
            self.param_groups = all_groups

        def zero_grad(self, set_to_none=True):
            self.comp_muon.zero_grad(set_to_none=set_to_none)
            self.fp_muon.zero_grad(set_to_none=set_to_none)
            self.adamw.zero_grad(set_to_none=set_to_none)

        def step(self, closure=None):
            # fp_muon is not bound to the scheduler; sync its LR to comp_muon's
            # (which is updated by the cosine-warmup scheduler before each step).
            comp_lr = self.comp_muon.param_groups[0]["lr"]
            for g in self.fp_muon.param_groups:
                g["lr"] = comp_lr
            self.comp_muon.step(closure)
            self.fp_muon.step(closure)
            self.adamw.step(closure)

        def state_dict(self):
            return {
                "comp_muon": self.comp_muon.state_dict(),
                "fp_muon":   self.fp_muon.state_dict(),
                "adamw":     self.adamw.state_dict(),
            }

        def load_state_dict(self, sd):
            self.comp_muon.load_state_dict(sd["comp_muon"])
            self.fp_muon.load_state_dict(sd["fp_muon"])
            self.adamw.load_state_dict(sd["adamw"])

    return CombinedCompMuonFPMuonAdamW(comp_muon, fp_muon, adamw)


class SoftCompMuonMetricsCallback(TrainerCallback):
    """Log SoftCompPreservingMuon ADMM diagnostics to stdout and wandb."""

    def __init__(self, soft_comp_muon):
        self.soft_comp_muon = soft_comp_muon

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0:
            return control
        metrics = self.soft_comp_muon.get_constraint_metrics()
        if not metrics:
            return control

        drift_mean = metrics.get("soft_comp_muon/core_drift_mean", float("nan"))
        drift_max  = metrics.get("soft_comp_muon/core_drift_max",  float("nan"))
        h_mean     = metrics.get("soft_comp_muon/H_norm_mean",     float("nan"))
        iters_mean = metrics.get("soft_comp_muon/dual_iters_mean", float("nan"))
        current_lr = self.soft_comp_muon.param_groups[0]["lr"]
        print(
            f"[SoftCompMuon step={state.global_step}] "
            f"lr={current_lr:.3e}  "
            f"core_drift={drift_mean:.3e}(max={drift_max:.3e})  "
            f"H_norm={h_mean:.3e}  dual_iters={iters_mean:.1f}",
            flush=True,
        )
        print(self.soft_comp_muon.get_fallback_summary(), flush=True)
        metrics["soft_comp_muon/lr"] = current_lr

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step, commit=False)
        except Exception:
            pass
        return control


def create_soft_comp_muon_optimizer(model, args):
    """
    3-way combined optimizer:
      - SoftCompPreservingMuon for attention Q/K/V/O params
        (preserves top-k singular core of W_Q^T W_K and W_O W_V per head)
      - FeaturePreservingMuon for MLP params (gate/up/down projections)
        (projects gradient orthogonal to dominant right singular directions)
      - AdamW for 1-D params, embed_tokens, lm_head

    LR is synced from SoftCompMuon → FPMuon at every step so the cosine
    scheduler (bound to SoftCompMuon) drives both.
    """
    skip_names = {"embed_tokens.weight", "lm_head.weight"}
    _attn_pats = [
        re.compile(r".*\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"),
        re.compile(r".*\.layers\.(\d+)\.attention\.(q|k|v|o)_proj\.weight"),
    ]

    attn_named: list = []
    mlp_params: list = []
    scalar_params: list = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(s in name for s in skip_names) or p.ndim < 2:
            scalar_params.append(p)
        elif any(pat.match(name) for pat in _attn_pats):
            attn_named.append((name, p))
        else:
            mlp_params.append(p)

    print(
        f"[SoftCompMuon] attn params={len(attn_named)} → SoftCompPreservingMuon "
        f"(lr={args.learning_rate}, head_dim={args.soft_comp_head_dim}, "
        f"num_kv_heads={args.soft_comp_num_kv_heads}, "
        f"qk_energy={args.soft_comp_qk_energy_threshold}, "
        f"ov_energy={args.soft_comp_ov_energy_threshold}, "
        f"admm_steps={args.soft_comp_admm_steps}, dual_lr={args.soft_comp_dual_lr})"
    )
    print(
        f"[SoftCompMuon] mlp params={len(mlp_params)} → FeaturePreservingMuon "
        f"(energy={args.soft_comp_fp_energy_threshold}, ur_alpha={args.soft_comp_fp_ur_alpha}, "
        f"fixed_subspaces={args.soft_comp_fp_fixed_subspaces}, n_dual_iter=0, "
        f"scale_to_proj_norm=True)"
    )
    print(
        f"[SoftCompMuon] scalar/embed params={len(scalar_params)} → AdamW "
        f"(lr={args.soft_comp_adamw_lr})"
    )

    use_sylvester = not args.soft_comp_no_sylvester_fallback
    print(
        f"[SoftCompMuon] Sylvester fallback: {'ON' if use_sylvester else 'OFF'}"
        + (f"  fallback_tol={args.soft_comp_fallback_tol}"
           f"  hard_fallback_tol={args.soft_comp_hard_fallback_tol}"
           f"  damping={args.soft_comp_sylvester_damping}" if use_sylvester else "")
    )

    soft_comp = SoftCompPreservingMuon(
        attn_named,
        lr=args.learning_rate,
        head_dim=args.soft_comp_head_dim,
        num_kv_heads=args.soft_comp_num_kv_heads,
        qk_energy_threshold=args.soft_comp_qk_energy_threshold,
        ov_energy_threshold=args.soft_comp_ov_energy_threshold,
        preserve_qk=not args.soft_comp_no_preserve_qk,
        preserve_ov=not args.soft_comp_no_preserve_ov,
        admm_steps=args.soft_comp_admm_steps,
        dual_lr=args.soft_comp_dual_lr,
        dual_tol=args.soft_comp_dual_tol,
        ns_steps=8,
        eps=1e-7,
        debug=args.soft_comp_debug,
        use_sylvester_fallback=use_sylvester,
        fallback_tol=args.soft_comp_fallback_tol,
        hard_fallback_tol=args.soft_comp_hard_fallback_tol,
        sylvester_damping=args.soft_comp_sylvester_damping,
        sylvester_skip_tol=args.soft_comp_sylvester_skip_tol,
    )

    fixed_subspaces = None
    if args.soft_comp_fp_fixed_subspaces and args.soft_comp_fp_ur_alpha > 0.0:
        fixed_subspaces = FeaturePreservingMuon.compute_fixed_subspaces(
            mlp_params, energy_threshold=args.soft_comp_fp_energy_threshold
        )
        print(f"[SoftCompMuon] Computed fixed U_r/V_r for {len(fixed_subspaces)} MLP params "
              f"(energy={args.soft_comp_fp_energy_threshold})")

    fp_muon = FeaturePreservingMuon(
        mlp_params,
        lr=args.learning_rate,
        energy_threshold=args.soft_comp_fp_energy_threshold,
        vr_bypass_alpha=0.0,
        ur_alpha=args.soft_comp_fp_ur_alpha,
        fixed_subspaces=fixed_subspaces,
        n_dual_iter=0,
        residual_fallback_tol=0.1,
        ns_steps=8,
        eps=1e-7,
        scale_to_proj_norm=True,
        debug=args.soft_comp_debug,
    )

    adamw = torch.optim.AdamW(
        scalar_params, lr=args.soft_comp_adamw_lr,
        betas=(0.9, 0.999), weight_decay=0.01,
    )

    class CombinedSoftCompFPMuonAdamW(torch.optim.Optimizer):
        def __init__(self, soft_comp, fp_muon, adamw):
            self.soft_comp = soft_comp
            self.fp_muon = fp_muon
            self.adamw = adamw
            all_groups = soft_comp.param_groups + fp_muon.param_groups + adamw.param_groups
            super().__init__([dict(g) for g in all_groups], {"lr": soft_comp.param_groups[0]["lr"]})
            self.param_groups = all_groups

        def zero_grad(self, set_to_none=True):
            self.soft_comp.zero_grad(set_to_none=set_to_none)
            self.fp_muon.zero_grad(set_to_none=set_to_none)
            self.adamw.zero_grad(set_to_none=set_to_none)

        def step(self, closure=None):
            # Run soft_comp first so it sees the LR that the scheduler already
            # set (via scheduler.step() at the end of the previous iteration).
            self.soft_comp.step(closure)
            # Copy the same LR to FPMuon immediately after soft_comp has used it,
            # so both sub-optimizers always see the identical scheduled value.
            sc_lr = self.soft_comp.param_groups[0]["lr"]
            for g in self.fp_muon.param_groups:
                g["lr"] = sc_lr
            self.fp_muon.step(closure)
            self.adamw.step(closure)

        def state_dict(self):
            return {
                "soft_comp": self.soft_comp.state_dict(),
                "fp_muon":   self.fp_muon.state_dict(),
                "adamw":     self.adamw.state_dict(),
            }

        def load_state_dict(self, sd):
            self.soft_comp.load_state_dict(sd["soft_comp"])
            self.fp_muon.load_state_dict(sd["fp_muon"])
            self.adamw.load_state_dict(sd["adamw"])

    return CombinedSoftCompFPMuonAdamW(soft_comp, fp_muon, adamw)


def evaluate_all_tasks_extended(model, tokenizer, device, args, task_order=None):
    """Evaluate model on all tasks with extended metrics."""
    if task_order is None:
        task_order = TASK_ORDER_QWEN3

    results = {}

    for task_name in task_order:
        try:
            _, eval_ds = load_task_dataset(task_name, seed=args.seed)
            if eval_ds:
                print(f"\n[Eval] Evaluating {task_name}...")
                task_result = evaluate_task(
                    task_name, model, tokenizer, eval_ds, device,
                    max_samples=args.eval_max_samples, verbose=False
                )
                results[task_name] = task_result
                print(f"  {task_name}: {task_result['accuracy']:.2%} ({task_result['correct']}/{task_result['total']})")
        except Exception as e:
            print(f"  {task_name}: Error - {e}")
            results[task_name] = {"accuracy": 0.0, "error": str(e)}

    return results


def train_dft(model, ref_model, tokenizer, train_dataset, task_name, args):
    """Train using DFT (Distillation Fine-Tuning) with frozen teacher."""
    # RepeatSampler uses mini_repeat_count=num_generations, making the effective
    # dataloader length len(dataset) * num_generations. Without this factor the
    # cosine scheduler completes in 1/8th of training and cycles 8 more times.
    _num_generations = 8  # matches DistilConfig / GRPOConfig default
    estimated_steps = (len(train_dataset) * _num_generations // args.gradient_accumulation_steps) * args.num_train_epochs

    if args.adamw_only:
        optimizer_label = "AdamW"
    elif args.use_muon:
        optimizer_label = "Muon+AdamW"
    elif args.use_fp_muon:
        optimizer_label = f"FeaturePreservingMuon(energy={args.fp_energy_threshold})+AdamW"
    elif args.use_comp_muon:
        optimizer_label = f"CompositionPreservingMuon(qk_e={args.comp_qk_energy_threshold},ov_e={args.comp_ov_energy_threshold})+AdamW"
    elif args.use_soft_comp_muon:
        optimizer_label = (
            f"SoftCompPreservingMuon(qk_e={args.soft_comp_qk_energy_threshold},"
            f"ov_e={args.soft_comp_ov_energy_threshold})"
            f"+FPMuon(e={args.soft_comp_fp_energy_threshold})+AdamW"
        )
    else:
        optimizer_label = "ProjectedGradient"
    print(f"[DFT] Estimated steps: {estimated_steps}, save_strategy={args.save_strategy}, optimizer: {optimizer_label}")

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
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        max_grad_norm=1,
        sync_ref_model=(args.method == "sdft"),
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        report_to="wandb",
        run_name=f"exp{args.exp_id}_dft_{task_name}",
        log_completions=False,
    )

    if args.adamw_only:
        optimizers = (None, None)
    elif args.use_muon:
        optimizer = create_muon_optimizer(model, args)
        optimizers = (optimizer, None)
    elif args.use_fp_muon:
        from transformers import get_constant_schedule_with_warmup
        optimizer = create_fp_muon_optimizer(model, args)
        # Linear warmup then constant LR (no decay) for fp_muon.
        # Bound directly to fp_muon's param_groups to bypass AcceleratedOptimizer
        # wrapping that breaks param_group reference sharing.
        warmup_steps = max(1, int(estimated_steps * 0.1))
        scheduler = get_constant_schedule_with_warmup(
            optimizer.fp_muon,
            num_warmup_steps=warmup_steps,
        )
        optimizers = (optimizer, scheduler)
    elif args.use_comp_muon:
        from transformers import get_cosine_schedule_with_warmup
        optimizer = create_comp_muon_optimizer(model, args)
        warmup_steps = max(1, int(estimated_steps * 0.1))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer.comp_muon,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, estimated_steps),
        )
        optimizers = (optimizer, scheduler)
    elif args.use_soft_comp_muon:
        from transformers import get_cosine_schedule_with_warmup
        optimizer = create_soft_comp_muon_optimizer(model, args)
        warmup_steps = max(1, int(estimated_steps * 0.1))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer.soft_comp,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, estimated_steps),
        )
        optimizers = (optimizer, scheduler)
    else:
        optimizer = create_projected_gradient_optimizer(model, args)
        optimizers = (optimizer, None)

    trainer = DistilTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        optimizers=optimizers,
    )

    if not args.adamw_only and not args.use_muon and not args.use_fp_muon \
            and not args.use_comp_muon and not args.use_soft_comp_muon:
        trainer.add_callback(ProjectedOptimizerMetricsCallback(optimizer))
    elif args.use_fp_muon:
        trainer.add_callback(FPMuonMetricsCallback(optimizer.fp_muon))
    elif args.use_comp_muon:
        trainer.add_callback(CompMuonMetricsCallback(optimizer.comp_muon))
        trainer.add_callback(FPMuonMetricsCallback(optimizer.fp_muon))
    elif args.use_soft_comp_muon:
        trainer.add_callback(SoftCompMuonMetricsCallback(optimizer.soft_comp))
        trainer.add_callback(FPMuonMetricsCallback(optimizer.fp_muon))

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def train_sft(model, tokenizer, train_dataset, task_name, args):
    """Train using SFT with Projected Gradient."""
    def format_for_sft(example):
        prompt = example["prompt"]
        response = example["response"]
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        messages.append({"role": "assistant", "content": response})
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode(text, truncation=True, max_length=2048)
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        return {"text": text}

    sft_dataset = train_dataset.map(format_for_sft, remove_columns=train_dataset.column_names)

    estimated_steps = (len(sft_dataset) // args.gradient_accumulation_steps) * args.num_train_epochs
    if args.adamw_only:
        optimizer_label = "AdamW"
    elif args.use_fp_muon:
        optimizer_label = f"FeaturePreservingMuon(energy={args.fp_energy_threshold})+AdamW"
    else:
        optimizer_label = "ProjectedGradient"
    print(f"[SFT] Estimated steps: {estimated_steps}, optimizer: {optimizer_label}")

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
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        max_grad_norm=1,
        report_to="wandb",
        run_name=f"exp{args.exp_id}_sft_{task_name}",
        seed=args.seed,
    )

    if args.adamw_only:
        optimizers = (None, None)
        callbacks = []
    elif args.use_fp_muon:
        optimizer = create_fp_muon_optimizer(model, args)
        optimizers = (optimizer, None)
        callbacks = [FPMuonMetricsCallback(optimizer.fp_muon)]
    else:
        optimizer = create_projected_gradient_optimizer(model, args)
        optimizers = (optimizer, None)
        callbacks = [ProjectedOptimizerMetricsCallback(optimizer)]

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


def _build_hyperparams(args) -> dict:
    base = {"learning_rate": args.learning_rate, "method": args.method}
    if args.use_soft_comp_muon:
        return {**base,
                "optimizer": "SoftCompPreservingMuon+FPMuon",
                "soft_comp_qk_energy_threshold": args.soft_comp_qk_energy_threshold,
                "soft_comp_ov_energy_threshold": args.soft_comp_ov_energy_threshold,
                "soft_comp_admm_steps": args.soft_comp_admm_steps,
                "soft_comp_dual_lr": args.soft_comp_dual_lr,
                "soft_comp_fp_energy_threshold": args.soft_comp_fp_energy_threshold,
                "soft_comp_adamw_lr": args.soft_comp_adamw_lr}
    if args.use_comp_muon:
        return {**base,
                "optimizer": "CompositionPreservingMuon",
                "comp_qk_energy_threshold": args.comp_qk_energy_threshold,
                "comp_ov_energy_threshold": args.comp_ov_energy_threshold,
                "comp_adamw_lr": args.comp_adamw_lr}
    if args.use_fp_muon:
        return {**base,
                "optimizer": "FeaturePreservingMuon",
                "fp_energy_threshold": args.fp_energy_threshold,
                "fp_vr_bypass_alpha": args.fp_vr_bypass_alpha,
                "fp_n_dual_iter": args.fp_n_dual_iter,
                "fp_dual_step_size": args.fp_dual_step_size,
                "fp_adamw_lr": args.fp_adamw_lr}
    if args.use_muon:
        return {**base,
                "optimizer": "Muon",
                "muon_lr": args.muon_lr,
                "muon_momentum": args.muon_momentum,
                "muon_adamw_lr": args.muon_adamw_lr}
    if args.adamw_only:
        return {**base, "optimizer": "AdamW"}
    return {**base,
            "optimizer": "ProjectedGradient",
            "energy_threshold": args.proj_energy_threshold,
            "stiefel_update": args.stiefel_update,
            "stiefel_max_rms": args.stiefel_max_rms,
            "uv_lr_scale": args.uv_lr_scale,
            "b_lr_scale": args.b_lr_scale,
            "admm_steps": args.proj_admm_steps}


def main():
    args = parse_args()

    setup_distributed_port()

    if args.exp_id is None:
        args.exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    method_display = {"dft": "DFT", "sdft": "SDFT", "sft": "SFT"}[args.method]

    print(f"\n{'='*60}")
    print(f"Training: {method_display} on {args.task}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    if args.use_fp_muon:
        print(f"Optimizer: FeaturePreservingMuon (energy={args.fp_energy_threshold}, lr={args.learning_rate})")
    elif args.use_comp_muon:
        print(f"Optimizer: CompositionPreservingMuon (qk_energy={args.comp_qk_energy_threshold}, ov_energy={args.comp_ov_energy_threshold}, lr={args.learning_rate})")
    elif args.use_soft_comp_muon:
        print(f"Optimizer: SoftCompPreservingMuon+FPMuon "
              f"(qk_energy={args.soft_comp_qk_energy_threshold}, "
              f"ov_energy={args.soft_comp_ov_energy_threshold}, "
              f"admm_steps={args.soft_comp_admm_steps}, dual_lr={args.soft_comp_dual_lr}, "
              f"mlp_energy={args.soft_comp_fp_energy_threshold}, lr={args.learning_rate})")
    else:
        print(f"Optimizer: ProjectedGradient (energy={args.proj_energy_threshold})")
    print(f"Hyperparams: lr={args.learning_rate}, stiefel={args.stiefel_update}")
    print(f"{'='*60}\n")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load ref model for DFT / SDFT
    ref_model = None
    if args.method in ("dft", "sdft"):
        print("Loading reference model...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Evaluate before training (skip if --skip_before_eval)
    device = next(model.parameters()).device
    before_results = {}
    if not args.skip_before_eval:
        if args.eval_all_tasks:
            task_order = args.eval_tasks.split(",") if args.eval_tasks else None
            print("\n[Eval] Before training (all tasks):")
            before_results = evaluate_all_tasks_extended(model, tokenizer, device, args, task_order=task_order)
        else:
            print("\n[Eval] Before training (current task only):")
            _, eval_ds = load_task_dataset(args.task, seed=args.seed)
            before_results = {args.task: evaluate_task(
                args.task, model, tokenizer, eval_ds, device,
                max_samples=args.eval_max_samples
            )}
            print(f"  {args.task}: {before_results[args.task]['accuracy']:.2%}")
    else:
        print("\n[Eval] Skipping before-training evaluation")

    # Load task dataset
    train_dataset, eval_dataset = load_task_dataset(args.task, seed=args.seed, max_samples=args.max_samples)
    print(f"\nTraining on {len(train_dataset)} examples")

    # Train
    if args.method in ("dft", "sdft"):
        train_dft(model, ref_model, tokenizer, train_dataset, args.task, args)
    else:
        train_sft(model, tokenizer, train_dataset, args.task, args)

    # Cleanup and reload for evaluation
    del model
    if ref_model is not None:
        del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.output_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Evaluate after training
    if args.eval_all_tasks:
        task_order = args.eval_tasks.split(",") if args.eval_tasks else None
        print("\n[Eval] After training (all tasks):")
        after_results = evaluate_all_tasks_extended(model, tokenizer, device, args, task_order=task_order)
    else:
        print("\n[Eval] After training (current task):")
        after_results = {args.task: evaluate_task(
            args.task, model, tokenizer, eval_dataset, device,
            max_samples=args.eval_max_samples
        )}
        print(f"  {args.task}: {after_results[args.task]['accuracy']:.2%}")

    # Full evaluation on training task
    print(f"\n[Eval] Full {args.task} test set evaluation:")
    task_full_results = evaluate_task(
        args.task, model, tokenizer, eval_dataset, device,
        max_samples=min(100, len(eval_dataset)), verbose=False
    )
    print(f"  {args.task} full: {task_full_results['accuracy']:.2%} ({task_full_results['correct']}/{task_full_results['total']})")

    # Log to wandb
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                f"eval/{args.task}_full_accuracy": task_full_results["accuracy"],
                f"eval/{args.task}_full_correct": task_full_results["correct"],
            })
            wandb.run.summary[f"{args.task}_full_accuracy"] = task_full_results["accuracy"]
    except Exception as e:
        print(f"  Warning: Could not log to wandb: {e}")

    # Save results
    results = {
        "task": args.task,
        "method": args.method,
        "model": args.model_name,
        "hyperparams": _build_hyperparams(args),
        "before": {k: {"accuracy": v.get("accuracy", 0), "correct": v.get("correct", 0), "total": v.get("total", 0)} for k, v in before_results.items()},
        "after": {k: {"accuracy": v.get("accuracy", 0), "correct": v.get("correct", 0), "total": v.get("total", 0)} for k, v in after_results.items()},
        f"{args.task}_full": {
            "accuracy": task_full_results["accuracy"],
            "correct": task_full_results["correct"],
            "total": task_full_results["total"],
        },
    }

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()

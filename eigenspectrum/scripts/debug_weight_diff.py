#!/usr/bin/env python3
"""
Debug script to verify weights are actually different between checkpoints.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eigenspectrum.scripts.compare_sft_sdft_subspaces import (
    load_weight, svd, alignment_score, get_checkpoint_paths, PARAM_SHORT
)

DATA_ROOT = "/data/saket/continual/Self-Distillation"


def debug_single_param(layer: int = 0, param: str = 'self_attn.q_proj.weight', data_root: str = DATA_ROOT):
    """Debug weight loading and alignment for a single param."""

    paths = get_checkpoint_paths(data_root)
    labels = ['base', 'tooluse', 'gsm8k', 'mbpp']

    print(f"\n{'='*70}")
    print(f"Layer {layer}, {PARAM_SHORT[param]}")
    print('='*70)

    # Load all weights for SFT
    print("\nLoading SFT checkpoints...")
    sft_weights = []
    for i, (label, path) in enumerate(zip(labels, paths['sft'])):
        W = load_weight(path, layer, param)
        sft_weights.append(W)
        print(f"  {label}: shape={W.shape}, norm={np.linalg.norm(W):.4f}, "
              f"mean={W.mean():.6f}, std={W.std():.6f}")

    # Load all weights for SDFT
    print("\nLoading SDFT checkpoints...")
    sdft_weights = []
    for i, (label, path) in enumerate(zip(labels, paths['sdft'])):
        W = load_weight(path, layer, param)
        sdft_weights.append(W)
        print(f"  {label}: shape={W.shape}, norm={np.linalg.norm(W):.4f}, "
              f"mean={W.mean():.6f}, std={W.std():.6f}")

    # Check if weights are actually different
    print("\n" + "-"*70)
    print("Weight differences (Frobenius norm of W_i - W_j):")
    print("-"*70)

    print("\nSFT checkpoints:")
    for i in range(4):
        diffs = []
        for j in range(4):
            diff = np.linalg.norm(sft_weights[i] - sft_weights[j])
            diffs.append(f"{diff:.4f}")
        print(f"  {labels[i]:>8}: " + " ".join(f"{d:>10}" for d in diffs))

    print("\nSDFT checkpoints:")
    for i in range(4):
        diffs = []
        for j in range(4):
            diff = np.linalg.norm(sdft_weights[i] - sdft_weights[j])
            diffs.append(f"{diff:.4f}")
        print(f"  {labels[i]:>8}: " + " ".join(f"{d:>10}" for d in diffs))

    # Check SFT base vs SDFT base (should be identical)
    print(f"\nSFT base vs SDFT base: {np.linalg.norm(sft_weights[0] - sdft_weights[0]):.6f}")

    # Compute SVDs and check alignment
    print("\n" + "-"*70)
    print("SVD Analysis:")
    print("-"*70)

    sft_svds = [svd(W, gpu=True) for W in sft_weights]
    sdft_svds = [svd(W, gpu=True) for W in sdft_weights]

    # Check if U is orthonormal
    print("\nOrthonormality check (should be ~0):")
    for i, label in enumerate(labels):
        U = sft_svds[i][0]
        err = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))
        print(f"  SFT {label}: ||U'U - I|| = {err:.6e}")

    # Alignment matrix for base vs mbpp
    print("\n" + "-"*70)
    print("Alignment Analysis (base vs mbpp):")
    print("-"*70)

    U_base = sft_svds[0][0]
    U_mbpp = sft_svds[3][0]

    print(f"\nU_base shape: {U_base.shape}")
    print(f"U_mbpp shape: {U_mbpp.shape}")

    # Full alignment matrix
    k = 50
    A = U_base[:, :k].T @ U_mbpp[:, :k]

    print(f"\nAlignment matrix A = U_base[:,:k].T @ U_mbpp[:,:k] (k={k}):")
    print(f"  Shape: {A.shape}")
    print(f"  Diagonal (first 10): {np.diag(A)[:10]}")
    print(f"  |Diagonal| (first 10): {np.abs(np.diag(A))[:10]}")
    print(f"  Mean |diagonal|: {np.mean(np.abs(np.diag(A))):.6f}")
    print(f"  Min |diagonal|: {np.min(np.abs(np.diag(A))):.6f}")
    print(f"  Max |diagonal|: {np.max(np.abs(np.diag(A))):.6f}")

    # Check off-diagonal
    mask = ~np.eye(k, dtype=bool)
    off_diag = np.abs(A[mask])
    print(f"\n  Mean |off-diagonal|: {np.mean(off_diag):.6f}")
    print(f"  Max |off-diagonal|: {np.max(off_diag):.6f}")

    # Singular values of A (should all be 1 if subspaces identical)
    s = np.linalg.svd(A, compute_uv=False)
    print(f"\n  Singular values of A (first 10): {s[:10]}")
    print(f"  Singular values of A (last 10): {s[-10:]}")

    # Compare singular VALUES (not vectors)
    print("\n" + "-"*70)
    print("Singular Value Changes:")
    print("-"*70)

    S_base = sft_svds[0][1]
    S_mbpp = sft_svds[3][1]

    print(f"\nSFT Base singular values (first 10): {S_base[:10]}")
    print(f"SFT MBPP singular values (first 10): {S_mbpp[:10]}")
    print(f"Relative change (first 10): {(S_mbpp[:10] - S_base[:10]) / S_base[:10]}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--param', type=str, default='self_attn.q_proj.weight')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    args = parser.parse_args()

    debug_single_param(args.layer, args.param, args.data_root)

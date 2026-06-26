"""
Singular Vector Alignment Analysis

Analyzes how singular vectors U, V of a weight matrix drift over training
using alignment matrices:
    A_U(t) = U_0^T @ U_t
    A_V(t) = V_0^T @ V_t

If training preserves singular directions, these should be close to identity
(or signed permutation matrices if directions swap/flip).

Key insight: Singular vectors are defined only up to sign. For repeated
singular values, they're defined only up to rotation within the eigenspace.
We handle this via:
1. Absolute value alignment |A_U|, |A_V|
2. Sign correction that maximizes diagonal
3. Permutation-aware metrics using Hungarian algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SVDResult:
    """SVD decomposition of a matrix."""
    U: np.ndarray      # Left singular vectors (m x k)
    S: np.ndarray      # Singular values (k,)
    Vt: np.ndarray     # Right singular vectors transposed (k x n)

    @property
    def V(self) -> np.ndarray:
        return self.Vt.T

    @property
    def rank(self) -> int:
        return len(self.S)


@dataclass
class AlignmentResult:
    """Alignment analysis between reference and target SVD."""
    # Raw alignment matrices
    A_U: np.ndarray  # U_ref^T @ U_target
    A_V: np.ndarray  # V_ref^T @ V_target

    # Sign-corrected alignment (maximize diagonal)
    A_U_corrected: np.ndarray
    A_V_corrected: np.ndarray

    # Scalar metrics
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrajectoryResult:
    """Results for a full training trajectory."""
    checkpoint_ids: List[str]
    alignments: List[AlignmentResult]
    reference_svd: SVDResult
    metrics_over_time: Dict[str, List[float]] = field(default_factory=dict)


# =============================================================================
# Core SVD and Alignment Functions
# =============================================================================

def compute_svd(
    W: np.ndarray,
    full_matrices: bool = False,
    use_gpu: bool = False,
) -> SVDResult:
    """
    Compute SVD of a weight matrix.

    Args:
        W: 2D array (m x n)
        full_matrices: If False, returns truncated SVD
        use_gpu: Use torch GPU SVD if available

    Returns:
        SVDResult with U, S, Vt
    """
    if use_gpu:
        import torch
        if torch.cuda.is_available():
            W_torch = torch.tensor(W, dtype=torch.float32, device='cuda')
            U, S, Vh = torch.linalg.svd(W_torch, full_matrices=full_matrices)
            return SVDResult(
                U=U.cpu().numpy(),
                S=S.cpu().numpy(),
                Vt=Vh.cpu().numpy(),
            )

    # CPU fallback
    W = np.asarray(W, dtype=np.float64)
    U, S, Vt = np.linalg.svd(W, full_matrices=full_matrices)
    return SVDResult(U=U, S=S, Vt=Vt)


def compute_alignment_matrices(
    svd_ref: SVDResult,
    svd_target: SVDResult,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute alignment matrices between reference and target SVDs.

    A_U = U_ref^T @ U_target  (shape: k x k)
    A_V = V_ref^T @ V_target  (shape: k x k)

    If singular vectors haven't moved, A_U and A_V should be close to
    identity (or signed identity due to sign ambiguity).

    Args:
        svd_ref: Reference SVD (usually t=0)
        svd_target: Target SVD (checkpoint t)
        top_k: Only consider top-k singular directions

    Returns:
        (A_U, A_V) alignment matrices
    """
    k = min(svd_ref.rank, svd_target.rank)
    if top_k is not None:
        k = min(k, top_k)

    U_ref = svd_ref.U[:, :k]
    U_target = svd_target.U[:, :k]
    V_ref = svd_ref.V[:, :k]
    V_target = svd_target.V[:, :k]

    A_U = U_ref.T @ U_target
    A_V = V_ref.T @ V_target

    return A_U, A_V


def correct_signs(A: np.ndarray) -> np.ndarray:
    """
    Correct sign ambiguity to maximize diagonal entries.

    Singular vectors are defined up to sign: if u is a singular vector,
    so is -u. This means A_U[i,i] could be +1 or -1 for perfect alignment.

    We flip signs of columns to make diagonal entries positive.

    Args:
        A: Alignment matrix (k x k)

    Returns:
        Sign-corrected alignment matrix
    """
    A_corrected = A.copy()
    # For each column, if diagonal is negative, flip the column
    for j in range(A.shape[1]):
        if A_corrected[j, j] < 0:
            A_corrected[:, j] *= -1
    return A_corrected


def correct_signs_optimal(A: np.ndarray) -> np.ndarray:
    """
    Optimal sign correction using greedy approach on absolute diagonal.

    For each singular vector pair, choose sign that maximizes |A[i,i]|.
    This is equivalent to: sign_j = sign(A[j,j])
    """
    signs = np.sign(np.diag(A))
    signs[signs == 0] = 1  # Handle exact zeros
    return A * signs[np.newaxis, :]


# =============================================================================
# Alignment Metrics
# =============================================================================

def compute_alignment_metrics(
    A: np.ndarray,
    top_k: Optional[int] = None,
    use_absolute: bool = True,
) -> Dict[str, float]:
    """
    Compute scalar summary metrics from alignment matrix.

    Args:
        A: Alignment matrix (k x k)
        top_k: Focus on top-k x top-k block
        use_absolute: Use |A| for metrics (handles sign ambiguity)

    Returns:
        Dictionary of metrics
    """
    if top_k is not None:
        A = A[:top_k, :top_k]

    k = A.shape[0]

    # Use absolute values to handle sign ambiguity
    A_abs = np.abs(A) if use_absolute else A

    # Diagonal entries (should be close to 1 for preserved directions)
    diag = np.diag(A_abs)
    mean_diag = float(np.mean(diag))
    min_diag = float(np.min(diag))
    max_diag = float(np.max(diag))

    # Off-diagonal entries (should be close to 0)
    mask = ~np.eye(k, dtype=bool)
    off_diag = A_abs[mask]
    sum_off_diag = float(np.sum(off_diag))
    mean_off_diag = float(np.mean(off_diag))
    max_off_diag = float(np.max(off_diag))

    # Diagonal mass ratio: sum(|diag|) / sum(|A|)
    total_mass = float(np.sum(A_abs))
    diag_mass = float(np.sum(diag))
    diag_ratio = diag_mass / total_mass if total_mass > 0 else 0.0

    # Frobenius distance from identity
    I = np.eye(k)
    frob_dist_identity = float(np.linalg.norm(A_abs - I, 'fro'))

    # Effective rank of alignment (should be k for perfect alignment)
    # Using nuclear norm / spectral norm
    nuclear = float(np.sum(np.linalg.svd(A, compute_uv=False)))
    spectral = float(np.max(np.linalg.svd(A, compute_uv=False)))
    effective_rank = nuclear / spectral if spectral > 0 else 0.0

    return {
        'mean_diag': mean_diag,
        'min_diag': min_diag,
        'max_diag': max_diag,
        'sum_off_diag': sum_off_diag,
        'mean_off_diag': mean_off_diag,
        'max_off_diag': max_off_diag,
        'diag_ratio': diag_ratio,
        'frob_dist_identity': frob_dist_identity,
        'effective_rank': effective_rank,
    }


def compute_permutation_aware_alignment(A: np.ndarray) -> Dict[str, float]:
    """
    Compute alignment score accounting for possible permutations.

    If singular values are close, their vectors might swap order.
    We use the Hungarian algorithm to find optimal matching.

    Args:
        A: Alignment matrix (k x k)

    Returns:
        Permutation-aware metrics
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Fallback without scipy
        return {'permutation_score': float(np.mean(np.abs(np.diag(A))))}

    k = A.shape[0]
    # Cost matrix: we want to maximize |A[i,j]|, so minimize -|A|
    cost = -np.abs(A)

    row_ind, col_ind = linear_sum_assignment(cost)

    # Optimal matching score
    matched_values = np.abs(A[row_ind, col_ind])
    permutation_score = float(np.mean(matched_values))

    # How far is the matching from identity?
    # Count how many indices are not on diagonal
    num_swaps = int(np.sum(row_ind != col_ind))

    return {
        'permutation_score': permutation_score,
        'num_swaps': num_swaps,
        'swap_fraction': num_swaps / k,
    }


# =============================================================================
# Full Alignment Analysis
# =============================================================================

def analyze_alignment(
    svd_ref: SVDResult,
    svd_target: SVDResult,
    top_k: int = 50,
) -> AlignmentResult:
    """
    Full alignment analysis between reference and target SVDs.

    Args:
        svd_ref: Reference SVD (t=0)
        svd_target: Target SVD (checkpoint t)
        top_k: Number of top singular directions to analyze

    Returns:
        AlignmentResult with matrices and metrics
    """
    # Compute raw alignment matrices
    A_U, A_V = compute_alignment_matrices(svd_ref, svd_target, top_k=top_k)

    # Sign-corrected versions
    A_U_corrected = correct_signs(A_U)
    A_V_corrected = correct_signs(A_V)

    # Compute metrics
    metrics = {}

    # U metrics
    u_metrics = compute_alignment_metrics(A_U_corrected, use_absolute=True)
    for key, val in u_metrics.items():
        metrics[f'U_{key}'] = val

    # V metrics
    v_metrics = compute_alignment_metrics(A_V_corrected, use_absolute=True)
    for key, val in v_metrics.items():
        metrics[f'V_{key}'] = val

    # Permutation-aware metrics
    u_perm = compute_permutation_aware_alignment(A_U)
    v_perm = compute_permutation_aware_alignment(A_V)
    metrics['U_permutation_score'] = u_perm['permutation_score']
    metrics['V_permutation_score'] = v_perm['permutation_score']
    metrics['U_num_swaps'] = u_perm.get('num_swaps', 0)
    metrics['V_num_swaps'] = v_perm.get('num_swaps', 0)

    return AlignmentResult(
        A_U=A_U,
        A_V=A_V,
        A_U_corrected=A_U_corrected,
        A_V_corrected=A_V_corrected,
        metrics=metrics,
    )


# =============================================================================
# Trajectory Analysis
# =============================================================================

def analyze_trajectory(
    matrices: List[np.ndarray],
    checkpoint_ids: Optional[List[str]] = None,
    reference_idx: int = 0,
    top_k: int = 50,
    use_gpu: bool = False,
) -> TrajectoryResult:
    """
    Analyze singular vector alignment across a training trajectory.

    Args:
        matrices: List of weight matrices W_t at each checkpoint
        checkpoint_ids: Names/labels for each checkpoint
        reference_idx: Index of reference checkpoint (default: 0)
        top_k: Number of top singular directions to track
        use_gpu: Use GPU for SVD computation

    Returns:
        TrajectoryResult with all alignments and time series
    """
    n_checkpoints = len(matrices)

    if checkpoint_ids is None:
        checkpoint_ids = [f't={i}' for i in range(n_checkpoints)]

    # Compute reference SVD
    svd_ref = compute_svd(matrices[reference_idx], use_gpu=use_gpu)

    # Analyze each checkpoint
    alignments = []
    metrics_over_time = {}

    for t, W_t in enumerate(matrices):
        svd_t = compute_svd(W_t, use_gpu=use_gpu)
        alignment = analyze_alignment(svd_ref, svd_t, top_k=top_k)
        alignments.append(alignment)

        # Collect metrics over time
        for key, val in alignment.metrics.items():
            if key not in metrics_over_time:
                metrics_over_time[key] = []
            metrics_over_time[key].append(val)

    return TrajectoryResult(
        checkpoint_ids=checkpoint_ids,
        alignments=alignments,
        reference_svd=svd_ref,
        metrics_over_time=metrics_over_time,
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_alignment_heatmap(
    A: np.ndarray,
    title: str = "Alignment Matrix",
    ax: Optional[plt.Axes] = None,
    cmap: str = 'RdBu_r',
    vmin: float = -1.0,
    vmax: float = 1.0,
    show_colorbar: bool = True,
    annotate: bool = False,
) -> plt.Axes:
    """
    Plot heatmap of alignment matrix.

    Args:
        A: Alignment matrix (k x k)
        title: Plot title
        ax: Matplotlib axes (created if None)
        cmap: Colormap (RdBu_r centers 0 at white)
        vmin, vmax: Color scale limits
        show_colorbar: Whether to show colorbar
        annotate: Whether to annotate cells with values

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(A, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if annotate and A.shape[0] <= 20:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if abs(A[i,j]) < 0.5 else 'white')

    ax.set_xlabel('Target singular vector index')
    ax.set_ylabel('Reference singular vector index')
    ax.set_title(title)

    return ax


def plot_alignment_comparison(
    alignment: AlignmentResult,
    top_k: int = 50,
    checkpoint_id: str = "",
    save_path: Optional[str] = None,
):
    """
    Plot side-by-side comparison of U and V alignment matrices.

    Shows both raw and sign-corrected versions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    k = min(top_k, alignment.A_U.shape[0])

    # Raw alignments
    plot_alignment_heatmap(
        alignment.A_U[:k, :k],
        title=f'A_U (raw) - {checkpoint_id}',
        ax=axes[0, 0],
    )
    plot_alignment_heatmap(
        alignment.A_V[:k, :k],
        title=f'A_V (raw) - {checkpoint_id}',
        ax=axes[0, 1],
    )

    # Sign-corrected (absolute value for visualization)
    plot_alignment_heatmap(
        np.abs(alignment.A_U_corrected[:k, :k]),
        title=f'|A_U| (sign-corrected) - {checkpoint_id}',
        ax=axes[1, 0],
        vmin=0, vmax=1,
        cmap='viridis',
    )
    plot_alignment_heatmap(
        np.abs(alignment.A_V_corrected[:k, :k]),
        title=f'|A_V| (sign-corrected) - {checkpoint_id}',
        ax=axes[1, 1],
        vmin=0, vmax=1,
        cmap='viridis',
    )

    # Add metrics as text
    metrics = alignment.metrics
    text = (
        f"U: diag={metrics['U_mean_diag']:.3f}, "
        f"off-diag={metrics['U_mean_off_diag']:.3f}, "
        f"ratio={metrics['U_diag_ratio']:.3f}\n"
        f"V: diag={metrics['V_mean_diag']:.3f}, "
        f"off-diag={metrics['V_mean_off_diag']:.3f}, "
        f"ratio={metrics['V_diag_ratio']:.3f}"
    )
    fig.text(0.5, 0.02, text, ha='center', fontsize=10, family='monospace')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_trajectory_metrics(
    trajectory: TrajectoryResult,
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot scalar alignment metrics over training trajectory.

    Args:
        trajectory: TrajectoryResult from analyze_trajectory
        metrics_to_plot: Which metrics to plot (default: key metrics)
        save_path: Path to save figure
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            'U_mean_diag', 'V_mean_diag',
            'U_diag_ratio', 'V_diag_ratio',
            'U_frob_dist_identity', 'V_frob_dist_identity',
            'U_permutation_score', 'V_permutation_score',
        ]

    # Filter to available metrics
    metrics_to_plot = [m for m in metrics_to_plot if m in trajectory.metrics_over_time]

    n_metrics = len(metrics_to_plot)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    x = range(len(trajectory.checkpoint_ids))

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = trajectory.metrics_over_time[metric_name]

        color = 'blue' if metric_name.startswith('U_') else 'red'
        ax.plot(x, values, 'o-', color=color, linewidth=2, markersize=4)

        ax.set_xlabel('Checkpoint')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)

        # Reference line at 1.0 for diagonal metrics
        if 'diag' in metric_name.lower() or 'ratio' in metric_name.lower():
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Hide unused axes
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_trajectory_heatmaps(
    trajectory: TrajectoryResult,
    checkpoint_indices: Optional[List[int]] = None,
    top_k: int = 30,
    which: str = 'U',  # 'U', 'V', or 'both'
    save_path: Optional[str] = None,
):
    """
    Plot alignment heatmaps for selected checkpoints.

    Args:
        trajectory: TrajectoryResult from analyze_trajectory
        checkpoint_indices: Which checkpoints to plot (default: evenly spaced)
        top_k: Size of alignment block to show
        which: 'U', 'V', or 'both'
        save_path: Path to save figure
    """
    n_checkpoints = len(trajectory.alignments)

    if checkpoint_indices is None:
        # Select ~6 evenly spaced checkpoints
        if n_checkpoints <= 6:
            checkpoint_indices = list(range(n_checkpoints))
        else:
            checkpoint_indices = np.linspace(0, n_checkpoints - 1, 6, dtype=int).tolist()

    n_plots = len(checkpoint_indices)

    if which == 'both':
        fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8))
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        axes = axes.reshape(1, -1)

    for col, t in enumerate(checkpoint_indices):
        alignment = trajectory.alignments[t]
        checkpoint_id = trajectory.checkpoint_ids[t]

        if which in ['U', 'both']:
            row = 0 if which == 'both' else 0
            A = np.abs(alignment.A_U_corrected[:top_k, :top_k])
            plot_alignment_heatmap(
                A, title=f'|A_U| @ {checkpoint_id}',
                ax=axes[row, col], vmin=0, vmax=1, cmap='viridis',
                show_colorbar=(col == n_plots - 1)
            )

        if which in ['V', 'both']:
            row = 1 if which == 'both' else 0
            A = np.abs(alignment.A_V_corrected[:top_k, :top_k])
            plot_alignment_heatmap(
                A, title=f'|A_V| @ {checkpoint_id}',
                ax=axes[row, col], vmin=0, vmax=1, cmap='viridis',
                show_colorbar=(col == n_plots - 1)
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Checkpoint Loading (Modular Interface)
# =============================================================================

def load_matrices_from_checkpoints(
    checkpoint_paths: List[str],
    layer_idx: int,
    param_name: str,
    loader_fn: Optional[Callable] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load weight matrices from a sequence of checkpoints.

    Args:
        checkpoint_paths: List of paths to checkpoints
        layer_idx: Which transformer layer
        param_name: Parameter name (e.g., 'self_attn.q_proj.weight')
        loader_fn: Custom function(path, layer_idx, param_name) -> np.ndarray
                   If None, uses default HuggingFace loader

    Returns:
        (matrices, checkpoint_ids)
    """
    if loader_fn is None:
        loader_fn = _default_hf_loader

    matrices = []
    checkpoint_ids = []

    for path in checkpoint_paths:
        W = loader_fn(path, layer_idx, param_name)
        matrices.append(np.asarray(W, dtype=np.float64))
        checkpoint_ids.append(Path(path).name)

    return matrices, checkpoint_ids


def _default_hf_loader(
    checkpoint_path: str,
    layer_idx: int,
    param_name: str,
) -> np.ndarray:
    """
    Default loader for HuggingFace model checkpoints.

    Uses state_dict loading for efficiency (avoids full model instantiation).
    """
    import torch
    from safetensors import safe_open
    import glob
    import os

    path = Path(checkpoint_path)

    # Build the full parameter name
    full_param_name = f"model.layers.{layer_idx}.{param_name}"

    # Try to load from safetensors (faster) or pytorch bin
    if path.exists() and path.is_dir():
        # Local checkpoint
        safetensor_files = list(path.glob("*.safetensors"))
        pytorch_files = list(path.glob("*.bin")) + list(path.glob("pytorch_model*.bin"))

        if safetensor_files:
            # Load from safetensors (memory-efficient, loads only requested tensors)
            for sf_path in safetensor_files:
                with safe_open(sf_path, framework="pt", device="cpu") as f:
                    if full_param_name in f.keys():
                        W = f.get_tensor(full_param_name).float().numpy()
                        return W
            raise KeyError(f"Parameter {full_param_name} not found in safetensors files")

        elif pytorch_files:
            # Load from pytorch bin (loads full state dict)
            for pt_path in pytorch_files:
                state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)
                if full_param_name in state_dict:
                    W = state_dict[full_param_name].float().numpy()
                    del state_dict
                    return W
            raise KeyError(f"Parameter {full_param_name} not found in pytorch files")

        else:
            raise FileNotFoundError(f"No model files found in {path}")

    else:
        # HuggingFace hub - need to download/cache and load
        from huggingface_hub import hf_hub_download, list_repo_files

        try:
            # Try safetensors first
            files = list_repo_files(checkpoint_path)
            safetensor_files = [f for f in files if f.endswith('.safetensors')]

            if safetensor_files:
                for sf_file in safetensor_files:
                    local_path = hf_hub_download(checkpoint_path, sf_file)
                    with safe_open(local_path, framework="pt", device="cpu") as f:
                        if full_param_name in f.keys():
                            W = f.get_tensor(full_param_name).float().numpy()
                            return W

            # Fallback to full model load
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map='cpu',
                trust_remote_code=True,
            )
            layer = model.model.layers[layer_idx]
            parts = param_name.split('.')
            obj = layer
            for part in parts[:-1]:
                obj = getattr(obj, part)
            weight = getattr(obj, parts[-1])
            W = weight.data.float().numpy()
            del model
            torch.cuda.empty_cache()
            return W

        except Exception as e:
            raise RuntimeError(f"Failed to load {full_param_name} from {checkpoint_path}: {e}")


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_full_analysis(
    checkpoint_paths: List[str],
    layer_idx: int,
    param_name: str,
    output_dir: str,
    top_k: int = 50,
    loader_fn: Optional[Callable] = None,
):
    """
    Run complete singular vector alignment analysis.

    Args:
        checkpoint_paths: Ordered list of checkpoint paths
        layer_idx: Transformer layer to analyze
        param_name: Weight matrix to analyze
        output_dir: Directory to save outputs
        top_k: Number of top singular directions
        loader_fn: Custom matrix loader function
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(checkpoint_paths)} checkpoints...")
    matrices, checkpoint_ids = load_matrices_from_checkpoints(
        checkpoint_paths, layer_idx, param_name, loader_fn
    )

    print(f"Matrix shape: {matrices[0].shape}")
    print(f"Analyzing top-{top_k} singular directions...")

    # Run trajectory analysis
    trajectory = analyze_trajectory(
        matrices,
        checkpoint_ids=checkpoint_ids,
        reference_idx=0,
        top_k=top_k,
    )

    # Save metrics to JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'checkpoint_ids': trajectory.checkpoint_ids,
            'metrics_over_time': trajectory.metrics_over_time,
            'layer_idx': layer_idx,
            'param_name': param_name,
            'top_k': top_k,
        }, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Plot metrics over time
    plot_trajectory_metrics(
        trajectory,
        save_path=output_dir / 'metrics_over_time.png',
    )

    # Plot heatmaps for selected checkpoints
    plot_trajectory_heatmaps(
        trajectory,
        top_k=min(top_k, 30),
        which='both',
        save_path=output_dir / 'alignment_heatmaps.png',
    )

    # Detailed plots for first and last checkpoints
    for idx in [0, len(trajectory.alignments) - 1]:
        plot_alignment_comparison(
            trajectory.alignments[idx],
            top_k=min(top_k, 30),
            checkpoint_id=trajectory.checkpoint_ids[idx],
            save_path=output_dir / f'alignment_detail_{idx}.png',
        )

    print(f"\nAnalysis complete. Results saved to {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    final = trajectory.alignments[-1].metrics
    print(f"\nFinal checkpoint alignment (vs reference):")
    print(f"  U: mean_diag={final['U_mean_diag']:.4f}, "
          f"diag_ratio={final['U_diag_ratio']:.4f}, "
          f"frob_dist={final['U_frob_dist_identity']:.4f}")
    print(f"  V: mean_diag={final['V_mean_diag']:.4f}, "
          f"diag_ratio={final['V_diag_ratio']:.4f}, "
          f"frob_dist={final['V_frob_dist_identity']:.4f}")

    return trajectory


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze singular vector alignment over training'
    )
    parser.add_argument(
        '--checkpoints', type=str, nargs='+', required=True,
        help='Paths to checkpoints in temporal order'
    )
    parser.add_argument(
        '--layer', type=int, required=True,
        help='Transformer layer index'
    )
    parser.add_argument(
        '--param', type=str, default='self_attn.q_proj.weight',
        help='Parameter name within layer'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--top_k', type=int, default=50,
        help='Number of top singular directions to analyze'
    )

    args = parser.parse_args()

    run_full_analysis(
        checkpoint_paths=args.checkpoints,
        layer_idx=args.layer,
        param_name=args.param,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

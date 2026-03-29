"""Block-wise eigenspectrum analyzer for LLMs."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

from .lanczos_torch import stochastic_lanczos_quadrature
from .hessian_torch import (
    get_hvp_fn_for_param,
    get_hvp_fn_for_param_sdft,
    get_hvp_fn_for_param_checkpointed,
    get_hvp_fn_for_param_sdft_checkpointed,
    list_weight_matrices,
    get_block_param_names,
)
from .density_numpy import tridiag_to_density, tridiag_to_eigv, effective_rank_from_eigv


@dataclass
class SpectrumResult:
    """Result of eigenspectrum analysis for a single weight matrix."""
    param_name: str
    param_shape: Tuple[int, ...]
    num_params: int
    eigenvalues: np.ndarray  # (num_samples, order)
    weights: np.ndarray  # (num_samples, order)
    density: np.ndarray  # (grid_len,)
    grids: np.ndarray  # (grid_len,)
    effective_rank: float
    top_eigenvalue: float
    trace_estimate: float

    def to_dict(self) -> dict:
        return {
            "param_name": self.param_name,
            "param_shape": list(self.param_shape),
            "num_params": self.num_params,
            "effective_rank": float(self.effective_rank),
            "top_eigenvalue": float(self.top_eigenvalue),
            "trace_estimate": float(self.trace_estimate),
        }


class SpectrumAnalyzer:
    """Analyze eigenspectrum of individual weight matrices in LLMs."""

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_batches: int = 10,
        ref_model: nn.Module = None,
        loss_type: str = "sft",
        use_gradient_checkpointing: bool = False,
    ):
        """Initialize the analyzer.

        Args:
            model: The model to analyze.
            dataloader: DataLoader providing training data for Hessian.
            device: Device for computation.
            dtype: Data type (float16 saves memory).
            max_batches: Number of batches for Hessian estimation.
            ref_model: Reference model for SDFT loss (required if loss_type='sdft').
            loss_type: 'sft' for cross-entropy, 'sdft' for KL divergence.
            use_gradient_checkpointing: If True, use gradient checkpointing with
                reverse-mode HVP. This significantly reduces memory usage but is
                ~2x slower due to recomputing activations.
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        self.max_batches = max_batches
        self.ref_model = ref_model
        self.loss_type = loss_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.model.eval()

        if loss_type == "sdft" and ref_model is None:
            raise ValueError("ref_model is required for SDFT loss")
        if ref_model is not None:
            self.ref_model.eval()

    def list_params(self, layer_idx: int = None) -> List[Tuple[str, Tuple, int]]:
        """List available parameters to analyze.

        Args:
            layer_idx: If provided, only list params from this layer.

        Returns:
            List of (name, shape, num_params) tuples, sorted by size.
        """
        return list_weight_matrices(self.model, layer_idx)

    def get_layer_blocks(self, layer_idx: int) -> Dict[str, List[str]]:
        """Get parameter names organized by block type.

        Args:
            layer_idx: Layer index.

        Returns:
            Dict mapping 'attention', 'mlp', 'layernorm' to param names.
        """
        return get_block_param_names(self.model, layer_idx)

    def analyze_param(
        self,
        param_name: str,
        lanczos_order: int = 50,
        num_samples: int = 5,
        sigma_squared: float = 1e-5,
        grid_len: int = 10000,
        verbose: bool = True,
    ) -> SpectrumResult:
        """Analyze eigenspectrum of a single parameter.

        Args:
            param_name: Full parameter name (e.g., 'model.layers.0.self_attn.q_proj.weight').
            lanczos_order: Number of Lanczos iterations.
            num_samples: Number of independent Lanczos runs to average.
            sigma_squared: Smoothing for density estimation.
            grid_len: Grid resolution for density.
            verbose: Print progress.

        Returns:
            SpectrumResult with eigenvalues, density, and summary statistics.
        """
        if verbose:
            print(f"Analyzing: {param_name} (loss_type={self.loss_type}, checkpointing={self.use_gradient_checkpointing})")

        # Get HVP function for this parameter
        if self.use_gradient_checkpointing:
            # Use reverse-mode HVP with gradient checkpointing (lower memory)
            if self.loss_type == "sdft":
                hvp_fn, num_params = get_hvp_fn_for_param_sdft_checkpointed(
                    self.model,
                    self.ref_model,
                    param_name,
                    self.dataloader,
                    max_batches=self.max_batches,
                    device=self.device,
                    gradient_checkpointing=True,
                )
            else:
                hvp_fn, num_params = get_hvp_fn_for_param_checkpointed(
                    self.model,
                    param_name,
                    self.dataloader,
                    max_batches=self.max_batches,
                    device=self.device,
                    gradient_checkpointing=True,
                )
        else:
            # Use forward-over-reverse HVP (faster but more memory)
            if self.loss_type == "sdft":
                hvp_fn, num_params = get_hvp_fn_for_param_sdft(
                    self.model,
                    self.ref_model,
                    param_name,
                    self.dataloader,
                    max_batches=self.max_batches,
                    device=self.device,
                )
            else:
                hvp_fn, num_params = get_hvp_fn_for_param(
                    self.model,
                    param_name,
                    self.dataloader,
                    max_batches=self.max_batches,
                    device=self.device,
                )

        if verbose:
            print(f"  Parameters: {num_params:,}")
            mem_gb = num_params * lanczos_order * 4 / 1e9  # float32
            print(f"  Est. memory for Lanczos vectors: {mem_gb:.2f} GB")

        # Run Stochastic Lanczos Quadrature
        if verbose:
            print(f"  Running {num_samples} Lanczos iterations (order={lanczos_order})...")

        tridiag_list = stochastic_lanczos_quadrature(
            hvp_fn,
            dim=num_params,
            order=lanczos_order,
            num_samples=num_samples,
            device=self.device,
            dtype=self.dtype,
        )

        # Convert to numpy for density estimation (convert to float32 first, numpy doesn't support bfloat16)
        tridiag_np = np.stack([t.float().numpy() for t in tridiag_list])

        # Get eigenvalues and density
        eigenvalues, weights = tridiag_to_eigv(tridiag_np)
        density, grids = tridiag_to_density(
            tridiag_np, sigma_squared=sigma_squared, grid_len=grid_len
        )

        # Compute summary statistics
        effective_rank = float(effective_rank_from_eigv(eigenvalues))
        top_eigenvalue = float(np.max(eigenvalues))
        trace_estimate = float(np.mean(np.sum(eigenvalues * weights, axis=1)))

        # Get param shape
        param_dict = dict(self.model.named_parameters())
        param_shape = tuple(param_dict[param_name].shape)

        if verbose:
            print(f"  Top eigenvalue: {top_eigenvalue:.4e}")
            print(f"  Trace estimate: {trace_estimate:.4e}")
            print(f"  Effective rank: {effective_rank:.2f}")

        return SpectrumResult(
            param_name=param_name,
            param_shape=param_shape,
            num_params=num_params,
            eigenvalues=eigenvalues,
            weights=weights,
            density=density,
            grids=grids,
            effective_rank=effective_rank,
            top_eigenvalue=top_eigenvalue,
            trace_estimate=trace_estimate,
        )

    def analyze_layer(
        self,
        layer_idx: int,
        block_type: str = None,
        **kwargs,
    ) -> Dict[str, SpectrumResult]:
        """Analyze all weight matrices in a layer.

        Args:
            layer_idx: Layer index.
            block_type: If provided, only analyze 'attention', 'mlp', or 'layernorm'.
            **kwargs: Arguments passed to analyze_param.

        Returns:
            Dict mapping param_name to SpectrumResult.
        """
        blocks = self.get_layer_blocks(layer_idx)

        if block_type:
            param_names = blocks.get(block_type, [])
        else:
            param_names = sum(blocks.values(), [])

        results = {}
        for param_name in param_names:
            # Skip small params (like biases, norms)
            param_dict = dict(self.model.named_parameters())
            if param_dict[param_name].numel() < 1000:
                continue

            results[param_name] = self.analyze_param(param_name, **kwargs)

        return results

    def analyze_all_layers(
        self,
        layer_indices: List[int] = None,
        block_type: str = None,
        save_dir: str = None,
        **kwargs,
    ) -> Dict[str, SpectrumResult]:
        """Analyze weight matrices across multiple layers.

        Args:
            layer_indices: Which layers to analyze. If None, infers from model.
            block_type: If provided, only analyze 'attention' or 'mlp'.
            save_dir: If provided, save results incrementally.
            **kwargs: Arguments passed to analyze_param.

        Returns:
            Dict mapping param_name to SpectrumResult.
        """
        if layer_indices is None:
            # Infer number of layers
            params = list(dict(self.model.named_parameters()).keys())
            layer_indices = set()
            for p in params:
                for pattern in [".layers.", ".h.", ".layer."]:
                    if pattern in p:
                        try:
                            idx = int(p.split(pattern)[1].split(".")[0])
                            layer_indices.add(idx)
                        except (ValueError, IndexError):
                            pass
            layer_indices = sorted(layer_indices)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        all_results = {}
        for layer_idx in layer_indices:
            print(f"\n=== Layer {layer_idx} ===")
            layer_results = self.analyze_layer(layer_idx, block_type, **kwargs)
            all_results.update(layer_results)

            if save_dir:
                # Save incrementally
                for name, result in layer_results.items():
                    safe_name = name.replace(".", "_").replace("/", "_")
                    np.savez(
                        save_path / f"{safe_name}.npz",
                        eigenvalues=result.eigenvalues,
                        weights=result.weights,
                        density=result.density,
                        grids=result.grids,
                        **result.to_dict(),
                    )

        if save_dir:
            # Save summary
            summary = {name: r.to_dict() for name, r in all_results.items()}
            with open(save_path / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        return all_results


def quick_analyze(
    model: nn.Module,
    dataloader,
    param_name: str,
    lanczos_order: int = 30,
    num_samples: int = 3,
    device: str = "cuda",
) -> SpectrumResult:
    """Quick one-liner to analyze a single parameter.

    Example:
        >>> result = quick_analyze(model, train_loader,
        ...     "model.layers.0.self_attn.q_proj.weight")
        >>> plt.plot(result.grids, result.density)
    """
    analyzer = SpectrumAnalyzer(
        model, dataloader, device=torch.device(device), max_batches=5
    )
    return analyzer.analyze_param(
        param_name, lanczos_order=lanczos_order, num_samples=num_samples
    )

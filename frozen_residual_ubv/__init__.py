"""
Frozen Residual UBV: Rank-preserving continual fine-tuning for LLMs.

This package implements a UBV^T factorization with Riemannian optimization
for continual learning that prevents spectral collapse. The key idea is:

    W = W_res (frozen) + U @ B @ V^T (trainable, rank-preserving)

where:
    - U, V are on the Stiefel manifold (orthonormal columns)
    - B is symmetric positive definite (trainable spectrum)
    - W_res captures the frozen residual (tail singular components)

The optimizer applies geometric update rules:
    - U, V: Dual ascent on Stiefel (Modula/Bernstein)
    - B: Riemannian gradient on S_{++} with eigenvalue clamping

References:
    - He et al., "Spectral Collapse Drives Loss of Plasticity in Deep
      Continual Learning," arXiv:2509.22335
    - Mishra et al., "R3MC: A Riemannian three-factor algorithm for low-rank
      matrix completion," CDC 2014
    - Bernstein et al., "Modula: Manifold Optimization via Dual Ascent"

Example:
    >>> from frozen_residual_ubv import FactoredLinear, UBVOptimizer, convert_model_to_factored
    >>>
    >>> # Convert a pretrained model
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> model = convert_model_to_factored(model, r_star=64)
    >>>
    >>> # Create optimizer with geometric update rules
    >>> optimizer = UBVOptimizer(model.parameters(), lr_U=1e-2, lr_B=1e-3)
    >>>
    >>> # Train as usual - manifold constraints are enforced automatically
    >>> for x, y in dataloader:
    ...     optimizer.zero_grad()
    ...     loss = criterion(model(x), y)
    ...     loss.backward()
    ...     optimizer.step()
"""

from .factored_linear import FactoredLinear, create_factored_linear_from_linear
from .ubv_optimizer import UBVOptimizer
from .projected_gradient_optimizer import (
    ProjectedGradientOptimizer,
    create_projected_optimizer,
)
from .feature_preserving_muon import FeaturePreservingMuon
from .soft_comp_preserving_muon import (
    SoftCompPreservingMuon,
    _factored_svd,
    _factored_svd_energy,
    _energy_rank,
    _solve_sylvester_damped,
)
from .composition_preserving_muon import (
    CompositionPreservingMuon,
    topk_svd_lowrank_product,
    energy_svd_lowrank_product,
    constraint_residual,
    strict_composition_fallback,
    orth,
    project_left,
    project_right,
)
from .model_surgery import (
    convert_model_to_factored,
    get_factored_parameters,
    count_parameters,
    print_factored_summary,
    compute_energy_rank,
    compute_tolerance_rank,
    create_rank_fn,
)
from .utils import (
    msign,
    eigenvalue_clamp,
    stiefel_retraction_modula,
    polar_retraction,
    symmetric_part,
    skew_symmetric_part,
    stiefel_project_tangent,
    check_stiefel_constraint,
    check_spd_constraint,
)

__version__ = "0.1.0"
__author__ = "Self-Distillation Research"

__all__ = [
    # Core modules
    "FactoredLinear",
    "create_factored_linear_from_linear",
    "UBVOptimizer",
    # Projected gradient optimizer (no surgery)
    "ProjectedGradientOptimizer",
    "create_projected_optimizer",
    # Feature-preserving Muon
    "FeaturePreservingMuon",
    # Soft composition-preserving Muon
    "SoftCompPreservingMuon",
    "_factored_svd",
    "_factored_svd_energy",
    "_energy_rank",
    "_solve_sylvester_damped",
    # Composition-preserving Muon
    "CompositionPreservingMuon",
    "topk_svd_lowrank_product",
    "energy_svd_lowrank_product",
    "constraint_residual",
    "strict_composition_fallback",
    "orth",
    "project_left",
    "project_right",
    # Model surgery
    "convert_model_to_factored",
    "get_factored_parameters",
    "count_parameters",
    "print_factored_summary",
    "compute_energy_rank",
    "compute_tolerance_rank",
    "create_rank_fn",
    # Utilities
    "msign",
    "eigenvalue_clamp",
    "stiefel_retraction_modula",
    "polar_retraction",
    "symmetric_part",
    "skew_symmetric_part",
    "stiefel_project_tangent",
    "check_stiefel_constraint",
    "check_spd_constraint",
]

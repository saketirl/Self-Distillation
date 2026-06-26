"""Eigenspectrum analysis tools for LLMs."""

from .lanczos_torch import (
    lanczos_alg,
    lanczos_alg_memory_efficient,
    stochastic_lanczos_quadrature,
)
from .hessian_torch import (
    hvp_single_param,
    get_hvp_fn_for_param,
    list_weight_matrices,
    get_block_param_names,
)
from .density_numpy import (
    tridiag_to_density,
    tridiag_to_eigv,
    eigv_to_density,
    effective_rank_from_eigv,
)
from .spectrum_analyzer import (
    SpectrumAnalyzer,
    SpectrumResult,
    quick_analyze,
)
from .stiefel_distance import (
    spectral_distance_to_stiefel,
    analyze_layer,
    analyze_model,
    compare_models,
)

__all__ = [
    # Lanczos
    "lanczos_alg",
    "lanczos_alg_memory_efficient",
    "stochastic_lanczos_quadrature",
    # HVP
    "hvp_single_param",
    "get_hvp_fn_for_param",
    "list_weight_matrices",
    "get_block_param_names",
    # Density
    "tridiag_to_density",
    "tridiag_to_eigv",
    "eigv_to_density",
    "effective_rank_from_eigv",
    # High-level API
    "SpectrumAnalyzer",
    "SpectrumResult",
    "quick_analyze",
    # Stiefel distance
    "spectral_distance_to_stiefel",
    "analyze_layer",
    "analyze_model",
    "compare_models",
]

"""PyTorch implementation of Lanczos algorithm for eigenspectrum estimation."""

import torch
from typing import Callable, Tuple


def lanczos_alg(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    order: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lanczos algorithm for tridiagonalizing a real symmetric matrix.

    This function applies Lanczos algorithm of a given order with full
    reorthogonalization.

    Args:
        matrix_vector_product: Maps v -> Hv for a real symmetric matrix H.
            Input/Output must be of shape [dim].
        dim: Matrix H is [dim, dim].
        order: Number of Lanczos iterations (typically 50-100).
        device: Torch device for computation.
        dtype: Data type (float32 or float16 for memory savings).
        seed: Random seed for reproducibility.

    Returns:
        tridiag: Tridiagonal matrix of shape (order, order).
        vecs: Lanczos vectors of shape (order, dim).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is not None:
        torch.manual_seed(seed)

    tridiag = torch.zeros((order, order), device=device, dtype=dtype)
    vecs = torch.zeros((order, dim), device=device, dtype=dtype)

    # Initialize with random unit vector
    init_vec = torch.randn(dim, device=device, dtype=dtype)
    init_vec = init_vec / torch.linalg.norm(init_vec)
    vecs[0] = init_vec

    beta = 0.0
    for i in range(order):
        v = vecs[i]
        v_old = vecs[i - 1] if i > 0 else torch.zeros_like(v)

        # Hessian-vector product
        w = matrix_vector_product(v)
        assert w.shape == (dim,), f"Expected shape ({dim},), got {w.shape}"

        w = w - beta * v_old

        alpha = torch.dot(w, v)
        tridiag[i, i] = alpha
        w = w - alpha * v

        # Full reorthogonalization for numerical stability
        for j in range(i):
            tau = vecs[j]
            coeff = torch.dot(w, tau)
            w = w - coeff * tau

        beta = torch.linalg.norm(w)

        if i + 1 < order:
            if beta < 1e-6:
                # Lanczos vectors becoming linearly dependent
                # Reinitialize with new random vector orthogonal to existing
                w = torch.randn(dim, device=device, dtype=dtype)
                for j in range(i + 1):
                    tau = vecs[j]
                    coeff = torch.dot(w, tau)
                    w = w - coeff * tau
                beta = torch.linalg.norm(w)

            tridiag[i, i + 1] = beta
            tridiag[i + 1, i] = beta
            vecs[i + 1] = w / beta

    return tridiag, vecs


def lanczos_alg_memory_efficient(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    order: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    seed: int = None,
) -> torch.Tensor:
    """Memory-efficient Lanczos that only returns tridiagonal matrix.

    Uses selective reorthogonalization and doesn't store all Lanczos vectors.
    Suitable for very large parameter counts where storing (order, dim) is
    prohibitive.

    Args:
        matrix_vector_product: Maps v -> Hv for a real symmetric matrix H.
        dim: Matrix H is [dim, dim].
        order: Number of Lanczos iterations.
        device: Torch device.
        dtype: Data type.
        seed: Random seed.

    Returns:
        tridiag: Tridiagonal matrix of shape (order, order).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is not None:
        torch.manual_seed(seed)

    tridiag = torch.zeros((order, order), device=device, dtype=dtype)

    # Only store current and previous vectors
    v = torch.randn(dim, device=device, dtype=dtype)
    v = v / torch.linalg.norm(v)
    v_old = torch.zeros_like(v)

    # Store a few vectors for periodic reorthogonalization
    reorth_interval = max(1, order // 10)
    reorth_vecs = []

    beta = 0.0
    for i in range(order):
        w = matrix_vector_product(v)
        w = w - beta * v_old

        alpha = torch.dot(w, v)
        tridiag[i, i] = alpha
        w = w - alpha * v

        # Periodic reorthogonalization against stored vectors
        if i % reorth_interval == 0:
            reorth_vecs.append(v.clone())
        for rv in reorth_vecs:
            coeff = torch.dot(w, rv)
            w = w - coeff * rv

        beta = torch.linalg.norm(w)

        if i + 1 < order:
            if beta < 1e-6:
                w = torch.randn(dim, device=device, dtype=dtype)
                for rv in reorth_vecs:
                    coeff = torch.dot(w, rv)
                    w = w - coeff * rv
                beta = torch.linalg.norm(w)

            tridiag[i, i + 1] = beta
            tridiag[i + 1, i] = beta

            v_old = v
            v = w / beta

    return tridiag


def stochastic_lanczos_quadrature(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    order: int = 50,
    num_samples: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> list:
    """Stochastic Lanczos Quadrature for eigenspectrum estimation.

    Runs multiple independent Lanczos iterations with different random
    starting vectors. This is the recommended approach for robust
    density estimation.

    Args:
        matrix_vector_product: Maps v -> Hv.
        dim: Parameter dimension.
        order: Lanczos iterations per sample.
        num_samples: Number of independent runs to average.
        device: Torch device.
        dtype: Data type.

    Returns:
        tridiag_list: List of tridiagonal matrices, one per sample.
    """
    tridiag_list = []

    for i in range(num_samples):
        tridiag = lanczos_alg_memory_efficient(
            matrix_vector_product,
            dim=dim,
            order=order,
            device=device,
            dtype=dtype,
            seed=i * 12345,  # Different seed per sample
        )
        tridiag_list.append(tridiag.cpu())

    return tridiag_list

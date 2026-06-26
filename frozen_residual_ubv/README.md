# Frozen Residual UBV

A PyTorch package for continual fine-tuning of pretrained LLMs using rank-preserving UBV^T factorization with Riemannian optimization.

## Overview

This package addresses the problem of **spectral collapse** during continual learning, where fine-tuning causes weight matrices to lose effective rank, leading to catastrophic forgetting and loss of plasticity.

The key insight is to decompose pretrained weights as:

```
W = W_res + U B V^T
```

Where:
- **W_res** (frozen): The residual capturing components outside the top-r singular space
- **U** (m × r): Left singular vectors, constrained to Stiefel manifold St(r, m)
- **B** (r × r): Diagonal matrix of singular values, constrained to S_{++}
- **V** (n × r): Right singular vectors, constrained to Stiefel manifold St(r, n)

By optimizing U, V on the Stiefel manifold and B on the symmetric positive definite cone with eigenvalue clamping, we maintain the spectral structure during training.

## Installation

```bash
# From the repository root
pip install -e .
```

Or simply add the package directory to your Python path.

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM
from frozen_residual_ubv import convert_model_to_factored, UBVOptimizer

# Load a pretrained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Convert attention layers to factored form
convert_model_to_factored(
    model,
    r_star=16,  # Target rank
    include_patterns=['c_attn', 'c_proj'],  # Attention layers
    freeze_residual=True,
)

# Create the Riemannian optimizer
optimizer = UBVOptimizer(
    model.parameters(),
    lr_U=1e-3,      # Stiefel learning rate
    lr_B=5e-4,      # S_{++} learning rate
    lr_V=1e-3,      # Stiefel learning rate
    lambda_min=1e-4,  # Spectrum floor
)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()  # Riemannian updates with constraint preservation
```

## Components

### FactoredLinear

A drop-in replacement for `nn.Linear` that maintains the UBV^T factorization:

```python
from frozen_residual_ubv import FactoredLinear

# Create from a pretrained weight matrix
W = torch.randn(512, 768)
layer = FactoredLinear(W, r_star=32)

# Forward pass: equivalent to original linear layer
x = torch.randn(16, 768)
y = layer(x)  # Shape: (16, 512)
```

### UBVOptimizer

A Riemannian optimizer that maintains manifold constraints:

- **U, V**: Riemannian gradient descent on Stiefel manifold with QR retraction
- **B**: Riemannian gradient on S_{++} with affine-invariant metric + eigenvalue clamping
- **bias**: Standard SGD

```python
optimizer = UBVOptimizer(
    model.parameters(),
    lr_U=1e-2,        # Learning rate for U (Stiefel)
    lr_B=1e-3,        # Learning rate for B (S_{++})
    lr_V=1e-2,        # Learning rate for V (Stiefel)
    lr_bias=1e-3,     # Learning rate for bias (SGD)
    lambda_min=1e-4,  # Minimum eigenvalue (spectrum floor)
    momentum=0.0,     # Momentum for B updates
)
```

### Model Surgery

Convert existing models to factored form:

```python
from frozen_residual_ubv import convert_model_to_factored

# Convert specific layers by pattern
count = convert_model_to_factored(
    model,
    r_star=16,
    include_patterns=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    exclude_patterns=['lm_head'],
    freeze_residual=True,
)
print(f"Converted {count} layers")
```

## Mathematical Background

### Stiefel Manifold Optimization

The Stiefel manifold St(r, m) consists of m × r matrices with orthonormal columns:
```
St(r, m) = {U ∈ R^{m×r} : U^T U = I_r}
```

We optimize using Riemannian gradient descent:
1. Project Euclidean gradient onto tangent space: `G_tan = G - U @ Sym(U^T G)`
2. Take step in tangent direction: `U_temp = U - lr * G_tan`
3. Retract back to manifold via QR decomposition

### S_{++} Optimization

The symmetric positive definite cone S_{++} consists of SPD matrices. We use the affine-invariant metric (Mishra et al. 2014):

```
B_new = B - lr * B @ Sym(G_B) @ B
```

After the update, eigenvalues are clamped to maintain the spectrum floor:
```
B = Q @ diag(max(λ_i, λ_min)) @ Q^T
```

This prevents spectral collapse by ensuring no eigenvalue drops below `lambda_min`.

## Testing

Run the test suite:

```bash
pytest frozen_residual_ubv/tests/ -v
```

Key tests verify:
- Forward pass equivalence (UBV^T ≈ W)
- Stiefel constraint preservation (U^T U = I after many steps)
- Spectrum floor maintenance (min eigenvalue ≥ lambda_min)
- Effective rank preservation (±20% of initial)
- Numerical stability (no NaN/Inf)

## Example: Continual Learning

See `examples/finetune_gpt2_continual.py` for a complete example of continual fine-tuning GPT-2 across multiple tasks while preserving spectral properties.

```bash
# Task 0
python frozen_residual_ubv/examples/finetune_gpt2_continual.py --task_id 0

# Task 1 (continual, loads from task 0)
python frozen_residual_ubv/examples/finetune_gpt2_continual.py --task_id 1

# Task 2 (continual, loads from task 1)
python frozen_residual_ubv/examples/finetune_gpt2_continual.py --task_id 2
```

## References

- Mishra et al., "R3MC: A Riemannian three-factor algorithm for low-rank matrix completion," CDC 2014
- Bernstein et al., "Modula: Manifold Optimization via Dual Ascent"
- He et al., "Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning," arXiv:2509.22335

## License

MIT License

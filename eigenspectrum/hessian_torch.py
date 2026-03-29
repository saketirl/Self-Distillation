"""PyTorch Hessian-vector product computation for individual weight matrices."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp
from typing import Callable, Tuple, Dict, Any, Optional
from functools import partial


def hvp_single_param_reverse(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    param: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute Hessian-vector product using reverse-over-reverse mode.

    Uses grad(grad(L) · v) which requires two backward passes but is
    compatible with gradient checkpointing.

    Args:
        loss_fn: Function mapping param -> scalar loss.
        param: The parameter tensor.
        v: Vector to multiply with Hessian, shape [num_params].

    Returns:
        Hv: Hessian-vector product, shape [num_params].
    """
    # Create a copy that requires gradients
    param_flat = param.view(-1).detach().clone()
    param_flat.requires_grad_(True)
    v_flat = v.view(-1).detach()

    # Compute loss
    loss = loss_fn(param_flat.view(param.shape))

    # First backward pass: compute gradient with graph preserved
    (g,) = torch.autograd.grad(loss, param_flat, create_graph=True)

    # Compute g · v (scalar)
    gv = torch.dot(g.view(-1), v_flat)

    # Second backward pass: gradient of g·v gives HVP
    (hvp,) = torch.autograd.grad(gv, param_flat)

    return hvp.detach()


def hvp_single_param(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    param: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute Hessian-vector product for a single parameter tensor.

    Uses forward-over-reverse mode autodiff.

    Args:
        loss_fn: Function mapping param -> scalar loss.
        param: The parameter tensor (will be flattened internally).
        v: Vector to multiply with Hessian, shape [num_params].

    Returns:
        Hv: Hessian-vector product, shape [num_params].
    """
    param_flat = param.view(-1)
    v = v.view(param.shape)

    def flat_loss(p_flat):
        return loss_fn(p_flat.view(param.shape))

    # Forward-over-reverse: jvp of grad
    def grad_fn(p):
        return grad(flat_loss)(p)

    _, hvp = jvp(grad_fn, (param_flat,), (v.view(-1),))
    return hvp


def create_single_param_loss_fn(
    model: nn.Module,
    target_param_name: str,
    data_batch: Tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable = None,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, int]:
    """Create a loss function that depends only on a single parameter.

    Args:
        model: The neural network model.
        target_param_name: Name of the parameter to analyze (e.g.,
            'model.layers.0.self_attn.q_proj.weight').
        data_batch: Tuple of (input_ids, labels) for loss computation.
        loss_fn: Optional custom loss function. If None, uses model's
            forward which should return loss.

    Returns:
        param_loss_fn: Function mapping param_value -> scalar loss.
        original_param: The original parameter tensor (detached).
        num_params: Number of parameters in this tensor.
    """
    # Get the target parameter
    param_dict = dict(model.named_parameters())
    if target_param_name not in param_dict:
        raise ValueError(
            f"Parameter '{target_param_name}' not found. "
            f"Available: {list(param_dict.keys())[:10]}..."
        )

    original_param = param_dict[target_param_name].detach().clone()
    num_params = original_param.numel()

    # Create all other params dict (frozen)
    frozen_params = {
        name: p.detach()
        for name, p in param_dict.items()
        if name != target_param_name
    }

    input_ids, labels = data_batch

    def param_loss_fn(param_value: torch.Tensor) -> torch.Tensor:
        """Compute loss with target param set to param_value."""
        # Combine frozen params with the variable param
        all_params = {**frozen_params, target_param_name: param_value}

        # Use functional_call to evaluate model with custom params
        outputs = functional_call(model, all_params, (input_ids,), dict(labels=labels))

        if hasattr(outputs, "loss"):
            return outputs.loss
        elif loss_fn is not None:
            return loss_fn(outputs, labels)
        else:
            raise ValueError("Model output has no 'loss' attribute and no loss_fn provided")

    return param_loss_fn, original_param, num_params


def get_hvp_fn_for_param(
    model: nn.Module,
    target_param_name: str,
    dataloader,
    max_batches: int = 10,
    device: torch.device = None,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Get HVP function for a specific parameter, averaged over data batches.

    Args:
        model: The model.
        target_param_name: Which parameter to analyze.
        dataloader: DataLoader providing (input_ids, labels) batches.
        max_batches: Maximum batches to use for Hessian estimation.
        device: Device for computation.

    Returns:
        hvp_fn: Function mapping v -> Hv where H is the Hessian.
        num_params: Dimension of the parameter.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Collect batches
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids).to(device)
        else:
            input_ids, labels = batch[0].to(device), batch[1].to(device)
        batches.append((input_ids, labels))

    if not batches:
        raise ValueError("Dataloader yielded no batches")

    # Get param info from first batch
    _, original_param, num_params = create_single_param_loss_fn(
        model, target_param_name, batches[0]
    )

    def hvp_fn(v: torch.Tensor, num_batches_per_hvp: int = 5) -> torch.Tensor:
        """Compute Hessian-vector product averaged over random batch subset.

        Uses stochastic estimation: randomly samples a few batches per HVP
        call instead of using all batches. Much faster with minimal accuracy loss.
        """
        import random
        v = v.to(device)
        hvp_sum = torch.zeros_like(v)

        # Randomly sample batches for this HVP (stochastic estimation)
        batch_subset = random.sample(batches, min(num_batches_per_hvp, len(batches)))

        for batch in batch_subset:
            loss_fn, param, _ = create_single_param_loss_fn(
                model, target_param_name, batch
            )

            # Compute HVP for this batch
            hvp_batch = hvp_single_param(loss_fn, param, v)
            hvp_sum = hvp_sum + hvp_batch.detach()

        # Clear cache periodically (not every batch)
        torch.cuda.empty_cache()

        return hvp_sum / len(batch_subset)

    return hvp_fn, num_params


def get_hvp_fn_for_param_checkpointed(
    model: nn.Module,
    target_param_name: str,
    dataloader,
    max_batches: int = 10,
    device: torch.device = None,
    gradient_checkpointing: bool = True,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Get HVP function using reverse-mode autodiff with gradient checkpointing.

    This version uses reverse-over-reverse mode (two backward passes) instead of
    forward-over-reverse (jvp of grad). This is compatible with gradient
    checkpointing and uses significantly less memory.

    Args:
        model: The model.
        target_param_name: Which parameter to analyze.
        dataloader: DataLoader providing (input_ids, labels) batches.
        max_batches: Maximum batches to use for Hessian estimation.
        device: Device for computation.
        gradient_checkpointing: Whether to enable gradient checkpointing.

    Returns:
        hvp_fn: Function mapping v -> Hv where H is the Hessian.
        num_params: Dimension of the parameter.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Enable gradient checkpointing if requested
    checkpointing_was_enabled = getattr(model, 'is_gradient_checkpointing', False)
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"  Gradient checkpointing enabled")

    # Collect batches
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids).to(device)
        else:
            input_ids, labels = batch[0].to(device), batch[1].to(device)
        batches.append((input_ids, labels))

    if not batches:
        raise ValueError("Dataloader yielded no batches")

    # Get param info from first batch
    _, original_param, num_params = create_single_param_loss_fn(
        model, target_param_name, batches[0]
    )

    def hvp_fn(v: torch.Tensor, num_batches_per_hvp: int = 5) -> torch.Tensor:
        """Compute HVP using reverse-over-reverse mode (two backward passes).

        This is compatible with gradient checkpointing.
        """
        import random
        v = v.to(device)
        hvp_sum = torch.zeros_like(v)

        # Randomly sample batches for this HVP
        batch_subset = random.sample(batches, min(num_batches_per_hvp, len(batches)))

        for batch in batch_subset:
            loss_fn, param, _ = create_single_param_loss_fn(
                model, target_param_name, batch
            )

            # Compute HVP using reverse-mode (compatible with checkpointing)
            hvp_batch = hvp_single_param_reverse(loss_fn, param, v)
            hvp_sum = hvp_sum + hvp_batch.detach()

        torch.cuda.empty_cache()

        return hvp_sum / len(batch_subset)

    # Store cleanup function
    def cleanup():
        if not checkpointing_was_enabled and hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()

    hvp_fn.cleanup = cleanup

    return hvp_fn, num_params


def create_single_param_kl_loss_fn(
    model: nn.Module,
    ref_model: nn.Module,
    target_param_name: str,
    data_batch: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, int]:
    """Create a KL divergence loss function (SDFT-style) for a single parameter.

    Args:
        model: The student model being analyzed.
        ref_model: The reference/teacher model (frozen).
        target_param_name: Name of the parameter to analyze.
        data_batch: Tuple of (input_ids, labels) for loss computation.

    Returns:
        param_loss_fn: Function mapping param_value -> scalar KL loss.
        original_param: The original parameter tensor (detached).
        num_params: Number of parameters in this tensor.
    """
    # Get the target parameter
    param_dict = dict(model.named_parameters())
    if target_param_name not in param_dict:
        raise ValueError(
            f"Parameter '{target_param_name}' not found. "
            f"Available: {list(param_dict.keys())[:10]}..."
        )

    original_param = param_dict[target_param_name].detach().clone()
    num_params = original_param.numel()

    # Create all other params dict (frozen)
    frozen_params = {
        name: p.detach()
        for name, p in param_dict.items()
        if name != target_param_name
    }

    input_ids, labels = data_batch

    # Get teacher logits (computed once, frozen - no gradients)
    # Teacher output is only used as a target, gradients do not flow through it
    with torch.no_grad():
        ref_outputs = ref_model(input_ids)
        ref_logits = ref_outputs.logits.detach().clone()
        # Pre-compute teacher log probs (frozen target)
        teacher_log_probs = F.log_softmax(ref_logits, dim=-1).detach()

    def param_loss_fn(param_value: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss with target param set to param_value."""
        # Combine frozen params with the variable param
        all_params = {**frozen_params, target_param_name: param_value}

        # Get student logits (this is the only part that has gradients)
        outputs = functional_call(model, all_params, (input_ids,))
        student_logits = outputs.logits

        # Compute KL divergence: KL(teacher || student)
        # Only student_log_probs has gradients, teacher_log_probs is frozen
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # KL divergence per token (gradient only flows through student)
        kl_div = F.kl_div(student_log_probs, teacher_log_probs,
                         reduction='none', log_target=True)

        # Create mask from labels (ignore padding, typically -100)
        mask = (labels != -100).float()

        # Sum over vocab dimension, mean over valid tokens
        kl_per_token = kl_div.sum(dim=-1)  # [batch, seq_len]
        masked_kl = kl_per_token * mask
        loss = masked_kl.sum() / mask.sum().clamp(min=1.0)

        return loss

    return param_loss_fn, original_param, num_params


def get_hvp_fn_for_param_sdft(
    model: nn.Module,
    ref_model: nn.Module,
    target_param_name: str,
    dataloader,
    max_batches: int = 10,
    device: torch.device = None,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Get HVP function using SDFT (KL divergence) loss.

    Args:
        model: The student model to analyze.
        ref_model: The reference/teacher model (frozen).
        target_param_name: Which parameter to analyze.
        dataloader: DataLoader providing (input_ids, labels) batches.
        max_batches: Maximum batches to use for Hessian estimation.
        device: Device for computation.

    Returns:
        hvp_fn: Function mapping v -> Hv where H is the Hessian of KL loss.
        num_params: Dimension of the parameter.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ref_model.eval()

    # Collect batches
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids).to(device)
        else:
            input_ids, labels = batch[0].to(device), batch[1].to(device)
        batches.append((input_ids, labels))

    if not batches:
        raise ValueError("Dataloader yielded no batches")

    # Get param info from first batch
    _, original_param, num_params = create_single_param_kl_loss_fn(
        model, ref_model, target_param_name, batches[0]
    )

    def hvp_fn(v: torch.Tensor, num_batches_per_hvp: int = 5) -> torch.Tensor:
        """Compute Hessian-vector product of KL loss, averaged over random batch subset."""
        import random
        v = v.to(device)
        hvp_sum = torch.zeros_like(v)

        # Randomly sample batches for this HVP (stochastic estimation)
        batch_subset = random.sample(batches, min(num_batches_per_hvp, len(batches)))

        for batch in batch_subset:
            loss_fn, param, _ = create_single_param_kl_loss_fn(
                model, ref_model, target_param_name, batch
            )

            # Compute HVP for this batch
            hvp_batch = hvp_single_param(loss_fn, param, v)
            hvp_sum = hvp_sum + hvp_batch.detach()

        torch.cuda.empty_cache()

        return hvp_sum / len(batch_subset)

    return hvp_fn, num_params


def get_hvp_fn_for_param_sdft_checkpointed(
    model: nn.Module,
    ref_model: nn.Module,
    target_param_name: str,
    dataloader,
    max_batches: int = 10,
    device: torch.device = None,
    gradient_checkpointing: bool = True,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Get HVP function using SDFT loss with gradient checkpointing.

    Uses reverse-over-reverse mode for HVP computation, which is compatible
    with gradient checkpointing and uses less memory.

    Args:
        model: The student model to analyze.
        ref_model: The reference/teacher model (frozen).
        target_param_name: Which parameter to analyze.
        dataloader: DataLoader providing (input_ids, labels) batches.
        max_batches: Maximum batches to use for Hessian estimation.
        device: Device for computation.
        gradient_checkpointing: Whether to enable gradient checkpointing.

    Returns:
        hvp_fn: Function mapping v -> Hv where H is the Hessian of KL loss.
        num_params: Dimension of the parameter.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ref_model.eval()

    # Enable gradient checkpointing on student model only
    checkpointing_was_enabled = getattr(model, 'is_gradient_checkpointing', False)
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"  Gradient checkpointing enabled on student model")

    # Collect batches
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids).to(device)
        else:
            input_ids, labels = batch[0].to(device), batch[1].to(device)
        batches.append((input_ids, labels))

    if not batches:
        raise ValueError("Dataloader yielded no batches")

    # Get param info from first batch
    _, original_param, num_params = create_single_param_kl_loss_fn(
        model, ref_model, target_param_name, batches[0]
    )

    def hvp_fn(v: torch.Tensor, num_batches_per_hvp: int = 5) -> torch.Tensor:
        """Compute HVP of KL loss using reverse-over-reverse mode."""
        import random
        v = v.to(device)
        hvp_sum = torch.zeros_like(v)

        batch_subset = random.sample(batches, min(num_batches_per_hvp, len(batches)))

        for batch in batch_subset:
            loss_fn, param, _ = create_single_param_kl_loss_fn(
                model, ref_model, target_param_name, batch
            )

            # Compute HVP using reverse-mode (compatible with checkpointing)
            hvp_batch = hvp_single_param_reverse(loss_fn, param, v)
            hvp_sum = hvp_sum + hvp_batch.detach()

        torch.cuda.empty_cache()

        return hvp_sum / len(batch_subset)

    def cleanup():
        if not checkpointing_was_enabled and hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()

    hvp_fn.cleanup = cleanup

    return hvp_fn, num_params


def get_block_param_names(model: nn.Module, layer_idx: int) -> Dict[str, list]:
    """Get parameter names organized by block type for a transformer layer.

    Args:
        model: HuggingFace transformer model.
        layer_idx: Which layer to analyze.

    Returns:
        Dict mapping block type to list of param names.
    """
    param_names = list(dict(model.named_parameters()).keys())

    # Common patterns for transformer models
    layer_prefix_patterns = [
        f"model.layers.{layer_idx}.",  # Llama, Qwen, Mistral
        f"transformer.h.{layer_idx}.",  # GPT-2
        f"encoder.layer.{layer_idx}.",  # BERT
    ]

    layer_prefix = None
    for pattern in layer_prefix_patterns:
        if any(p.startswith(pattern) for p in param_names):
            layer_prefix = pattern
            break

    if layer_prefix is None:
        raise ValueError(f"Could not find layer {layer_idx} in model")

    layer_params = [p for p in param_names if p.startswith(layer_prefix)]

    # Organize by block
    blocks = {
        "attention": [],
        "mlp": [],
        "layernorm": [],
    }

    attention_keywords = ["attn", "attention", "self_attn", "q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_keywords = ["mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]
    norm_keywords = ["norm", "ln_"]

    for p in layer_params:
        p_lower = p.lower()
        if any(kw in p_lower for kw in attention_keywords):
            blocks["attention"].append(p)
        elif any(kw in p_lower for kw in mlp_keywords):
            blocks["mlp"].append(p)
        elif any(kw in p_lower for kw in norm_keywords):
            blocks["layernorm"].append(p)
        else:
            # Default to MLP if unclear
            blocks["mlp"].append(p)

    return blocks


def list_weight_matrices(model: nn.Module, layer_idx: int = None) -> list:
    """List all weight matrices in the model, optionally filtered by layer.

    Args:
        model: The model.
        layer_idx: If provided, only list params from this layer.

    Returns:
        List of (param_name, shape, num_params) tuples.
    """
    result = []
    for name, param in model.named_parameters():
        if layer_idx is not None:
            # Check if this param belongs to the specified layer
            layer_patterns = [
                f".layers.{layer_idx}.",
                f".h.{layer_idx}.",
                f".layer.{layer_idx}.",
            ]
            if not any(p in name for p in layer_patterns):
                continue

        result.append((name, tuple(param.shape), param.numel()))

    return sorted(result, key=lambda x: -x[2])  # Sort by size descending

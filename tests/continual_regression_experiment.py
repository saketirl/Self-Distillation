"""
Continual regression experiment: Adam vs FeaturePreservingMuon.

Data
----
Two tasks, each a mixture of two Gaussians with scalar outputs:

    x_i = mu_{c_i} + eps * N(0, I_64)
    y_i = t_{c_i}  (scalar target for category c_i)

Task A and Task B have different means and different targets.

Protocol
--------
Both methods start from the same random initialisation.

  Adam -> Adam  : train Task A with Adam, then Task B with Adam.
                  Shows catastrophic forgetting of Task A.

  Adam -> FPMuon: train Task A with Adam, then Task B with FPMuon.
                  FPMuon computes V_r from the Task A weights (lazy init
                  on first step), then projects all Task B gradients
                  orthogonal to those directions.

Model
-----
4-layer deep linear network, no bias, no nonlinearity.
    dims = [64, 64, 64, 64, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frozen_residual_ubv import FeaturePreservingMuon


# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 0
torch.manual_seed(SEED)

# ── Geometry ─────────────────────────────────────────────────────────────────
DIM       = 64
N_TRAIN   = 512
EPS       = 0.25        # isotropic noise scale
STEPS_A   = 300         # steps for Task A
STEPS_B   = 300         # steps for Task B
LR_ADAM   = 1e-3
LR_FP     = 1e-3        # matched to Adam for fair comparison
ENERGY    = 0.8         # fraction of Task-A spectral energy to protect
BATCH     = N_TRAIN     # full-batch for simplicity

# ── Fixed task parameters ────────────────────────────────────────────────────
torch.manual_seed(1)
mu1_A = torch.randn(DIM)          # class-0 mean, Task A
mu2_A = torch.randn(DIM)          # class-1 mean, Task A
t1_A, t2_A = +1.0, +3.0           # scalar targets, Task A

mu1_B = torch.randn(DIM)          # class-0 mean, Task B  (different from A)
mu2_B = torch.randn(DIM)          # class-1 mean, Task B
t1_B, t2_B = -2.0, -4.0           # scalar targets, Task B (different from A)


# ── Dataset generator ────────────────────────────────────────────────────────
def make_dataset(n, mu1, mu2, t1, t2, eps=EPS, seed=0):
    """
    Sample n points from a 2-component isotropic Gaussian mixture.
    Returns X: [n, DIM], y: [n, 1].
    """
    g = torch.Generator()
    g.manual_seed(seed)
    labels = torch.randint(0, 2, (n,), generator=g)          # 0 or 1
    means  = torch.stack([mu1, mu2])                          # [2, DIM]
    X = means[labels] + eps * torch.randn(n, mu1.shape[0], generator=g)
    y = torch.where(labels == 0,
                    torch.full((n,), t1),
                    torch.full((n,), t2)).float().unsqueeze(1)  # [n, 1]
    return X, y


# ── Model factory ────────────────────────────────────────────────────────────
def make_model(seed=SEED):
    torch.manual_seed(seed)
    dims = [64, 64, 64, 64, 1]
    return nn.Sequential(*[nn.Linear(dims[i], dims[i+1], bias=False)
                            for i in range(len(dims) - 1)])


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, optimizer, X, y, steps, tag=""):
    losses = []
    for step in range(steps):
        pred = model(X)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"  [{tag}] step {step+1:3d}/{steps}  loss={loss.item():.4f}")
    return losses


def evaluate(model, X, y, label):
    with torch.no_grad():
        loss = F.mse_loss(model(X), y).item()
    print(f"  {label}: MSE = {loss:.4f}")
    return loss


# ── Data ─────────────────────────────────────────────────────────────────────
X_A, y_A = make_dataset(N_TRAIN, mu1_A, mu2_A, t1_A, t2_A, seed=10)
X_B, y_B = make_dataset(N_TRAIN, mu1_B, mu2_B, t1_B, t2_B, seed=20)

# ── Shared Task-A initialisation ──────────────────────────────────────────────
# Train Task A once; clone the checkpoint for both baselines.
print("=" * 60)
print("Pre-training Task A with Adam (shared for both conditions)")
print("=" * 60)
torch.manual_seed(SEED)
model_pretrain = make_model()
opt_pretrain = torch.optim.Adam(model_pretrain.parameters(), lr=LR_ADAM)
train(model_pretrain, opt_pretrain, X_A, y_A, STEPS_A, tag="TaskA-Adam")

print("\nTask A checkpoint evaluation:")
loss_A_after_taskA = evaluate(model_pretrain, X_A, y_A, "Task A loss (after Task A)")
loss_B_before_taskB = evaluate(model_pretrain, X_B, y_B, "Task B loss (before Task B)")

# Clone the Task-A checkpoint for both conditions
ckpt_adam  = copy.deepcopy(model_pretrain)
ckpt_fpmuon = copy.deepcopy(model_pretrain)


# ═══════════════════════════════════════════════════════════════════════════
# Condition 1: Adam → Adam  (catastrophic forgetting baseline)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Condition 1: Adam → Adam")
print("=" * 60)
opt_adam_B = torch.optim.Adam(ckpt_adam.parameters(), lr=LR_ADAM)
train(ckpt_adam, opt_adam_B, X_B, y_B, STEPS_B, tag="TaskB-Adam")

print("\nAdam→Adam final evaluation:")
loss_A_adam = evaluate(ckpt_adam, X_A, y_A, "Task A loss")
loss_B_adam = evaluate(ckpt_adam, X_B, y_B, "Task B loss")


# ═══════════════════════════════════════════════════════════════════════════
# Condition 2: Adam → FPMuon  (feature-preserving)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Condition 2: Adam → FPMuon  (energy_threshold={ENERGY})")
print("=" * 60)

# FPMuon for hidden layers (layers 0-2), Adam for the read-out (layer 3).
# Layer 3 is [1×64] — rank-1, so FPMuon can't protect it meaningfully.
hidden_params  = list(ckpt_fpmuon.parameters())[:-1]   # layers 0, 1, 2
readout_params = list(ckpt_fpmuon.parameters())[-1:]    # layer 3

opt_fp    = FeaturePreservingMuon(hidden_params, lr=LR_FP, energy_threshold=ENERGY)
opt_adam_readout = torch.optim.Adam(readout_params, lr=LR_ADAM)

# Combined step: FPMuon on hidden, Adam on read-out.
# Every LOG_EVERY steps, compute and report the constraint residual
# ||A @ V_r||_F per hidden layer, where A = (W_before - W_after) / lr.
LOG_EVERY = 50

def train_combined(model, opt_fp, opt_adam, X, y, steps, lr_fp, tag=""):
    losses = []
    hidden_params = [p for p in opt_fp.param_groups[0]["params"]]

    for step in range(steps):
        # Snapshot weights before step for constraint profiling
        W_before = [p.data.clone() for p in hidden_params]

        pred = model(X)
        loss = F.mse_loss(pred, y)
        opt_fp.zero_grad()
        opt_adam.zero_grad()
        loss.backward()
        opt_fp.step()
        opt_adam.step()
        losses.append(loss.item())

        if (step + 1) % LOG_EVERY == 0:
            # Constraint residual: ||A @ V_r|| per layer
            residuals = []
            for p, W_b in zip(hidden_params, W_before):
                s = opt_fp.state.get(p, {})
                if s.get("skip", True):
                    continue
                A = (W_b - p.data).float() / lr_fp   # [m, n]
                V_r = s["V_r"]                         # [n, r]
                res = torch.linalg.norm(A @ V_r).item()
                residuals.append(res)

            res_str = "  ".join(f"L{i}={r:.2e}" for i, r in enumerate(residuals))
            print(f"  [{tag}] step {step+1:3d}/{steps}"
                  f"  loss={loss.item():.4f}"
                  f"  ||A@V_r||: {res_str}")
    return losses

train_combined(ckpt_fpmuon, opt_fp, opt_adam_readout,
               X_B, y_B, STEPS_B, LR_FP, tag="TaskB-FPMuon")

print("\nProtected ranks per layer (computed from Task-A weights):")
for i, p in enumerate(ckpt_fpmuon.parameters()):
    s = opt_fp.state.get(p, {})
    if not s.get("skip", True):
        print(f"  Layer {i}: shape={tuple(p.shape)}  r={s['r']}  "
              f"energy_captured={s['captured_energy']:.3f}")

print("\nAdam→FPMuon final evaluation:")
loss_A_fp = evaluate(ckpt_fpmuon, X_A, y_A, "Task A loss")
loss_B_fp = evaluate(ckpt_fpmuon, X_B, y_B, "Task B loss")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Task A loss after Task A training:          {loss_A_after_taskA:.4f}")
print()
print(f"  After Task B (Adam):    Task A={loss_A_adam:.4f}  Task B={loss_B_adam:.4f}")
print(f"  After Task B (FPMuon):  Task A={loss_A_fp:.4f}  Task B={loss_B_fp:.4f}")
print()
forgetting_adam  = loss_A_adam  - loss_A_after_taskA
forgetting_fp    = loss_A_fp    - loss_A_after_taskA
print(f"  Forgetting (Adam):   ΔTask A = +{forgetting_adam:.4f}")
print(f"  Forgetting (FPMuon): ΔTask A = +{forgetting_fp:.4f}")

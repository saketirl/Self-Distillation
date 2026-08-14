"""P-Gram ICL sweep runner (clone of icl_combined_optimizer.py +
PGramHalfSplitMuon kinds). See docs/halfsplit_constraint.md §7.

Optimizer assignment (the 'sensible design'):
  - SCM  (SoftCompPreservingMuon) -> q_proj, k_proj (QK) + v_proj, o_proj (OV)
         preserves the attention compositions W_Q^T W_K and W_O W_V (pairs intact)
  - GramFlow (GramFlowMuon)       -> gate_proj, up_proj (MLP tall, Stiefel)
  - AdamW                          -> down_proj, embed, head, LayerNorm

Baselines: adam (all AdamW); gramflow (GramFlow on tall q/gate/up + AdamW);
scm (SCM on q/k/v/o + AdamW). Task 0 pretrained once with Adam and reused.
Reports the forgetting matrix F[i,j] = ICL MSE on task j after task i.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_support"))
sys.path.insert(0, str(ROOT / "eigenspectrum" / "scripts"))
torch.set_default_dtype(torch.float64)          # SCM ADMM/Sylvester is precision-sensitive
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import test_soft_comp_continual as T
import icl_chain_common as C
C.DEVICE = DEV
from frozen_residual_ubv import GramFlowMuon, PartialGramMuon
from frozen_residual_ubv.compositional_dual_muon import CompositionalDualMuon
from frozen_residual_ubv.compositional_halfsplit_muon import CompositionalHalfSplitMuon
from frozen_residual_ubv.soft_comp_preserving_muon import SoftCompPreservingMuon
from icl_gqa_model import ICLGQATransformer
from frozen_residual_ubv.pgram_halfsplit_muon import PGramHalfSplitMuon

D, D_MODEL, N_HEADS, N_LAYERS, D_FF, N_CTX, BATCH = 20, 128, 4, 2, 256, 32, 64
HEAD_DIM = D_MODEL // N_HEADS
# n_kv_heads < N_HEADS makes k_proj/v_proj WIDE, reproducing Qwen3-4B's coverage
# gap (only q/gate/up constrained). Set from --n_kv_heads; N_HEADS = stock model.
N_KV_HEADS = N_HEADS
# --seed varies model init, data order, AND the task covariances themselves
# (random orthogonal rotation + task permutation, see C.make_chols). seed=0
# keeps the original axis-aligned family so prior results reproduce exactly.
SEED = 0


def fresh_model():
    torch.manual_seed(42 + 1000 * SEED)
    if N_KV_HEADS == N_HEADS:
        return T.ICLTransformer(D, D_MODEL, N_HEADS, N_LAYERS, D_FF).to(DEV)
    return ICLGQATransformer(D, D_MODEL, N_HEADS, N_LAYERS, D_FF,
                             n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM).to(DEV)


def _attn_named(m):
    return [(n, p) for n, p in m.named_parameters()
            if ".self_attn." in n and "_proj.weight" in n]           # q,k,v,o


def _mlp_tall(m):
    return [p for n, p in m.named_parameters()
            if ".mlp." in n and ("gate_proj.weight" in n or "up_proj.weight" in n)]


def _gram_tall(m):
    return [p for n, p in m.named_parameters()
            if p.ndim == 2 and p.shape[0] >= p.shape[1] and "_proj.weight" in n
            and ("self_attn" in n or "mlp" in n)]                    # q,gate,up


def _down_proj(m):
    return [p for n, p in m.named_parameters() if ".mlp.down_proj.weight" in n]  # wide


def _partial_gram_groups(params, energy):
    """PartialGramMuon groups protecting the top-r eigenblock of the base-model
    input Gram G0 = W0^T W0, r set by energy fraction of trace.

    Works for a WIDE W (m<n) with no transpose: the full input Gram W^T W = I_n
    is infeasible (rank <= m < n), but preserving the top-r block AT ITS ACTUAL
    VALUE Lambda_p is feasible for r <= rank, and W U_p is (m, r) tall.
    r is capped at the numerical rank -- past it Lambda_p holds ~zero
    eigenvalues, lam.rsqrt() explodes and the constraint breaks down.
    U/lam must be float32: PartialGramMuon casts X to float32 internally.
    """
    groups = []
    for p in params:
        Gp = (p.data.t() @ p.data).to(torch.float32)
        ev, evec = torch.linalg.eigh(0.5 * (Gp + Gp.T))
        ev = ev.clamp_min(1e-12)
        order = torch.argsort(ev, descending=True)
        rank = int((ev > ev.max() * 1e-6).sum().item())
        csum = ev[order].cumsum(0) / ev.sum()
        r = min(int((csum < energy).sum().item()) + 1, rank, min(p.shape))
        idx = order[:r]
        groups.append({"params": [p], "U": evec[:, idx].contiguous(),
                       "lam": ev[idx].contiguous(), "core": True, "ceiling": None})
        print(f"  [partialgram] {tuple(p.shape)} energy={energy} -> r={r} "
              f"(rank={rank}, n={p.shape[1]})", flush=True)
    return groups


def _rest(m, used):
    ids = {id(p) for p in used}
    return [p for p in m.parameters() if id(p) not in ids]


def _make_scm(an, args):
    """SCM with the stabilizer knobs exposed.

    scm_renorm='msign' (the library default) DIVERGES on this chain: after the
    Sylvester correction subtracts the protected-core component, msign re-inflates
    every surviving direction back to unit singular value -- including the near-null
    ones the projection just suppressed -- and a second _project subtracts again.
    'none' applies the corrected direction as-is and is stable (see
    icl_scm_stabilize.py). Damping is NOT the lever; it diverges at 1e-2 too.
    """
    return SoftCompPreservingMuon(
        an, lr=args.scm_lr, head_dim=HEAD_DIM, num_kv_heads=N_KV_HEADS,
        attn_name_patterns=[r"(?:.*\.)?layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"],
        qk_energy_threshold=args.scm_energy, ov_energy_threshold=args.scm_energy,
        preserve_qk=True, preserve_ov=True, ns_steps=8,
        renorm_mode=args.scm_renorm, sylvester_damping=args.scm_damping,
        hard_fallback_tol=args.scm_fallback_tol,
        sylvester_skip_tol=args.scm_skip_tol)


def make_opts(m, kind, args):
    if kind == "adam":
        return [torch.optim.Adam(list(m.parameters()), lr=args.adam_lr)]
    if kind == "gramflow":
        g = _gram_tall(m)
        return [GramFlowMuon(g, lr=args.gram_lr, stiefel_rate=args.gamma),
                torch.optim.AdamW(_rest(m, g), lr=args.adamw_lr)]
    if kind == "scm":
        an = _attn_named(m); ap = [p for _, p in an]
        return [_make_scm(an, args),
                torch.optim.AdamW(_rest(m, ap), lr=args.adamw_lr)]
    if kind == "combined":
        an = _attn_named(m); ap = [p for _, p in an]; mt = _mlp_tall(m)
        return [_make_scm(an, args),
                GramFlowMuon(mt, lr=args.gram_lr, stiefel_rate=args.gamma),
                torch.optim.AdamW(_rest(m, ap + mt), lr=args.adamw_lr)]
    if kind == "combined_full":
        # like 'combined' but ALSO GramFlow the wide down_proj (transpose_wide=True
        # -> constrains output Gram W W^T = I, preserving the MLP's output directions)
        an = _attn_named(m); ap = [p for _, p in an]
        mlp_all = [p for n, p in m.named_parameters()
                   if ".mlp." in n and "_proj.weight" in n]      # gate, up, down
        return [_make_scm(an, args),
                GramFlowMuon(mlp_all, lr=args.gram_lr, stiefel_rate=args.gamma,
                             transpose_wide=True),
                torch.optim.AdamW(_rest(m, ap + mlp_all), lr=args.adamw_lr)]
    if kind in ("compdual", "compdual_gramflow"):
        # Joint compositional trust region ||dM||_op <= eps on QK and OV, solved
        # via its dual (docs/compositional_muon_dual.md). Unlike SCM this is
        # spectrally bounded BY CONSTRUCTION -- the ball projection is the last
        # op before the factor split, not a post-hoc cap.
        an = _attn_named(m); ap = [p for _, p in an]
        cd = CompositionalDualMuon(
            an, lr=args.cd_lr, eps=args.cd_eps, head_dim=HEAD_DIM,
            num_kv_heads=N_KV_HEADS, n_outer=args.cd_outer, n_inner=args.cd_inner,
            debug=args.cd_debug,
            attn_name_patterns=[r"(?:.*\.)?layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"])
        if kind == "compdual":
            return [cd, torch.optim.AdamW(_rest(m, ap), lr=args.adamw_lr)]
        mt = _mlp_tall(m)
        return [cd, GramFlowMuon(mt, lr=args.gram_lr, stiefel_rate=args.gamma),
                torch.optim.AdamW(_rest(m, ap + mt), lr=args.adamw_lr)]
    if kind in ("halfsplit", "halfsplit_gramflow"):
        # Tilde's closed-form half-split: ONE msign per factor, no iteration.
        # Same coverage as compdual (q/k/v/o via the compositions) but Muon-class
        # cost -- the joint solve is ~12.6 min/step at Qwen geometry, this is not.
        an = _attn_named(m); ap = [p for _, p in an]
        hs = CompositionalHalfSplitMuon(
            an, lr=args.hs_lr, eps=args.hs_eps, head_dim=HEAD_DIM,
            num_kv_heads=N_KV_HEADS, damping=args.hs_damping,
            whitening=args.hs_whitening,
            attn_name_patterns=[r"(?:.*\.)?layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"])
        if kind == "halfsplit":
            return [hs, torch.optim.AdamW(_rest(m, ap), lr=args.adamw_lr)]
        mt = _mlp_tall(m)
        return [hs, GramFlowMuon(mt, lr=args.gram_lr, stiefel_rate=args.gamma),
                torch.optim.AdamW(_rest(m, ap + mt), lr=args.adamw_lr)]
    if kind == "gramflow_pdown":
        # NO SCM. gramflow control + PartialGramMuon on the wide down_proj:
        # preserves the top-r block of its INPUT Gram (same quantity GramFlow
        # protects on gate/up), leaving the other n-r input dims free.
        g = _gram_tall(m); dp = _down_proj(m)
        return [GramFlowMuon(g, lr=args.gram_lr, stiefel_rate=args.gamma),
                PartialGramMuon(_partial_gram_groups(dp, args.protect_energy),
                                lr=args.pg_lr, dual_steps=args.pg_dual_steps),
                torch.optim.AdamW(_rest(m, g + dp), lr=args.adamw_lr)]
    if kind == "gramflow_tdown":
        # NO SCM. gramflow control + transpose_wide GramFlow on down_proj --
        # the treatment combined_full used, isolated from SCM.
        g = _gram_tall(m); dp = _down_proj(m)
        return [GramFlowMuon(g + dp, lr=args.gram_lr, stiefel_rate=args.gamma,
                             transpose_wide=True),
                torch.optim.AdamW(_rest(m, g + dp), lr=args.adamw_lr)]
    if kind in ("pgram", "pgram_gramflow"):
        # P-Gram half-split: composed-Gram (M^T M, N^T N) top-r subspace
        # preservation on QK and OV, joint two-leg dual, half-split balls.
        an = _attn_named(m); ap = [p for _, p in an]
        pg = PGramHalfSplitMuon(
            an, lr=args.pgram_lr, eps=args.pgram_eps, head_dim=HEAD_DIM,
            num_kv_heads=N_KV_HEADS, damping=args.pgram_damping,
            rank=args.pgram_rank, gamma=args.pgram_gamma,
            dual_steps=args.pgram_dual_steps, dual_lr=args.pgram_dual_lr,
            track_every=10,
            attn_name_patterns=[r"(?:.*\.)?layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"])
        if kind == "pgram":
            return [pg, torch.optim.AdamW(_rest(m, ap), lr=args.adamw_lr)]
        mt = _mlp_tall(m)
        return [pg, GramFlowMuon(mt, lr=args.gram_lr, stiefel_rate=args.gamma),
                torch.optim.AdamW(_rest(m, ap + mt), lr=args.adamw_lr)]
    raise ValueError(kind)


def run(kind, task0_state, chols, args):
    m = fresh_model(); m.load_state_dict(task0_state)
    opts = make_opts(m, kind, args)
    lf = torch.nn.MSELoss()
    states = [{k: v.detach().clone() for k, v in m.state_dict().items()}]
    diag = [float(T._icl_mse(m, chols[0], D, N_CTX))]
    t0 = time.time()
    for t in range(1, args.n_tasks):
        m.train()
        for s in range(args.n_steps):
            Z, y = T._make_icl_batch(N_CTX, D, chols[t], BATCH,
                                     seed=t * 100000 + s + 10_000_000 * SEED)
            for o in opts:
                o.zero_grad()
            lf(m(Z), y).backward()
            for o in opts:
                o.step()
        mse_i = float(T._icl_mse(m, chols[t], D, N_CTX))
        diag.append(mse_i)
        states.append({k: v.detach().clone() for k, v in m.state_dict().items()})
        finite = all(torch.isfinite(p).all() for p in m.parameters())
        # max|W| is the LEADING divergence indicator -- the ICL MSE lags a weight
        # explosion by several tasks (SCM hit max|W|=3.4e34 while MSE still read 7.7).
        wmax = max(float(p.detach().abs().max()) for p in m.parameters())
        extra = ""
        for o in opts:
            if hasattr(o, "get_metrics"):
                mt = o.get_metrics()
                if mt:
                    extra = "  " + " ".join(f"{k.split('/')[-1]}={v:.3e}"
                                            for k, v in mt.items())
        print(f"  [{kind}] T{t:02d}: MSE={mse_i:.3f} finite={finite} "
              f"max|W|={wmax:.3e} ({(time.time()-t0)/60:.1f}m){extra}", flush=True)
        if not finite:
            print(f"  [{kind}] DIVERGED T{t}", flush=True); break
    n = args.n_tasks
    F = np.full((n, n), np.nan)
    for i, st in enumerate(states):
        m.load_state_dict(st)
        for j in range(i + 1):
            F[i, j] = float(T._icl_mse(m, chols[j], D, N_CTX))
    return F, np.array(diag)


def summarize(kind, F):
    n = F.shape[0]; done = int(np.sum(~np.isnan(np.diag(F))))
    dm = np.nanmedian(np.diag(F)); t0 = F[done - 1, 0]
    pe = np.nanmean(F[done - 1, :done - 1]) if done > 1 else np.nan
    pc = np.nanmean([np.nanmean(F[i, :i]) for i in range(1, done)]) if done > 1 else np.nan
    print(f"  [{kind}] done={done}/{n}  diag_med={dm:.2f}  task0_end={t0:.2f}  "
          f"past_end={pe:.2f}  past_chain={pc:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=10)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--tau0", type=float, default=4.0)
    ap.add_argument("--n_steps_0", type=int, default=5000)
    ap.add_argument("--n_steps", type=int, default=400)
    ap.add_argument("--adam0_lr", type=float, default=3e-4)
    ap.add_argument("--adam_lr", type=float, default=3e-3)
    ap.add_argument("--adamw_lr", type=float, default=3e-3)
    ap.add_argument("--gram_lr", type=float, default=3e-3)
    ap.add_argument("--scm_lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.01)
    # CompositionalDualMuon: (5,3) reaches 99.3% of the optimal <Y,X> at ~1/14 the
    # cost of (25,12) — see the cost/quality sweep in the benchmark.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_kv_heads", type=int, default=N_HEADS)   # < N_HEADS -> GQA
    ap.add_argument("--head_dim", type=int, default=D_MODEL // N_HEADS)  # >d_model/n_h -> wide o_proj
    ap.add_argument("--hs_lr", type=float, default=1e-3)
    ap.add_argument("--hs_eps", type=float, default=1.0)
    ap.add_argument("--hs_damping", type=float, default=1e-6)
    ap.add_argument("--hs_whitening", default="ns", choices=["ns", "eigh"])
    ap.add_argument("--cd_lr", type=float, default=1e-3)
    ap.add_argument("--cd_eps", type=float, default=1.0)   # trust region on ||dM||_op
    ap.add_argument("--cd_outer", type=int, default=5)
    ap.add_argument("--cd_inner", type=int, default=3)
    ap.add_argument("--cd_debug", action="store_true")   # SVD invariants (slow)
    # SCM stabilizers — 'none' is the stable renorm; 'msign' (lib default) diverges
    ap.add_argument("--scm_renorm", default="none",
                    choices=["none", "msign", "frobenius", "spectral"])
    ap.add_argument("--scm_damping", type=float, default=1e-4)
    ap.add_argument("--scm_energy", type=float, default=0.8)
    ap.add_argument("--scm_fallback_tol", type=float, default=0.5)
    ap.add_argument("--scm_skip_tol", type=float, default=0.0)
    ap.add_argument("--protect_energy", type=float, default=0.8)   # PartialGram rank
    ap.add_argument("--pg_lr", type=float, default=3e-3)
    ap.add_argument("--pg_dual_steps", type=int, default=1)
    ap.add_argument("--pgram_lr", type=float, default=1e-3)
    ap.add_argument("--pgram_eps", type=float, default=1.0)
    ap.add_argument("--pgram_damping", type=float, default=1e-6)
    ap.add_argument("--pgram_rank", type=int, default=8)
    ap.add_argument("--pgram_gamma", type=float, default=1e-3)
    ap.add_argument("--pgram_dual_steps", type=int, default=2)
    ap.add_argument("--pgram_dual_lr", type=float, default=0.25)
    ap.add_argument("--configs", default="adam,gramflow,scm,combined")
    ap.add_argument("--out", default=str(ROOT / "eigenspectrum" / "outputs" / "icl_combined.npz"))
    args = ap.parse_args()
    global N_KV_HEADS, HEAD_DIM, SEED
    SEED = args.seed
    N_KV_HEADS = args.n_kv_heads
    HEAD_DIM = args.head_dim

    chols = [c.to(DEV) for c in C.make_chols(args.n_tasks, args.k, tau=args.tau0,
                                             seed=(args.seed or None))]
    # task 0 pretrain once (Adam), reused
    m0 = fresh_model(); lf = torch.nn.MSELoss()
    ad0 = torch.optim.Adam(list(m0.parameters()), lr=args.adam0_lr)
    for s in range(args.n_steps_0):
        Z, y = T._make_icl_batch(N_CTX, D, chols[0], BATCH,
                                 seed=s + 500_000_000 * SEED)
        ad0.zero_grad(); lf(m0(Z), y).backward(); ad0.step()
    task0 = {k: v.detach().clone() for k, v in m0.state_dict().items()}
    print(f"[combined-icl] task0 MSE={float(T._icl_mse(m0, chols[0], D, N_CTX)):.3f}; "
          f"configs={args.configs}", flush=True)

    out = {}
    for kind in args.configs.split(","):
        print(f"\n=== {kind} ===", flush=True)
        F, diag = run(kind, task0, chols, args)
        summarize(kind, F)
        out[f"F_{kind}"] = F; out[f"diag_{kind}"] = diag
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()

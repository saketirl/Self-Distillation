"""Measure composition movement over the ICL chain.

Reruns the 20-task GQA chain (identical seeds -> identical trajectory) for
given configs and, at every task boundary, logs how far the composed operators
have moved, per layer/head, relative to the CHAIN START (post-task0 weights):

  relM      = ||M_t - M_0||_F / ||M_0||_F          (score map drift)
  relGram   = ||M_t^T M_t - M_0^T M_0||_F / ||.||  (composed-Gram drift)
  relGram_r = same restricted to the top-r subspace V_r of M_0^T M_0
              (the part P-Gram explicitly protects; r matches the optimizer)
plus the same three for N = W_O W_V. Saved per task to npz.
"""
import sys, argparse, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_support"))
sys.path.insert(0, str(ROOT / "eigenspectrum" / "scripts"))
torch.set_default_dtype(torch.float64)

import icl_pgram_sweep as R                      # reuse harness verbatim
import test_soft_comp_continual as T
import icl_chain_common as C

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C.DEVICE = DEV
R.DEV = DEV


def _heads(m):
    """Per layer: (WQ, WK, WO, WV) head stacks in math convention."""
    out = []
    d_h, H_kv = R.HEAD_DIM, R.N_KV_HEADS
    sd = {n: p.detach() for n, p in m.named_parameters()}
    for li in range(R.N_LAYERS):
        pre = None
        for cand in (f"layers.{li}.self_attn.", f"model.layers.{li}.self_attn."):
            if cand + "q_proj.weight" in sd:
                pre = cand
                break
        if pre is None:
            continue
        q, k = sd[pre + "q_proj.weight"], sd[pre + "k_proj.weight"]
        v, o = sd[pre + "v_proj.weight"], sd[pre + "o_proj.weight"]
        H_q = q.shape[0] // d_h
        grp = max(H_q // H_kv, 1)
        d = q.shape[1]
        WQ = q.view(H_q, d_h, d).mT
        WK = k.view(H_kv, d_h, d).mT.repeat_interleave(grp, 0)
        WO = o.view(d, H_q, d_h).permute(1, 0, 2)
        WV = v.view(H_kv, d_h, d).repeat_interleave(grp, 0)
        out.append((WQ, WK, WO, WV))
    return out


def _comps(m):
    Ms, Ns = [], []
    for WQ, WK, WO, WV in _heads(m):
        Ms.append(WQ @ WK.mT)                     # (H, d, d)
        Ns.append(WO @ WV)                        # (H, d, d)
    return Ms, Ns


def _topr(Gs, r):
    Vs = []
    for G in Gs:
        ev, V = torch.linalg.eigh(0.5 * (G + G.mT))
        Vs.append(V[..., -r:])
    return Vs


def _rel(a, b):
    return float(((a - b).norm() / b.norm().clamp_min(1e-12)))


def measure(kind, args):
    torch.manual_seed(0)
    chols = [c.to(DEV) for c in C.make_chols(args.n_tasks, 1, tau=4.0, seed=None)]
    m0 = R.fresh_model()
    lf = torch.nn.MSELoss()
    ad0 = torch.optim.Adam(list(m0.parameters()), lr=3e-4)
    for s in range(args.n_steps_0):
        Z, y = T._make_icl_batch(R.N_CTX, R.D, chols[0], R.BATCH, seed=s)
        ad0.zero_grad(); lf(m0(Z), y).backward(); ad0.step()
    task0 = {k_: v.detach().clone() for k_, v in m0.state_dict().items()}

    m = R.fresh_model(); m.load_state_dict(task0)
    opts = R.make_opts(m, kind, args)
    M0, N0 = _comps(m)
    GM0 = [Mi.mT @ Mi for Mi in M0]
    GN0 = [Ni.mT @ Ni for Ni in N0]
    VrM = _topr(GM0, args.pgram_rank)
    VrN = _topr(GN0, args.pgram_rank)

    recs = []
    for t in range(1, args.n_tasks):
        m.train()
        for s in range(args.n_steps):
            Z, y = T._make_icl_batch(R.N_CTX, R.D, chols[t], R.BATCH,
                                     seed=t * 100000 + s)
            for o in opts:
                o.zero_grad()
            lf(m(Z), y).backward()
            for o in opts:
                o.step()
        Ms, Ns = _comps(m)
        row = {"task": t}
        for name, cur, ref, Gref, Vr in (("M", Ms, M0, GM0, VrM),
                                          ("N", Ns, N0, GN0, VrN)):
            rel, relg, relr = [], [], []
            for Ci, Ri, Gi, Vi in zip(cur, ref, Gref, Vr):
                rel.append(_rel(Ci, Ri))
                Gc = Ci.mT @ Ci
                relg.append(_rel(Gc, Gi))
                relr.append(_rel(Vi.mT @ Gc @ Vi, Vi.mT @ Gi @ Vi))
            row[f"rel{name}"] = float(np.mean(rel))
            row[f"relGram_{name}"] = float(np.mean(relg))
            row[f"relGram_{name}_topr"] = float(np.mean(relr))
        mse0 = float(T._icl_mse(m, chols[0], R.D, R.N_CTX))
        row["task0_mse"] = mse0
        recs.append(row)
        print(f"[{kind}] T{t:02d} relM={row['relM']:.3f} "
              f"gramM={row['relGram_M']:.3f} gramM_top{args.pgram_rank}="
              f"{row['relGram_M_topr']:.3f} | relN={row['relN']:.3f} "
              f"gramN={row['relGram_N']:.3f} gramN_topr={row['relGram_N_topr']:.3f} "
              f"| task0_mse={mse0:.2f}", flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=20)
    ap.add_argument("--n_steps_0", type=int, default=5000)
    ap.add_argument("--n_steps", type=int, default=400)
    ap.add_argument("--kinds", default="pgram_gramflow,adam")
    ap.add_argument("--out", default=str(ROOT / "eigenspectrum" / "outputs" /
                                         "pgram_sweep" / "composition_drift.json"))
    # winning point defaults
    ap.add_argument("--pgram_lr", type=float, default=1e-3)
    ap.add_argument("--adamw_lr", type=float, default=3e-4)
    ap.add_argument("--adam_lr", type=float, default=3e-3)
    ap.add_argument("--gram_lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--pgram_eps", type=float, default=1.0)
    ap.add_argument("--pgram_damping", type=float, default=1e-6)
    ap.add_argument("--pgram_rank", type=int, default=8)
    ap.add_argument("--pgram_gamma", type=float, default=1e-3)
    ap.add_argument("--pgram_dual_steps", type=int, default=2)
    ap.add_argument("--pgram_dual_lr", type=float, default=0.25)
    args = ap.parse_args()
    R.N_KV_HEADS = 1
    R.HEAD_DIM = 48
    R.SEED = 0
    out = {}
    for kind in args.kinds.split(","):
        print(f"=== measuring {kind} ===", flush=True)
        out[kind] = measure(kind, args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()

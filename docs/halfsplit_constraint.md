# The Half-Split as a Constrained Optimization Problem

How `CompositionalHalfSplitMuon` is defined mathematically: the constraint it
imposes, why that constraint (and not the one we actually want), and the
mechanism by which it is *maintained exactly at every step, by construction* —
no dual loop, no projection, no accumulated error.

Implementation: `frozen_residual_ubv/compositional_halfsplit_muon.py`.
Source: Tilde's Compositional Muon (blog.tilderesearch.com/blog/compositional-muon).
The coupled key-leg extension lives in
`frozen_residual_ubv/coupled_key_halfsplit_muon.py` (see §6).

Conventions: per attention head, math orientation. $W_Q, W_K \in
\mathbb{R}^{d \times d_h}$ (columns are head dimensions, $d = d_{\text{model}}$),
$W_O \in \mathbb{R}^{d \times d_h}$, $W_V \in \mathbb{R}^{d_h \times d}$.
$\operatorname{msign}(X) = UV^\top$ for the reduced SVD $X = U\Sigma V^\top$
(computed with Polar Express Newton–Schulz, batched over heads).

---

## 1. The problem we want (and cannot solve in closed form)

The loss does not see $W_Q$ or $W_K$ individually; it sees their composition —
the score bilinear form and the value-write map:

$$
M \;=\; W_Q W_K^\top \in \mathbb{R}^{d\times d}, \qquad
N \;=\; W_O W_V \in \mathbb{R}^{d\times d}.
$$

A first-order change of the factors induces

$$
\Delta M \;=\; \Delta W_Q\, W_K^\top \;+\; W_Q\, \Delta W_K^\top ,
$$

so the *right* trust region for one optimizer step is on the composed operator:

$$
\textbf{(P)}\qquad
\min_{\Delta W_Q,\,\Delta W_K}\;
\langle G_Q, \Delta W_Q\rangle + \langle G_K, \Delta W_K\rangle
\quad\text{s.t.}\quad
\lVert \Delta M \rVert_{\mathrm{op}} \;\le\; \varepsilon .
$$

**(P) has no msign-style closed form.** Achievable perturbations satisfy
$\operatorname{rank}(\Delta M) \le 2 d_h \ll d$, while the unconstrained
minimizer of a linear functional over the op-norm ball,
$-\varepsilon\,\operatorname{msign}(G_M)$, is generically full-rank — a rank
obstruction (numerically: forcing the Hölder form anyway gives ~75%
reconstruction error; see `docs/compositional_muon_dual.md` §5).

## 2. The half-split constraint

Give each *leg* of $\Delta M$ its own budget of $\varepsilon/2$:

$$
\boxed{\;
\lVert \Delta W_Q\, W_K^\top \rVert_{\mathrm{op}} \;\le\; \tfrac{\varepsilon}{2}
\qquad\text{and}\qquad
\lVert W_Q\, \Delta W_K^\top \rVert_{\mathrm{op}} \;\le\; \tfrac{\varepsilon}{2}
\;}
$$

By the triangle inequality these **imply** the joint constraint of (P):

$$
\lVert \Delta M\rVert_{\mathrm{op}}
\;\le\;
\lVert \Delta W_Q W_K^\top\rVert_{\mathrm{op}}
+ \lVert W_Q \Delta W_K^\top\rVert_{\mathrm{op}}
\;\le\; \varepsilon .
$$

The half-split feasible set is therefore a **conservative inner approximation**
of (P)'s: every half-split-feasible update is (P)-feasible, not conversely.
The measured slack is ~30% — the realized $\lVert\Delta M\rVert$ uses only
0.65–0.8 of the joint budget (wandb `qk_composed_ratio`, `ov_composed_ratio`) —
which is the price paid for what the split buys: each leg separately *does*
have a closed form.

Note what the constraint is on: the **update**, not the weights. There is no
manifold the weights must stay on, hence no feasibility to restore, no anchor,
and no drift-correction term. Each step's constraint is defined fresh at the
current point and discharged within that step.

## 3. How the constraint is maintained: whitening + Hölder

Take the $\Delta W_Q$ leg (the others are symmetric). The constraint norm is
*skewed* by the partner matrix. Un-skew it with the partner's Gram root:

$$
C_K \;=\; \big(W_K^\top W_K + \lambda I\big)^{1/2} \in \mathbb{R}^{d_h\times d_h},
\qquad
\lVert \Delta W_Q\, W_K^\top \rVert_{\mathrm{op}}
= \lVert \Delta W_Q\, C_K \rVert_{\mathrm{op}}
\quad (\lambda = 0),
$$

which holds because $W_K = U_K C_K$ with $U_K$ orthonormal-column (polar
decomposition), and multiplying by $U_K^\top$ on the right preserves singular
values. Substituting $Y = \Delta W_Q C_K$ turns the leg subproblem into a
linear functional over a *plain* spectral ball:

$$
\min_Y\; \big\langle G_Q C_K^{-1},\, Y \big\rangle
\quad\text{s.t.}\quad \lVert Y\rVert_{\mathrm{op}} \le \tfrac{\varepsilon}{2},
$$

whose minimizer, by Hölder duality
($\langle A,B\rangle \ge -\lVert A\rVert_* \lVert B\rVert_{\mathrm{op}}$, tight
at the matrix sign), is $Y^\star = -\tfrac{\varepsilon}{2}
\operatorname{msign}(G_Q C_K^{-1})$. Un-substituting:

$$
\boxed{\;
\Delta W_Q \;=\; -\tfrac{\varepsilon}{2}\,
\operatorname{msign}\!\big(G_Q\, C_K^{-1}\big)\, C_K^{-1}
\;}
$$

and symmetrically $\Delta W_K = -\tfrac{\varepsilon}{2}\operatorname{msign}(G_K
C_Q^{-1})\, C_Q^{-1}$. For OV the asymmetry of the circuit (V acts head-locally
before aggregation, O globally after) gives the mirrored forms

$$
\Delta W_O = -\tfrac{\varepsilon}{2}\operatorname{msign}\!\big(G_O C_V^{-1}\big) C_V^{-1},
\qquad
\Delta W_V = -\tfrac{\varepsilon}{2}\, C_O^{-1} \operatorname{msign}\!\big(C_O^{-1} G_V\big),
$$

with $C_V = (W_V W_V^\top + \lambda I)^{1/2}$, $C_O = (W_O^\top W_O + \lambda I)^{1/2}$.

**Why this maintains the constraint exactly.** The solution *saturates* the
budget by construction: $\lVert \Delta W_Q W_K^\top\rVert_{\mathrm{op}} =
\lVert Y^\star\rVert_{\mathrm{op}} = \varepsilon/2$ (msign output has unit
spectral norm). Enforcement is not iterative and cannot fail to converge —
the constraint is satisfied *identically* by the functional form of the update.
Measured: `leg_ratio_mean` $= 0.9999$, `leg_ratio_max` $= 1.000$ across
training. Three refinements:

1. **Damping is conservative.** With $\lambda>0$, $C_K^2 \succeq W_K^\top W_K$
   implies $\lVert \Delta W_Q W_K^\top\rVert \le \lVert \Delta W_Q
   C_K\rVert = \varepsilon/2$ — the realized leg norm is *at most* the budget.
   ($\lambda$ is relative, $\lambda = \texttt{damping}\cdot
   \overline{\operatorname{diag}}$; $10^{-3}$ is required for the
   Newton–Schulz inverse-root to converge at high condition numbers.)
2. **The learning rate multiplies afterwards**: the applied update is
   $\text{lr}\cdot\Delta W$, so the effective per-step composed budget is
   $\text{lr}\cdot\varepsilon$. Only the product matters (`hs_lr` and
   `hs_eps` are degenerate).
3. **msign is approximate** (8 Polar-Express iterations): spectral norm of the
   output is $1 + O(10^{-3})$, hence the 1.000-not-0.999 max ratios.

## 4. Grouped-query attention: whitening by the group-summed Gram

Under GQA a K/V head is shared by a group $\mathcal{G}$ of query heads, and the
leg constraint must hold **for every** $h \in \mathcal{G}$:
$\lVert W_{Q,h}\, \Delta W_K^\top \rVert_{\mathrm{op}} \le \varepsilon/2$.
Computing per-$h$ updates and averaging them does *not* satisfy this (each term
is bounded only against its own $W_{Q,h}$; measured violation up to $4.4\times$
on random weights). Instead whiten once by the **group-summed Gram**:

$$
C_{\mathcal G} = \Big(\textstyle\sum_{h\in\mathcal G} W_{Q,h}^\top W_{Q,h}
+ \lambda I\Big)^{1/2},
\qquad
\Delta W_K = -\tfrac{\varepsilon}{2}\operatorname{msign}\!\big(G_K\,
C_{\mathcal G}^{-1}\big)\, C_{\mathcal G}^{-1}.
$$

Since $W_{Q,h}^\top W_{Q,h} \preceq C_{\mathcal G}^2$ for each $h$,
$\lVert W_{Q,h}\Delta W_K^\top\rVert \le \lVert C_{\mathcal G}\Delta
W_K^\top\rVert = \varepsilon/2$ — the per-head guarantee is restored, with one
solve per KV head. (`CoupledKeyHalfSplitMuon` uses this; the parent class
retains the historical average form. The same construction applies one level up
for Gemma-4-style cross-*layer* KV sharing, summing Grams over the sharing set —
not yet implemented.)

## 5. What is and is not guaranteed

- **Guaranteed, per step, exactly:** each leg $\le \varepsilon/2$ (hence
  $\lVert\Delta M\rVert_{\mathrm{op}} \le \varepsilon$,
  $\lVert\Delta N\rVert_{\mathrm{op}} \le \varepsilon$), in the whitened
  metric, conservatively under damping.
- **Not constrained:** the cumulative drift $\sum_t \Delta M_t$ — the
  half-split bounds steps, not trajectories. Retention over a task sequence is
  an empirical claim, not an algebraic one (contrast FPMuon/SCM subspace
  equalities, which anchor state).
- **Not optimal:** the $\varepsilon/2{+}\varepsilon/2$ split of the joint
  budget is the tractable inner approximation; ~30% of the joint budget goes
  unused (§2).

## 6. The coupled key leg (extension)

When the key matrix answers to a second atom — the key self-Gram
$M_{kk} = W_K W_K^\top$, functional under gated-delta-net recurrences and,
partially, under QK-norm — the k-leg gains an *equality* constraint on a
protected eigensubspace $U_r$ of the task-start anchor:

$$
\min_{\Delta W_K} \langle G_K, \Delta W_K\rangle
\quad\text{s.t.}\quad
\lVert W_Q \Delta W_K^\top\rVert_{\mathrm{op}} \le \tfrac{\varepsilon}{2},
\qquad
U_r^\top\!\big(\Delta W_K W_K^\top + W_K \Delta W_K^\top\big) U_r
= -\gamma\, E_r ,
$$

with leak $E_r = U_r^\top (M_{kk} - M_{kk}^{\text{anchor}}) U_r$. Unlike the
inequality (maintained by construction, §3), the equality is enforced by a
**warm-started dual ascent**: a multiplier $\Lambda_r \in \mathrm{Sym}(r)$
enters *inside* the whitened msign,

$$
\Delta W_K(\Lambda_r) = -\tfrac{\varepsilon}{2}
\operatorname{msign}\!\Big(\big(\hat G_K + 2\,U_r \Lambda_r U_r^\top W_K\big)
C_{\mathcal G}^{-1}\Big)\, C_{\mathcal G}^{-1},
\qquad
\Lambda_r \leftarrow \Lambda_r + \alpha\, \frac{r(\Lambda_r)}{s},
$$

$K{=}2$ iterations per step, best-iterate selection (msign chattering), duals
persisted across steps — soft enforcement that tightens over tens of steps
(GramFlow discipline), measured `k_leak_ratio` $\approx 0.04$. Derivation and
compute audit: session notes 2026-08-13; tests in
`tests/test_coupled_key_halfsplit.py`.

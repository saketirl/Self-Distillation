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

---

## 7. Preserving the Gram of the composition: $M^\top M$ (proposed)

Instead of only bounding $\Delta M$, additionally *preserve the Gram of the
composed operator* — the composition-level analogue of Stiefel's
$W^\top W = I$ and of the gram-manifold view
(blog.tilderesearch.com/vignettes/gram-space), which treats the manifold
$\{W : W^\top W = G\}$ for a single weight matrix. Here the manifold lives in
**composed space**, as the quadratic preimage
$\{(W_Q, W_K) : (W_Q W_K^\top)^\top (W_Q W_K^\top) = G_0\}$:

$$
\textbf{(P-Gram)}\qquad
\min_{\Delta W_Q,\,\Delta W_K}\;
\langle G_Q, \Delta W_Q\rangle + \langle G_K, \Delta W_K\rangle
\quad\text{s.t.}\quad
\lVert \Delta M \rVert_{\mathrm{op}} \le \varepsilon,
\qquad
\operatorname{sym}\!\big(M^\top \Delta M\big) = 0 ,
$$

using $D(M^\top M)[\Delta] = 2\operatorname{sym}(M^\top \Delta M)$ — the
standard gram-manifold tangency with $W \to M$, pulled back through
$\Delta M = \Delta W_Q W_K^\top + W_Q \Delta W_K^\top$. In factor space:

$$
\operatorname{sym}\Big(
W_K \big(W_Q^\top \Delta W_Q\big) W_K^\top
\;+\;
W_K\, C_Q^2\, \Delta W_K^\top
\Big) = 0,
\qquad C_Q^2 = W_Q^\top W_Q .
$$

### 7.1 What it preserves

$M^\top M = V \Sigma^2 V^\top$ is the right singular structure of the score
map: **the key-side directions the head can attend over, and their gains**.
Tangency freezes $V, \Sigma$ to first order and leaves the query-side frame
$U$ free — retention of *what can be attended to*, plasticity in *what
attends*. Note the identity

$$
M^\top M \;=\; W_K\, C_Q^{2}\, W_K^\top
$$

— the composed Gram **is** the key self-Gram measured in the query metric.

### 7.2 Gauge invariance (the decisive property)

Under the attention gauge $(W_Q, W_K) \to (W_Q R^{-\top},\, W_K R)$, $M$ — and
hence $M^\top M$ — is invariant. The constraint therefore spends budget only
on functional directions. This corrects the weakest point of §6: the raw key
self-Gram $M_{kk} = W_K W_K^\top$ is *not* gauge-invariant in pure softmax
attention (QK-norm only partially breaks the gauge), so §6's anchor can bind
non-functional motion. **§6 is exactly the $C_Q = I$ approximation of
(P-Gram)** — same machinery, unweighted metric.

### 7.3 Rigidity of the exact constraint

Splitting by the projector $P_K$ onto $\operatorname{col}(W_K)$: the
$\perp$–$K$ cross-block of the equality forces
$(I - P_K)\,\Delta W_K\, C_Q^2 = 0$, i.e. $\Delta W_K$'s columns must remain
in $\operatorname{col}(W_K)$ — **no new key directions, ever** — plus a
$d_h \times d_h$ core equality that couples $\Delta W_Q$ and $\Delta W_K$.
Same over-rigidity as §6's full tangency, with one added consequence: the
coupling breaks the per-leg separability that made the half-split closed-form.

### 7.4 Practical (soft, subspace) form and solver

Protect the top-$r$ right singular subspace $V_r$ of the **task-start** $M$,
with contraction toward the anchored core
$E_r = V_r^\top (M^\top M - (M^\top M)^{\text{anchor}}) V_r$:

$$
V_r^\top\, \operatorname{sym}(M^\top \Delta M)\, V_r \;=\; -\gamma\, E_r .
$$

Dualizing with one shared $\Lambda_r \in \mathrm{Sym}(r)$ per head tilts
**both** legs (the pairing $\langle \Lambda, \operatorname{sym}(M^\top \Delta
M)\rangle$ contributes to each factor's gradient):

$$
\tilde G_Q = G_Q + 2\, M\, \bar\Lambda\, W_K, \qquad
\tilde G_K = G_K + 2\, \bar\Lambda\, M^\top W_Q,
\qquad \bar\Lambda = V_r \Lambda_r V_r^\top ,
$$

all thin products (no $d\times d$ object: $MV_r$, $W_K^\top V_r$ etc. are
$d\times r$ / $d_h\times r$). Each tilted leg is then solved by the §3
whitened msign under its $\varepsilon/2$ ball, and $\Lambda_r$ ascends on the
$r\times r$ residual of the **joint** realized $\Delta M$ — $K$ warm-started
iterations, best-iterate selection, exactly the §6 discipline except the
residual must be computed from both legs together (they are no longer
independent given $\Lambda_r$).

Cost over §6: one extra $d_h\times d_h$ Gram ($C_Q^2$) per head per step, and
$V_r$ once per task from the thin SVD of the factor pair
($M = (W_Q)(W_K)^\top$, QR of each thin factor, SVD of the $d_h\times d_h$
core — never forming $M$).

Status: formulated; not yet implemented. The delta from
`CoupledKeyHalfSplitMuon` is (i) anchor $V_r$ of $M$ instead of eigenvectors
of $W_KW_K^\top$, (ii) the $C_Q^2$ weighting, (iii) the joint two-leg residual
in the dual loop.

---

## 8. Deriving the Dion3-inspired modifications (not adopting them as recipes)

Each modification must fall out of the same variational template as §§1–7:
*choose the estimator of the linear functional, choose the norm(s), dualize,
solve by Hölder.* Otherwise it has no business inside a constrained optimizer.

### 8.1 Momentum = shrinkage estimation of the linear functional
The per-step program (P) minimizes $\langle G_t, \Delta\rangle$; but $G_t$
is a noisy estimate of the true descent functional $g = \mathbb E[G]$. Choose
instead the exponentially-weighted maximum-likelihood estimator

$$
M_t \;=\; \arg\min_m \sum_{s\le t} \mu^{\,t-s}\, \lVert m - G_s\rVert_F^2
\;=\; (1-\mu)\sum_{s\le t} \mu^{\,t-s} G_s ,
$$

i.e. the EMA (Dion3's buffer at $f{=}1$ is this up to normalization, with its
$M \leftarrow \mu M$ decay-after-use). Substituting $M_t$ for $G_t$ changes
**only the coefficient of a linear objective** — every downstream piece
(whitening, the $\varepsilon/2$ Hölder step, the tilts $2W\Lambda$, the dual
ascent) is invariant to this substitution, because none of them depend on the
objective being the instantaneous gradient. Hence momentum requires *no*
recertification: the applied update is still exactly
$-\tfrac{\varepsilon}{2}\operatorname{msign}(\cdot)C^{-1}$. At $f<1$,
Dion3's error feedback is the same estimator with a low-rank observation
operator: unapplied components remain in the state until observed —
in our setting the observation operator must additionally annihilate the
protected subspace, or the estimator accumulates exactly the motion the
constraint exists to forbid (§ noted; f=1 sidesteps this).

### 8.2 Per-neuron scaling = an intersected trust region, derived
Adam's derivation, one level up: gradient noise is heteroscedastic across
output neurons; let $v_i$ estimate the second moment of row $i$. A trust
region that equalizes estimation *risk* rather than motion allots each neuron
budget $\eta_i \propto 1/\sqrt{v_i}$. The derived program is the ball
INTERSECTION

$$
\min_\Delta \langle M, \Delta\rangle
\quad\text{s.t.}\quad
\lVert W_Q \Delta^\top\rVert_{\mathrm{op}} \le \tfrac{\varepsilon}{2}
\;\;\wedge\;\;
\lVert e_i^\top \Delta\rVert \le c/\sqrt{v_i}\;\;\forall i .
$$

KKT: a matrix multiplier for the composed ball plus one scalar multiplier per
neuron — no closed form for the exact intersection (same rank-obstruction
flavor as §1). The tractable inner approximation, in the spirit of the
half-split itself: (i) take the composed-ball solution, (ii) apply the
diagonal reweighting $D^{-1} = \operatorname{diag}(1/(\sqrt{v_i}+\epsilon))$
(Frobenius-rescaled: the budget is redistributed, not shrunk), (iii) project
back onto the composed ball **along the ray** — i.e. rescale by
$\min\!\big(1, \tfrac{\varepsilon/2}{\lVert W_Q \hat\Delta^\top\rVert}\big)$.
Step (iii) is the exact Euclidean projection onto the ball along the search
direction (one Dykstra half-step onto the intersection), so the applied update
is feasible for BOTH constraint sets by construction. This is precisely the
implemented `_apply_leg` + recertification: not a safety patch but the
inner-approximate solution of the derived intersection program. (A purist
would iterate (ii)–(iii) to the Dykstra fixed point; one step is our standing
half-split philosophy of conservative inner approximations.)

### 8.3 Gram Newton–Schulz = an exact algebraic rewriting (nothing to re-derive)
Dion3 Thm 2: every Newton–Schulz iterate is an odd polynomial of the initial
matrix, so all iterates commute and the recursion transfers to the small Gram
$R = XX^\top$:
$z_t = a I + b R_t + c R_t^2$, $R_{t+1} = R_t z_t^2$, $Q_{t+1} = z_t Q_t$,
$\operatorname{msign}(X) \approx Q_T X$. Identical operator, fewer big-side
FLOPs (ratio $\sim$ aspect); constraint semantics untouched. Verified to
$<5\%$ of the Polar-Express output at our step counts, fp32, no restart
needed.

### 8.4 What was *not* derived and therefore *not* adopted
Dion3's row-subset selection ($f<1$) and its $\eta\propto 1/\sqrt f$
transfer rule are performance devices whose interaction with the protected
subspace (via error feedback, §8.1) changes constraint semantics; they stay
out until the annihilating observation operator is written down and tested.

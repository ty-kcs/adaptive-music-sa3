# Optimization (T / S corners)

The UI sliders are **T** (temporal regularity) and **S** (spectral /
harmonic coherence). For each `(T, S)` we run
**FlowEdit** with a small parameter dictionary θ (`t_start`, `tgt_cfg`,
…) and a prompt from a fixed table. Optimization only searches those
FlowEdit settings.

**Generate** (including `--demo`) loads a precomputed parameteer json file and edits the upload at the requested `(T, S)`. (so simply running the sa-3 model. fast.)

## Phase 0 and Phase 1

**Phase 0** takes time. Bayesian optimization tries
many candidate maps on 11 different types of ambient clips, keeps the one with the
best mean score $J$, and writes `g0_ts.json`. That file is meant to
be reused.

**Phase 1** is a per-clip **Optimize** button: a short local search
around `g0_ts.json` on a 3×3 grid. It is implemented, but so far it
does not clearly improve from Phase 0, so `--demo` does not show it.

## What one trial for bayesian optimization does

1. Propose eight numbers: `t_start` and `tgt_cfg` at each of the four
   corners. `src_cfg` and `t_stop` stay fixed for now.
2. Turn those into four θ dictionaries. Any other `(T, S)` is a
   bilinear mix of the four (see Map).
3. Edit the search clips on a grid of commanded `(T, S)` — 5×5 in
   Phase 0, 3×3 in Phase 1.
4. Measure DSP `tc` / `hc` and latent drift from the source. Fold that
   into a scalar $J$ (higher is better).
5. Repeat. `skopt.gp_minimize` maximizes mean $J$ (it minimizes $-J$).
   Phase 0 then re-ranks a shortlist on a held-out validation set.

## Map

We store four FlowEdit dictionaries, one at each corner of the unit
square: `(T,S) = (0,0)`, `(1,0)`, `(0,1)`, `(1,1)`.

Anywhere between those corners, parameters are a bilinear mix. For
example `(0.5, 0.5)` is an equal blend; `(1, 0)` is exactly the T-high
/ S-low corner.

$$
\theta(T,S)=(1-T)(1-S)\theta_{00}+T(1-S)\theta_{10}+(1-T)S\theta_{01}+TS\theta_{11}.
$$

Prompt: `"{genre}, {T phrase}, {S phrase}"` (five rungs on each axis).
The Phase-0 grid is every pair of T and S in
`{0, 0.25, 0.5, 0.75, 1}` (25 edits).

## Objective $J$

$J$ asks: did the knobs do what we asked? Higher is better.

$$
\begin{aligned}
J &= \rho(T,\mathrm{tc})+\rho(S,\mathrm{hc}) \\
&\quad -\lambda_x\big(|\rho(T,\mathrm{hc})|+|\rho(S,\mathrm{tc})|\big) \\
&\quad +\rho(\max(T,S),\mathrm{drift}) \\
&\quad -\lambda_m\big(\mathrm{viol}_T+\mathrm{viol}_S\big) \\
&\quad -\lambda_c\sum_{c}\max(0,\,d_{00}+m-d_c)
\end{aligned}
$$

$\rho$ is Spearman rank correlation (0 if a series is constant).
Defaults: $\lambda_x=0.5$, $\lambda_m=0.3$, $\lambda_c=1.0$,
margin $m=0.08$.

**On-axis.** Commanded T should track DSP `tc`; commanded S should
track `hc`. This is the actual T/S control.

**Cross-talk.** T should *not* drag `hc`, and S should *not* drag
`tc`. A map that swaps the axes, or that only moves a single
“musicality” diagonal, scores worse.

**Amount of change.** Drift from the source should rise when *either*
knob is high — $\max(T,S)$, not an average. We also count
non-monotonic steps of `tc` along T (fixed S) and of `hc` along S
(fixed T).

**Loud-corner hinge.** `(1,0)`, `(0,1)`, and `(1,1)` (edited ones) must drift more
than `(0,0)` by at least margin $m$. (An earlier $J$ used
$M=0.5T+0.5S$ instead of $\max(T,S)$ and had no hinge; then the
S-high corner could stay almost unedited because $(1,0)$ and $(0,1)$
share the same $M$.)

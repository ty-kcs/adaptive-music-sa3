# Optimization (T / S corners)

Phase 0 searches FlowEdit **corners** so commanded T and S
move two DSP axes separately: `tc` (temporal regularity) and `hc`
(spectral / harmonic coherence). We do **not** train SA3 weights.
Prompts are fixed tables, not searched.

## Map

Four FlowEdit dictionaries at (T,S)\in0,1^2. Interior points:


\theta(T,S)=(1-T)(1-S)\theta_{00}+T(1-S)\theta_{10}
+(1-T)S\theta_{01}+TS\theta_{11}.


BO moves only `t_start` **and** `tgt_cfg` **at each corner** (8-D).
`src_cfg` and `t_stop` are copied from the fitted 1-D FlowEdit
\theta_0 and shared.

Prompt: `"{genre}, {T phrase}, {S phrase}"` (five rungs on each axis).
Each trial scores a **5×5 grid** \(\{0, 0.25, 0.5, 0.75, 1\}^2\) (25 edits).

Writes `artifacts/g0_ts.json`. Gradio **Generate** loads that file and
edits at independent `(T, S)` (bilinear θ + T/S prompts). **Optimize**
is Phase 1: local 8-D BO on a **3×3** grid (not 5×5) around `g0_ts`.

## Objective

One trial: render the 25 cells, then


\begin{aligned}
J &= \rho(T,\mathrm{tc})+\rho(S,\mathrm{hc})
&\quad -\lambda_x\big(|\rho(T,\mathrm{hc})|+|\rho(S,\mathrm{tc})|\big)
&\quad +\rho(\max(T,S),\mathrm{drift})
&\quad -\lambda_m\big(\mathrm{viol}_T+\mathrm{viol}_S\big)
&\quad -\lambda_c\sum_{c\in\{10,01,11\}}\max(0,\,d_{00}+m-d_c)
\end{aligned}


Defaults: \(\lambda_x=0.5\), \(\lambda_m=0.3\), \(\lambda_c=1.0\),
margin \(m=0.08\). \(\rho\) is Spearman (0 if a series is constant).
Monotonicity is checked along T at each fixed S (for `tc`) and along
S at each fixed T (for `hc`). Drift hinge uses the nearest cell to
`(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` (0 if a corner is missing).
Phase 0 uses **mean J** over search clips, then re-ranks a shortlist
on a validation set.

Existing `artifacts/g0_ts.json` was fit with the previous
\(\rho(M,\mathrm{drift})\) term (\(M=0.5T+0.5S\), no hinge). Phase 1
Optimize (non-demo UI) already uses this \(J\).

Optimizer: `skopt.gp_minimize` on -J. Seed: `00←θ0`, `11←θ1`,
off-diagonal = midpoint.

## Time

Wall time is the **edits**, not the GP. Phase 0: one trial is 25
FlowEdit passes × search clips. Phase 1 (UI Optimize): 9 edits ×
local BO calls on this clip. 

```bash
uv run python -m adaptive_music_sa3.fit --device cuda --ts-corners
```



## Limits

- Low command on ambient audio often means “barely edit,” not
“destroy pulse on a rhythmic source.”
- 15 calls in 8-D is a short search.
- `tc` / `hc` are DSP proxies, not a listening test.


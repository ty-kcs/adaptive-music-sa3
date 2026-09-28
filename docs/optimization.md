# Optimization (T / S corners)

Phase 0 searches FlowEdit **corners** so commanded T and S
move two DSP axes separately: `tc` (temporal regularity) and `hc`
(spectral / harmonic coherence). 
Prompts are fixed tables, not searched.

## Map

We store four FlowEdit parameter dictionaries, one at each corner of
the unit square: `(T,S) = (0,0)`, `(1,0)`, `(0,1)`, `(1,1)`.

Anywhere between those corners, FlowEdit parameters are a bilinear mix
of the four dictionaries. For example `(0.5, 0.5)` is an equal blend;
`(1, 0)` is exactly the T-high / S-low corner.

$$
\theta(T,S)=(1-T)(1-S)\theta_{00}+T(1-S)\theta_{10}+(1-T)S\theta_{01}+TS\theta_{11}.
$$

Bayesian optimization moves only `t_start` and `tgt_cfg` at each
corner (eight numbers). `src_cfg` and `t_stop` are fixed for now.

Prompt: `"{genre}, {T phrase}, {S phrase}"` (five rungs on each axis).
Each trial scores a **5×5 grid**: every pair of T and S in
`{0, 0.25, 0.5, 0.75, 1}` (25 edits).

Writes `artifacts/g0_ts.json`. Gradio **Generate** loads that file and
edits at independent `(T, S)` (bilinear θ + T/S prompts). **Optimize**
is Phase 1: local Bayesian optimization of those eight numbers on a
**3×3** grid (not 5×5) around `g0_ts`. That local search is still
weak in practice, so the `--demo` UI hides it.

## Objective

One trial renders the 25 cells, then scores how well the knobs did
what we asked. Higher $J$ is better.

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

**On-axis (first line).** Commanded T should track DSP temporal
regularity `tc`; commanded S should track spectral / harmonic
coherence `hc`. This is the actual T/S control.

**Cross-talk (second line).** T should *not* drag `hc` with it, and S
should *not* drag `tc`. Absolute correlation is penalized so a map
that swaps the axes, or that only moves a single “musicality”
diagonal, scores worse.

**Amount of change (third line).** Latent drift from the source should
rise when *either* knob is high — $\max(T,S)$, not an average.
Monotonicity of `tc` along T (at each fixed S) and of `hc` along S
(at each fixed T) is counted as violations and subtracted.

**Loud-corner hinge (last line).** The three "edited" corners `(1,0)`,
`(0,1)`, `(1,1)` must drift more than `(0,0)` by at least margin $m$.
Lookup is the nearest cell to each corner (works for 5×5 and the
Phase-1 3×3). If a corner is missing, the hinge is 0.

### How this $J$ got here

The 1-D prototype used a single knob $M$ and rewarded
$\rho(M,\widehat{M})$ plus $\rho(M,\mathrm{drift})$. When we split
the knob into T and S, the first T/S objective kept that drift term
but with $M=0.5T+0.5S$, and added the on-axis / cross-talk Spearmans
so the two DSP axes would not collapse onto one diagonal.

That was not enough. $(1,0)$ and $(0,1)$ share the same $M=0.5$, so
$\rho(M,\mathrm{drift})$ does not care whether *both* axes actually
edit. Phase 0 could park the S-high corner $\theta_{01}$ at a weak
bound (low `t_start`, `tgt_cfg` ≈ `src_cfg`). Listening then matched
the math: S at 1 with T low barely moved the clip, because bilinear
interpolation sat on that weak corner.

Spearman also does not demand a numeric gap. A loud corner can drift
almost as little as `(0,0)` and still look “ranked” if the rest of
the grid is orderly.

So the average $M$ was replaced by $\max(T,S)$ (“at least one knob
high means more change”), and a hinge was added so the three loud
corners must beat `(0,0)` by a margin. Independence terms are
unchanged.

Phase 0 maximizes **mean $J$** over search clips, then re-ranks a
shortlist on a validation set. Existing `artifacts/g0_ts.json` was
fit with the older $\rho(M,\mathrm{drift})$ (no hinge). Phase 1
Optimize (non-demo UI) already uses this $J$.

Optimizer: `skopt.gp_minimize` on $-J$. The search starts from a
quiet corner, a loud corner, and two midpoints.

## Time

Wall time is the **edits**, not the Gaussian process. Phase 0: one
trial is 25 FlowEdit passes × search clips. Phase 1 (UI Optimize):
9 edits × local Bayesian-optimization calls on this clip.

```bash
uv run python -m adaptive_music_sa3.fit --device cuda --ts-corners
```

## Limits

- Low command on ambient audio often means “barely edit,” not
“destroy pulse on a rhythmic source.”
- 15 calls in 8-D is a short search.
- `tc` / `hc` are DSP proxies, not a listening test.

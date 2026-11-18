# Task 2: comparison with alternative difficulty signals (Reviewer 2), corrected data

Rule: corrected logging; probe refit on corrected probe-fit episodes (paper recipe); published γ 0.95/0.70/0.90 and EMARec α 0.02/0.5/0.02; readout = probe probability σ(vᵀh+b), logit stored; canonical 50-direction null; trajectory bootstrap 1,000 (seed 7); primary model per task; 100 evaluation episodes = 46,700 sites per task. Script: `run_t123.py`; raw: `T123_raw.json`.

Signals, all on the same 46,700 evaluation sites per task:
- KL_t, Rec_t
- EMARec (published α)
- EMA-KL, with α chosen on probe-fit episodes to maximise R² with E_{t,10}: cartpole 0.1, reacher 0.3, pendulum 0.9
- prior and posterior entropy of z_t
- C_t (Section 6 definition)
- the readout (probability, and logit)

## Flags

- **The full-set CIs for C_t barely exclude zero.** Their lower bounds are 5×10⁻⁷ (cartpole) and 7×10⁻⁶ (reacher). In-sample ΔR² is never negative, so its CI excludes zero almost by construction (see the Task 1 flags). The rule is applied literally below, but the cartpole and reacher C_t increments are of order 10⁻⁴.
- **On pendulum, EMA-KL's calibrated α is 0.9,** which makes it numerically almost identical to KL_t.

## Standalone association with E_{t,10}

| Signal | cartpole r [CI] | R² | reacher r [CI] | R² | pendulum r [CI] | R² |
|---|---|---|---|---|---|---|
| KL | +0.782 [+0.746, +0.812] | 0.612 | +0.695 [+0.637, +0.751] | 0.483 | +0.552 [+0.523, +0.579] | 0.305 |
| Rec | +0.713 [+0.656, +0.763] | 0.509 | +0.662 [+0.618, +0.748] | 0.439 | +0.626 [+0.611, +0.641] | 0.392 |
| EMARec | +0.638 [+0.584, +0.689] | 0.407 | +0.662 [+0.622, +0.747] | 0.438 | +0.282 [+0.225, +0.349] | 0.079 |
| EMAKL | +0.798 [+0.761, +0.827] | 0.637 | +0.749 [+0.693, +0.803] | 0.560 | +0.552 [+0.523, +0.579] | 0.305 |
| prior_entropy | +0.258 [+0.203, +0.318] | 0.067 | +0.092 [+0.010, +0.174] | 0.009 | -0.151 [-0.174, -0.129] | 0.023 |
| posterior_entropy | -0.803 [-0.822, -0.781] | 0.644 | -0.717 [-0.778, -0.654] | 0.514 | -0.210 [-0.245, -0.173] | 0.044 |
| C_t | +0.638 [+0.602, +0.675] | 0.408 | +0.524 [+0.495, +0.559] | 0.274 | +0.192 [+0.174, +0.211] | 0.037 |
| readout_prob | +0.695 [+0.673, +0.717] | 0.482 | +0.479 [+0.446, +0.516] | 0.229 | +0.509 [+0.460, +0.550] | 0.259 |
| readout_logit | +0.781 [+0.752, +0.806] | 0.611 | +0.578 [+0.527, +0.632] | 0.335 | +0.548 [+0.506, +0.582] | 0.300 |

R² of the six standard signals together: cartpole 0.755, reacher 0.646, pendulum 0.611.

## ΔR² beyond the other signals (95% CI)

"All others" means the six standard signals plus the other member of the {C_t, readout} pair. It is the primary row for the pre-registered reading.

| ΔR² of … beyond … | cartpole | reacher | pendulum |
|---|---|---|---|
| **C_t beyond all others** (incl. readout, probability) | +0.00039 [+0.00000, +0.00285] | +0.00046 [+0.00001, +0.00168] | +0.00280 [+0.00144, +0.00475] |
| C_t beyond all others (incl. readout, logit) | +0.00078 [+0.00000, +0.00420] | +0.00035 [+0.00000, +0.00144] | +0.00137 [+0.00048, +0.00268] |
| C_t beyond the 6 standard signals | +0.00159 [+0.00003, +0.00569] | +0.00010 [+0.00000, +0.00095] | +0.00303 [+0.00163, +0.00497] |
| **readout (probability) beyond all others** (incl. C_t) | +0.00917 [+0.00402, +0.01723] | +0.00288 [+0.00098, +0.00623] | +0.00084 [+0.00019, +0.00194] |
| readout (logit) beyond all others (incl. C_t) | +0.01429 [+0.00758, +0.02335] | +0.00562 [+0.00260, +0.01064] | +0.00318 [+0.00120, +0.00604] |
| readout (probability) beyond the 6 standard signals | +0.01037 [+0.00486, +0.01892] | +0.00252 [+0.00079, +0.00588] | +0.00106 [+0.00034, +0.00222] |
| readout (logit) beyond the 6 standard signals | +0.01511 [+0.00824, +0.02402] | +0.00537 [+0.00250, +0.01044] | +0.00484 [+0.00217, +0.00841] |

## Pre-registered reading

The paper may say C_t or the readout adds information beyond standard uncertainty signals only on tasks where the full-set CI excludes zero.

- **Readout (probability, and logit): passes on all three tasks.** Beyond all other signals including C_t:
  - probability: cartpole +0.0092 [+0.0040, +0.0172], reacher +0.0029 [+0.0010, +0.0062], pendulum +0.0008 [+0.0002, +0.0019]
  - logit: +0.0143, +0.0056 and +0.0032, all CIs > 0
- **C_t: passes literally on all three tasks, but marginally on two.** Beyond all others including the readout (probability):
  - cartpole +0.0004, CI lower bound 5×10⁻⁷
  - reacher +0.0005, lower bound 7×10⁻⁶
  - pendulum +0.0028 [+0.0014, +0.0047]; only pendulum is clearly above zero
- **Sizes:** all increments are small (≤ 0.015). The six standard signals together already explain 0.755 / 0.646 / 0.611 of the variance.
- **Strongest single signals:**
  - cartpole: posterior entropy (|r| 0.80) and EMA-KL (r 0.80)
  - reacher: EMA-KL (r 0.75)
  - pendulum: Rec (r 0.63)
  - The readout's logit is close to KL on cartpole (0.78), lower on reacher (0.58), and below Rec on pendulum (0.55).

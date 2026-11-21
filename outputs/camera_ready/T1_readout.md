# Task 1: direct predictive value of the learned readout

Rule: corrected logging; probe refit on corrected probe-fit episodes (paper recipe); published γ 0.95/0.70/0.90 and EMARec α 0.02/0.5/0.02; readout = probe probability σ(vᵀh+b), logit stored; canonical 50-direction null; trajectory bootstrap 1,000 (seed 7); primary model per task; 100 evaluation episodes = 46,700 sites per task. Script: `run_t123.py`; raw: `T123_raw.json`.

## Flags

- **The probability null is degenerate on all three tasks.** Each null direction u gets the same readout as v: σ(‖w‖·uᵀh + b_u), where b_u matches v's mean logit on probe-fit states. Because ‖w‖ is large, these null readouts are saturated (probability < 0.01 or > 0.99) at a median of 99.5% / 97.4% / 99.998% of sites (cartpole / reacher / pendulum); all 50 directions are > 90% saturated. Their probability is effectively a 0/1 step, so the null tests on the probability are reported but should not be relied on; use the logit rows.
- **v's own probability is also heavily saturated:** 72% of cartpole sites, 11% of reacher sites and 90% of pendulum sites (pendulum logits reach about 2×10⁵). On pendulum this is why the probability gives a larger ΔR² than the logit.
- **In-sample ΔR² is never negative,** because adding a regressor cannot lower in-sample R². Its bootstrap CI therefore excludes zero almost automatically. The same holds for Table 4.

Controls are as in Table 4: {KL, Rec, EMARec} plus ensemble disagreement on cartpole. C_t is the Section-6 definition (identical to the final Table 4).

## Results (E_{t,10})

| Task | Signal | r [95% CI] | R² | ΔR² beyond controls [95% CI] | null z (\|r\|) / pct | null z (R²) / pct | null z (ΔR²) / pct |
|---|---|---|---|---|---|---|---|
| cartpole | readout (probability) | +0.695 [+0.673, +0.717] | 0.482 | +0.0420 [+0.0278, +0.0603] | +3.41 / 100 (degenerate) | +5.78 / 100 (degenerate) | +1.78 / 90 (degenerate) |
| cartpole | readout (logit) | +0.781 [+0.752, +0.806] | 0.611 | +0.0428 [+0.0308, +0.0567] | +3.31 / 100 | +5.47 / 100 | +0.49 / 78 |
| cartpole | C_t (Table 4) | +0.638 [+0.602, +0.675] | 0.408 | +0.0168 [+0.0079, +0.0288] | — | +3.37 / 98 (logit null) | -0.28 / 60 (logit null) |
| reacher | readout (probability) | +0.479 [+0.446, +0.516] | 0.229 | +0.0030 [+0.0010, +0.0062] | +10.26 / 100 (degenerate) | +36.15 / 100 (degenerate) | +0.71 / 86 (degenerate) |
| reacher | readout (logit) | +0.578 [+0.527, +0.632] | 0.335 | +0.0047 [+0.0020, +0.0092] | +9.73 / 100 | +32.32 / 100 | +0.65 / 82 |
| reacher | C_t (Table 4) | +0.524 [+0.495, +0.559] | 0.274 | +0.0053 [+0.0015, +0.0084] | — | +26.35 / 100 (logit null) | +0.83 / 82 (logit null) |
| pendulum | readout (probability) | +0.509 [+0.460, +0.550] | 0.259 | +0.0250 [+0.0197, +0.0293] | +1.65 / 92 (degenerate) | +1.88 / 92 (degenerate) | +0.07 / 74 (degenerate) |
| pendulum | readout (logit) | +0.548 [+0.506, +0.582] | 0.300 | +0.0170 [+0.0114, +0.0231] | +1.58 / 92 | +1.69 / 92 | -0.42 / 46 |
| pendulum | C_t (Table 4) | +0.192 [+0.174, +0.211] | 0.037 | +0.0022 [+0.0009, +0.0039] | — | -0.53 / 54 (logit null) | -0.82 / 16 (logit null) |

Baseline R² of the controls alone: cartpole 0.702, reacher 0.591, pendulum 0.492.

## Reading (descriptive; Task 1 has no pre-registered rule)

- **Without the null:** on all three tasks the readout predicts E_{t,10} and adds variance beyond the Table 4 controls, under both definitions. On cartpole and pendulum it adds more than C_t:
  - cartpole: +0.042 (probability) / +0.043 (logit), against C_t +0.017
  - pendulum: +0.025 / +0.017, against +0.002
  - reacher: +0.003 / +0.005, against +0.005, so slightly less than C_t.
- **Against the random-direction null (logit):**
  - The readout's *standalone* association clears the null on cartpole (R² z +5.47) and reacher (+32.3), but not on pendulum (+1.69).
  - Its *incremental* value (ΔR²) does **not** clear the null on any task: z = +0.49, +0.65 and −0.42.
  - Projections onto random directions of h_t add about as much variance beyond the controls as v does.
- **C_t against the same null:** its ΔR² is also inside the null on every task (z −0.28 / +0.83 / −0.82).
- **Change from the earlier old-pipeline Task 1:** reacher's readout previously cleared the ΔR² null (z +4.0); on corrected data it does not.

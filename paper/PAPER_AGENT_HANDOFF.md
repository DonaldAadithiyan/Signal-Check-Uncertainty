# Paper-writing agent handoff: ICAART 2027 submission

**Paper:** *Endogenous Predictive-Difficulty Representations in Recurrent World Models*
**Target:** ICAART regular-paper round, deadline **29 September 2026**
**Repo branch:** `claude/dazzling-gates-5wg23z` (DonaldAadithiyan/Signal-Check-Uncertainty)
**Source of truth for the text:** `paper/icaart_main.tex`

---

## 1. Read this first: current state

- **The text is final in structure. It contains no final numbers yet.** Every number is a placeholder written as `\res{<key>}{<format>}`, and each key points into `outputs/sprint/results.json`.
- **`results.json` does not exist yet for real models.** The corrected analysis pipeline is written, tested and smoke-tested on randomly initialised models. It has **not** been run on the trained checkpoints, which were never uploaded to the cloud session. The user either uploads them or runs `./run_sprint_all.sh` locally.
- **Once results exist**, run `python paper/fill_paper.py`. It writes `paper/icaart_main_filled.tex` and lists any unresolved keys. **Edit prose in `icaart_main.tex`, compile `icaart_main_filled.tex`.** Re-run the fill after every prose edit.
- Sentences whose wording depends on how a result comes out are marked `\confirm{...}` and render in orange. Your main job after the fill is resolving them (section 5).
- Before submission, set `\draftmodefalse`. From then on, any unfilled `\res` stops compilation with an error, and every `\confirm` disappears.

## 2. Hard rules (non-negotiable)

1. **Never mix pipelines within one claim.** A number comes from `outputs/sprint/results.json` (the corrected pipeline) or it is removed. Never re-insert an old number from the previous draft, `PAPER.md`, `RESULTS_CONSOLIDATED.md`, `GAP_CLOSING_*`, `README.md` or any old deliverable, even as "for comparison". The old numbers were produced by a buggy pipeline (section 3).
2. **Never type a number by hand** that has a `results.json` key. Use `\res{key}{fmt}`, so the fill script stays the only source.
3. **Report results as they come out.** If a `\confirm` condition fails, rewrite the claim to match the data. Anything that weakens a claim is listed at the top of `outputs/sprint/RESULTS.md` under "Weakening results". Every item there must be reflected in the text.
4. **Do not re-add removed material** (section 4.2) unless a corrected rerun of it exists in `results.json`.
5. **No world-model training** was or will be done for this revision. Don't describe any new training.

## 3. Why the paper changed (background you need)

A code audit (`outputs/deliverables/review_response_lookups.md`) found problems in the pipeline behind the previous draft:

| ID | Problem | Consequence for the paper |
|---|---|---|
| **W1** | Every state-logging loop fed the model the action *about to be taken* (a_t) together with o_t. Training uses a_{t-1}, as the paper's Eq. 1 says. | Every KL_t, C_t, h_t, probe and downstream number from the old draft was computed under the wrong convention. The whole analysis is being re-run with the correct pairing. A pre-registered **gate** (Step 2) checks how large the effect is; see `outputs/sprint/gate.json` and the top of `RESULTS.md`. |
| **W2** | Phase 1's C_t threshold was the median of each whole episode, which includes future steps. | Old Table 3 values used a regressor that looks at the future. Now: calibration-split median. |
| **W3** | "88° from the top-50 PCs" was a *mean of per-PC angles*. A random 256-d vector scores 87.1° on that measure. | The angle claim is replaced by the projection fraction ‖proj_U v‖² against a 1,000-random-vector baseline. |
| **W4** | Table 5 "incremental R² carried" re-encoded the ablated state at the same step, which zeroes the atom by construction. | Row removed. |
| **W5** | γ was chosen on the same data its R² was reported on; train/test splits were by individual step within episodes. | Now: episode-level splits (probe-fit 100 / calibration 50 / evaluation 100 episodes, fixed seeds). Every threshold, γ and α is chosen on calibration, and everything is reported on evaluation. |
| **W6** | The z≈−22 ablation null and the 512-dim check were never seeded. | Everything is now seeded through explicit generators. The 512-dim check is not rerun and is not in the paper. |

The review also asked for:
- C_t^past (excluding the current step, which is the probe's own label);
- an EMA-of-KL baseline;
- robustness of C_t's definition;
- stronger nulls;
- PC-scrubbed and MLP probes;
- a KL autocorrelation diagnostic.

All of these are now in the pipeline and the paper.

## 4. What changed in `icaart_main.tex` (vs. the draft the user pasted)

### 4.1 Changed or added
- **Abstract, contributions and conclusion:** three contributions (emergence, external meaning, causal dissociation of the dense direction). The sparse-component and partial-observability claims are gone.
- **Keywords:** "Sparse Autoencoders" removed.
- **§3 Setup:** Eq. 1 pairing stated explicitly; new "Data Splits" subsection; Eq. 2 now defines both C_t and C_t^past with the calibration median τ; γ chosen on calibration from {0.5, 0.7, 0.8, 0.9, 0.95}. The effective-memory numbers (13/3/10 steps) were removed, and site counts are `\res` keys.
- **§3 statistical protocol:** 50-direction null plus the 500-direction and subspace-matched nulls; episode bootstrap; explicit seeding.
- **§4 Emergence:** probe AUROC (clean and noisy), PC-scrubbed and MLP probes, KL-matched contrast, R² for KL / C_t / **C_t^past** and partial R² beyond the current flag, the ridge scramble test with the current flag fixed (Fig. 1d), ablation z, and **geometry as a projection fraction vs. random**. Table 2 was rebuilt: primary model plus the range over all 13 models.
- **§5 External:** the control set is now KL + Rec + EMARec + **EMAKL** (no ensemble term), with α chosen on calibration. Table 3 has C_t and C_t^past columns. There is a cross-fit (out-of-sample) ΔR², head-to-heads against both EMARec and EMAKL, the external scramble z, and a **new §5.4 robustness section with Fig. 5** (8 definitions × fixed-γ grid).
- **§6 Causal (was §7):** dense steering only. Table 5 comes from mean ± sd over models. §7.2 "Independent replication" is replaced by "Robustness of the Null" (500-direction null, subspace-matched null, ablation).
- **§7 Interpretation (was §8):** the Kalman simulation is kept unchanged (no world model involved). The cross-task paragraph now uses the KL autocorrelation diagnostic (P6), whose prediction was stated in advance. A new "Scope" paragraph covers the out-of-scope items (pixels, larger models, learned policies, predictive-state methods).
- **LaTeX:** the broken `\widefig{H}{fig4_dissociation.pdf}…` (5 arguments) is fixed. `\verify` is replaced by `\res` and `\confirm`.

### 4.2 Removed (do not re-add without a corrected rerun)
| Removed | Reason |
|---|---|
| §8.4 partial observability, and the abstract/discussion clause "stronger when the model must filter unobserved state" | The POMDP scripts use the W1 pairing; not rerun |
| §7.2 POMDP ablation replication and noise-shift replication (max \|z\| = 1.43) | W1 pairing; not rerun |
| §6 mechanism, Fig. 3, Table 4, the atom rows of Table 6, §7.3 sparse-component intervention, Table 6 itself | SAE/atom rerun (sprint step P5) not done; old atom numbers must not be reused |
| "Incremental R² carried" row | Tautological (W4) |
| Berger drift claim ("C_t lower in reward-inflated drift states") | Old pipeline; not rerun |
| VIF / partial-correlation sentence on pendulum | Old pipeline; not rerun |
| "substituting another held-out state's projection" | Not rerun |
| "88°", "cos 88° ≈ 3.5%" | Wrong statistic (W3) |
| Effective memories "13, 3 and 10 steps" | Old γ values; recompute from the new γ if wanted (n ≈ ln 0.5 / ln γ) |

If P5 (the SAE rerun) is later completed, the atom material can come back **only** with numbers from the P5 keys in `results.json`, and the paper-map rules apply (`paper/SPRINT_PAPER_MAP.md`).

## 5. `\confirm` items: decision rules

Look up each one in `outputs/sprint/RESULTS.md` (sections 1 and 2a), rewrite as needed, then delete the marker.

| # | Where | Condition to check | If it fails |
|---|---|---|---|
| 1 | §4.1 KL-matched contrast | AUROC > 0.5 on cartpole | Say the probe *anti*-ranks the groups, and tie it to the within-bin r sign (#4). |
| 2 | §4.2 | On every task, the CI of R²(readout, C_t^past) − R²(readout, KL_t) is above 0, and the CI of the partial R² beyond the flag excludes 0 | **This is the review's key test.** If it fails on a task, say so plainly, and restrict "beyond the current step" to the tasks where it holds. |
| 3 | §4.3 geometry | w's fraction is below the random 5th percentile (percentile column < 5) | Drop "under-represented in the dominant variance"; report the fraction neutrally. If it holds on some tasks only, name them. Also update the abstract/intro wording. |
| 4 | §4.4, after Table 2 | Sign of the KL-matched AUROC − 0.5 agrees with the sign of within-bin r(Rec, C_t) on each task | Remove the "follows the sign" explanation. |
| 5 | §5.1 horizons | r(C_t, E_K) > 0 at every K on each task; describe the trend (flat or decaying) per task | Rewrite with the actual signs and trends. |
| 6 | §5.2, before the bold claim | Which tasks have a Table 3 CI excluding 0, for C_t and for C_t^past | Scope the bold sentence, the abstract and the contribution to those tasks. If none, the external-meaning contribution must be reframed. |
| 7 | §5.3 head-to-heads | "C_t beyond rival" > "rival beyond C_t" (RESULTS.md head-to-head table) | Report both directions and say the smoothed baseline carries more. Soften "not reducible to smoothing". |
| 8 | §5.4 robustness | Sign and CI of ΔR² across all 8 variants and all γ per task (Fig. 5) | Write one or two sentences summarising where the effect holds and where it doesn't. |
| 9 | §6.1 bold claim | Every readout z > 2 (k=0 and k=10) and every E^state \|z\| < 2, over all models | If any E^state \|z\| ≥ 2 or any readout z < 2, rewrite the paragraph, abstract and conclusion accordingly. |
| 10 | §6.2 null robustness | Same conclusion under null500 and subspace50 | State which null changes the conclusion. |
| 11 | §7.4 cross-task | Does the ordering of lag-1 ACF match the ordering of ΔR²(C_t^past)? | Report the actual pattern. It's descriptive either way, with n = 3 tasks. |

(A 12th marker in the preamble comment is only documentation.)

## 6. Figures

All are regenerated from `results.json` by `paper/figures/src/make_sprint_figs.py` into `paper/figures/`. The captions in the `.tex` already describe these layouts.

| File | Panels |
|---|---|
| `fig1_emergence.pdf` | (a) readout vs C_t scatter, cartpole; (b) R² of the readout on KL / C_t / C_t^past (bars = primary models, dots = replicates); (c) top-50-PC share of v vs the random band; (d) ridge h→C_t real vs scrambled, with z. |
| `fig2_external.pdf` | (a) r(C_t, E_K) vs K with bands; (b) ΔR² C_t (filled) / C_t^past (open) with CIs; (c) external ΔR² real vs scrambled, with z. |
| `fig4_dissociation.pdf` | Steering z vs the 50-direction null: readout k=0 (hollow), k=10 (filled), E^state (vermillion); one marker per model; grey band \|z\| < 2. |
| `fig5_robustness.pdf` | (a) ΔR² across the 8 C_t definitions; (b) ΔR² vs fixed γ for C_t and C_t^past. |
| `fig3_mechanism.pdf` | **Not used** (atoms removed). Delete it from `paper/figures/` before submission, or leave it unreferenced. |
| `fig1_overview.pdf`, `fig6_filtering.pdf` | Old files, **not referenced** in the current `.tex`. Don't add them back unless they are regenerated from corrected data (fig6 is the Kalman simulation, so it would be allowed). |

Check heights and placement against the compiled PDF. The `\widefig` placement comments assume page breaks that have now moved.

## 7. Files you will use

| File | Purpose |
|---|---|
| `paper/icaart_main.tex` | Source to edit (prose only; numbers via `\res`). |
| `paper/fill_paper.py` | Fills `\res` from results.json → `icaart_main_filled.tex`. `*` and `\|mean/sd/min/max/absmax/n` aggregate over models. Negative numbers are emitted as `\ensuremath{-}`. |
| `outputs/sprint/RESULTS.md` | Human-readable results. **Section 1** = the W1 gate plus auto-detected weakening results; **2a** = tables in the paper's layout; **3** = claim map and `[VERIFY]` answers. |
| `outputs/sprint/results.json` | Every number, keyed `w1_gate`, `p1.<task>.<model>`, `p2.<task>`, `p3.<task>.<model>`, `p4.<task>`, `p6.<task>`. |
| `paper/SPRINT_PAPER_MAP.md` | Claim-by-claim status vs. the *previous* draft, plus the `[VERIFY]` answers (reference only). |
| `outputs/deliverables/review_response_lookups.md` | The audit behind W1–W6. |

Key naming, in case you need to add a `\res`: `p1.<task>.primary.variants.<C|C_past|C_p60|C_p60_past|C_p70|C_p70_past|Ccont|Ccont_past>.{r2.{point,lo,hi}, gamma, grid.<γ>}`, `p1.<task>.primary.geometry.<standardized|raw>_top<10|50>.{frac, random_mean, random_p5, percentile_vs_random, subspace_angle_deg}`, `p2.<task>.table3.<C|C_past>.delta_r2.{point,lo,hi}`, `p2.<task>.variants.<name>.grid.<γ>.{point,lo,hi}`, `p3.<task>.<model>.nulls.<null50|null500|subspace50>.<slope_r0|slope_r10|slope_es|ablate_r0|ablate_r10|ablate_es>.z`. When unsure, open `results.json`.

## 8. `[VERIFY]` answers (all resolved in the text)

1. **E^state:** L2 norm per step, averaged over k = 1..K. It uses the real actions of the continuation, and a_t is fed exactly once at the handoff (unit-tested).
2. **Geometry:** the old 88° was a mean per-PC angle, not arccos(‖proj_U v‖/‖v‖). Replaced by the projection fraction.
3. **Top-10 SAE share:** P5 only; the figure was removed.
4. **Ensemble in Table 3:** the old table included it for cartpole only. The new table has **no ensemble term on any task**, and the caption says so.
5. **#156 r(activation, E^state):** removed with the atoms.
6. **"Incremental R² carried" removal operation:** tautological (same-step re-encoding). Row removed.
7. **k=10 readout:** posterior updates with the real observations. E^state uses prior imagination. The text says so.

## 9. Still open and not yours to fix

- If the W1 gate (`gate.json`, top of `RESULTS.md`) says **STOP**, the September 29 submission is off: the central quantity changes too much for a two-day rewrite. The user switches to the October 22 round. Don't write around a failed gate.
- P3 across all 13 models is optional compute. If only the primary models ran, Table 5's `n` column fills with 1, and the text "across all models" must become "on each task's primary model".
- Task B (training-necessity experiment) is paused and not in the paper.

## 10. Final checklist before submission

- [ ] `outputs/sprint/gate.json` shows `proceed_sept29: true`.
- [ ] `python paper/fill_paper.py` reports **0 unresolved**.
- [ ] Every `\confirm` is resolved and deleted (`grep -n confirm paper/icaart_main.tex` shows only the macro definition).
- [ ] Every item in `RESULTS.md` "Weakening results" is reflected in the text.
- [ ] Abstract, contributions, discussion and conclusion match the final `\confirm` decisions (especially #2, #3, #6 and #9).
- [ ] `\draftmodefalse` is set; `icaart_main_filled.tex` compiles with SCITEPRESS.sty and `icaart_refs.bib`.
- [ ] Figures regenerated after the final run; fig3/fig1_overview/fig6 not referenced unless regenerated from corrected data.
- [ ] No number anywhere in the PDF comes from the pre-sprint pipeline.
- [ ] Double-blind: no names, no repo URL (the acknowledgements stay in `\iffalse`).

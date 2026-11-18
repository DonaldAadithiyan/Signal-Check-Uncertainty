# Camera-ready status — ICAART 2027 Paper #388

- **Camera-ready + registration deadline:** 27 November 2026 (AOE) — Regular Papers, 1st stage (https://icaart.scitevents.org/ImportantDates.aspx, read 2026-11-16)
- **Page limit:** 12 pages (Full Paper; acceptance letter + https://icaart.scitevents.org/Guidelines.aspx); up to 4 extra pages at €50/page (RegistrationFees.aspx)

## Scope
2026-11-16: the handoff was replaced by a four-task version (Tasks 1-4 only, on the
paper's original data, checkpoints, logging and definitions; no re-logging).
The earlier Part A (corrected-pipeline consistency check) was stopped part-way at that
point. Its partial outputs remain in outputs/camera_ready/{states,A_raw,logs}; they
are not deliverables of the current spec and were not completed.

## Item 1 (corrected-logging re-check, requested 2026-11-17)
- DONE 2026-11-17: A_consistency.md / .json (+ A_report_paperprobe/, A_report_frozenprobe/, control_oldpairing/)

## Items A-D (2026-11-17)
- DONE: camera_ready_numbers.md/.json, B_sae_corrected.json, figures/ (fig1_emergence, fig2_external, fig2_external_globalthreshold)
- Item A: training log uses old pairing (o_t with a_t) -> not recomputed (stopped as instructed)
- OPEN DECISIONS: (1) paper probe at chance on corrected reacher (AUROC 0.43-0.56); cartpole C_t no longer ahead of KL; (2) Fig 2c pendulum sign depends on C_t threshold definition
- Task 4 training started 2026-11-17 (walker-walk, cheetah-run; outputs/t4_generality/)

## Final numbers (refit rule, 2026-11-17)
- DONE: camera_ready_numbers.md/.json (refit probe/PCA/SAE on corrected probe-fit episodes; probability readout, logit stored; published gamma/alpha)
- Figures NOT regenerated (on hold per instruction); outputs/camera_ready/figures/ are from the superseded paper-probe pass

## Progress
- Task 1 (readout predictive value): REDONE on corrected data 2026-11-17 (T1_readout.md/.json); old-pipeline version kept as *_oldpipeline_SUPERSEDED
- Task 2 (comparison): DONE 2026-11-17 (T2_comparison.md/.json): readout passes 3/3; C_t passes literally 3/3 (marginal on cartpole/reacher)
- Task 3 (decision example): DONE 2026-11-17 (T3_decision.md/.json): rule FAILS on 3/3 (readout and C_t worse than lagged Rec/EMARec); figure on hold
- Task 4 (new environments): DONE 2026-11-17 (T4_generality.md/.json): walker steering E z +2.30 (dissociation fails on walker); C_t not ahead of KL on either task; Table-4 ΔR² (in-sample and CV) > 0 on both
- Out-of-sample ΔR² (grouped 5-fold CV): DONE for all 5 tasks (CV_delta_r2.md/.json)
- Figures FINAL 2026-11-17: outputs/camera_ready/figures_final/ (11 PDFs + PNGs + FIGURE_MANIFEST.md); old folder renamed figures_SUPERSEDED/

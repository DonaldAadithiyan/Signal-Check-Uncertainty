# Task 3: decision example, selective trust in imagination

Rule: corrected logging; probe refit on corrected probe-fit episodes (paper recipe); published γ 0.95/0.70/0.90 and EMARec α 0.02/0.5/0.02; readout = probe probability σ(vᵀh+b), logit stored; canonical 50-direction null; trajectory bootstrap 1,000 (seed 7); primary model per task; 100 evaluation episodes = 46,700 sites per task. Script: `run_t123.py`; raw: `T123_raw.json`.

## Protocol (pre-registered)

- **Decision:** at each evaluation site, trust the 10-step imagined rollout or fall back. A signal ranks the 46,700 sites; at coverage c the agent trusts the fraction c with the lowest signal.
- **Risk:** mean E_{t,10} over the trusted sites, for c = 0.1, …, 1.0. AURC is the mean risk over these 10 coverages (lower is better). Risk at c = 0.5 and 0.8 is also reported.
- **Signals, all decision-time available (nothing uses o_t):**
  - readout of h_t, which is computed before o_t
  - C_t from KL_{t−1} and earlier, using the calibration median (a per-trajectory median would use future steps)
  - KL_{t−1}, Rec_{t−1}, EMARec_{t−1}, EMA-KL_{t−1} (α as in Task 2)
  - prior entropy at t
  - random ranking (mean of 100 seeded permutations)
  - the 50 canonical null directions, each oriented like v (sign of corr(uᵀh, KL) on probe-fit episodes)
  - oracle (true E_{t,10})
- **Ranking:** the readout is ranked by its logit, because the probability saturates. Ties are broken by a seeded random permutation and counted.
- **CIs:** trajectory bootstrap of AURC with the ranking held fixed. The best non-oracle baseline is chosen by point AURC.
- **Null z:** z = (null mean AURC − AURC)/null sd, so positive means better than the random directions.
- **Figure:** `figures_final/fig07{a,b,c}_risk_coverage_*.pdf`. The full curves are in `T3_decision.json` (`risk.<signal>.curve`).

## Results

### cartpole (best non-oracle baseline: Rec_tm1)

| Signal | AURC [95% CI] | risk @ c=0.5 | risk @ c=0.8 | tied sites |
|---|---|---|---|---|
| readout | 0.1875 [0.1720, 0.2043] | 0.1406 | 0.2738 | 0 |
| C_t_past | 0.2048 [0.1904, 0.2203] | 0.1351 | 0.2774 | 14612 |
| KL_tm1 | 0.1905 [0.1755, 0.2070] | 0.1432 | 0.2744 | 49 |
| Rec_tm1 | 0.1816 [0.1671, 0.1970] | 0.1366 | 0.2628 | 8 |
| EMARec_tm1 | 0.2025 [0.1872, 0.2188] | 0.1606 | 0.2749 | 0 |
| EMAKL_tm1 | 0.1904 [0.1754, 0.2064] | 0.1362 | 0.2762 | 0 |
| prior_entropy_t | 0.3377 [0.3086, 0.3675] | 0.3130 | 0.3333 | 808 |
| random | 0.3854 [0.3576, 0.4141] | 0.3855 | 0.3853 | — |
| oracle | 0.1664 [0.1541, 0.1790] | 0.1225 | 0.2331 | 34 |

readout: AURC − best baseline = +0.0058 [+0.0033, +0.0087]; null z +2.62 (null AURC 0.3529 ± 0.0631; better than 100% of the 50 directions); **rule FAIL**

C_t_past: AURC − best baseline = +0.0232 [+0.0195, +0.0270]; null z +2.35 (null AURC 0.3529 ± 0.0631; better than 100% of the 50 directions); **rule FAIL**

Tied sites if the readout were ranked by probability instead of logit: **640** (0 by logit).

### reacher (best non-oracle baseline: EMARec_tm1)

| Signal | AURC [95% CI] | risk @ c=0.5 | risk @ c=0.8 | tied sites |
|---|---|---|---|---|
| readout | 0.1676 [0.1617, 0.1733] | 0.1625 | 0.1844 | 0 |
| C_t_past | 0.1624 [0.1573, 0.1679] | 0.1533 | 0.1825 | 2197 |
| KL_tm1 | 0.1638 [0.1583, 0.1693] | 0.1561 | 0.1799 | 109 |
| Rec_tm1 | 0.1599 [0.1542, 0.1661] | 0.1526 | 0.1785 | 20 |
| EMARec_tm1 | 0.1585 [0.1529, 0.1646] | 0.1512 | 0.1779 | 0 |
| EMAKL_tm1 | 0.1595 [0.1545, 0.1647] | 0.1515 | 0.1773 | 0 |
| prior_entropy_t | 0.2112 [0.2004, 0.2241] | 0.2112 | 0.2122 | 1250 |
| random | 0.2152 [0.2053, 0.2252] | 0.2152 | 0.2152 | — |
| oracle | 0.1434 [0.1390, 0.1479] | 0.1340 | 0.1675 | 82 |

readout: AURC − best baseline = +0.0091 [+0.0069, +0.0113]; null z +5.12 (null AURC 0.2129 ± 0.0089; better than 100% of the 50 directions); **rule FAIL**

C_t_past: AURC − best baseline = +0.0039 [+0.0022, +0.0056]; null z +5.70 (null AURC 0.2129 ± 0.0089; better than 100% of the 50 directions); **rule FAIL**

Tied sites if the readout were ranked by probability instead of logit: **0** (0 by logit).

### pendulum (best non-oracle baseline: EMARec_tm1)

| Signal | AURC [95% CI] | risk @ c=0.5 | risk @ c=0.8 | tied sites |
|---|---|---|---|---|
| readout | 0.1706 [0.1555, 0.1889] | 0.1530 | 0.1971 | 0 |
| C_t_past | 0.1840 [0.1659, 0.2070] | 0.1619 | 0.2278 | 23971 |
| KL_tm1 | 0.1695 [0.1539, 0.1884] | 0.1544 | 0.1993 | 48 |
| Rec_tm1 | 0.1714 [0.1580, 0.1863] | 0.1593 | 0.1889 | 41 |
| EMARec_tm1 | 0.1662 [0.1533, 0.1809] | 0.1518 | 0.2014 | 0 |
| EMAKL_tm1 | 0.1696 [0.1540, 0.1886] | 0.1544 | 0.1994 | 0 |
| prior_entropy_t | 0.2760 [0.2563, 0.2963] | 0.2574 | 0.2331 | 1434 |
| random | 0.2351 [0.2177, 0.2530] | 0.2349 | 0.2351 | — |
| oracle | 0.1305 [0.1188, 0.1432] | 0.1142 | 0.1710 | 53 |

readout: AURC − best baseline = +0.0044 [+0.0001, +0.0088]; null z +1.50 (null AURC 0.2149 ± 0.0295; better than 92% of the 50 directions); **rule FAIL**

C_t_past: AURC − best baseline = +0.0178 [+0.0113, +0.0263]; null z +1.05 (null AURC 0.2149 ± 0.0295; better than 80% of the 50 directions); **rule FAIL**

Tied sites if the readout were ranked by probability instead of logit: **7804** (0 by logit).

## Pre-registered rule

The rule: the readout or C_t has lower AURC than the best non-oracle baseline, with a CI on the difference that excludes zero, **and** beats the random-direction null at z > 2, on at least 2 of 3 tasks.

**Result: FAIL, a negative result on all three tasks.**
- **Readout:** worse than the best baseline on every task. The AURC difference is positive (worse) with CIs above zero:
  - cartpole +0.0058 vs Rec_{t−1}
  - reacher +0.0091 vs EMARec_{t−1}
  - pendulum +0.0044 [+0.0001, +0.0088] vs EMARec_{t−1}
- **C_t:** also worse than the best baseline on every task (+0.023, +0.004 and +0.018).
- **The null criterion alone** would pass on cartpole (readout z +2.62) and reacher (+5.12), but not on pendulum (+1.50). So the readout ranks sites much better than random directions in h_t, yet not better than lagged reconstruction error or its moving average.
- **Oracle headroom is small:** the best baseline's AURC is within 0.015–0.036 of the oracle on every task.

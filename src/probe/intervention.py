"""
Shared infrastructure for Phase-1b causal-hardening experiments (Tasks A, B, C).

Provides:
  * Faithful manual GRU step that exposes/overrides the update gate z_t
    (verified to match nn.GRUCell to 1e-7).
  * Confusion-direction extraction from a trained probe / a C_t regression.
  * A rollout helper that continues an RSSM trajectory from a (possibly
    intervened) h_t, so downstream probe/decoder/next-KL can be measured.
  * Bootstrap CI + paired-bootstrap utilities so every new number ships with
    a sample size and a measure of spread.

Nothing here trains a model. All interventions are inference-time on frozen weights.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ─── GRU gate access / override ──────────────────────────────────────────────

def gru_gates(gru_cell, inp, h):
    """Return (z_gate, r_gate, n_candidate) for one nn.GRUCell step.

    Matches nn.GRUCell exactly. Convention: h' = (1-z)*n + z*h.
    inp: (B, in_dim)  h: (B, deter)
    """
    W_ih, W_hh = gru_cell.weight_ih, gru_cell.weight_hh
    b_ih, b_hh = gru_cell.bias_ih,   gru_cell.bias_hh
    d = h.shape[-1]
    pre_ih = inp @ W_ih.T + b_ih
    pre_hh = h   @ W_hh.T + b_hh
    r = torch.sigmoid(pre_ih[:, :d]        + pre_hh[:, :d])
    z = torch.sigmoid(pre_ih[:, d:2 * d]   + pre_hh[:, d:2 * d])
    n = torch.tanh(   pre_ih[:, 2 * d:]    + r * pre_hh[:, 2 * d:])
    return z, r, n


def gru_step(gru_cell, inp, h, z_override=None):
    """One GRU step with optional update-gate override.

    z_override: None (natural), a scalar, or a tensor broadcastable to (B, deter).
    Returns h_next. When z_override is None this reproduces nn.GRUCell exactly.
    """
    z, r, n = gru_gates(gru_cell, inp, h)
    if z_override is not None:
        if not torch.is_tensor(z_override):
            z_override = torch.full_like(z, float(z_override))
        z = z_override
    return (1.0 - z) * n + z * h, z


def rssm_observe_with_override(rssm, h, z, action, embed, z_override=None):
    """rssm.observe_step but with an optional GRU update-gate override.

    Returns h_next, z_next(sample), prior_logits, post_logits, z_gate.
    """
    inp = torch.cat([z, action], dim=-1)
    h_next, z_gate = gru_step(rssm.gru, inp, h, z_override=z_override)
    prior_logits = rssm.prior_net(h_next)
    post_logits  = rssm.post_net(torch.cat([h_next, embed], dim=-1))
    z_next = rssm._straight_through_sample(post_logits)
    return h_next, z_next, prior_logits, post_logits, z_gate


# ─── Confusion-direction extraction ──────────────────────────────────────────

def probe_direction(clf, scaler):
    """Unit confusion direction in RAW h_t space from a trained logistic probe.

    The probe operates on standardised features x = (h - mu) / sigma, with
    decision fn w·x. In raw h space the gradient direction is w / sigma.
    Returns a unit vector v in raw h_t coordinates.
    """
    w = clf.coef_[0]
    v_raw = w / scaler.scale_          # undo the standardisation scaling
    v_raw = v_raw / np.linalg.norm(v_raw)
    return v_raw.astype(np.float32)


def regression_direction(ridge, scaler):
    """Unit direction in raw h_t space from a Ridge regression on C_t."""
    w = ridge.coef_
    v_raw = w / scaler.scale_
    v_raw = v_raw / np.linalg.norm(v_raw)
    return v_raw.astype(np.float32)


def random_matched_direction(rng, dim):
    """A unit random direction in h_t space (norm-matched control)."""
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def project_out(h, v):
    """Ablate direction v (unit) from h. h:(N,D), v:(D,) → (N,D)."""
    return h - np.outer(h @ v, v)


def amplify(h, v, alpha):
    """Push h along unit v by alpha. h:(N,D), v:(D,) → (N,D)."""
    return h + alpha * v[None, :]


# ─── C_t (discounted confusion integral) ─────────────────────────────────────

def compute_ct(kl, traj_id, gamma=0.95, max_lag=50, kl_median=None):
    """C_t = Σ_{i≥0} γ^i · 1[KL_{t-i} > median], reset at trajectory boundaries."""
    N = len(kl)
    if kl_median is None:
        kl_median = np.median(kl)
    high = (kl > kl_median).astype(np.float32)
    ct = np.zeros(N, dtype=np.float32)
    for i in range(N):
        val = 0.0
        for lag in range(max_lag):
            j = i - lag
            if j < 0 or traj_id[j] != traj_id[i]:
                break
            val += (gamma ** lag) * high[j]
        ct[i] = val
    return ct


# ─── Bootstrap utilities ─────────────────────────────────────────────────────

def bootstrap_ci(values, stat_fn=np.mean, n_boot=1000, alpha=0.05, seed=0):
    """Bootstrap CI of a statistic over a 1-D sample.

    Returns (point, lo, hi) using percentile method. `values` is resampled
    with replacement n_boot times.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    point = stat_fn(values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = stat_fn(values[idx])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(point), float(lo), float(hi)


def bootstrap_auroc_ci(y, scores, n_boot=1000, alpha=0.05, seed=0):
    """Bootstrap CI for AUROC by resampling (y, score) pairs with replacement."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores)
    n = len(y)
    point = roc_auc_score(y, scores)
    boots = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], scores[idx]))
    boots = np.array(boots)
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(point), float(lo), float(hi)


def paired_bootstrap_diff(y, scores_a, scores_b, metric='auroc',
                          n_boot=1000, seed=0):
    """Paired bootstrap of metric(a) - metric(b) on the SAME resampled indices.

    Returns (diff_point, lo, hi, p_two_sided) where p is the fraction of boot
    diffs on the opposite side of 0 (×2), i.e. an achieved-significance level.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    sa, sb = np.asarray(scores_a), np.asarray(scores_b)
    n = len(y)

    def m(yy, ss):
        return roc_auc_score(yy, ss)

    point = m(y, sa) - m(y, sb)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(m(y[idx], sa[idx]) - m(y[idx], sb[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided achieved significance level
    frac_le0 = np.mean(boots <= 0)
    p = 2 * min(frac_le0, 1 - frac_le0)
    return float(point), float(lo), float(hi), float(min(p, 1.0))

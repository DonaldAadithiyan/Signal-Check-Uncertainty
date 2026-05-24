"""
Linear probes and ensemble baseline.
All probes use logistic regression (scikit-learn).
Model inference runs on the configured device (MPS/CPU).
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ─── helpers ─────────────────────────────────────────────────────────────────

def binarise_by_median(values: np.ndarray) -> np.ndarray:
    return (values > np.median(values)).astype(np.int32)


def train_probe(X_train, y_train, C=1.0, max_iter=2000):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs', random_state=0)
    clf.fit(X_s, y_train)
    return clf, scaler


def auroc(clf, scaler, X, y):
    if len(np.unique(y)) < 2:
        return float('nan')
    probs = clf.predict_proba(scaler.transform(X))[:, 1]
    return roc_auc_score(y, probs)


def auroc_direct(scores, y):
    """AUROC from raw scores (no clf)."""
    if len(np.unique(y)) < 2:
        return float('nan')
    return roc_auc_score(y, scores)


def _split(X, y):
    """60% train / 40% test. Returns X_tr, y_tr, X_te, y_te."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=0)
    return X_tr, y_tr, X_te, y_te


def _eval_labels(kl_array, median):
    """Binarise with given median; fall back to per-array median if degenerate."""
    y = (kl_array > median).astype(np.int32)
    if len(np.unique(y)) < 2:
        y = binarise_by_median(kl_array)
    return y


# ─── Probe A: KL gap ─────────────────────────────────────────────────────────

def run_probe_a(training_states, set_a, set_b, set_c):
    h, kl = training_states['h'], training_states['kl']
    y = binarise_by_median(kl)
    kl_median = np.median(kl)

    X_tr, y_tr, X_te, y_te = _split(h, y)
    clf, scaler = train_probe(X_tr, y_tr)

    return {
        'auroc_id': auroc(clf, scaler, X_te, y_te),
        'auroc_a':  auroc(clf, scaler, set_a['h'], _eval_labels(set_a['kl'], kl_median)),
        'auroc_b':  auroc(clf, scaler, set_b['h'], _eval_labels(set_b['kl'], kl_median)),
        'auroc_c':  auroc(clf, scaler, set_c['h'], set_c['labels']),
        'clf': clf, 'scaler': scaler, 'kl_median': kl_median,
    }


# ─── Probe B: rollout variance ────────────────────────────────────────────────

def _batch_rollout_var(model, h_array, cfg, batch=512):
    device = next(model.parameters()).device
    var_list = []
    for i in range(0, len(h_array), batch):
        h_b = torch.tensor(h_array[i:i + batch], dtype=torch.float32, device=device)
        v   = model.rollout_variance(h_b,
                                     n_samples=cfg['rollout_samples'],
                                     horizon=cfg['rollout_horizon'])
        var_list.append(v.cpu().numpy())
    return np.concatenate(var_list)


def run_probe_b(model, training_states, set_a, set_b, set_c, cfg):
    print("  Probe B: computing training rollout variance...")
    train_var  = _batch_rollout_var(model, training_states['h'], cfg)
    y_train    = binarise_by_median(train_var)
    var_median = np.median(train_var)

    X_tr, y_tr, X_te, y_te = _split(training_states['h'], y_train)
    clf, scaler = train_probe(X_tr, y_tr)

    print("  Probe B: evaluating on Sets A, B, C...")
    a_var = _batch_rollout_var(model, set_a['h'], cfg)
    b_var = _batch_rollout_var(model, set_b['h'], cfg)

    return {
        'auroc_id': auroc(clf, scaler, X_te,       y_te),
        'auroc_a':  auroc(clf, scaler, set_a['h'],
                          _eval_labels(a_var, var_median)),
        'auroc_b':  auroc(clf, scaler, set_b['h'],
                          _eval_labels(b_var, var_median)),
        'auroc_c':  auroc(clf, scaler, set_c['h'], set_c['labels']),
    }


# ─── Probe C: recon error sanity check ───────────────────────────────────────

def run_probe_c(training_states, set_a, set_b, set_c):
    h, recon = training_states['h'], training_states['recon']
    y = binarise_by_median(recon)
    recon_median = np.median(recon)

    X_tr, y_tr, X_te, y_te = _split(h, y)
    clf, scaler = train_probe(X_tr, y_tr)

    return {
        'auroc_id': auroc(clf, scaler, X_te, y_te),
        'auroc_a':  auroc(clf, scaler, set_a['h'],
                          _eval_labels(set_a['recon'], recon_median)),
        'auroc_b':  auroc(clf, scaler, set_b['h'],
                          _eval_labels(set_b['recon'], recon_median)),
        'auroc_c':  auroc(clf, scaler, set_c['h'], set_c['labels']),
    }


# ─── Block contribution ───────────────────────────────────────────────────────

def run_block_analysis(training_states, set_a):
    h, kl = training_states['h'], training_states['kl']
    y = binarise_by_median(kl)
    kl_median = np.median(kl)

    n_q = 4
    q   = h.shape[1] // n_q
    results = {}

    for i in range(n_q):
        sl   = slice(i * q, (i + 1) * q)
        X_tr, y_tr, X_te, y_te = _split(h[:, sl], y)
        clf, scaler = train_probe(X_tr, y_tr)

        y_a    = _eval_labels(set_a['kl'], kl_median)
        auroc_a_val = auroc(clf, scaler, set_a['h'][:, sl], y_a)

        results[f'Q{i + 1}'] = {
            'auroc_train': auroc(clf, scaler, X_te, y_te),
            'auroc_a':     auroc_a_val,
            'dims':        f'{i * q}-{(i + 1) * q}',
        }
    return results


# ─── h_t vs z_t ──────────────────────────────────────────────────────────────

def run_ht_vs_zt(training_states, set_a, set_c):
    h, z, kl = training_states['h'], training_states['z'], training_states['kl']
    y = binarise_by_median(kl)
    kl_median = np.median(kl)

    results = {}
    for name, feats, a_feats, c_feats in [
        ('h_t', h, set_a['h'], set_c['h']),
        ('z_t', z, set_a['z'], set_c['z']),
    ]:
        X_tr, y_tr, X_te, y_te = _split(feats, y)
        clf, scaler = train_probe(X_tr, y_tr)
        results[name] = {
            'auroc_train': auroc(clf, scaler, X_te,    y_te),
            'auroc_a':     auroc(clf, scaler, a_feats,
                                 _eval_labels(set_a['kl'], kl_median)),
            'auroc_c':     auroc(clf, scaler, c_feats, set_c['labels']),
        }
    return results


# ─── Ensemble baseline ────────────────────────────────────────────────────────

def ensemble_disagreement(models, set_c, cfg):
    """
    Variance of each model's one-step prior prediction across the ensemble.
    Returns (disagreement_scores, auroc_on_set_c).
    """
    device = next(models[0].parameters()).device
    batch  = 512
    all_preds = []

    for model in models:
        model.eval()
        preds = []
        h_arr = set_c['h']
        with torch.no_grad():
            for i in range(0, len(h_arr), batch):
                h_b = torch.tensor(h_arr[i:i + batch], dtype=torch.float32, device=device)
                N   = h_b.shape[0]
                z_b = torch.zeros(N, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
                a_b = torch.zeros(N, cfg['act_dim'], device=device)
                h_n, z_n, _ = model.rssm.imagine_step(h_b, z_b, a_b)
                dec = model.decoder(torch.cat([h_n, z_n], dim=-1))
                preds.append(dec.cpu().numpy())
        all_preds.append(np.concatenate(preds, axis=0))

    stacked      = np.stack(all_preds, axis=0)         # (n_models, N, obs_dim)
    disagreement = stacked.var(axis=0).mean(axis=-1)   # (N,)
    auc          = auroc_direct(disagreement, set_c['labels'])
    return disagreement, auc

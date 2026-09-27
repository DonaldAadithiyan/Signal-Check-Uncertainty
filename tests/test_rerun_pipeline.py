"""Unit tests for the sprint pipeline (src/rerun/pipeline.py).

The W1 test instruments the real GRU cell and fails if obs_t is ever paired
with a_t: the action columns of every GRU input at step t must equal a_{t-1}
(zeros at t=0), and the imagination handoff must feed a_t exactly once.
"""

import inspect

import numpy as np
import pytest
import torch

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.rerun import pipeline as P


def _toy(E=3, T=15, obs_dim=5, act_dim=2, seed=0):
    torch.manual_seed(seed)
    cfg = dict(XS_CONFIG)
    m = WorldModel(obs_dim, act_dim, cfg).eval()
    rng = np.random.default_rng(seed)
    ep = dict(obs=rng.standard_normal((E, T, obs_dim)).astype(np.float32),
              act=rng.uniform(-1, 1, (E, T, act_dim)).astype(np.float32))
    return m, ep


class _GruRecorder:
    def __init__(self, gru, act_dim):
        self.act_dim, self.calls = act_dim, []
        self.handle = gru.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.calls.append(args[0][:, -self.act_dim:].detach().clone().numpy())

    def remove(self):
        self.handle.remove()


def test_logging_pairs_obs_t_with_previous_action():
    m, ep = _toy()
    rec = _GruRecorder(m.rssm.gru, ep['act'].shape[-1])
    P.log_states(m, ep, torch_seed=1, convention='correct')
    rec.remove()
    T = ep['act'].shape[1]
    assert len(rec.calls) == T
    for t, fed in enumerate(rec.calls):
        expected = np.zeros_like(ep['act'][:, 0]) if t == 0 else ep['act'][:, t - 1]
        np.testing.assert_array_equal(fed, expected, err_msg=f'step {t} fed wrong action')
        if t < T - 1:
            assert not np.array_equal(fed, ep['act'][:, t]), f'obs_{t} paired with a_{t}'


def test_buggy_convention_is_detected():
    """Sanity check on the test itself: the old pairing must fail the W1 check."""
    m, ep = _toy()
    rec = _GruRecorder(m.rssm.gru, ep['act'].shape[-1])
    P.log_states(m, ep, torch_seed=1, convention='buggy')
    rec.remove()
    assert any(np.array_equal(fed, ep['act'][:, t]) for t, fed in enumerate(rec.calls))


def test_imagination_handoff_uses_a_t_once():
    m, ep = _toy(E=2, T=40)
    st = P.log_states(m, ep, torch_seed=1)
    rec = _GruRecorder(m.rssm.gru, ep['act'].shape[-1])
    e_idx, t_idx = np.array([0, 1]), np.array([12, 15])
    h0 = torch.from_numpy(st['h'][e_idx, t_idx])
    z0 = P.z_from_idx(m.rssm, st['z_idx'][e_idx, t_idx])
    P.imagine_errors(m, ep, h0, z0, t_idx, e_idx, K=5, gen=torch.Generator().manual_seed(0))
    rec.remove()
    for k, fed in enumerate(rec.calls, start=1):
        np.testing.assert_array_equal(fed, ep['act'][e_idx, t_idx + k - 1])


def test_logging_is_deterministic_and_order_independent():
    m, ep = _toy()
    a = P.log_states(m, ep, torch_seed=5)
    torch.rand(100)  # disturb the global RNG
    b = P.log_states(m, ep, torch_seed=5)
    np.testing.assert_array_equal(a['h'], b['h'])
    np.testing.assert_array_equal(a['kl'], b['kl'])


def test_compute_ct_requires_threshold_and_past_excludes_current():
    assert 'thresh' in inspect.signature(P.compute_ct).parameters
    assert inspect.signature(P.compute_ct).parameters['thresh'].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        P.compute_ct(np.zeros((1, 5)), gamma=0.9)  # no threshold
    kl = np.array([[0., 2., 2., 0., 2.]])
    c = P.compute_ct(kl, 1.0, 0.5)
    cp = P.compute_ct(kl, 1.0, 0.5, include_current=False)
    np.testing.assert_allclose(c[0], [0, 1, 1.5, 0.75, 1.375])
    np.testing.assert_allclose(cp[0], [0, 0, 0.5, 0.75, 0.375])
    # C^past_t never depends on KL_t
    kl2 = kl.copy(); kl2[0, 4] = -5
    np.testing.assert_allclose(P.compute_ct(kl2, 1.0, 0.5, include_current=False)[0, 4], cp[0, 4])


def test_splits_disjoint():
    seen = set()
    for name, r in P.SPLITS.items():
        assert not (seen & set(r)), name
        seen |= set(r)


def test_gram_r2_matches_direct_ols_and_bootstrap_rows():
    from src.rerun.stats import GramSet
    rng = np.random.default_rng(0)
    E, T = 6, 30
    a, b = rng.standard_normal((E, T)), rng.standard_normal((E, T))
    y = 0.5 * a - 0.2 * b + rng.standard_normal((E, T))
    mask = np.ones((E, T), bool); mask[:, :5] = False
    g = GramSet(dict(a=a, b=b, y=y), mask)
    m = mask
    assert abs(g.r2('y', ['a', 'b']) - P.ols_r2([a[m], b[m]], y[m])) < 1e-10
    w = np.array([2, 0, 1, 0, 3, 0], float)
    rows = np.concatenate([np.where(m[e])[0] + e * T for e in range(E) for _ in range(int(w[e]))])
    fa, fb, fy = a.ravel()[rows], b.ravel()[rows], y.ravel()[rows]
    assert abs(g.r2('y', ['a'], w) - P.ols_r2([fa], fy)) < 1e-10

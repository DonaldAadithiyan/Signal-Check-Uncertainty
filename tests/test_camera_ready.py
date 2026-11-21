"""Preconditions: action pairing, logging consistency, determinism. Must pass before any analysis run."""
import numpy as np
import torch

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.model.pomdp_world_model import PomdpWorldModel
from src.camera_ready import core as C
from src.rerun import pipeline as P


def _toy(kind='wm', E=3, T=30, obs_dim=5, act_dim=2, seed=0):
    torch.manual_seed(seed)
    cfg = dict(XS_CONFIG)
    if kind == 'wm':
        m = WorldModel(obs_dim, act_dim, cfg).eval()
        m.obs_dim_in = None
    else:
        m = PomdpWorldModel(3, obs_dim, act_dim, cfg).eval()
        m.obs_dim_in = 3
    rng = np.random.default_rng(seed)
    ep = dict(obs=rng.standard_normal((E, T, obs_dim)).astype(np.float32),
              act=rng.uniform(-1, 1, (E, T, act_dim)).astype(np.float32),
              rew=rng.uniform(0, 1, (E, T)).astype(np.float32))
    return m, ep


class _Rec:
    def __init__(self, gru, act_dim):
        self.act_dim, self.calls = act_dim, []
        self.h = gru.register_forward_pre_hook(lambda mod, a: self.calls.append(
            a[0][:, -self.act_dim:].detach().clone().numpy()))


def _check_pairing(kind):
    m, ep = _toy(kind)
    r = _Rec(m.rssm.gru, ep['act'].shape[-1])
    C.log_states_full(m, ep, torch_seed=1)
    r.h.remove()
    T = ep['act'].shape[1]
    assert len(r.calls) == T
    for t, fed in enumerate(r.calls):
        exp = np.zeros_like(ep['act'][:, 0]) if t == 0 else ep['act'][:, t - 1]
        np.testing.assert_array_equal(fed, exp, err_msg=f'{kind}: step {t} fed wrong action')
        if t < T - 1:
            assert not np.array_equal(fed, ep['act'][:, t]), f'{kind}: obs_{t} paired with a_{t}'


def test_action_convention():
    """obs_t is paired with a_{t-1} (zeros at t=0), never with a_t; imagination
    from site t feeds a_t exactly once at the handoff."""
    _check_pairing('wm')
    _check_pairing('pomdp')
    m, ep = _toy('wm', E=2, T=40)
    st = C.log_states_full(m, ep, torch_seed=1)
    r = _Rec(m.rssm.gru, ep['act'].shape[-1])
    e_idx, t_idx = np.array([0, 1]), np.array([12, 15])
    z0 = P.z_from_idx(m.rssm, st['z_idx'][e_idx, t_idx]).numpy()
    C.imagine_from(m, ep['obs'], ep['act'], st['h'][e_idx, t_idx], z0, e_idx, t_idx, 5,
                   torch.Generator().manual_seed(0))
    r.h.remove()
    for k, fed in enumerate(r.calls, start=1):
        np.testing.assert_array_equal(fed, ep['act'][e_idx, t_idx + k - 1])


def test_logging_matches_sprint_pipeline():
    """h/kl/rec identical to the reviewed sprint logger for the same seed."""
    m, ep = _toy('wm')
    a = C.log_states_full(m, ep, torch_seed=3)
    b = P.log_states(m, ep, torch_seed=3)
    for k in ('h', 'kl', 'rec', 'z_idx'):
        np.testing.assert_array_equal(a[k], b[k])


def test_deterministic_and_order_independent():
    m, ep = _toy('wm')
    a = C.log_states_full(m, ep, torch_seed=5)
    torch.rand(1000)
    np.random.rand(1000)
    b = C.log_states_full(m, ep, torch_seed=5)
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), k


def test_prior_entropy_is_decision_time():
    """Prior entropy at t must not depend on o_t."""
    m, ep = _toy('wm')
    a = C.log_states_full(m, ep, torch_seed=5)
    ep2 = {k: v.copy() for k, v in ep.items()}
    ep2['obs'][:, 20] += 5.0
    b = C.log_states_full(m, ep2, torch_seed=5)
    np.testing.assert_array_equal(a['ent_prior'][:, :21], b['ent_prior'][:, :21])
    assert not np.array_equal(a['ent_post'][:, 20], b['ent_post'][:, 20])

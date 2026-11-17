"""Interventions on h_t at evaluation sites (corrected convention), adapted from
run_sprint_p3.py. For an intervened state h'_t at site t:
  z_t ~ posterior(h'_t, o_t)                         (as in the paper's scripts)
  readout k=0  : probe probability(h'_t) [r0] and projection v.h'_t [p0]
  readout k=10 : probe(h_{t+10}) after posterior updates with real o_{t+1..t+10}
                 and actions a_t..a_{t+9}
  E (K=10)     : prior-only imagination from (h'_t, z_t), actions a_t..a_{t+9},
                 mean L2 to the real obs_{t+1..t+10}
Every condition at a site uses the same torch generator seed (common random numbers),
and v and every null direction are evaluated on the SAME sites.
"""
import numpy as np
import torch

from src.rerun import pipeline as P
from src.camera_ready.core import enc_input

LAMBDAS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
CRN_SEED = 77
K_READ = 10
CANON = 'outputs/canonical_null/random_directions_50_dim256.npz'


def canonical_null():
    d = np.load(CANON)['directions'].astype(np.float64)
    if d.shape != (50, 256):
        raise AssertionError(f'canonical null has shape {d.shape}')
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@torch.no_grad()
def run_condition(model, clf, sc, obs_in, obs_tgt, act, e_idx, t_idx, h_new, v):
    rssm = model.rssm
    gen = torch.Generator().manual_seed(CRN_SEED)
    e = torch.as_tensor(e_idx, dtype=torch.long)
    t = torch.as_tensor(t_idx, dtype=torch.long)
    h = torch.as_tensor(h_new, dtype=torch.float32)
    z0, _ = P._sample_onehot(rssm, rssm.post_net(torch.cat([h, model.encoder(enc_input(model, obs_in[e, t]))], -1)), gen)
    r0 = clf.predict_proba(sc.transform(h.numpy()))[:, 1]
    p0 = h.numpy().astype(np.float64) @ v
    hc, zc = h, z0
    for k in range(1, K_READ + 1):
        hc = P._gru(rssm, hc, zc, act[e, t + k - 1])
        emb = model.encoder(enc_input(model, obs_in[e, t + k]))
        zc, _ = P._sample_onehot(rssm, rssm.post_net(torch.cat([hc, emb], -1)), gen)
    r10 = clf.predict_proba(sc.transform(hc.numpy()))[:, 1]
    p10 = hc.numpy().astype(np.float64) @ v
    hi, zi, dist = h, z0, []
    for k in range(1, 11):
        hi = P._gru(rssm, hi, zi, act[e, t + k - 1])
        zi, _ = P._sample_onehot(rssm, rssm.prior_net(hi), gen)
        dec = model.decoder(torch.cat([hi, zi], -1))
        dist.append(torch.linalg.vector_norm(dec - obs_tgt[e, t + k], dim=-1))
    es = torch.stack(dist, 1).mean(1).numpy()
    return dict(r0=r0, r10=r10, p0=p0, p10=p10, es=es)


class Sites:
    def __init__(self, model, clf, sc, h_eval, ep, n_sites, seed, noisy_obs=None, pick=None, v=None):
        E, T = h_eval.shape[:2]
        if pick is None:
            rng = np.random.default_rng(seed)
            ts = np.arange(P.MIN_T, T - max(P.HORIZONS) - 1)
            pool = np.stack(np.meshgrid(np.arange(E), ts, indexing='ij'), -1).reshape(-1, 2)
            pick = pool[rng.choice(len(pool), n_sites, replace=False)]
        self.e, self.t = pick[:, 0], pick[:, 1]
        self.H = h_eval[self.e, self.t].astype(np.float64)
        self.obs_tgt = torch.from_numpy(ep['obs'])
        self.obs_in = torch.from_numpy(noisy_obs) if noisy_obs is not None else self.obs_tgt
        self.act = torch.from_numpy(ep['act'])
        self.model, self.clf, self.sc = model, clf, sc
        if v is None:
            raise ValueError('probe direction v required (projection readout)')
        self.v = np.asarray(v, np.float64)
        self.base = self.run(self.H)

    def run(self, Hn):
        return run_condition(self.model, self.clf, self.sc, self.obs_in, self.obs_tgt, self.act,
                             self.e, self.t, Hn, self.v)

    def effects(self, u, sigma=None, steer=True):
        res = {}
        if steer:
            means = {k: [] for k in ('r0', 'r10', 'p0', 'p10', 'es')}
            for lam in LAMBDAS:
                out = self.base if lam == 0 else self.run(self.H + lam * sigma * u[None])
                for k in means:
                    means[k].append(float(out[k].mean()))
            for k in means:
                res[f'slope_{k}'] = float(np.polyfit(LAMBDAS, means[k], 1)[0])
                res[f'means_{k}'] = means[k]
        abl = self.run(self.H - np.outer(self.H @ u, u))
        for k in ('r0', 'r10', 'p0', 'p10', 'es'):
            res[f'ablate_{k}'] = float((abl[k] - self.base[k]).mean())
        return res


def z_against(stat, null_stats):
    n = np.asarray(null_stats, np.float64)
    return dict(value=float(stat), null_mean=float(n.mean()), null_std=float(n.std()),
                z=float((stat - n.mean()) / (n.std() + 1e-12)),
                percentile=float((n < stat).mean() * 100), n=int(len(n)))


def score(eff_target, effs_null):
    return {k: z_against(eff_target[k], [e[k] for e in effs_null])
            for k in eff_target if not k.startswith('means_')}

"""Fast OLS R^2 with trajectory (episode) bootstrap via per-episode Gram matrices.

For columns M = [1, x_1, ..., x_p] (targets are ordinary columns too), each
episode e contributes G_e = M_e^T M_e over its valid sites. A bootstrap
resample with episode multiplicities w gives G = sum_e w_e G_e, from which the
in-sample OLS R^2 of any target on any subset of columns follows exactly. This
equals refitting OLS on the stacked resampled rows, without touching the rows.
"""

import numpy as np

from src.rerun.pipeline import N_BOOT, BOOT_SEED


class GramSet:
    def __init__(self, columns, mask):
        """columns: dict name -> (E,T) array; mask: (E,T) bool of valid sites
        (sites with any non-finite column value are dropped)."""
        self.names = ['_one'] + list(columns)
        self.idx = {n: i for i, n in enumerate(self.names)}
        E = mask.shape[0]
        stack = np.stack([np.ones(mask.shape)] + [np.asarray(columns[n], np.float64)
                                                  for n in columns], -1)
        valid = mask & np.all(np.isfinite(stack), -1)
        self.G = np.zeros((E, len(self.names), len(self.names)))
        self.n = valid.sum(1)
        for e in range(E):
            M = stack[e][valid[e]]
            self.G[e] = M.T @ M
        self.E = E

    def _r2_from_G(self, G, y, xs):
        yi = self.idx[y]
        xi = [0] + [self.idx[x] for x in xs]
        n = G[0, 0]
        sst = G[yi, yi] - G[0, yi] ** 2 / n
        Gxx = G[np.ix_(xi, xi)]
        Gxy = G[xi, yi]
        beta = np.linalg.solve(Gxx, Gxy)
        ssr = G[yi, yi] - Gxy @ beta
        return 1.0 - ssr / sst

    def r2(self, y, xs, w=None):
        G = self.G.sum(0) if w is None else np.tensordot(w, self.G, 1)
        return float(self._r2_from_G(G, y, xs))

    def boot_weights(self, n_boot=N_BOOT, seed=BOOT_SEED):
        rng = np.random.default_rng(seed)
        return np.stack([np.bincount(rng.integers(0, self.E, self.E), minlength=self.E)
                         for _ in range(n_boot)]).astype(np.float64)

    def stat_ci(self, fn, W):
        """fn(w) -> float for w=None (point) and each bootstrap weight row."""
        point = fn(None)
        boots = np.array([fn(w) for w in W])
        boots = boots[np.isfinite(boots)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return dict(point=float(point), lo=float(lo), hi=float(hi), n_boot=int(len(boots)))

    def r2_ci(self, y, xs, W):
        return self.stat_ci(lambda w: self.r2(y, xs, w), W)

    def delta_r2_ci(self, y, add, base, W):
        return self.stat_ci(lambda w: self.r2(y, base + add, w) - self.r2(y, base, w), W)

    def partial_r2_ci(self, y, add, base, W):
        def f(w):
            rb = self.r2(y, base, w)
            return (self.r2(y, base + add, w) - rb) / (1 - rb)
        return self.stat_ci(f, W)


def pearson_ci(g, a, b, W):
    """Signed Pearson r(a, b) with CI (sign from the covariance)."""
    def f(w):
        G = g.G.sum(0) if w is None else np.tensordot(w, g.G, 1)
        i, j = g.idx[a], g.idx[b]
        n = G[0, 0]
        cov = G[i, j] - G[0, i] * G[0, j] / n
        va = G[i, i] - G[0, i] ** 2 / n
        vb = G[j, j] - G[0, j] ** 2 / n
        return cov / np.sqrt(va * vb)
    return g.stat_ci(f, W)


def acf_within(x, mask, lags):
    """Autocorrelation at each lag, pooled over episodes, using only pairs where
    both sites are valid; mean/var from all valid sites."""
    v = x[mask]
    mu, var = v.mean(), v.var()
    out = {}
    for L in lags:
        a, b = x[:, :-L], x[:, L:]
        m = mask[:, :-L] & mask[:, L:]
        out[L] = float(((a[m] - mu) * (b[m] - mu)).mean() / var)
    return out

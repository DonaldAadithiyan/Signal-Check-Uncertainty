"""Camera-ready Figures 1 and 2 (ICAART 2027 #388), regenerated from
outputs/camera_ready/camera_ready_numbers.json with the fixed choices (corrected
logging, paper probe, projection readout v^T h_t, published gamma/alpha).
Same panels and design as the published figures, with two requested changes:
  Fig 1c: share of ||v||^2 in the top-10 PCs (standardized h, Table 3 statistic)
          against the random-direction band (mean, 5th-95th percentile);
  Fig 2a/2c: same sites (all 46,700 evaluation sites) and the same C_t definition
          (Section 6: per-trajectory median, published gamma), so (c)'s real value
          equals (a)'s K=10 value.
A variant of Fig 2 with the global-threshold C_t in panel (c) is also written
(fig2_external_globalthreshold.pdf), because the pendulum scramble z changes sign
between the two definitions.
Output: outputs/camera_ready/figures/ (paper/figures/ is not modified).
Usage (repo root): python3.11 paper/figures/src/make_cr_figs.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('CR_PROBE', 'paper')

import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from style import letter, TASKS, COL, MRK, NULL, plt  # noqa: E402

from src.camera_ready import core as C  # noqa: E402
from src.rerun import pipeline as P  # noqa: E402
from run_sprint_p1p2 import ridge_fit, r2  # noqa: E402
from run_cr_final_numbers import PAPER13, GAMMA  # noqa: E402

OUT = os.path.join(REPO, 'outputs', 'camera_ready', 'figures')
CACHE = os.path.join(OUT, '_cache_scrambles.json')
CM = 1 / 2.54
W = 6.77 * CM          # SCITEPRESS column (~192 pt)
NUM = json.load(open('outputs/camera_ready/camera_ready_numbers.json'))
FD = json.load(open('outputs/camera_ready/fig_data.json'))
M = C.MIN_T


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'MISMATCH'}] {label}: fig={got:.4f} numbers-file={want:.4f}")
    if not ok:
        raise AssertionError(f'must-match failed: {label}')


def save(f, name):
    os.makedirs(OUT, exist_ok=True)
    f.savefig(os.path.join(OUT, name), format='pdf', pad_inches=0.01)
    prev = os.environ.get('FIG_PREVIEW_DIR')
    if prev:
        os.makedirs(prev, exist_ok=True)
        f.savefig(os.path.join(prev, name.replace('.pdf', '.png')), dpi=300)
    plt.close(f)
    print('wrote', os.path.join(OUT, name))


def scramble_lists():
    """20-scramble null values for Fig 1d (ridge h->C_t) and Fig 2c (r(C_t,E10)),
    recomputed with the same seeds/definitions as the numbers file and checked."""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    out = {}
    for t in TASKS:
        g = GAMMA[t]
        S = {s: C.load_states(t, 'primary', s) for s in ('probe', 'calibration', 'evaluation')}
        thr = np.median(S['calibration']['kl'][:, M:])
        # Fig 1d -- identical procedure to A_ridge_scramble_paper_gamma.json
        hp = S['probe']['h'][:, M:].reshape(-1, 256)
        he = S['evaluation']['h'][:, M:].reshape(-1, 256)
        f = ridge_fit(hp, P.compute_ct(S['probe']['kl'], thr, g)[:, M:].ravel())
        real = r2(P.compute_ct(S['evaluation']['kl'], thr, g)[:, M:].ravel(), f(he))
        rng = np.random.default_rng(2027)

        def scr(kl):
            fl = (kl > thr).astype(float)
            return P.discounted(np.stack([r_[rng.permutation(len(r_))] for r_ in fl]), g)
        null = []
        for _ in range(20):
            fs = ridge_fit(hp, scr(S['probe']['kl'])[:, M:].ravel())
            null.append(r2(scr(S['evaluation']['kl'])[:, M:].ravel(), fs(he)))
        out[f'{t}_1d'] = dict(real=real, null=null)
        # Fig 2c -- identical procedure to run_cr_final_numbers.scramble_r (both definitions)
        kl = S['evaluation']['kl'].astype(np.float64)
        E = S['evaluation']['estate_10']
        mask = P.site_mask(kl.shape, C.KMAX)
        rng2 = np.random.default_rng(2027 + 1)
        for name, thr_fn in (('2c_sec6', lambda k: np.median(k[:, M:], axis=1, keepdims=True)),
                             ('2c_global', lambda k: thr)):
            flags = (kl > thr_fn(kl)).astype(float)
            real = pearsonr(P.discounted(flags, g)[mask], E[mask])[0]
            nl = []
            for _ in range(20):
                fl = np.stack([r_[rng2.permutation(len(r_))] for r_ in flags])
                nl.append(pearsonr(P.discounted(fl, g)[mask], E[mask])[0])
            out[f'{t}_{name}'] = dict(real=float(real), null=[float(x) for x in nl])
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(CACHE, 'w'))
    return out


def random_band_top10(t):
    """Random-unit-vector share in the top-10 PCs of standardized training-log h
    (the paper probe's fitting states), primary model: mean, 5th, 95th pct."""
    from sklearn.decomposition import PCA
    clf, sc, v, hfit = C.paper_probe(t, 'primary')
    pcs = PCA(n_components=10, random_state=0).fit(sc.transform(hfit)).components_
    R = np.random.default_rng(11).standard_normal((1000, 256))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    fr = ((R @ pcs.T) ** 2).sum(1) * 100
    return float(fr.mean()), float(np.percentile(fr, 5)), float(np.percentile(fr, 95))


def zval(d):
    n = np.array(d['null'])
    return (d['real'] - n.mean()) / n.std()


def grey_box(ax, vals, pos, width):
    bp = ax.boxplot(vals, positions=[pos], widths=width, patch_artist=True, showfliers=False)
    for b in bp['boxes']:
        b.set(facecolor=NULL, edgecolor='#777777', lw=0.5)
    for k in ('whiskers', 'caps', 'medians'):
        for b in bp[k]:
            b.set(color='#777777', lw=0.5)


def fig1(SL):
    E = NUM['emergence']
    rs = NUM['ridge_scramble']
    for t in TASKS:
        check(f'{t} Fig1d z', zval(SL[f'{t}_1d']), rs[f'corrected/{t}']['z'], 0.05)
    f = plt.figure(figsize=(W, 6.4 * CM))
    L, Rr, B, T, w, hh = 0.17, 0.66, 0.12, 0.57, 0.31, 0.33
    # (a)
    ax = f.add_axes([L, T, w, hh])
    ct, p = np.array(FD['scatter']['C']), np.array(FD['scatter']['readout'])
    ax.hexbin(ct, p, gridsize=26, cmap='Blues', mincnt=1, bins='log', linewidths=0.1)
    A = np.column_stack([ct, np.ones_like(ct)])
    coef = np.linalg.lstsq(A, p, rcond=None)[0]
    xs = np.linspace(ct.min(), ct.max(), 10)
    ax.plot(xs, coef[0] * xs + coef[1], color='k', lw=0.8)
    r2a = E['cartpole']['primary']['r2_readout_C']['point']
    ax.text(0.04, 0.96, f'$R^2={r2a:.2f}$', transform=ax.transAxes, va='top', fontsize=7)
    ax.set_xlabel(r'$C_t$ ($\gamma{=}0.95$)', labelpad=1)
    ax.set_ylabel(r'readout $v^\top h_t$', labelpad=1)
    letter(ax, 'a', x=-0.48, y=1.0)
    # (b)
    ax = f.add_axes([Rr, T, w, hh])
    x, bw = np.arange(3), 0.36
    for i, t in enumerate(TASKS):
        klv = [E[t][l]['r2_readout_kl']['point'] for l in PAPER13[t]]
        cv = [E[t][l]['r2_readout_C']['point'] for l in PAPER13[t]]
        ax.bar(x[i] - bw / 2, np.mean(klv), bw, color='white', edgecolor=COL[t], hatch='////', lw=0.6)
        ax.bar(x[i] + bw / 2, np.mean(cv), bw, color=COL[t], edgecolor=COL[t], lw=0.6)
        ax.scatter(np.full(len(klv), x[i] - bw / 2), klv, s=3, color='k', marker=MRK[t], zorder=3, lw=0)
        ax.scatter(np.full(len(cv), x[i] + bw / 2), cv, s=3, color='k', marker=MRK[t], zorder=3, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels(['cart', 'reach', 'pend'])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel(r'$R^2$ of readout', labelpad=1)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='white', edgecolor='k', hatch='////', lw=0.5, label='KL$_t$'),
                       Patch(facecolor='#777777', label='$C_t$')], loc='upper left', ncol=2,
              handlelength=1.2, columnspacing=0.8, borderaxespad=0.1, bbox_to_anchor=(0, 1.02))
    letter(ax, 'b', x=-0.42, y=1.0)
    # (c) share of v in top-10 PCs vs random band
    ax = f.add_axes([L, B, w, hh])
    for i, t in enumerate(TASKS):
        mean, p5, p95 = random_band_top10(t)
        ax.fill_between([i - 0.3, i + 0.3], p5, p95, color=NULL, alpha=0.6, lw=0)
        ax.plot([i - 0.3, i + 0.3], [mean] * 2, color='#777777', lw=0.6)
        vals = [E[t][l]['geometry']['standardized_top10']['frac'] * 100 for l in PAPER13[t]]
        check(f'{t} random mean (fig vs numbers)', mean, E[t]['primary']['geometry']['standardized_top10']['random_mean'] * 100, 0.01)
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=5, color=COL[t],
                   marker=MRK[t], lw=0, zorder=3)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['cart', 'reach', 'pend'])
    ax.set_ylabel(r'% of $\|v\|^2$ in top-10 PCs', labelpad=1)
    letter(ax, 'c', x=-0.48, y=1.0)
    # (d)
    ax = f.add_axes([Rr, B, w, hh])
    for i, t in enumerate(TASKS):
        d = SL[f'{t}_1d']
        grey_box(ax, d['null'], i - 0.17, 0.25)
        ax.scatter([i + 0.17], [d['real']], marker=MRK[t], s=12, color=COL[t], zorder=3)
        ax.text(i, 1.02, f'{zval(d):+.1f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['cart', 'reach', 'pend'])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel(r'$R^2(h_t \rightarrow C_t)$', labelpad=1)
    ax.text(1.0, -0.2, 'grey: scrambled; colour: real; top: $z$', transform=ax.transAxes,
            ha='right', va='top', fontsize=6)
    letter(ax, 'd', x=-0.42, y=1.0)
    save(f, 'fig1_emergence.pdf')


def fig2(SL, variant):
    X = NUM['external']
    Ks = [1, 5, 10, 20]
    key = '2c_sec6' if variant == 'sec6' else '2c_global'
    for t in TASKS:
        check(f'{t} Fig2a K=10 r = Fig2c real (sec6)', SL[f'{t}_2c_sec6']['real'], X[t]['horizons']['10']['point'], 1e-6)
        jk = 'scramble_section6_def' if variant == 'sec6' else 'scramble_global_threshold'
        check(f'{t} Fig2c z ({variant})', zval(SL[f'{t}_{key}']), X[t][jk]['z'], 0.05)
    f = plt.figure(figsize=(W, 7.6 * CM))
    ax1 = f.add_axes([0.19, 0.765, 0.78, 0.195])
    for t in TASKS:
        h = X[t]['horizons']
        ax1.fill_between(Ks, [h[str(k)]['lo'] for k in Ks], [h[str(k)]['hi'] for k in Ks], color=COL[t], alpha=0.25, lw=0)
        ax1.plot(Ks, [h[str(k)]['point'] for k in Ks], color=COL[t], marker=MRK[t], ms=3, label=t)
    ax1.set_xticks(Ks)
    ax1.set_xlabel('imagination horizon $K$', labelpad=1)
    ax1.set_ylabel(r'$r(C_t, E_{t,K})$')
    ax1.set_ylim(0, 0.9)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax1.legend(loc='upper right', ncol=3, handlelength=1.5, columnspacing=0.8, bbox_to_anchor=(1.0, 1.08))
    letter(ax1, 'a', x=-0.22, y=0.97)
    ax2 = f.add_axes([0.19, 0.465, 0.44, 0.20])
    y = np.arange(3)[::-1]
    for yi, t in zip(y, TASKS):
        d = X[t]['delta_r2_C']
        ax2.errorbar(d['point'], yi, xerr=[[d['point'] - d['lo']], [d['hi'] - d['point']]], fmt=MRK[t],
                     color=COL[t], ms=3.5, capsize=2, elinewidth=0.8, capthick=0.8)
    ax2.axvline(0, color='k', lw=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(TASKS)
    hi = max(X[t]['delta_r2_C']['hi'] for t in TASKS)
    ax2.set_xlim(-0.1 * hi, 1.1 * hi)
    ax2.set_ylim(-0.6, 2.6)
    ax2.set_xlabel(r'$\Delta R^2$ from $C_t$ ($K{=}10$, 95% CI)', labelpad=1)
    letter(ax2, 'b', x=-0.42, y=0.97)
    ax2i = f.add_axes([0.72, 0.465, 0.25, 0.17])
    d = X['pendulum']['delta_r2_C']
    ax2i.errorbar(d['point'], 0, xerr=[[d['point'] - d['lo']], [d['hi'] - d['point']]], fmt=MRK['pendulum'],
                  color=COL['pendulum'], ms=3.5, capsize=2, elinewidth=0.8, capthick=0.8)
    ax2i.axvline(0, color='k', lw=0.5)
    ax2i.set_xlim(-0.001, 1.15 * d['hi'])
    ax2i.set_yticks([])
    ax2i.spines['left'].set_visible(False)
    ax2i.set_ylim(-1, 1)
    ax2i.tick_params(axis='x', labelsize=6)
    ax2i.set_title('pendulum (zoom)', fontsize=6, pad=1)
    for i, t in enumerate(TASKS):
        ax = f.add_axes([0.19 + i * 0.275, 0.065, 0.21, 0.17])
        d = SL[f'{t}_{key}']
        null = np.array(d['null'])
        grey_box(ax, null, 0, 0.5)
        ax.scatter(np.random.default_rng(0).uniform(-0.12, 0.12, len(null)), null, s=1.5, color='#777777', zorder=3)
        ax.scatter([1], [d['real']], marker=MRK[t], s=16, color=COL[t], zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['scr.', 'real'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.05, 0.9)
        ax.set_yticks([0, 0.4, 0.8])
        if i == 0:
            ax.set_ylabel(r'$r(C_t, E_{t,10})$')
            letter(ax, 'c', x=-0.8, y=1.12)
        else:
            ax.set_yticklabels([])
        ax.text(0.5, 1.0, f'{t}\n$z={zval(d):+.1f}$', transform=ax.transAxes, ha='center', va='bottom', fontsize=6)
    save(f, 'fig2_external.pdf' if variant == 'sec6' else 'fig2_external_globalthreshold.pdf')


if __name__ == '__main__':
    SL = scramble_lists()
    fig1(SL)
    fig2(SL, 'sec6')
    fig2(SL, 'global')

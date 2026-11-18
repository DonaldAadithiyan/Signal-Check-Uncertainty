"""Camera-ready figures (ICAART 2027 #388), final rule. PLOTTING ONLY: every plotted value
is read from the final JSON files and checked against them (1e-9) after drawing.

Sizes from SCITEPRESS.sty (no LaTeX installed): \\textwidth = 6.221 in = 15.80 cm,
\\columnsep = 0.8 cm, two-column -> \\columnwidth = 7.50 cm (cross-checked against the
column offset in the template's Example.pdf: 235.3 bp = column + gap).
Single-column plots: 7.50 x 4.65 cm (0.62 x width). Figure 7: (15.80 - 0.6)/3 = 5.07 cm
wide, 4.31 cm high (0.85 x width).
Output: outputs/camera_ready/figures_final/{*.pdf,*.png}. Usage: python3.11 <this file>
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CR = os.path.join(REPO, 'outputs', 'camera_ready')
OUT = os.path.join(CR, 'figures_final')
CM = 1 / 2.54
TEXTW_CM = 6.221 * 2.54
COLW_CM = (TEXTW_CM - 0.8) / 2
W1, H1 = COLW_CM * CM, 0.62 * COLW_CM * CM
W7 = (TEXTW_CM - 2 * 0.3) / 3
W7I, H7I = W7 * CM, 0.85 * W7 * CM

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'],
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 7, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 3.5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
})
TCOL = {'cartpole': '#0072B2', 'reacher': '#E69F00', 'pendulum': '#009E73', 'walker': '#CC79A7', 'cheetah': '#56B4E9'}
ORIG, NEW = ['cartpole', 'reacher', 'pendulum'], ['walker', 'cheetah']
SCOL = {'readout': '#D55E00', 'C_t_past': '#0072B2', 'Rec_tm1': '#555555', 'EMARec_tm1': '#888888',
        'KL_tm1': '#AAAAAA', 'random': '#BBBBBB', 'oracle': 'black'}
P13 = {'cartpole': ['primary', 'seed_1', 'seed_2', 'seed_3', 'seed_4'],
       'reacher': ['primary', 'reacher_seed1', 'reacher_seed2', 'reacher_seed3'],
       'pendulum': ['primary', 'pendulum_seed1', 'pendulum_seed2', 'pendulum_seed3']}


def J(name):
    return json.load(open(os.path.join(CR, name)))


NUM, CV, T3, T4, B, FD = (J('camera_ready_numbers.json'), J('CV_delta_r2.json'), J('T3_decision.json')['results'],
                          J('T4_generality.json'), J('B_sae_corrected_refit.json'), J('fig_data.json'))
CHECKS = []
MANIFEST = {}


def check(label, plotted, source):
    p, s = np.asarray(plotted, float).ravel(), np.asarray(source, float).ravel()
    ok = p.shape == s.shape and np.all(np.abs(p - s) <= 1e-9)
    CHECKS.append((label, int(s.size), bool(ok)))
    if not ok:
        raise AssertionError(f'value check failed: {label}')


def save(fig, name, meta):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), format='pdf')
    fig.savefig(os.path.join(OUT, name.replace('.pdf', '.png')), dpi=300)
    w, h = fig.get_size_inches()
    meta['size_cm'] = (round(w * 2.54, 2), round(h * 2.54, 2))
    MANIFEST[name] = meta
    plt.close(fig)
    print('wrote', name)


def divider(ax, x, label=True, where='top'):
    ax.axvline(x, color='#999999', lw=0.5, ls=':')
    if label and where == 'topleft':
        ax.text(x - 0.06, 1.0, r'new tasks $\rightarrow$', transform=ax.get_xaxis_transform(), fontsize=7,
                color='#666666', va='top', ha='right')
    elif label:
        ax.text(x + 0.08, 1.0, 'new tasks', transform=ax.get_xaxis_transform(), fontsize=7, color='#666666',
                va='top', ha='left')


def mstyle(task):
    return dict(facecolor='none' if task in NEW else TCOL[task], edgecolor=TCOL[task])


# ── Figure 1 ─────────────────────────────────────────────────────────────────
def fig1():
    f, ax = plt.subplots(figsize=(W1, H1))
    f.subplots_adjust(left=0.15, right=0.97, bottom=0.21, top=0.96)
    c, p = np.array(FD['scatter']['C']), np.array(FD['scatter']['readout_prob'])
    hb = ax.hexbin(c, p, gridsize=(28, 14), cmap='Blues', mincnt=1, bins='log', linewidths=0.1)
    A = np.column_stack([c, np.ones_like(c)])
    coef = np.linalg.lstsq(A, p, rcond=None)[0]
    xs = np.array([c.min(), c.max()])
    ln, = ax.plot(xs, coef[0] * xs + coef[1], color='black', lw=1.0)
    r2 = NUM['emergence']['cartpole']['primary']['r2_readout_C']['point']
    ax.text(0.03, 0.95, f'$R^2 = {r2:.2f}$', transform=ax.transAxes, va='top', fontsize=7)
    ax.set_xlabel(r'$C_t$ ($\gamma = 0.95$)')
    ax.set_ylabel('readout')
    ax.set_ylim(-0.03, 1.03)
    check('fig1 scatter n', [len(c), len(p)], [3000, 3000])
    check('fig1 hexbin total count', [hb.get_array().sum()], [len(c)])
    check('fig1 R2 annotation (2 dp)', [round(r2, 2)], [round(NUM['emergence']['cartpole']['primary']['r2_readout_C']['point'], 2)])
    save(f, 'fig01_readout_vs_Ct.pdf', dict(
        section='5 (probe direction tracks accumulated difficulty)',
        source='fig_data.json scatter (3,000 seeded evaluation sites, written by run_cr_final_numbers.py under the final rule); '
               'camera_ready_numbers.json emergence.cartpole.primary.r2_readout_C (R^2 on all 46,700 sites)',
        key=f'R^2 = {r2:.3f} (annotated 0.84); least-squares line fit to the 3,000 plotted points (slope {coef[0]:.4f})',
        caption='Probe readout against accumulated difficulty $C_t$ on cartpole (3,000 evaluation sites); line: least-squares fit; $R^2$ over all sites.',
        notes='Hexbin (log counts) used because 3,000 points overplot heavily. fig_data.json is not in the handoff\'s source list, '
              'but it is written by the same final-rule script (run_cr_final_numbers.py) and contains only these 3,000 (C_t, readout) pairs.'))


# ── Figure 2 ─────────────────────────────────────────────────────────────────
def fig2():
    f, ax = plt.subplots(figsize=(W1, H1))
    f.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.88)
    tasks = ORIG + NEW
    xpos = {'cartpole': 0, 'reacher': 1, 'pendulum': 2, 'walker': 3.4, 'cheetah': 4.4}
    bw = 0.36
    plotted, src = [], []
    for t in tasks:
        if t in ORIG:
            cv = [NUM['emergence'][t][l]['r2_readout_C']['point'] for l in P13[t]]
            kv = [NUM['emergence'][t][l]['r2_readout_kl']['point'] for l in P13[t]]
        else:
            cv, kv = [T4[t]['r2_readout_C']['point']], [T4[t]['r2_readout_kl']['point']]
        x = xpos[t]
        bc = ax.bar(x - bw / 2, np.mean(cv), bw, color=TCOL[t], edgecolor=TCOL[t], lw=0.6)
        bk = ax.bar(x + bw / 2, np.mean(kv), bw, color='white', edgecolor=TCOL[t], hatch='/////', lw=0.6)
        sc1 = ax.scatter(np.full(len(cv), x - bw / 2), cv, s=6, marker='o', zorder=3, lw=0.6,
                         facecolor='black' if t in ORIG else 'white', edgecolor='black')
        sc2 = ax.scatter(np.full(len(kv), x + bw / 2), kv, s=6, marker='o', zorder=3, lw=0.6,
                         facecolor='black' if t in ORIG else 'white', edgecolor='black')
        plotted += [bc[0].get_height(), bk[0].get_height()] + list(sc1.get_offsets()[:, 1]) + list(sc2.get_offsets()[:, 1])
        src += [np.mean(cv), np.mean(kv)] + cv + kv
    divider(ax, 2.7)
    ax.set_xticks([xpos[t] for t in tasks])
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r'$R^2$ of readout')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#777777', label=r'$C_t$'),
                       Patch(facecolor='white', edgecolor='#777777', hatch='/////', label=r'KL$_t$')],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2, handlelength=1.4)
    check('fig2 bars and dots', plotted, src)
    prim = {t: (NUM['emergence'][t]['primary']['r2_readout_C']['point'], NUM['emergence'][t]['primary']['r2_readout_kl']['point']) for t in ORIG}
    means = {t: (np.mean([NUM['emergence'][t][l]['r2_readout_C']['point'] for l in P13[t]]),
                 np.mean([NUM['emergence'][t][l]['r2_readout_kl']['point'] for l in P13[t]])) for t in ORIG}
    save(f, 'fig02_R2_Ct_vs_KL.pdf', dict(
        section='5',
        source='camera_ready_numbers.json emergence.<task>.<model>.r2_readout_C / r2_readout_kl (13 original models); '
               'T4_generality.json <task>.r2_readout_C / r2_readout_kl (walker, cheetah: 1 model each)',
        key='bars = mean over models, C_t vs KL: ' + '; '.join(f'{t} {means[t][0]:.2f} vs {means[t][1]:.2f}' for t in ORIG) +
            f"; walker {T4['walker']['r2_readout_C']['point']:.2f} vs {T4['walker']['r2_readout_kl']['point']:.2f}; "
            f"cheetah {T4['cheetah']['r2_readout_C']['point']:.2f} vs {T4['cheetah']['r2_readout_kl']['point']:.2f}. "
            'Primary models (dots): ' + '; '.join(f'{t} {prim[t][0]:.2f} vs {prim[t][1]:.2f}' for t in ORIG),
        caption='Variance of the readout explained by $C_t$ (filled) and current KL (hatched); bars: model means, dots: individual models.',
        notes='As specified, bars are model means, so the original-task bars show 0.83 / 0.52 / 0.77 (C_t), not the primary-model values '
              '0.84 / 0.50 / 0.89 listed in the spec (those are the primary-model dots). New-task dots are open markers.'))


# ── Figure 3 ─────────────────────────────────────────────────────────────────
def fig3():
    f, ax = plt.subplots(figsize=(W1, H1))
    f.subplots_adjust(left=0.14, right=0.97, bottom=0.21, top=0.88)
    Ks = [1, 5, 10, 20]
    for t in ORIG:
        h = NUM['external'][t]['horizons']
        y = [h[str(k)]['point'] for k in Ks]
        lo, hi = [h[str(k)]['lo'] for k in Ks], [h[str(k)]['hi'] for k in Ks]
        band = ax.fill_between(Ks, lo, hi, color=TCOL[t], alpha=0.22, lw=0)
        ln, = ax.plot(Ks, y, color=TCOL[t], marker='o', ms=3.5, label=t)
        check(f'fig3 {t} r', ln.get_ydata(), y)
        verts = band.get_paths()[0].vertices
        check(f'fig3 {t} band lo/hi present', [np.min(verts[:, 1]), np.max(verts[:, 1])], [min(lo), max(hi)])
    ax.set_xticks(Ks)
    ax.set_xlabel(r'imagination horizon $K$')
    ax.set_ylabel(r'$r(C_t, E_{t,K})$')
    ax.set_ylim(0, 0.8)
    ax.axhline(0, color='black', lw=0.4)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2, handlelength=1.6)
    vals = {t: ' / '.join(f"{NUM['external'][t]['horizons'][str(k)]['point']:+.2f}" for k in Ks) for t in ORIG}
    save(f, 'fig03_horizon.pdf', dict(
        section='6 (C_t predicts later imagination error)',
        source='camera_ready_numbers.json external.<task>.horizons.<K> (Section-6 C_t: per-trajectory median, published gamma; '
               '46,700 sites, 100 episodes; trajectory-bootstrap 95% bands)',
        key='; '.join(f'{t} K=1/5/10/20: {vals[t]}' for t in ORIG),
        caption=r'Correlation of $C_t$ with imagination error $E_{t,K}$ by horizon $K$; bands: 95% trajectory-bootstrap CIs.',
        notes='Walker and cheetah not plotted. Their logs do contain E_{t,K} for all four K (outputs/camera_ready/states/*_evaluation.npz), '
              'but r(C_t, E_{t,K}) for those tasks is not in any final JSON (T4_generality.json has only r(C_t, E_{t,10})); '
              'plotting them would require computing new numbers, which the figure handoff excludes. Can be added on request.'))


# ── Figure 4 ─────────────────────────────────────────────────────────────────
def fig4():
    f, ax = plt.subplots(figsize=(W1, H1))
    f.subplots_adjust(left=0.17, right=0.97, bottom=0.14, top=0.95)
    tasks = ORIG + NEW
    xpos = {'cartpole': 0, 'reacher': 1, 'pendulum': 2, 'walker': 3.4, 'cheetah': 4.4}
    pts, src = [], []
    for t in tasks:
        r = CV[t]['a_T4_C_beyond_controls']['cv']
        eb = ax.errorbar(xpos[t], r['point'], yerr=[[r['point'] - r['lo']], [r['hi'] - r['point']]], fmt='o', ms=4,
                         color=TCOL[t], mfc='white' if t in NEW else TCOL[t], mec=TCOL[t], capsize=2, elinewidth=1.0, capthick=1.0)
        seg = eb[2][0].get_segments()[0]
        pts += [eb[0].get_ydata()[0], seg[0][1], seg[1][1]]
        src += [r['point'], r['lo'], r['hi']]
    ax.axhline(0, color='black', lw=0.4)
    divider(ax, 2.7, where='topleft')
    ax.set_xticks([xpos[t] for t in tasks])
    ax.set_xticklabels(tasks)
    ax.set_ylabel(r'$\Delta R^2_{\mathrm{cv}}$ of $C_t$')
    ax.set_xlim(-0.5, 4.9)
    ax.set_ylim(-0.003, 0.04)
    # inset: pendulum and walker
    ins = f.add_axes([0.74, 0.62, 0.23, 0.30])
    for i, t in enumerate(['pendulum', 'walker']):
        r = CV[t]['a_T4_C_beyond_controls']['cv']
        eb = ins.errorbar(i, r['point'], yerr=[[r['point'] - r['lo']], [r['hi'] - r['point']]], fmt='o', ms=3.5,
                          color=TCOL[t], mfc='white' if t in NEW else TCOL[t], mec=TCOL[t], capsize=1.5, elinewidth=0.9)
        seg = eb[2][0].get_segments()[0]
        pts += [eb[0].get_ydata()[0], seg[0][1], seg[1][1]]
        src += [r['point'], r['lo'], r['hi']]
    ins.axhline(0, color='black', lw=0.4)
    ins.set_xlim(-0.6, 1.6)
    ins.set_ylim(-0.0005, 0.004)
    ins.set_xticks([0, 1])
    ins.set_xticklabels(['pend.', 'walker'], fontsize=6)
    ins.set_yticks([0, 0.002, 0.004])
    ins.set_yticklabels(['0', '.002', '.004'], fontsize=6)
    ins.tick_params(length=1.5)
    check('fig4 points and CIs (main + inset)', pts, src)
    save(f, 'fig04_cv_deltaR2_Ct.pdf', dict(
        section='6 (headline result)',
        source="CV_delta_r2.json <task>.a_T4_C_beyond_controls.cv {point, lo, hi} (grouped 5-fold CV by trajectory, seed 55; "
               "trajectory-bootstrap 95% CI over fixed out-of-fold predictions; 46,700 sites per task)",
        key='; '.join(f"{t} {CV[t]['a_T4_C_beyond_controls']['cv']['point']:+.4f} [{CV[t]['a_T4_C_beyond_controls']['cv']['lo']:+.4f}, "
                      f"{CV[t]['a_T4_C_beyond_controls']['cv']['hi']:+.4f}]" for t in tasks),
        caption=r'Out-of-sample $\Delta R^2_{\mathrm{cv}}$ of $C_t$ for $E_{t,10}$ beyond KL, Rec and EMARec; 95% CIs; inset enlarges pendulum and walker.',
        notes='Inset added because the pendulum and walker CIs are not readable on the main linear scale. '
              'Controls include ensemble disagreement on cartpole only (no ensembles exist for the other tasks).'))


# ── Figure 5 ─────────────────────────────────────────────────────────────────
def fig5():
    f, ax = plt.subplots(figsize=(W1, H1))
    f.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.88)
    pts, src = [], []
    for i, t in enumerate(ORIG):
        s = NUM['external'][t]['scramble_section6_def']
        eb = ax.errorbar(i - 0.15, s['scrambled_mean'], yerr=s['scrambled_sd'], fmt='o', ms=3.5, color='#888888',
                         mfc='white', capsize=2, elinewidth=1.0)
        rp = ax.scatter([i + 0.15], [s['real']], s=18, color=TCOL[t], zorder=3)
        ax.text(i, 0.80, f"$z = {s['z']:+.1f}$", ha='center', va='bottom', fontsize=7)
        seg = eb[2][0].get_segments()[0]
        pts += [eb[0].get_ydata()[0], seg[1][1] - eb[0].get_ydata()[0], rp.get_offsets()[0, 1]]
        src += [s['scrambled_mean'], s['scrambled_sd'], s['real']]
        # check that the real value equals Figure 3's K=10 value (same sites, same C_t)
        check(f'fig5 {t} real r == fig3 K=10', [s['real']], [NUM['external'][t]['horizons']['10']['point']])
    check('fig5 real / scrambled mean / sd', pts, src)
    ax.axhline(0, color='black', lw=0.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(ORIG)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.05, 0.92)
    ax.set_ylabel(r'$r(C_t, E_{t,10})$')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#888888', mfc='white', label='scrambled order (mean ± sd, 20)'),
                       Line2D([], [], marker='o', ls='', color='#444444', label='real order')],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2, handletextpad=0.3)
    zs = {t: NUM['external'][t]['scramble_section6_def'] for t in ORIG}
    save(f, 'fig05_order_scramble.pdf', dict(
        section='6',
        source='camera_ready_numbers.json external.<task>.scramble_section6_def {real, scrambled_mean, scrambled_sd, z} '
               '(Section-6 C_t; same 46,700 sites as Figure 3; 20 within-trajectory permutations of the high-KL flags)',
        key='; '.join(f"{t}: real {zs[t]['real']:+.3f}, scrambled {zs[t]['scrambled_mean']:+.3f} ± {zs[t]['scrambled_sd']:.3f}, z {zs[t]['z']:+.1f}" for t in ORIG),
        caption=r'Real-order $C_t$ vs. $C_t$ from time-scrambled high-KL flags (20 permutations, mean ± sd) in predicting $E_{t,10}$, with $z$.',
        notes='Scrambles are stored as mean and sd only, so mean ± sd is drawn rather than a box. The scrambled sd is small '
              '(0.002-0.017), so the error bars are about the marker size. A check confirms each real r equals Figure 3\'s K=10 value.'))


# ── Figure 6 ─────────────────────────────────────────────────────────────────
def fig6():
    f, ax = plt.subplots(figsize=(W1, 1.25 * H1))
    f.subplots_adjust(left=0.40, right=0.97, bottom=0.15, top=0.99)
    rows = [('b_T1_readout_prob_beyond_controls', 'readout | KL, Rec, EMARec', 'readout'),
            ('c_T2_readout_prob_beyond_6_standard', 'readout | 6 standard signals', 'readout'),
            ('c_T2_C_beyond_6_standard', r'$C_t$ | 6 standard signals', 'C_t')]
    y, labels, pts, src, keys = 0, [], [], [], []
    yt = []
    for t in ORIG:
        ax.text(1.0, -y + 0.75, t, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7,
                color=TCOL[t], fontweight='bold')
        for k, lab, sig in rows:
            r = CV[t][k]['cv']
            col = SCOL['readout'] if sig == 'readout' else SCOL['C_t_past']
            filled = r['lo'] > 0 or r['hi'] < 0
            eb = ax.errorbar(r['point'], -y, xerr=[[r['point'] - r['lo']], [r['hi'] - r['point']]], fmt='o', ms=3.5,
                             color=col, mfc=col if filled else 'white', mec=col, capsize=1.5, elinewidth=1.0)
            seg = eb[2][0].get_segments()[0]
            pts += [eb[0].get_xdata()[0], seg[0][0], seg[1][0]]
            src += [r['point'], r['lo'], r['hi']]
            labels.append(lab)
            yt.append(-y)
            keys.append(f"{t} {lab.replace('$', '')}: {r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{' *' if filled else ''}")
            y += 1
        y += 0.9
    ax.axvline(0, color='black', lw=0.4)
    ax.set_yticks(yt)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(r'$\Delta R^2_{\mathrm{cv}}$ for $E_{t,10}$', labelpad=1)
    ax.set_ylim(-y + 0.5, 1.3)
    check('fig6 points and CIs', pts, src)
    save(f, 'fig06_comparison_forest.pdf', dict(
        section='7 (comparison with standard uncertainty signals; Reviewer 2)',
        source='CV_delta_r2.json <task>.{b_T1_readout_prob_beyond_controls, c_T2_readout_prob_beyond_6_standard, c_T2_C_beyond_6_standard}.cv '
               '(readout = probability; 46,700 sites per task; 5-fold grouped CV; trajectory-bootstrap CI)',
        key='; '.join(keys) + '  (* = CI excludes zero: filled marker)',
        caption=r'Out-of-sample $\Delta R^2_{\mathrm{cv}}$ (95% CI) of the readout and $C_t$ beyond standard uncertainty signals; open markers: CI includes zero.',
        notes='Height 1.25 x the 0.62 x width standard (5.81 cm instead of 4.65 cm): nine rows plus task labels were not legible at 4.65 cm. '
              'Shown as specified: reacher rows 1-2 and the cartpole/reacher C_t rows are open (CI includes zero).'))


# ── Figure 7 ─────────────────────────────────────────────────────────────────
def fig7():
    order = [('readout', 'readout', '-'), ('C_t_past', r'$C_t$ (past)', '-'), ('Rec_tm1', r'Rec$_{t-1}$', '-'),
             ('EMARec_tm1', r'EMARec$_{t-1}$', '-'), ('KL_tm1', r'KL$_{t-1}$', '-'), ('random', 'random', '--'),
             ('oracle', 'oracle', ':')]
    for letter, t in zip('abc', ORIG):
        f, ax = plt.subplots(figsize=(W7I, H7I))
        f.subplots_adjust(left=0.24, right=0.97, bottom=0.21, top=0.97)
        o = T3[t]
        cov = o['coverages']
        for k, lab, ls in order:
            y = o['risk'][k]['curve']
            ln, = ax.plot(cov, y, color=SCOL[k], ls=ls, lw=1.0, label=lab,
                          zorder=3 if k in ('readout', 'C_t_past') else 2)
            check(f'fig7{letter} {t} {k} curve', ln.get_ydata(), y)
            check(f'fig7{letter} {t} coverage', ln.get_xdata(), cov)
        ax.set_xlabel('coverage $c$')
        ax.set_ylabel(r'mean $E_{t,10}$ (trusted)')
        ax.set_xlim(0.08, 1.02)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        if t == 'cartpole':
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.92), fontsize=6, handlelength=1.6, labelspacing=0.15,
                      borderaxespad=0.2)
        ax.text(0.97, 0.04, t, transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        a = o['aurc']
        save(f, f'fig07{letter}_risk_coverage_{t}.pdf', dict(
            section='7 (decision example: selective trust in imagination; Reviewer 1)',
            source=f'T3_decision.json results.{t}.risk.<signal>.curve at coverages {cov} (46,700 sites; readout ranked by logit)',
            key='AURC: ' + ', '.join(f"{k} {a[k]['point']:.4f}" for k, _, _ in order) +
                f"; best non-oracle baseline {o['best_baseline']}; rule FAIL (readout AURC - best = "
                f"{o['readout']['diff_vs_best_baseline']['point']:+.4f})",
            caption=f'Risk–coverage on {t}: mean imagination error of the trusted fraction $c$ of sites, ranked by each decision-time signal.',
            notes='Legend only in the cartpole file (fontsize 6 to fit 7 entries in a 5.07 cm panel; the other text is 7-8 pt). '
                  'The readout, C_t, Rec, EMARec and KL curves lie close together and partly overlap at this size; readout and '
                  'C_t are drawn on top. EMA-KL and prior entropy are omitted as specified.'))


# ── Figure 8 ─────────────────────────────────────────────────────────────────
def fig8():
    f, ax = plt.subplots(figsize=(W1, H1))
    f.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.80)
    ax.axhspan(-2, 2, color='#E5E5E5', lw=0, zorder=0)
    ax.axhline(0, color='black', lw=0.4, zorder=1)
    qs = [('slope_r0', 'readout, $k{=}0$'), ('slope_r10', 'readout, $k{=}10$'), ('slope_es', r'$E$')]
    models = [(t, l) for t in ORIG for l in P13[t]] + [(t, 'primary') for t in NEW]
    rng = np.random.default_rng(8)
    jit = rng.uniform(-0.22, 0.22, len(models))
    pts, src = [], []
    for qi, (k, _) in enumerate(qs):
        for mi, (t, l) in enumerate(models):
            z = NUM['causal'][t][l][k]['z'] if t in ORIG else T4[t]['steering_z'][k]
            sc = ax.scatter([qi + jit[mi]], [z], s=12, zorder=3, lw=0.8, **mstyle(t))
            pts.append(sc.get_offsets()[0, 1]); src.append(z)
    check('fig8 steering z (15 models x 3)', pts, src)
    ax.set_xticks(range(3))
    ax.set_xticklabels([q[1] for q in qs])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel(r'steering $z$')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', ms=3.5, mfc=TCOL[t] if t in ORIG else 'none', mec=TCOL[t], label=t)
                       for t in ORIG + NEW], loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2,
              handletextpad=0.1, columnspacing=0.9, labelspacing=0.2)
    e = [NUM['causal'][t][l]['slope_es']['z'] for t in ORIG for l in P13[t]]
    r10 = [NUM['causal'][t][l]['slope_r10']['z'] for t in ORIG for l in P13[t]] + [T4[t]['steering_z']['slope_r10'] for t in NEW]
    save(f, 'fig08_steering.pdf', dict(
        section='8 (steering dissociates the readout from imagination error)',
        source='camera_ready_numbers.json causal.<task>.<model>.{slope_r0, slope_r10, slope_es}.z (13 original models; readout = probability); '
               'T4_generality.json <task>.steering_z (walker, cheetah); 500 sites per model; canonical 50-direction null with CRN',
        key=f"E: 13 original models |z| <= {max(abs(x) for x in e):.2f}, walker {T4['walker']['steering_z']['slope_es']:+.2f}, "
            f"cheetah {T4['cheetah']['steering_z']['slope_es']:+.2f}; readout k=10 range {min(r10):+.1f}..{max(r10):+.1f} "
            f"(below 2 on {sum(x < 2 for x in r10)} of 15 models)",
        caption=r'Steering the probe direction: $z$ against 50 random directions for the readout ($k{=}0$, $k{=}10$) and imagination error $E$; grey: $|z|<2$.',
        notes='Spec supports "moves the readout, not imagination error, on 14 of 15 models". That holds for k=0 and E. At k=10 the readout '
              f'z is below 2 on {sum(x < 2 for x in r10)} of 15 models (pendulum 4/4, reacher 3/4, cartpole 1/5), which is visible in the middle column. '
              'No annotation on walker\'s E point (+2.30), as specified.'))


# ── Figure 9 ─────────────────────────────────────────────────────────────────
def fig9():
    f, ax = plt.subplots(figsize=(W1, H1))
    f.subplots_adjust(left=0.13, right=0.97, bottom=0.21, top=0.80)
    marks = {0: 'o', 1: 's', 2: '^'}
    pts, src, cos_list = [], [], []
    for t in ORIG:
        for r in B[t]['runs']:
            x, y = r['ablate_readout_logit']['z'], r['ablate_E']['z']
            sc = ax.scatter([x], [y], marker=marks[r['sae_run']], s=16, color=TCOL[t], zorder=3)
            pts += list(sc.get_offsets()[0]); src += [x, y]
            cos_list.append(f"{t} run {r['sae_run']} #{r['atom']}: |cos| {abs(r['cos_atom_v']):.3f}, readout z {x:+.2f}, E z {y:+.2f}")
    for v in (-2, 2):
        ax.axvline(v, color='#888888', ls='--', lw=0.6)
        ax.axhline(v, color='#888888', ls='--', lw=0.6)
    ax.axhline(0, color='black', lw=0.4)
    ax.axvline(0, color='black', lw=0.4)
    check('fig9 atom ablation z (9 atoms)', pts, src)
    ax.set_xlabel(r'ablation $z$ on readout (logit)')
    ax.set_ylabel(r'ablation $z$ on $E$')
    ax.set_xlim(-4, 11)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=TCOL[t], label=t) for t in ORIG] +
              [Line2D([], [], marker=marks[s], ls='', color='#666666', mfc='none', label=f'run {s}') for s in range(3)],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2, handletextpad=0.05, columnspacing=0.9,
              labelspacing=0.2)
    nE = sum(r['ablate_E']['z'] > 2 for t in ORIG for r in B[t]['runs'])
    save(f, 'fig09_atoms.pdf', dict(
        section='9 (error-related SAE atoms)',
        source='B_sae_corrected_refit.json <task>.runs[*].{ablate_readout_logit.z, ablate_E.z, cos_atom_v} '
               '(3 retrained SAEs; atom per run by the paper\'s rule; 800 sites; canonical 50-direction null)',
        key=f'E z > 2 for {nE} of 9 atoms; ' + '; '.join(cos_list),
        caption=r'Ablating error-correlated SAE atoms (3 tasks × 3 SAE runs): $z$ on imagination error vs. on the readout (logit); dashed: $|z|=2$.',
        notes='Readout on the logit because the stored probability null is degenerate. Its sd is 0.03-0.06 in the atom runs '
              'against saturated random-direction readouts; see camera_ready_numbers.md. |cos(atom, v)| labels were left off '
              'the plot (they cluttered 9 points in 7.5 cm) and are listed above; all are <= 0.08. The x-axis cuts at 11; '
              'reacher run 1 (readout z +9.50, E z +33.98) is inside the range.'))


def font_report():
    import pypdf
    rep = {}
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith('.pdf'):
            continue
        fonts = {}
        for page in pypdf.PdfReader(os.path.join(OUT, fn)).pages:
            res = page.get('/Resources') or {}
            for _, ref in (res.get('/Font') or {}).items():
                fo = ref.get_object()
                name = str(fo.get('/BaseFont'))
                desc = fo.get('/FontDescriptor')
                if desc is None and fo.get('/DescendantFonts'):
                    desc = fo['/DescendantFonts'][0].get_object().get('/FontDescriptor')
                desc = desc.get_object() if desc is not None else {}
                emb = any(k in desc for k in ('/FontFile', '/FontFile2', '/FontFile3'))
                fonts[name] = dict(subtype=str(fo.get('/Subtype')), embedded=emb)
        rep[fn] = fonts
    return rep


if __name__ == '__main__':
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        fn()
    fonts = font_report()
    json.dump(dict(manifest=MANIFEST, checks=CHECKS, fonts=fonts,
                   widths=dict(textwidth_cm=TEXTW_CM, columnwidth_cm=COLW_CM, fig7_width_cm=W7)),
              open(os.path.join(OUT, '_figure_meta.json'), 'w'), indent=1)
    n = sum(c[1] for c in CHECKS)
    print(f'value checks: {len(CHECKS)} groups, {n} values, all pass: {all(c[2] for c in CHECKS)}')
    print('all fonts embedded:', all(v['embedded'] for f in fonts.values() for v in f.values()))

#!/usr/bin/env python3
r"""Fill \res{<key>}{<format>} placeholders in icaart_main.tex from
outputs/sprint/results.json and write icaart_main_filled.tex.

Key syntax: dotted path into results.json, e.g.
    p2.cartpole.table3.C.delta_r2.point
'*' matches every key at that level (e.g. every model of a task); a trailing
'|op' aggregates the matched values: mean, sd, min, max, absmax, n.
Format: a Python format spec ('+.4f', '.2f', 'd'); 'd' casts to int.

Unresolved keys are left in place (they render in red in the draft) and are
listed on stdout; the exit status is 1 if any remain.

Usage: python paper/fill_paper.py [--results outputs/sprint/results.json]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r'\\res\{([^{}]+)\}\{([^{}]*)\}')


def walk(node, parts):
    if not parts:
        return [node]
    head, rest = parts[0], parts[1:]
    if head == '*':
        return [v for k in sorted(node) for v in walk(node[k], rest)] if isinstance(node, dict) else []
    if isinstance(node, dict) and head in node:
        return walk(node[head], rest)
    return []


def resolve(R, key):
    path, _, op = key.partition('|')
    vals = walk(R, path.split('.'))
    vals = [v for v in vals if isinstance(v, (int, float)) and v is not None]
    if not vals:
        return None
    if not op:
        return vals[0] if len(vals) == 1 else None
    a = np.asarray(vals, float)
    return {'mean': a.mean(), 'sd': a.std(ddof=1) if len(a) > 1 else 0.0,
            'min': a.min(), 'max': a.max(), 'absmax': np.abs(a).max(), 'n': len(a)}[op]


def fmt(v, spec):
    if spec == 'd':
        return str(int(round(v)))
    s = format(v, spec)
    return '\\ensuremath{-}' + s[1:] if s.startswith('-') else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, '..', 'outputs', 'sprint', 'results.json'))
    ap.add_argument('--tex', default=os.path.join(HERE, 'icaart_main.tex'))
    ap.add_argument('--out', default=os.path.join(HERE, 'icaart_main_filled.tex'))
    a = ap.parse_args()
    with open(a.results) as f:
        R = json.load(f)
    src = open(a.tex).read()
    missing = []

    def sub(m):
        v = resolve(R, m.group(1))
        if v is None:
            missing.append(m.group(1))
            return m.group(0)
        return fmt(v, m.group(2))

    out = '\n'.join(l if l.lstrip().startswith('%') else PAT.sub(sub, l) for l in src.split('\n'))
    with open(a.out, 'w') as f:
        f.write(out)
    print(f'wrote {a.out}; {sum(len(PAT.findall(l)) for l in src.split(chr(10)) if not l.lstrip().startswith('%')) - len(missing)} filled, {len(missing)} unresolved')
    for k in sorted(set(missing)):
        print('  unresolved:', k)
    sys.exit(1 if missing else 0)


if __name__ == '__main__':
    main()

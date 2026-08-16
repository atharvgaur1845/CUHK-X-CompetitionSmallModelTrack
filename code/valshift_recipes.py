#!/usr/bin/env python3
"""Recover the exact geometric-fusion recipe behind the champion (preboth045)
and its ResNet50 twin (r50both045), so the same recipe can be replayed in OOF space."""
import itertools, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, 'research', 'artifacts')
EPS = 1e-12


def L(n):
    d = np.load(os.path.join(A, n + '.npz'), allow_pickle=True)
    return np.asarray(d['sids']), np.asarray(d['probs'], np.float64)


BASE = 'testprobs_astgcn_world25_int8'
MEMBERS = ['testprobs_visual_mil_f0123', 'testprobs_pre_r18_f0123', 'testprobs_pre_r50_f0123',
           'testprobs_pre_depth_f0123', 'testprobs_pre_ir_f0123', 'testprobs_pre_thermal_f0123']
TARGETS = ['testprobs_w25preboth045', 'testprobs_r50both045', 'testprobs_w25vg035_f0123',
           'testprobs_w25pre0123_035', 'testprobs_r50tri045', 'testprobs_w25preboth055']

sb, PB = L(BASE)
mem = {}
for m in MEMBERS:
    s, p = L(m)
    assert list(s) == list(sb), m
    mem[m] = p


def geo_multi(parts, ws):
    lg = np.zeros_like(PB)
    for p, w in zip(parts, ws):
        lg += w * np.log(np.maximum(p, EPS))
    lg -= lg.max(1, keepdims=True); e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


for tname in TARGETS:
    ts, TP = L(tname)
    assert list(ts) == list(sb)
    best = None
    # try: base^(1-w) * (geometric mean of k members)^w
    for k in range(1, 4):
        for combo in itertools.combinations(MEMBERS, k):
            for w in np.arange(0.05, 0.86, 0.005):
                parts = [PB] + [mem[c] for c in combo]
                ws = [1 - w] + [w / k] * k
                F = geo_multi(parts, ws)
                d = np.abs(F - TP).max()
                if best is None or d < best[0]:
                    best = (d, combo, round(float(w), 3), 'equal')
    print('%-30s best-fit  w=%-6s members=%s   max|delta|=%.2e  argmax-agree=%.4f'
          % (tname, best[2], [c.replace('testprobs_', '') for c in best[1]], best[0],
             (geo_multi([PB] + [mem[c] for c in best[1]],
                        [1 - best[2]] + [best[2] / len(best[1])] * len(best[1])).argmax(1) == TP.argmax(1)).mean()))

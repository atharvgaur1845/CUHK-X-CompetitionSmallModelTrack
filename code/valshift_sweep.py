#!/usr/bin/env python3
"""Controlled test: reconstruct the skeleton+visual geometric-fusion weight sweep
in OOF space and compare its shape to the KNOWN public curve.

Public curve (LEADERBOARD.md, 4-fold visual member, all with trans05 decoding):
    w=0.225 -> 123/201   w=0.35 -> 125/201   w=0.45 -> 123/201   w=0.55 -> 118/201
"""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, 'research', 'artifacts')
EPS = 1e-12

tr = pd.read_csv(os.path.join(ROOT, 'cache', 'meta_train.csv'))
tr['day'] = pd.to_datetime(tr.t0, unit='s').dt.strftime('%m-%d')
tr['uid'] = tr.user.str.replace('user', '').astype(int)
_dom = tr.dropna(subset=['t0']).groupby('uid').day.agg(lambda s: s.value_counts().idxmax())
tr.loc[tr.day.isna(), 'day'] = tr.loc[tr.day.isna(), 'uid'].map(_dom)
sid2user = dict(zip(tr.sample_id, tr.uid)); sid2day = dict(zip(tr.sample_id, tr.day))

COHORT = {}
for u in [1, 2, 3, 4, 5]: COHORT[u] = 0
for u in [6, 7, 8, 9]: COHORT[u] = 1
for u in [16, 17, 18, 19, 20]: COHORT[u] = 2
for u in [21, 22, 23, 24]: COHORT[u] = 3
folds = json.load(open(os.path.join(A, 'cv_folds_all18.json')))['folds']
user2fold = {int(u.replace('user', '')): f['fold'] for f in folds for u in f['val_users']}


def load_cat(names):
    s, p, l = [], [], []
    for n in names:
        d = np.load(os.path.join(A, n + '.npz'), allow_pickle=True)
        s.append(d['sids']); p.append(d['probs']); l.append(d['labels'])
    s = np.concatenate(s); p = np.concatenate(p).astype(np.float64); l = np.concatenate(l)
    _, i = np.unique(s, return_index=True)
    return dict(zip(s[i], p[i])), dict(zip(s[i], l[i]))


def geo(base, vis, w):
    lg = (1 - w) * np.log(np.maximum(base, EPS)) + w * np.log(np.maximum(vis, EPS))
    lg -= lg.max(1, keepdims=True); e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


VIS = ['oof_visual_mil_v1_f%d' % i for i in range(4)]
SKL = ['oof_astgcn_world25']
vp, vl = load_cat(VIS)
sp, sl = load_cat(SKL)
common = sorted(set(vp) & set(sp))
print('visual OOF rows %d | skeleton OOF rows %d | intersection %d' % (len(vp), len(sp), len(common)))
V = np.stack([vp[s] for s in common]); S = np.stack([sp[s] for s in common])
Y = np.array([vl[s] for s in common])
assert (Y == np.array([sl[s] for s in common])).all(), 'label mismatch'
U = np.array([sid2user[s] for s in common])
C = np.array([COHORT[u] for u in U]); F = np.array([user2fold[u] for u in U])
D = np.array([sid2day[s] for s in common])

PUBLIC = {0.225: 123, 0.35: 125, 0.45: 123, 0.55: 118}


def cand_metrics(ok):
    """All candidate local metrics, computed from a boolean correctness vector."""
    peruser = {u: ok[U == u].mean() for u in np.unique(U)}
    perfold = {f: ok[F == f].mean() for f in np.unique(F)}
    percoh = {c: ok[C == c].mean() for c in np.unique(C)}
    vals = sorted(peruser.values())
    m = {}
    m['a_micro'] = ok.mean()
    m['b_macro_user'] = float(np.mean(list(peruser.values())))
    m['c_worst_fold'] = min(perfold.values())
    m['d_worst_user'] = vals[0]
    m['e_worst3_user'] = float(np.mean(vals[:3]))
    m['f_worst6_user'] = float(np.mean(vals[:6]))
    m['g_worst_cohort'] = min(percoh.values())
    m['h_cvar25_user'] = float(np.mean(vals[:max(1, len(vals) // 4)]))
    m['i_user_std'] = float(np.std(list(peruser.values())))
    m['j_macro_minus_std'] = m['b_macro_user'] - m['i_user_std']
    return m


rows = []
for w in [0.0, 0.10, 0.15, 0.20, 0.225, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85, 1.0]:
    P = geo(S, V, w)
    ok = (P.argmax(1) == Y).astype(float)
    m = cand_metrics(ok); m['w'] = w; m['public'] = PUBLIC.get(w)
    rows.append(m)
df = pd.DataFrame(rows)
keys = [c for c in df.columns if c not in ('w', 'public')]
pd.set_option('display.width', 250)
print()
print('=== OOF fusion-weight sweep (skeleton^(1-w) * visual^w), n=%d ===' % len(Y))
print(df[['w', 'public'] + keys].to_string(index=False, float_format=lambda x: '%.4f' % x))

sub = df[df.public.notna()]
print()
print('=== On the 4 weights with KNOWN public scores ===')
print('public argmax weight  : w=%.3f (%d clips)' % (sub.loc[sub.public.idxmax(), 'w'], sub.public.max()))


def spearman(a, b):
    a = pd.Series(a).rank(); b = pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1])


print()
print('%-18s %10s %10s  %s' % ('metric', 'argmax w', 'spearman', 'values at w=0.225/0.35/0.45/0.55'))
for k in keys:
    v = sub[k].values
    print('%-18s %10.3f %10.3f  %s' % (k, sub.loc[sub[k].idxmax(), 'w'], spearman(v, sub.public.values),
                                       ' '.join('%.4f' % x for x in v)))
print()
print('public                       -            -   %s' % ' '.join('%d' % x for x in sub.public.values))

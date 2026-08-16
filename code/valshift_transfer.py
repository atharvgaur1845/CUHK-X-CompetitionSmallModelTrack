#!/usr/bin/env python3
"""OOF vs TEST behaviour for every model with a KNOWN public solo score.

Tests whether an *unlabelled* test-side statistic (confidence after temperature
calibration on OOF) predicts the public score better than OOF accuracy does.
"""
import json, os
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, 'research', 'artifacts')
EPS = 1e-12

tr = pd.read_csv(os.path.join(ROOT, 'cache', 'meta_train.csv'))
tr['day'] = pd.to_datetime(tr.t0, unit='s').dt.strftime('%m-%d')
tr['uid'] = tr.user.str.replace('user', '').astype(int)
_dom = tr.dropna(subset=['t0']).groupby('uid').day.agg(lambda s: s.value_counts().idxmax())
tr.loc[tr.day.isna(), 'day'] = tr.loc[tr.day.isna(), 'uid'].map(_dom)
sid2user = dict(zip(tr.sample_id, tr.uid))

tc = pd.read_csv(os.path.join(A, 'test_cohorts.csv'))
sid2sess = dict(zip(tc.sm_id, tc.session_id))
sid2date = dict(zip(tc.sm_id, tc.date))
TRAIN_DATES = set('2025-' + d for d in tr.day.unique())
COHORT = {**{u: 0 for u in [1, 2, 3, 4, 5]}, **{u: 1 for u in [6, 7, 8, 9]},
          **{u: 2 for u in [16, 17, 18, 19, 20]}, **{u: 3 for u in [21, 22, 23, 24]}}
folds = json.load(open(os.path.join(A, 'cv_folds_all18.json')))['folds']
user2fold = {int(u.replace('user', '')): f['fold'] for f in folds for u in f['val_users']}

MODELS = [
    # label, oof npz list, test npz, public
    ('visual_mil scratch 4fold', ['oof_visual_mil_v1_f%d' % i for i in range(4)],
     'testprobs_visual_mil_f0123', 0.38805),
    ('pre_r18 imagenet 4fold', ['oof_pre_r18_bn_f0', 'oof_pre_r18_bn_f1', 'oof_pre_r18_f2_bn', 'oof_pre_r18_bn_f3'],
     'testprobs_pre_r18_f0123', 0.31343),
    ('astgcn_world25 skeleton', ['oof_astgcn_world25'], 'testprobs_astgcn_world25', 0.53731),
    ('pre_r18 imagenet fold2', ['oof_pre_r18_f2_bn'], 'testprobs_pre_r18_f2_bn', 0.24378),
    ('v2_ssl fold2', ['oof_v2_ssl_f2_f2'], 'testprobs_v2ssl_f2', 0.28358),
    # no public solo score, listed for reference
    ('pre_r50 imagenet 4fold', ['oof_pre_r50_f%d' % i for i in range(4)], 'testprobs_pre_r50_f0123', None),
    ('v2_scratch fold2', ['oof_v2_scratch_f2_f2'], 'testprobs_v2scratch_f2', None),
]


def load_cat(names):
    s, p, l = [], [], []
    for n in names:
        d = np.load(os.path.join(A, n + '.npz'), allow_pickle=True)
        s.append(d['sids']); p.append(np.asarray(d['probs'], np.float64)); l.append(d['labels'])
    s = np.concatenate(s); p = np.concatenate(p); l = np.concatenate(l)
    _, i = np.unique(s, return_index=True)
    return s[i], p[i], l[i]


def temp_apply(P, T):
    lg = np.log(np.maximum(P, EPS)) / T
    lg -= lg.max(1, keepdims=True); e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def fit_T(P, Y):
    """Temperature that makes mean max-prob match accuracy on OOF (calibration)."""
    acc = (P.argmax(1) == Y).mean()
    f = lambda t: (temp_apply(P, t).max(1).mean() - acc) ** 2
    r = minimize_scalar(f, bounds=(0.05, 25.0), method='bounded')
    return r.x


rows = []
for label, oofs, tname, pub in MODELS:
    try:
        s, P, Y = load_cat(oofs)
    except Exception as e:
        print('SKIP', label, e); continue
    td = np.load(os.path.join(A, tname + '.npz'), allow_pickle=True)
    TS = np.asarray(td['sids']); TP = np.asarray(td['probs'], np.float64)
    oof_acc = (P.argmax(1) == Y).mean()
    T = fit_T(P, Y)
    Pc, TPc = temp_apply(P, T), temp_apply(TP, T)
    conf_oof, conf_test = Pc.max(1).mean(), TPc.max(1).mean()
    # per test session group
    dates = np.array([sid2date[x] for x in TS])
    seen = np.array([d in TRAIN_DATES for d in dates])
    u = np.array([sid2user[x] for x in s])
    peruser = {int(x): (P.argmax(1) == Y)[u == x].mean() for x in np.unique(u)}
    rows.append(dict(
        model=label, public=pub, n_oof=len(Y), oof_acc=oof_acc, T=T,
        conf_oof=conf_oof, conf_test=conf_test,
        shift=conf_test - oof_acc,
        conf_test_seenday=TPc.max(1)[seen].mean(), conf_test_newday=TPc.max(1)[~seen].mean(),
        worst_user=min(peruser.values()),
        macro_user=float(np.mean(list(peruser.values()))),
    ))

df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print('=== OOF accuracy vs OOF-calibrated TEST confidence ===')
print('(T = temperature that equates OOF mean-max-prob with OOF accuracy;')
print(' conf_test = mean max-prob on the 405 test clips under that same T)')
print()
cols = ['model', 'public', 'n_oof', 'oof_acc', 'T', 'conf_oof', 'conf_test', 'conf_test_seenday', 'conf_test_newday']
print(df[cols].to_string(index=False, float_format=lambda x: '%.4f' % x))

k = df[df.public.notna()].copy()


def sp(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


print()
print('=== Correlation with public score (n=%d models with known solo public) ===' % len(k))
for m in ['oof_acc', 'macro_user', 'worst_user', 'conf_test', 'conf_oof']:
    print('  %-14s  pearson %+.3f   spearman %+.3f' % (m, np.corrcoef(k[m], k.public)[0, 1], sp(k[m], k.public)))
print()
print('=== |predicted - actual| public accuracy, using conf_test as the estimate ===')
for _, r in k.iterrows():
    print('  %-26s public %.4f | oof_acc %.4f (err %+.4f) | conf_test %.4f (err %+.4f)'
          % (r.model, r.public, r.oof_acc, r.oof_acc - r.public, r.conf_test, r.conf_test - r.public))
print('  MAE  oof_acc   = %.4f' % np.abs(k.oof_acc - k.public).mean())
print('  MAE  conf_test = %.4f' % np.abs(k.conf_test - k.public).mean())

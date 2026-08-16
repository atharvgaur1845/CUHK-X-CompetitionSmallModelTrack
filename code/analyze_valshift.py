#!/usr/bin/env python3
"""Diagnose OOF->public inversion. Read-only analysis; writes nothing outside /tmp."""
import json, os, sys, glob
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, 'research', 'artifacts')

tr = pd.read_csv(os.path.join(ROOT, 'cache', 'meta_train.csv'))
te = pd.read_csv(os.path.join(ROOT, 'cache', 'meta_test.csv'))
tr['day'] = pd.to_datetime(tr.t0, unit='s').dt.strftime('%m-%d')
te['day'] = pd.to_datetime(te.t0, unit='s').dt.strftime('%m-%d')
tr['uid'] = tr.user.str.replace('user', '').astype(int)
# 2 clips have no frames -> NaN t0; assign them their user's dominant day
_dom = tr.dropna(subset=['t0']).groupby('uid').day.agg(lambda s: s.value_counts().idxmax())
tr.loc[tr.day.isna(), 'day'] = tr.loc[tr.day.isna(), 'uid'].map(_dom)

# --- cohort = connected component of users sharing a recording day ---
days = sorted(tr.day.unique())
users = sorted(tr.uid.unique())
# dominant day per user
dom = tr.groupby('uid').day.agg(lambda s: s.value_counts().idxmax())
# union-find over (user, day) with >=10% of that user's clips
share = {}
for u in users:
    sub = tr[tr.uid == u]
    for d, c in sub.day.value_counts().items():
        if c >= 0.10 * len(sub):
            share.setdefault(d, set()).add(u)
parent = {u: u for u in users}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for d, us in share.items():
    us = sorted(us)
    for u in us[1:]: union(us[0], u)
comp = {}
for u in users: comp.setdefault(find(u), []).append(u)
cohorts = {}
for i, (r, mem) in enumerate(sorted(comp.items(), key=lambda kv: min(kv[1]))):
    for u in mem: cohorts[u] = i
COHORT_NAME = {}
for i in sorted(set(cohorts.values())):
    mem = sorted([u for u in users if cohorts[u] == i])
    dd = sorted(tr[tr.uid.isin(mem)].day.unique())
    COHORT_NAME[i] = '%s..%s' % (dd[0], dd[-1])

sid2user = dict(zip(tr.sample_id, tr.uid))
sid2day = dict(zip(tr.sample_id, tr.day))

folds = json.load(open(os.path.join(A, 'cv_folds_all18.json')))['folds']
user2fold = {}
for f in folds:
    for u in f['val_users']:
        user2fold[int(u.replace('user', ''))] = f['fold']


def report_structure():
    print('=' * 78)
    print('SECTION 1  RECORDING-SESSION STRUCTURE')
    print('=' * 78)
    print('\nUser day-cohorts (connected components over shared recording days):')
    for i in sorted(set(cohorts.values())):
        mem = sorted([u for u in users if cohorts[u] == i])
        print('  cohort %d  days %-14s  users %s' % (i, COHORT_NAME[i], mem))
    print('\nTest clips by day (test has NO user/station column):')
    trdays = set(tr.day.unique())
    tot_seen = tot_unseen = 0
    for d, c in te.day.value_counts().sort_index().items():
        cu = sorted(tr[tr.day == d].uid.unique())
        seen = d in trdays
        tot_seen += c if seen else 0
        tot_unseen += 0 if seen else c
        print('  %s  n=%3d  train users that day: %s' % (d, c, cu if cu else 'NONE (unseen day)'))
    print('  -> test clips on days present in train : %d (%.1f%%)' % (tot_seen, 100 * tot_seen / len(te)))
    print('  -> test clips on days ABSENT from train: %d (%.1f%%)' % (tot_unseen, 100 * tot_unseen / len(te)))

    print('\nStation x user: is station confounded with user?')
    ct = pd.crosstab(tr.uid, tr.station)
    print('  every user appears at every station: %s (min cell = %d)' % ((ct.values > 0).all(), ct.values.min()))
    print('  -> leave-one-station-out is IMPOSSIBLE as a subject-shift proxy: station is fully crossed with user.')

    print('\nDoes the 4-fold CV leak cohort-mates into training?')
    for f in folds:
        vu = [int(u.replace('user', '')) for u in f['val_users']]
        tu = [int(u.replace('user', '')) for u in f['train_users']]
        tc = set(cohorts[u] for u in tu)
        rows = []
        for u in vu:
            mates = [x for x in tu if cohorts[x] == cohorts[u]]
            rows.append('u%d(c%d,%d mates)' % (u, cohorts[u], len(mates)))
        print('  fold %d val: %s' % (f['fold'], '  '.join(rows)))
    print('  -> EVERY held-out user has same-session cohort-mates in train, in every fold.')
    return


# ------------------------------------------------------------------ OOF models
def load_oof(names):
    """Concatenate a set of per-fold npz files into sid->pred/label arrays."""
    sids, probs, labs = [], [], []
    for n in names:
        p = os.path.join(A, n + '.npz')
        if not os.path.exists(p): return None
        d = np.load(p, allow_pickle=True)
        sids.append(d['sids']); probs.append(d['probs']); labs.append(d['labels'])
    sids = np.concatenate(sids); probs = np.concatenate(probs); labs = np.concatenate(labs)
    _, idx = np.unique(sids, return_index=True)
    return sids[idx], probs[idx], labs[idx]


MODELS = {
    # name : (fold npz list, public solo score, note)
    'visual_mil_v1 (scratch, 4fold)': (
        ['oof_visual_mil_v1_f0', 'oof_visual_mil_v1_f1', 'oof_visual_mil_v1_f2', 'oof_visual_mil_v1_f3'],
        0.38805, 'sub_visual_mil_v1.csv'),
    'pre_r18 imagenet (4fold)': (
        ['oof_pre_r18_bn_f0', 'oof_pre_r18_bn_f1', 'oof_pre_r18_f2_bn', 'oof_pre_r18_bn_f3'],
        0.31343, 'sub_pre_r18_f0123_solo.csv'),
    'astgcn_world25 skeleton': (['oof_astgcn_world25'], 0.53731, 'sub_astgcn_world25.csv'),
}
FOLD2_MODELS = {
    'v2_ssl fold2': (['oof_v2_ssl_f2_f2'], 0.28358, 'sub_v2ssl_f2_solo.csv'),
    'pre_r18 fold2': (['oof_pre_r18_f2_bn'], 0.24378, 'sub_pre_r18_f2_solo.csv'),
}
NOPUB = {
    'pre_r50 imagenet (4fold)': (['oof_pre_r50_f0', 'oof_pre_r50_f1', 'oof_pre_r50_f2', 'oof_pre_r50_f3'], None, 'fused 128 vs r18-fused 131'),
}


def metrics_for(sids, probs, labs):
    pred = probs.argmax(1)
    ok = (pred == labs).astype(float)
    u = np.array([sid2user[s] for s in sids])
    d = np.array([sid2day[s] for s in sids])
    fo = np.array([user2fold[x] for x in u])
    co = np.array([cohorts[x] for x in u])
    m = {}
    m['micro'] = ok.mean()
    m['macro_user'] = np.mean([ok[u == x].mean() for x in np.unique(u)])
    perfold = [ok[fo == k].mean() for k in np.unique(fo)]
    m['mean_fold'] = float(np.mean(perfold))
    m['worst_fold'] = float(np.min(perfold))
    peruser = {int(x): float(ok[u == x].mean()) for x in np.unique(u)}
    m['worst_user'] = float(min(peruser.values()))
    m['p10_user'] = float(np.percentile(list(peruser.values()), 10))
    m['worst3_user'] = float(np.mean(sorted(peruser.values())[:3]))
    percohort = {int(c): float(ok[co == c].mean()) for c in np.unique(co)}
    m['worst_cohort'] = float(min(percohort.values()))
    # late cohorts only (test days sit in / after cohorts 1 and 3)
    m['_peruser'] = peruser
    m['_percohort'] = percohort
    m['_perday'] = {dd: float(ok[d == dd].mean()) for dd in np.unique(d)}
    m['_n'] = len(ok)
    return m


def spearman(a, b):
    ra = pd.Series(a).rank().values; rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def report_models():
    print()
    print('=' * 78)
    print('SECTION 2  LOCAL METRICS vs KNOWN PUBLIC SOLO SCORES')
    print('=' * 78)
    rows = []
    for name, (fl, pub, note) in list(MODELS.items()) + list(NOPUB.items()):
        got = load_oof(fl)
        if got is None:
            print('  MISSING artifacts for', name); continue
        m = metrics_for(*got)
        m['name'] = name; m['public'] = pub; m['note'] = note
        rows.append(m)
    hdr = ['micro', 'macro_user', 'mean_fold', 'worst_fold', 'worst_user', 'worst3_user', 'worst_cohort']
    print('\n%-32s %7s | %s' % ('model', 'public', ' '.join('%10s' % h for h in hdr)))
    for m in rows:
        print('%-32s %7s | %s' % (m['name'][:32], ('%.5f' % m['public']) if m['public'] else '   n/a',
                                  ' '.join('%10.4f' % m[h] for h in hdr)))
    return rows


if __name__ == '__main__':
    report_structure()
    rows = report_models()
    np.save('/tmp/valshift_rows.npy', np.array(rows, dtype=object), allow_pickle=True)

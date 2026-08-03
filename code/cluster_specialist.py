#!/usr/bin/env python3
"""Confusion-cluster specialists for the all-18 fold-2 outer screen (X-02).

Motivation
----------
EXP-050b left 281 baseline errors on all-18 fold 2, and they are not spread
evenly: they concentrate inside a few tight groups of object classes that share
almost identical skeleton kinematics (table/kitchen actions, seated
screen/paper actions).  62 clips hold the truth at base rank 2.  A head that
only has to separate members of one such group solves a far easier problem
than the 40-way model.

Leakage discipline
------------------
The confusion structure observed while analysing fold 2 must NOT define the
clusters, because fold 2 is the scored split.  Clusters here are derived from
an inner cross-validation over the 14 fold-2 *training* users only.  Fold-2
validation labels are read exactly once, at final scoring.

Stages
------
``inner-oof``  4 inner folds over the 14 training users -> inner OOF.
``clusters``   derive confusion clusters from that inner OOF.

Usage
-----
    CUHKX_FOLD_FILE=research/artifacts/cv_folds_all18.json \
    python3 code/cluster_specialist.py inner-oof --fold 2
    python3 code/cluster_specialist.py clusters --fold 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import har_data
import har_models

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS = os.path.join(ROOT, "research", "artifacts")
N_CLASSES = 40
N_INNER = 4

# Fixed recipe, identical to EXP-040 / the matched baseline.
EPOCHS = 100
BATCH = 64
LR = 1e-3
LABEL_SMOOTHING = 0.1
WIDTH = 64
AUG = "trunc"
SEED = 0

# Predeclared cluster-derivation rule (fixed before any specialist is trained).
MIN_PAIR_ERRORS = 4     # symmetric confusions needed to link two classes
MIN_CLASS_SUPPORT = 10  # inner clips a class needs to take part
MAX_CLUSTER = 8         # cap so a component cannot swallow the label space


def base_tag(seed: int) -> str:
    """DA-006-A stored seed 0 without a suffix and seeds 1-3 as ``_s{n}``."""
    return "" if seed == 0 else f"_s{seed}"


def user_of(sample_id: str) -> str:
    match = re.search(r"user\d+", sample_id)
    if not match:
        raise ValueError(f"cannot parse user from {sample_id}")
    return match.group(0)


def inner_user_folds(users: list[str]) -> list[list[str]]:
    """Deterministic round-robin split of the outer-train users."""
    ordered = sorted(users, key=lambda u: int(u.replace("user", "")))
    return [ordered[i::N_INNER] for i in range(N_INNER)]


def make_dataset(fold: int, augment: bool, keep_users: set[str] | None):
    dataset = har_data.HARDataset(
        "train", "skelg", fold, "train",
        aug=(AUG if augment else False), seed=SEED,
        feat="jv", n_frames=8, person="first", imu_feat="raw",
        t_skel=32, skel_time="stretch",
    )
    if keep_users is not None:
        dataset.ids = [s for s in dataset.ids if user_of(s) in keep_users]
    return dataset


def train_one(train_set, val_set, device, n_cls=N_CLASSES, label_map=None):
    """Train the fixed recipe; return best-epoch val probabilities."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    loader = DataLoader(
        train_set, BATCH, shuffle=True, num_workers=4, drop_last=True,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_set, BATCH, shuffle=False, num_workers=2,
        pin_memory=device == "cuda",
    )
    model = har_models.build(
        "skelg", arch="adaptive", width=WIDTH, n_cls=n_cls
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, LR, epochs=EPOCHS, steps_per_epoch=len(loader),
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_accuracy, best_probs = -1.0, None
    for epoch in range(EPOCHS):
        model.train()
        for x, y, _ in loader:
            if label_map is not None:
                y = torch.as_tensor([label_map[int(v)] for v in y])
            loss = criterion(model(x.to(device)), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                logits = model(x.to(device))
                probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
                if label_map is not None:
                    y = torch.as_tensor([label_map[int(v)] for v in y])
                labels.append(y.numpy())
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        accuracy = float((probs.argmax(1) == labels).mean())
        if accuracy > best_accuracy:
            best_accuracy, best_probs = accuracy, probs
        if (epoch + 1) % 25 == 0:
            print(
                f"    ep{epoch + 1}: val {accuracy:.4f} (best {best_accuracy:.4f})",
                flush=True,
            )
    return best_accuracy, best_probs


def inner_oof_command(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = make_dataset(args.fold, False, None)
    users = sorted({user_of(s) for s in probe.ids})
    print(f"outer-train users ({len(users)}): {users}", flush=True)
    groups = inner_user_folds(users)

    all_probs, all_labels, all_ids = [], [], []
    for index, val_users in enumerate(groups):
        train_users = {u for u in users if u not in set(val_users)}
        train_set = make_dataset(args.fold, True, train_users)
        val_set = make_dataset(args.fold, False, set(val_users))
        print(
            f"  inner fold {index}: val={val_users} "
            f"({len(val_set.ids)} clips), train={len(train_set.ids)} clips",
            flush=True,
        )
        accuracy, probs = train_one(train_set, val_set, device)
        print(f"  inner fold {index}: best {accuracy:.4f}", flush=True)
        all_probs.append(probs)
        all_labels.append(
            np.array([val_set.labels[s] for s in val_set.ids], dtype=np.int64)
        )
        all_ids.extend(val_set.ids)

    probs = np.concatenate(all_probs).astype(np.float32)
    labels = np.concatenate(all_labels)
    output = os.path.join(ARTIFACTS, f"inner_oof_astgcn_f{args.fold}.npz")
    np.savez_compressed(
        output, probs=probs, labels=labels,
        sids=np.array(all_ids, dtype="<U15"),
    )
    print(
        f"inner OOF: {len(labels)} clips, micro "
        f"{(probs.argmax(1) == labels).mean():.4f} -> {output}",
        flush=True,
    )


def clusters_command(args) -> None:
    data = np.load(
        os.path.join(ARTIFACTS, f"inner_oof_astgcn_f{args.fold}.npz"),
        allow_pickle=True,
    )
    probs, labels = data["probs"], data["labels"]
    predictions = probs.argmax(1)
    support = Counter(labels.tolist())

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for truth, predicted in zip(labels.tolist(), predictions.tolist()):
        if truth != predicted:
            confusion[truth, predicted] += 1

    symmetric = confusion + confusion.T
    edges = [
        (i, j, int(symmetric[i, j]))
        for i in range(N_CLASSES)
        for j in range(i + 1, N_CLASSES)
        if symmetric[i, j] >= MIN_PAIR_ERRORS
        and support[i] >= MIN_CLASS_SUPPORT
        and support[j] >= MIN_CLASS_SUPPORT
    ]
    edges.sort(key=lambda e: -e[2])

    parent = list(range(N_CLASSES))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    members: dict[int, set[int]] = {c: {c} for c in range(N_CLASSES)}
    for i, j, _ in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if len(members[ri]) + len(members[rj]) > MAX_CLUSTER:
            continue
        parent[rj] = ri
        members[ri] |= members[rj]

    clusters = sorted(
        (sorted(group) for root, group in members.items()
         if find(root) == root and len(group) >= 2),
        key=lambda g: -len(g),
    )

    names = {}
    mapping = os.path.join(ROOT, "Small-Model-Track", "Training",
                           "class_mapping.csv")
    if os.path.isfile(mapping):
        import csv
        with open(mapping) as handle:
            for row in csv.DictReader(handle):
                values = list(row.values())
                names[int(values[0])] = values[1]

    covered = sum(
        1 for t, p in zip(labels.tolist(), predictions.tolist())
        if t != p and any(t in c and p in c for c in clusters)
    )
    errors = int((predictions != labels).sum())
    print(f"inner OOF errors: {errors}")
    print(
        f"errors inside a derived cluster: {covered} "
        f"({covered / max(1, errors):.1%})"
    )
    for group in clusters:
        print(f"  cluster ({len(group)}): {group}")
        for c in group:
            print(f"      {c:2d} {names.get(c, '?')}  n={support[c]}")

    output = os.path.join(ARTIFACTS, f"clusters_f{args.fold}.json")
    with open(output, "w") as handle:
        json.dump(
            {
                "fold": args.fold,
                "source": f"inner_oof_astgcn_f{args.fold}.npz",
                "rule": {
                    "min_pair_errors": MIN_PAIR_ERRORS,
                    "min_class_support": MIN_CLASS_SUPPORT,
                    "max_cluster": MAX_CLUSTER,
                },
                "inner_errors": errors,
                "errors_in_cluster": covered,
                "clusters": clusters,
            },
            handle,
            indent=2,
        )
    print(f"wrote {output}")


def train_specialist(train_set, device, n_cls, label_map):
    """Fixed recipe, final epoch, no checkpoint selection of any kind."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    batch = min(BATCH, max(8, len(train_set.ids) // 4))
    loader = DataLoader(
        train_set, batch, shuffle=True, num_workers=4, drop_last=False,
        pin_memory=device == "cuda",
    )
    model = har_models.build(
        "skelg", arch="adaptive", width=WIDTH, n_cls=n_cls
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, LR, epochs=EPOCHS, steps_per_epoch=len(loader),
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    for _ in range(EPOCHS):
        model.train()
        for x, y, _ in loader:
            y = torch.as_tensor([label_map[int(v)] for v in y])
            loss = criterion(model(x.to(device)), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
    model.eval()
    return model


def predict(model, dataset, device, batch=64):
    loader = DataLoader(dataset, batch, shuffle=False, num_workers=2,
                        pin_memory=device == "cuda")
    out = []
    with torch.no_grad():
        for x, _, _ in loader:
            out.append(torch.softmax(model(x.to(device)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def specialists_command(args) -> None:
    """Train one head per cluster and score fold 2 exactly once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(ARTIFACTS, f"clusters_f{args.fold}.json")) as handle:
        clusters = json.load(handle)["clusters"]

    base = np.load(
        os.path.join(
            ARTIFACTS,
            f"oof_astgcn_all18_f{args.fold}{base_tag(args.seed)}_best.npz",
        ),
        allow_pickle=True,
    )
    base_probs, labels = base["probs"].astype(np.float64), base["labels"]
    base_sids = base["sids"].tolist()

    val_set = har_data.HARDataset(
        "train", "skelg", args.fold, "val", aug=False, feat="jv",
        n_frames=8, person="first", imu_feat="raw", t_skel=32,
        skel_time="stretch",
    )
    if val_set.ids != base_sids:
        raise RuntimeError("validation row order differs from baseline OOF")

    train_all = make_dataset(args.fold, True, None)
    updated = base_probs.copy()
    report = []
    for cluster in clusters:
        members = sorted(cluster)
        label_map = {c: i for i, c in enumerate(members)}
        subset = make_dataset(args.fold, True, None)
        subset.ids = [s for s in train_all.ids
                      if subset.labels[s] in label_map]
        print(
            f"  cluster {members}: {len(subset.ids)} train clips",
            flush=True,
        )
        model = train_specialist(subset, device, len(members), label_map)
        spec = predict(model, val_set, device).astype(np.float64)

        # Predeclared fixed combination: equal-weight geometric mean of the
        # renormalized base posterior and the specialist, applied only where
        # the base top-1 already lies in this cluster. No tuning on fold 2.
        selected = np.isin(base_probs.argmax(1), members)
        restricted = base_probs[:, members]
        restricted = restricted / np.maximum(restricted.sum(1, keepdims=True), 1e-12)
        combined = np.sqrt(restricted * spec)
        winners = np.array(members)[combined.argmax(1)]
        changed = selected & (winners != base_probs.argmax(1))
        for row in np.where(selected)[0]:
            updated[row] = 0.0
            updated[row, winners[row]] = 1.0
        report.append((members, int(selected.sum()), int(changed.sum())))

    base_predictions = base_probs.argmax(1)
    new_predictions = updated.argmax(1)
    base_correct = base_predictions == labels
    new_correct = new_predictions == labels
    rescues = int((~base_correct & new_correct).sum())
    harms = int((base_correct & ~new_correct).sum())
    changed = int((base_predictions != new_predictions).sum())

    base_accuracy = float(base_correct.mean())
    new_accuracy = float(new_correct.mean())
    delta = new_accuracy - base_accuracy
    print()
    print(f"fold {args.fold} seed {args.seed} baseline   {base_accuracy:.4f}")
    print(f"fold {args.fold} seed {args.seed} specialist {new_accuracy:.4f} "
          f"(delta {delta:+.4f})")
    print(f"changed {changed} · rescues {rescues} · harms {harms} "
          f"· net {rescues - harms:+d} clips")
    print(f"predeclared gate: >= 0.5890 -> "
          f"{'PASS' if new_accuracy >= 0.5890 else 'FAIL'}")
    for members, n_sel, n_chg in report:
        print(f"  cluster {members}: routed {n_sel}, changed {n_chg}")

    # Paired result line, parsed by the EXP-054 aggregator.
    print(
        f"PAIRED_RESULT seed={args.seed} base={base_accuracy:.6f} "
        f"new={new_accuracy:.6f} delta={delta:+.6f} "
        f"rescues={rescues} harms={harms} changed={changed}",
        flush=True,
    )
    np.savez_compressed(
        os.path.join(
            ARTIFACTS,
            f"oof_cluster_specialist_f{args.fold}{base_tag(args.seed)}.npz",
        ),
        probs=updated.astype(np.float32), labels=labels,
        sids=np.array(base_sids, dtype="<U15"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    spec = commands.add_parser("specialists")
    spec.add_argument("--fold", type=int, default=2)
    spec.add_argument(
        "--seed", type=int, default=0,
        help="specialist training seed; also selects the matching DA-006-A "
             "base baseline so the delta stays paired",
    )
    spec.set_defaults(function=specialists_command)
    inner = commands.add_parser("inner-oof")
    inner.add_argument("--fold", type=int, default=2)
    inner.set_defaults(function=inner_oof_command)
    clusters = commands.add_parser("clusters")
    clusters.add_argument("--fold", type=int, default=2)
    clusters.set_defaults(function=clusters_command)
    args = parser.parse_args()
    if getattr(args, "seed", None) is not None:
        global SEED
        SEED = args.seed
    args.function(args)


if __name__ == "__main__":
    main()

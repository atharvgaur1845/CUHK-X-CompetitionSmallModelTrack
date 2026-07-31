#!/usr/bin/env python3
"""Assemble and validate upload-ready submission candidates from saved probabilities.

Profiles deliberately isolate one change at a time from the screenshot-verified
block8 stack:
  block8_g40       only raises the GCN-family weight from 0.30 to 0.40
  dyn_g40_clean    also appends the corrected dynamic-truncation TCN member
  dyn_g40_tta      uses held-out-gated temporal TTA for that new member only
  block8_g50_i175  weight control for the multi-stream candidate
  multitcn_g50     adds 7.5% of the joint/bone/motion multi-stream TCN
  astgcn_a20       adds 20% adaptive graph weight inside the skeleton block
  astgcn_world25   retains astgcn_a20 and adds 25% world-frame IMU
"""
import argparse
import csv
import hashlib
import json
import os

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
SUB = os.path.join(ROOT, "submissions")
TCN5 = (
    "skel_jvb_big",
    "skel_jvb_big_s1",
    "skel_jvb_big_s2",
    "skel_jvb_big_s3",
    "skel_jvb_big_s4",
)
GCN3 = ("stgcn_v1", "stgcn_s1", "stgcn_w96")
DYNAMIC_TAG = "skel_dynaug"
PASSES = 3


def load_npz(path, key="probs"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    labels = z["labels"] if "labels" in z.files else None
    return z[key], labels, list(z["sids"])


def load_standard(tag, split):
    prefix = "oof" if split == "train" else "testprobs"
    return load_npz(os.path.join(ART, f"{prefix}_{tag}.npz"))


def load_dynamic(split, use_tta):
    prefix = "oof" if split == "train" else "testprobs"
    path = os.path.join(ART, f"{prefix}_tjitter_v2_p{PASSES}_{DYNAMIC_TAG}.npz")
    return load_npz(path, key="probs" if use_tta else "clean_probs")


def load_probe(tag, split, use_tta=False):
    prefix = "oof" if split == "train" else "testprobs"
    path = os.path.join(ART, f"{prefix}_tjitter_v2_p{PASSES}_{tag}.npz")
    return load_npz(path, key="probs" if use_tta else "clean_probs")


def align(probs, sids, ref):
    order = {sid: i for i, sid in enumerate(sids)}
    if any(sid not in order for sid in ref):
        missing = [sid for sid in ref if sid not in order]
        raise ValueError(f"{len(missing)} samples missing; first={missing[0]}")
    return probs[[order[sid] for sid in ref]]


def block(split, gcn_weight, dynamic=None, multitcn_weight=0.0, imu_weight=0.2,
          adaptive_weight=0.0, world_weight=0.0):
    first, labels, ref = load_standard(TCN5[0], split)
    tcn = first.astype(np.float64)
    for tag in TCN5[1:]:
        p, _, s = load_standard(tag, split)
        tcn += align(p, s, ref)
    n_tcn = len(TCN5)
    if dynamic is not None:
        p, _, s = load_dynamic(split, use_tta=dynamic == "tta")
        tcn += align(p, s, ref)
        n_tcn += 1
    tcn /= n_tcn

    gcn = np.zeros_like(tcn)
    for tag in GCN3:
        p, _, s = load_standard(tag, split)
        gcn += align(p, s, ref)
    gcn /= len(GCN3)
    imu, _, imu_sids = load_standard("imu_inv4", split)
    imu = align(imu, imu_sids, ref)
    skeleton = (1.0 - gcn_weight) * tcn + gcn_weight * gcn
    if multitcn_weight:
        p, _, s = load_probe("skel_multitcn", split)
        multi = align(p, s, ref)
        skeleton = (1.0 - multitcn_weight) * skeleton + multitcn_weight * multi
    if adaptive_weight:
        p, _, s = load_probe("astgcn_v1", split)
        adaptive = align(p, s, ref)
        skeleton = (1.0 - adaptive_weight) * skeleton + adaptive_weight * adaptive
    fused = (1.0 - imu_weight) * skeleton + imu_weight * imu
    if world_weight:
        prefix = "oof" if split == "train" else "testprobs"
        world, _, world_sids = load_npz(
            os.path.join(ART, f"{prefix}_imu_world.npz")
        )
        world = align(world, world_sids, ref)
        fused = (1.0 - world_weight) * fused + world_weight * world
    return fused, labels, ref


def fold_masks(ref):
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as fh:
        user_of = {r["sample_id"]: r["user"] for r in csv.DictReader(fh)}
    with open(os.path.join(ART, "cv_folds.json")) as fh:
        folds = json.load(fh)["folds"]
    return [
        np.asarray([user_of[sid] in set(fold["val_users"]) for sid in ref])
        for fold in folds
    ]


def expected_test_ids():
    with open(os.path.join(ROOT, "cache", "meta_test.csv")) as fh:
        return sorted(r["sample_id"] for r in csv.DictReader(fh))


def write_submission(name, probs, sids):
    expected = expected_test_ids()
    if sorted(sids) != expected or len(sids) != 405 or len(set(sids)) != 405:
        raise ValueError("test IDs are incomplete, duplicated, or do not match meta_test.csv")
    pred = probs.argmax(1)
    if pred.min() < 0 or pred.max() > 39:
        raise ValueError("prediction outside [0, 39]")
    os.makedirs(SUB, exist_ok=True)
    path = os.path.join(SUB, f"sub_{name}.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("path", "prediction"))
        for sid, label in zip(sids, pred):
            writer.writerow((f"small_model_track_test/{sid}/", int(label)))
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return path, pred, digest


def known_block8_predictions():
    path = os.path.join(SUB, "sub_block8_gcnsoup.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return np.asarray([int(row["prediction"]) for row in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profiles",
        default=("block8_g40,dyn_g40_clean,dyn_g40_tta,"
                 "block8_g50_i175,multitcn_g50,astgcn_a20,"
                 "astgcn_world25"),
        help=("comma list: block8_g40,dyn_g40_clean,dyn_g40_tta,"
              "block8_g50_i175,multitcn_g50,astgcn_a20,"
                 "astgcn_world25"),
    )
    ap.add_argument(
        "--write-probabilities",
        action="store_true",
        help=(
            "persist each assembled profile's OOF/test probability arrays "
            "under research/artifacts for downstream structured decoding"
        ),
    )
    args = ap.parse_args()
    specs = {
        "block8_g40": (0.40, None, 0.0, 0.20, 0.0, 0.0),
        "dyn_g40_clean": (0.40, "clean", 0.0, 0.20, 0.0, 0.0),
        "dyn_g40_tta": (0.40, "tta", 0.0, 0.20, 0.0, 0.0),
        "block8_g50_i175": (0.50, None, 0.0, 0.175, 0.0, 0.0),
        "multitcn_g50": (0.50, None, 0.075, 0.175, 0.0, 0.0),
        "astgcn_a20": (0.50, None, 0.075, 0.175, 0.20, 0.0),
        "astgcn_world25": (0.50, None, 0.075, 0.175, 0.20, 0.25),
    }

    baseline_oof, labels, train_sids = block("train", 0.30)
    baseline_test, _, test_sids = block("test", 0.30)
    known = known_block8_predictions()
    if not np.array_equal(baseline_test.argmax(1), known):
        raise RuntimeError("block8 reconstruction does not match the verified CSV")
    masks = fold_masks(train_sids)
    base_pred = baseline_oof.argmax(1)
    print(
        f"verified block8 reconstruction: OOF={(base_pred == labels).mean():.5f}; "
        "CSV mismatch=0",
        flush=True,
    )

    for name in args.profiles.split(","):
        name = name.strip()
        if name not in specs:
            raise KeyError(f"unknown profile {name!r}")
        (
            gcn_weight,
            dynamic,
            multitcn_weight,
            imu_weight,
            adaptive_weight,
            world_weight,
        ) = specs[name]
        oof, candidate_labels, candidate_train_sids = block(
            "train", gcn_weight, dynamic, multitcn_weight, imu_weight,
            adaptive_weight, world_weight)
        test, _, candidate_test_sids = block(
            "test", gcn_weight, dynamic, multitcn_weight, imu_weight,
            adaptive_weight, world_weight)
        if candidate_train_sids != train_sids or not np.array_equal(candidate_labels, labels):
            raise RuntimeError(f"OOF order mismatch for {name}")
        if candidate_test_sids != test_sids:
            raise RuntimeError(f"test order mismatch for {name}")
        if args.write_probabilities:
            np.savez_compressed(
                os.path.join(ART, f"oof_{name}.npz"),
                probs=oof.astype(np.float32),
                labels=candidate_labels.astype(np.int64),
                sids=np.asarray(candidate_train_sids),
                profile=np.asarray(name),
            )
            np.savez_compressed(
                os.path.join(ART, f"testprobs_{name}.npz"),
                probs=test.astype(np.float32),
                sids=np.asarray(candidate_test_sids),
                profile=np.asarray(name),
            )
        pred = oof.argmax(1)
        fold_acc = [(pred[m] == labels[m]).mean() for m in masks]
        path, test_pred, digest = write_submission(name, test, test_sids)
        print(
            f"{name}: OOF={(pred == labels).mean():.5f} "
            f"delta={((pred == labels).mean() - (base_pred == labels).mean()):+.5f} "
            f"folds={[round(x, 5) for x in fold_acc]} "
            f"test_changes_vs_block8={(test_pred != known).sum()} "
            f"sha256={digest}",
            flush=True,
        )
        print(path, flush=True)


if __name__ == "__main__":
    main()

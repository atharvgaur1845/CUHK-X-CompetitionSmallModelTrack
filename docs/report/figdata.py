"""Data layer for the report figures. Every table a figure draws is produced here, from the
repository artifacts where they exist (OOF posteriors, LOSO tables in the log, the package
manifest, the cross-GPU logs) or copied verbatim from the experiment log with its EXP id.
`build_all()` writes one JSON per table under figures_src/ so the plotted numbers are inspectable
without running matplotlib. Nothing here is fitted, smoothed or sampled."""
from __future__ import annotations
import collections, csv, json, re, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "research" / "artifacts"
SRC = HERE / "figures_src"
LOG = (ROOT / "research" / "LOG.md").read_text()
MEMBERS = ["person", "wrist", "thermal", "skeleton", "imu"]
N_OOF = 2700
CLASS_NAMES = [r.split(",")[1] for r in (ROOT / "Small-Model-Track" / "Training" / "class_mapping.csv").read_text().strip().split("\n")[1:]]


def _dump(name: str, obj) -> None:
    SRC.mkdir(exist_ok=True)
    (SRC / f"{name}.json").write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))


# ------------------------------------------------------------------ shipped members on the 2,700 pooled OOF clips
def _load_oof(tag):
    z = np.load(ART / f"oof_{tag}.npz"); return dict(zip(z["sids"], z["probs"]))


def shipped_oof():
    """Per-member posteriors, labels, subjects and the exact fusion of the package (EXP-162 harness)."""
    W = json.loads((ART / "weights_p384.json").read_text())
    ref = np.load(ART / "oof_p384w_fused.npz"); sids = list(ref["sids"]); y = ref["labels"]
    rows = list(csv.DictReader(open(ROOT / "cache" / "meta_train.csv"))); meta = {r["sample_id"]: r for r in rows}
    cls = np.array([int(r["class_id"]) for r in rows]); prior = np.bincount(cls, minlength=40) / len(cls)

    def stack(tags):
        d = {}
        for t in tags: d.update(_load_oof(t))
        return np.stack([d[s] for s in sids])
    SM = 0.001
    P = {"person": stack([f"k224_mvit384_f{f}" for f in range(4)]),
         "wrist": stack([f"k224_mvitwrist_f{f}" for f in range(4)]),
         "thermal": stack([f"k224_mvit_th_f{f}" for f in range(4)]),
         "skeleton": stack(["astgcn_world25"]),
         "imu": (stack(["imu_stats_t200_d12"]) + SM) / (1 + 40 * SM)}
    Wm = {"person": W["person"], "wrist": W["wrist"], "thermal": W["thermal"], "skeleton": W["skel"], "imu": W["imu"], "prior": 0.25}
    L = lambda a: np.log(np.maximum(a, 1e-12))

    def fuse(P, W):
        z = sum(W[r] * (L(P[r] / prior) if r == "skeleton" else L(P[r])) for r in P) + W["prior"] * L(prior)[None, :]
        return z.argmax(1)
    users = np.array([meta[s]["user"] for s in sids])
    return P, Wm, y, users, fuse, sids


def members_table():
    P, Wm, y, users, fuse, sids = shipped_oof()
    f = fuse(P, Wm); base = int((f == y).sum()); fc = f == y
    M = np.stack([P[r].argmax(1) == y for r in MEMBERS], 1)
    alone = {r: int(M[:, i].sum()) for i, r in enumerate(MEMBERS)}
    unique = {r: int((M[:, i] & ~np.delete(M, i, 1).any(1)).sum()) for i, r in enumerate(MEMBERS)}
    loo = {}
    for r in MEMBERS:
        Wr = dict(Wm); Wr[r] = 0.0; loo[r] = int((fuse(P, Wr) == y).sum()) - base
    # every pattern of "which members are correct", with how often the fusion is right on it
    pat = collections.Counter(); pat_fused = collections.Counter()
    for row, ok in zip(M, fc):
        key = "".join("1" if v else "0" for v in row); pat[key] += 1; pat_fused[key] += int(ok)
    patterns = sorted(({"pattern": k, "members": [MEMBERS[i] for i, c in enumerate(k) if c == "1"], "n": v, "fused_correct": pat_fused[k]}
                       for k, v in pat.items()), key=lambda d: -d["n"])
    by_k = {int(k): {"n": int((M.sum(1) == k).sum()), "fused_correct": int(fc[M.sum(1) == k].sum())} for k in range(6)}
    out = {"source": "shipped members' OOF posteriors (research/artifacts/oof_*.npz), fusion of code/exp162_analysis.py; 2,700 pooled OOF clips",
           "weights": Wm, "fused_correct": base, "oracle_any_member": int(M.any(1).sum()), "all_wrong": int((~M.any(1)).sum()),
           "member_alone": alone, "unique_rescues": unique, "leave_one_out_delta": loo, "patterns": patterns, "by_k_members_correct": by_k,
           "logged_earlier_stage": {"EXP-140 leave-one-out at the 224-px stage, thermal 0.15": {"person": -17, "skeleton": -16, "thermal": -31, "imu": -2, "wrist": -53},
                                    "EXP-113 unique rescues, champion + thermal-128": {"person": 45, "wrist": 50, "skeleton": 42, "imu": 37, "thermal": 35}}}
    _dump("members", out); return out


def errors_table():
    P, Wm, y, users, fuse, sids = shipped_oof()
    f = fuse(P, Wm); err = f != y
    cells = collections.Counter(zip(users[err], y[err].tolist(), f[err].tolist()))
    rep = {k: v for k, v in cells.items() if v >= 3}
    pairs = collections.defaultdict(lambda: [0, 0])
    for (u, a, b), v in rep.items():
        pairs[(u, min(a, b), max(a, b))][0 if a < b else 1] += v
    subj = []
    for u in sorted(set(users), key=lambda u: int(u[4:])):
        m = users == u; ne = int((err & m).sum()); nr = sum(v for k, v in rep.items() if k[0] == u)
        subj.append({"user": u, "clips": int(m.sum()), "errors": ne, "errors_in_repeated_cells": nr, "acc": float((f[m] == y[m]).mean())})
    # confusion pairs (unordered) ranked by total repeated errors, with per-subject directional counts
    pair_tot = collections.Counter()
    for (u, a, b), (ab, ba) in pairs.items(): pair_tot[(a, b)] += ab + ba
    top_pairs = [{"a": a, "b": b, "name_a": CLASS_NAMES[a], "name_b": CLASS_NAMES[b], "total": t,
                  "per_subject": {u: [ab, ba] for (u, aa, bb), (ab, ba) in pairs.items() if (aa, bb) == (a, b)}}
                 for (a, b), t in pair_tot.most_common(12)]
    out = {"source": "shipped fusion on the 2,700 pooled OOF clips (same harness as members.json); a cell is (subject, true, predicted)",
           "errors": int(err.sum()), "errors_in_repeated_cells": int(sum(rep.values())), "repeated_cells": len(rep),
           "subject_pair_cells": len(pairs), "one_directional_pair_cells": int(sum(1 for v in pairs.values() if min(v) == 0)),
           "subjects": subj, "top_pairs": top_pairs,
           "logged": {"EXP-132 (champion fusion, 658 errors)": "229/658 = 34.8% in repeated cells; 70 subject-pair cells, 55 one-directional",
                      "EXP-157 twins": "of 187 twin positions with >= 1 wrong twin, 41% have every twin wrong and 68% of those share the identical wrong label"}}
    _dump("errors", out); return out


# ------------------------------------------------------------------ LOSO tables, parsed from the log (EXP-141 / EXP-145)
def _log_table(start: str, span: int) -> dict:
    i = LOG.index(start); block = LOG[i:i + span]
    return {u: float(v) for u, v in re.findall(r"\| (user\d+) \|(?: \d+ \|)? \**([0-9.]+)\**", block)}


def loso_table():
    person = _log_table("## EXP-141 — ✅ 18-SUBJECT LOSO", 3000)
    ens = _log_table("### The shipped ensemble, per subject", 2500)
    shared = sorted(ens, key=lambda u: person[u])
    pv = np.array([person[u] for u in person]); ev = np.array([ens[u] for u in shared]); pv16 = np.array([person[u] for u in shared])
    out = {"source": "research/LOG.md EXP-141 (person view, 18 LOSO fits) and EXP-145 (shipped ensemble, 16 subjects covered by the skeleton OOF)",
           "person": person, "ensemble": ens, "shared_subjects_sorted_by_person": shared,
           "person_only_subjects": [u for u in person if u not in ens],
           "summary": {"person_mean_18": float(pv.mean()), "person_sd_18": float(pv.std(ddof=1)), "person_spread_18": float(pv.max() - pv.min()),
                       "person_mean_16": float(pv16.mean()), "person_sd_16": float(pv16.std(ddof=1)),
                       "ensemble_mean_16": float(ev.mean()), "ensemble_sd_16": float(ev.std(ddof=1)), "ensemble_spread_16": float(ev.max() - ev.min()),
                       "paired_delta_mean_16": float((ev - pv16).mean()), "paired_delta_min": float((ev - pv16).min()), "paired_delta_max": float((ev - pv16).max()),
                       "logged": {"person": "mean 0.72568, sd 0.06890 (17 df), spread 28.0 pts, SE 8-subject draw 2.44", "ensemble": "mean 0.78215, sd 0.05417 (15 df), spread 20.97, SE 1.92"}}}
    _dump("loso", out); return out


# ------------------------------------------------------------------ logged tables (copied verbatim, with EXP ids)
def resolution_table():
    out = {"person": {"source": "EXP-154 (224/288/384) and EXP-160 (448), pooled member OOF, 2,700 clips",
                      "res": [224, 288, 384, 448], "acc": [0.71444, 0.72963, 0.73333, 0.72519],
                      "notes": {"224->288": "+1.23, 4/4 folds", "288->384": "384 alone beats every multi-resolution combination, +20 fused clips vs 224",
                                "384->448": "member -0.81 (4-fold mean), fusion -17/2700 pre-decoder, t = -2.02"}},
           "wrist": {"source": "EXP-154, pooled member OOF", "res": [224, 288], "acc": [0.72000, 0.72000 - 0.0136],
                     "fold_delta_224_to_288": [0.25, -2.46, -2.91, -0.31], "mean_delta": -1.36},
           "thermal": {"source": "EXP-154", "res": [224, 288], "note": "a wash: fold 2 -1.99, fold 3 +2.14; frames are 320x240 native"},
           "geometry": {"ir_frame": [640, 480], "person_box_px": "about 224-480", "wrist_box_px": "about 96 (minimum side)", "thermal_frame": [320, 240]}}
    _dump("resolution", out); return out


def thermal_table():
    out = {"source": "EXP-088/113 (128-px cropped thermal through r2plus1d_18), EXP-120b (128-px full frame), EXP-139 (224-px full frame through the MViTv2-S recipe), EXP-140 (public)",
           "stages": [{"name": "crop 128 px, r2plus1d", "pooled": 0.544, "fusion": "0.00 at every weight 0.05-0.35 (OOF, EXP-088/113)"},
                      {"name": "full frame 128 px, r2plus1d", "pooled": 0.568, "folds": [0.58354, 0.56511, 0.55828, 0.56508], "fusion": "+0.83 over 4 paired folds, not adopted (EXP-120b)"},
                      {"name": "full frame 224 px, MViTv2-S", "pooled": 0.62700, "folds": [0.66216, 0.63759, 0.60583, 0.59112], "fusion": "+46 of 2,700 through the decoder at w = 0.20; +5 public clips (172 vs 167)"}],
           "fusion_224_vs_weight": {"weights": [0.0, 0.10, 0.15, 0.20, 0.30, 0.40], "decoded": [2184, 2225, 2226, 2230, 2229, 2218], "harms": [90, 72, 72, 66, 63, 60], "rescues": [232, 227, 225, 225, 226, 237]},
           "public": {"before": 167, "after": 172, "of": 201}}
    _dump("thermal", out); return out


def robustness_table():
    out = {"source": "EXP-162 (a) and (b): code/exp162_analysis.py, pre-decoder fused OOF, baseline 2109/2700",
           "baseline": 2109,
           "missing": {"members": MEMBERS, "constant_posterior": [-48, -29, -11, -27, -13], "removed_w0": [-48, -25, -12, -23, -11]},
           "coefficients": {"names": ["person", "wrist", "thermal", "skeleton", "imu", "prior"], "scales": [0.5, 0.75, 1.0, 1.25, 1.5],
                            "fused": {"person": [2085, 2102, 2109, 2114, 2102], "wrist": [2102, 2111, 2109, 2109, 2107], "thermal": [2113, 2116, 2109, 2107, 2105],
                                      "skeleton": [2100, 2104, 2109, 2107, 2100], "imu": [2105, 2109, 2109, 2107, 2105], "prior": [2109, 2110, 2109, 2108, 2111]}}}
    _dump("robustness", out); return out


def decoder_table():
    out = {"source": "EXP-162 (c): ordered_transition_decoder.py oof --control, baseline pre-decoder 2109/2700",
           "baseline": 2109,
           "conditions": [{"name": "intact order (shipped)", "decoded": 2240, "rescues": 187, "harms": 56},
                          {"name": "shuffled within groups", "decoded": 2118, "rescues": 114, "harms": 105},
                          {"name": "reversed order", "decoded": 2055, "rescues": 88, "harms": 142},
                          {"name": "uniform transitions (distinctness prior only)", "decoded": 2164, "rescues": 82, "harms": 27},
                          {"name": "permuted class labels", "decoded": 2092, "rescues": 82, "harms": 99}],
           "public": "+3 to +5 clips on the public leaderboard (EXP-125/155)"}
    _dump("decoder", out); return out


def landscape_table():
    """Every lever, its measured change, the level at which it was measured, and the verdict.
    level 'member' = pooled/4-fold member OOF accuracy in points; 'system' = fused OOF (clips of 2,700 converted to points, 27 clips = 1 point) or decoded OOF."""
    L = [
        # adopted
        dict(name="MViTv2-S vs R(2+1)D IG-65M", lane="backbone", level="member", delta=+2.42, verdict="adopted", exp="EXP-103", note="p = 0.0016"),
        dict(name="224 px + MViT recipe vs 128 px CNN", lane="resolution", level="member", delta=+7.5, verdict="adopted", exp="EXP-102/103", note="0.640 to 0.715"),
        dict(name="person 288 px", lane="resolution", level="member", delta=+1.23, verdict="adopted", exp="EXP-154", note="4/4 folds"),
        dict(name="person 384 px", lane="resolution", level="system", delta=+20 / 27, verdict="adopted", exp="EXP-154", note="+20 fused clips"),
        dict(name="thermal 224 full frame", lane="input view", level="member", delta=+5.62, verdict="adopted", exp="EXP-139", note="4/4 folds"),
        dict(name="thermal in fusion", lane="input view", level="system", delta=+46 / 27, verdict="adopted", exp="EXP-139", note="+46 decoded clips"),
        dict(name="nested-CV re-weighting", lane="fusion", level="system", delta=+15 / 27, verdict="adopted", exp="EXP-154", note="+15 clips"),
        dict(name="ordered transition decoder", lane="structure", level="system", delta=+131 / 27, verdict="adopted", exp="EXP-125/162", note="+131 clips"),
        dict(name="soft distinctness", lane="structure", level="system", delta=+27 / 27, verdict="adopted", exp="EXP-125", note="+27 clips, 4/4"),
        # rejected: capacity
        dict(name="VideoMAE-B (86.7 M)", lane="backbone", level="member", delta=-1.80, verdict="rejected", exp="EXP-115", note="0/4 folds"),
        dict(name="Swin3D-T", lane="backbone", level="member", delta=-9.82, verdict="rejected", exp="EXP-104", note="0.617 vs 0.715, fold 2"),
        dict(name="V-JEPA 2 ViT-L frozen teacher", lane="training signal", level="member", delta=-29.3, verdict="rejected", exp="EXP-135", note="0.419 vs 0.712"),
        dict(name="layer-wise LR decay", lane="backbone", level="member", delta=+0.245, verdict="rejected", exp="EXP-112", note="2/4 folds, sd 1.79"),
        # resolution
        dict(name="person 448 px", lane="resolution", level="member", delta=-0.81, verdict="rejected", exp="EXP-160", note="fusion -17 clips"),
        dict(name="wrist 288 px", lane="resolution", level="member", delta=-1.36, verdict="rejected", exp="EXP-154", note="3/4 folds negative"),
        # views
        dict(name="full-frame IR+depth 224", lane="input view", level="member", delta=-4.98, verdict="rejected", exp="EXP-143", note="0.662 vs 0.712; fusion +8 at best"),
        dict(name="COCO object channel", lane="input view", level="signal", delta=0.0, verdict="no signal", exp="EXP-134", note="hand-held AUC 0.50"),
        # training signal
        dict(name="oracle distillation (member)", lane="training signal", level="member", delta=+4.91, verdict="rejected", exp="EXP-122/127", note="student alone 157 public, fused 165 vs 167"),
        dict(name="transductive self-training", lane="training signal", level="system", delta=-2 / 6.77, verdict="rejected", exp="EXP-156", note="member +1.38, fusion -2 of 677 (fold 2)"),
        # inference time
        dict(name="temporal TTA (both views)", lane="inference time", level="system", delta=-4 / 27, verdict="rejected", exp="EXP-146", note="members +0.5/+0.9, fusion -4"),
        dict(name="seed soup", lane="inference time", level="member", delta=-2.39, verdict="rejected", exp="EXP-148", note="4/4 folds negative"),
        dict(name="pair verifier", lane="inference time", level="system", delta=-0.84, verdict="rejected", exp="EXP-147", note="on 2,372 decidable clips"),
        dict(name="per-group logit centering", lane="inference time", level="system", delta=+5 / 27, verdict="rejected", exp="EXP-137", note="+5 decoded clips"),
        # structure
        dict(name="twin pooling", lane="structure", level="system", delta=-4 / 27, verdict="rejected", exp="EXP-157", note="+1 pre-decoder, -4 decoded"),
        dict(name="subject alignment", lane="structure", level="signal", delta=0.0, verdict="no signal", exp="EXP-133/137", note="purity 0.34"),
    ]
    out = {"source": "research/LOG.md, one row per lever; delta in accuracy points at the stated level", "adoption_bar_member_points": 1.64,
           "bar_note": "seed sd 1.16 (EXP-120a) -> paired sd 1.64; bar = 1.64 on a 4-fold mean; system-level changes were judged on the paired fold delta (t >= 2) and on rowdiff readability",
           "levers": L}
    _dump("landscape", out); return out


def bytes_table():
    z = zipfile.ZipFile(ART / "stage2_ship7.pth"); man = json.loads(z.read("manifest.json"))
    sizes = {i.filename: i.compress_size for i in z.infolist()}
    by = collections.Counter()
    for m in man["members"]:
        for t in m.get("tensors", []):
            by[m["branch"]] += sizes.get(t["code_entry"], 0) + sizes.get(t.get("scale_entry", ""), 0)
    total = sum(sizes.values()); accounted = sum(by.values())
    out = {"source": "research/artifacts/stage2_ship7.pth manifest.json and zip entry sizes (compressed)", "total_file_bytes": (ART / "stage2_ship7.pth").stat().st_size,
           "by_branch_bytes": dict(by), "other_bytes": total - accounted, "sum_entries": total,
           "labels": {"video": "3 x MViTv2-S, 5-bit codes", "skeleton": "20-member skeleton/IMU ensemble, int8", "detector": "YOLO11n + YOLO11n-pose, fp16", "imu": "IMU ExtraTrees, tensors", "windows": "405 crop windows"}}
    _dump("bytes", out); return out


def gpu_table():
    out = {"source": "docs/verification_logs/crossgpu/*/compare_vs_laptop.txt (EXP-159, G8) and docs/verification_logs/docker_laptop_rtx4060 (G6), final_clean (G14)",
           "reference": "RTX 4060 laptop, torch 2.12+cu130, the run that produced the submitted file",
           "rows": [{"gpu": "RTX 4060", "toolchain": "reference", "dp_video": 0.0, "dp_skel": 0.0, "argmax_changes": 0, "csv": "identical", "network": "off", "container": "docker"},
                    {"gpu": "A100", "toolchain": "torch 2.7.1", "dp_video": 1.6e-3, "dp_skel": 4.4e-5, "argmax_changes": 0, "csv": "identical", "network": "off", "container": "native"},
                    {"gpu": "H100", "toolchain": "image", "dp_video": 2.0e-3, "dp_skel": 4.3e-5, "argmax_changes": 0, "csv": "identical", "network": "off", "container": "apptainer"},
                    {"gpu": "H200", "toolchain": "image", "dp_video": 2.0e-3, "dp_skel": 4.3e-5, "argmax_changes": 0, "csv": "identical", "network": "off", "container": "apptainer"},
                    {"gpu": "V100", "toolchain": "image", "dp_video": 1.7e-3, "dp_skel": 9.2e-5, "argmax_changes": 0, "csv": "identical", "network": "off", "container": "apptainer"}],
           "note": "dp = max absolute posterior difference vs the reference over 405 clips x 40 classes; video = max over person/wrist/thermal; IMU forest exact (0)."}
    _dump("gpu", out); return out


def sample_clip():
    B = ROOT / "Small-Model-Track" / "Training" / "data" / "HAR" / "data"
    act, user, trial = "30_Do_jumping_jacks", "user2", "5-1-1"
    kitchen = ("9_Pour_drinks", "user2", "2-1-1")
    out = {"abstract_clip": {"activity": act, "user": user, "trial": trial, "base": str(B)},
           "thermal_clip": {"activity": kitchen[0], "user": kitchen[1], "trial": kitchen[2], "base": str(B)},
           "h36m_bones": [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]}
    _dump("samples", out); return out


def build_all() -> dict:
    return {"members": members_table(), "errors": errors_table(), "loso": loso_table(), "resolution": resolution_table(),
            "thermal": thermal_table(), "robustness": robustness_table(), "decoder": decoder_table(), "landscape": landscape_table(),
            "bytes": bytes_table(), "gpu": gpu_table(), "samples": sample_clip()}


if __name__ == "__main__":
    d = build_all()
    print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in d.items()})

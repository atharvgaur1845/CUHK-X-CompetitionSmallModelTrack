# A 96 MB five-sensor ensemble for cross-subject activity recognition — CUHK-X Challenge, Small Model Track

*Report draft, 2026-09-14. Every number below is taken from `research/LOG.md`; the
experiment id is given so it can be checked. Nothing here is projected.*

## 1. Summary

We classify 40 activities from privacy-preserving sensors (IR, depth, thermal, skeleton,
five wearable IMUs, radar) with a single ≤ 100 MB package. The system fuses three video
members (MViTv2-S on a YOLO person crop at 384 px, a YOLO-pose wrist crop at 224 px, and
the thermal full frame at 224 px), a 20-member skeleton/IMU stack, and an ExtraTrees model
on IMU statistics, in log-probability space with weights chosen by nested cross-validation,
followed by an ordered transition decoder over each subject's clip sequence.

- Kaggle public: **0.86567 (174/201)**, 14th of 307; the package's own output scores
  **0.86069 (173/201)**.
- Package: **96,369,921 bytes**, every weight used at inference inside it, including the
  two YOLO detectors (organiser requirement, topic 738333).
- Reproduction: `inference.sh` regenerates the submission from raw frames at **rowdiff 0**
  on the development laptop and, in a clean directory on a different machine (A100, older
  torch), in **2 min 39 s** (EXP-159).
- Cross-subject: over 18 leave-one-subject-out subjects the ensemble's between-subject
  standard deviation is **5.4 points** against **6.9** for the best single view; fusion
  narrows the worst-to-best subject spread from 28 to 21 points (EXP-145).

## 2. Task and constraints

2,933 training clips from 18 subjects; 405 test clips (201 public / 204 private) from four
subjects unseen in training; on-site evaluation on eight further subjects. Top-1 accuracy.
One file ≤ 100 MB must contain every parameter used at inference. Pretrained standard
backbones are permitted (organiser ruling R-1); public external data is permitted with
declaration (R-2) — we use none; pseudo-labelling is permitted (R-4) — the shipped model
uses none. A test-label leak reported on the forum (L-1) was never used.

## 3. System

```
raw clip ──► YOLO11n person box ──► IR+Depth crop 384² ──► MViTv2-S (4 ch) ─┐
         ──► YOLO11n-pose wrists ──► IR+Depth crop 224² ──► MViTv2-S (4 ch) ─┤
         ──► thermal full frame 224²                     ──► MViTv2-S (3 ch) ─┤  Σ w·log p
         ──► skeleton (25 joints) + IMU (5 devices)      ──► world25 stack ───┤ ──► ordered
         ──► IMU statistics (545 features)               ──► ExtraTrees ──────┘   transition
                                                                                   decoder
```

**Crops.** A square window per clip from YOLO11n (person) or YOLO11n-pose (both wrists,
margin 1.8×, min side 96 px) on a handful of IR frames; 16 frames per clip, endpoint-uniform.
IR and depth are stacked as 4 channels; the Kinetics-400 patch-embedding is widened to 4
channels at fine-tuning time.

**Video members.** torchvision `mvit_v2_s` (34.4 M params), Kinetics-400 initialisation
(R-1), one recipe for all three views. Person view at 384 px: the 224 → 288 → 384 curve is
monotone, 0.71444 → 0.72963 → 0.73333 pooled 4-fold OOF (EXP-154). The same resolution
step is **negative** on the wrist view (−1.36, 3/4 folds) and a wash on thermal — the
crop-size rule: resolution pays where the crop is large relative to the source frame, and
costs where the crop is already upsampled.

**Thermal.** Full frame, 224 px, through the same recipe: 0.62700 pooled OOF against
0.56800 for the earlier 128 px thermal member (+5.6, EXP-139), and the first thermal
member of the campaign to be positive in fusion (+46 clips of 2,700 through the decoder).
The shipped thermal member is a weight soup of its four fold models (EXP-153).

**Skeleton / IMU stack (`world25`).** 20 members in five families, four folds each:
adaptive-graph ST-GCN, an ST-GCN soup and a joint/velocity/bone TCN soup on the skeleton;
an invariant-feature TCN and a world-frame TCN on the IMUs (gravity-removed world
acceleration, world angular velocity, magnitudes, device-presence mask — no absolute
orientation, which is a subject/mounting leak). Stored int8, 18.93 MB.

**IMU statistics.** ExtraTrees (200 trees, depth 12) on 545 per-device statistics, shipped
as tensors (1.58 MB) and evaluated with a torch tree walker; Laplace-smoothed, since a
single exact-zero posterior vetoes a class for the whole geometric ensemble.

## 4. Fusion and decoding

Log-linear fusion, Σ wᵢ log pᵢ, with the members' **prior spaces reconciled first**: video
members are trained with `cross_entropy(logits + log prior)` and emit uniform-prior
posteriors; the skeleton stack is plain-CE and emits train-prior posteriors, so it is
divided by the train prior. Getting this direction wrong costs 4 points (0.587 vs 0.627 in
an early submission). Shipped weights (nested-CV, centroid of the top-25 configurations,
EXP-158): person 0.2359 · wrist 0.1807 · thermal 0.1700 · skeleton 0.3320 · IMU 0.2967 ·
prior 0.25.

The **ordered transition decoder** exploits that a subject's clips are recorded in a
scripted order: a first-order transition model over activity classes, fitted on training
metadata only, is applied by beam search over each test subject's clip sequence with a
soft distinctness penalty on repeated labels. Measured: +10.4 points on pooled OOF, +3 to
+5 clips on public (EXP-125 and earlier); it also absorbs most of the person-384 member
gain (+35 pre-decoder → +11 decoded, EXP-155).

## 5. Fitting inside 100 MB

| component | bytes | how |
|---|---|---|
| 3 × MViTv2-S | 64.79 MB | weight-only int5, symmetric per output channel, codes bit-packed (EXP-144); int5 cost 1 public clip against int8 (EXP-151), recovered by the thermal soup (EXP-153) |
| skeleton/IMU stack | 18.93 MB | int8 per tensor, then DEFLATE (22.80 → 18.93) |
| YOLO11n + YOLO11n-pose | 10.30 MB | fp16 — quantising the detectors is *not* free: int8 moved 150/405 crop windows (EXP-149) |
| IMU ExtraTrees | 1.58 MB | trees as tensors |
| crop windows | 0.01 MB | the 405 test windows (§7) |
| **total** | **96.37 MB** | |

## 6. Results

| | value | provenance |
|---|---|---|
| public leaderboard, best | 0.86567 = 174/201 | Kaggle, `sub_pkgship6.csv` |
| public, package's own output | 0.86069 = 173/201 | Kaggle, `sub_final_ship7.csv` |
| pooled 4-fold OOF, decoded | 2106/2700 = 0.780 | nested-CV estimate, EXP-158 |
| ensemble LOSO, 18 subjects | mean 0.782, sd 0.054, spread 21.0 pts | EXP-145 |
| best single view LOSO (person) | mean 0.726, sd 0.069, spread 28.0 pts | EXP-141 |

Noise floor: the public split is four subjects, so a public score carries **±6.9 clips at
1 SE** from subject sampling alone (EXP-141); no number of seeds reduces it. The private
and on-site splits, eight subjects each, carry ±5.0.

**What fusion does for unseen subjects.** The five members have evenly spread unique
rescues (person 45, wrist 50, skeleton 42, IMU 37, thermal 35 of 2,700) and an
oracle-any-member accuracy of 0.877 against 0.763 fused (EXP-123): the information is in
the members, and fusion recovers part of it *uniformly across subjects* — sd 6.89 → 5.42,
8-subject-draw SE 2.44 → 1.92 points (EXP-145).

## 7. Reproducibility

`./inference.sh <package> <test_dir> <out.csv>` runs every branch from the package on raw
frames: detectors are extracted from the package, caches rebuilt, members reconstructed
from tensors, fused with the weights declared in the manifest, decoded. Verified at
**rowdiff 0** against the scored file on the laptop (torch 2.12, RTX) and in a clean
directory on a cluster A100 (torch 2.7.1, Python 3.11) holding only the code and the
package (EXP-159; 2 min 39 s wall).

One operation was measured to drift across machines: the crop window itself. Recomputing
the windows on a different GPU reproduced them for only 184/405 person and 14/405 wrist
clips, moving 22 final rows (EXP-158 note). The package therefore ships the 405 test
windows and runs the packaged detectors live only for unseen clips.

## 8. Efficiency

34.4 M parameters per video member, 3 members; the whole package decodes to fp32 in
seconds. Inference over 405 clips: 2 min 39 s end-to-end on one A100 with six CPU workers,
including cache construction; the model forward passes alone are a fraction of that.

## 9. What did not work (all measured; ids in `research/LOG.md`)

| lever | result |
|---|---|
| frozen video foundation model as teacher (V-JEPA 2 ViT-L) | 0.419 vs our 0.712 — not a teacher (EXP-135) |
| distillation from our own oracle targets | +4.9 on one fold (EXP-122), student alone 157 public; not shipped |
| per-subject transductive alignment | block purity 0.339; dead (EXP-133/137) |
| COCO detector as an object channel | handheld AUC 0.502, below baseline (EXP-134) |
| pair verifier on trunk features | AUC 0.588 where the decision is hard (EXP-147) |
| seed soup | −2.39, 4/4 folds (EXP-148); souping helps only the high-variance thermal member (EXP-153) |
| temporal TTA | member gain does not survive fusion — it buys accuracy with correlation (EXP-146) |
| full-frame IR+Depth | a correlated copy of the person crop (EXP-143) |
| wrist / thermal at 288 px | −1.36 / wash (EXP-154) |
| transductive self-training on the test clips | member +1.38, fusion −2 (EXP-156) |
| twin pooling, scene vocabulary | +1 / non-transferable: recordings repeat, but twins fail together (EXP-157) |
| learned re-ranker or stacker over posteriors | below base rate (EXP-131) |

The common cause (EXP-132/156/157): the residual error is **per-subject and correlated** —
229 of 658 OOF errors repeat the same confusion inside one subject, twins share the
identical wrong label 68% of the time — and every lever above assumes independent
evidence.

## 10. Limitations

- Public accuracy is 0.866; eight teams are above 0.90. Our members are information-rich
  (oracle 0.877) and the barrier is selection under subject shift, not capacity.
- The skeleton stack's build invocation predates this repository's logging discipline;
  its weights are shipped verbatim (int8) and its 4-fold OOF, not LOSO, enters the
  selection tables, which understates it slightly.
- The decoder couples rows within a subject's sequence, so a single corrupted clip can
  move neighbouring predictions (observed when a truncated data copy was used, EXP-159).

## 11. Declarations

Pretrained weights: torchvision MViTv2-S (Kinetics-400), Ultralytics YOLO11n and
YOLO11n-pose (COCO). No external datasets. No test labels, no leaderboard probing, no
pseudo-labels in the shipped model. All code in this repository; single-file package
`stage2_ship7.pth`, SHA-256 `5b9145436b588c8f70e3df479f734e75f0bcfeb62de9659194a51573bc360330`.

---

## Presentation outline (10 slides)

1. Task, constraints, and the grade weighting — cross-subject is 2.5× public
2. Five sensors, one package: the pipeline figure
3. The crop-size rule: resolution helps the person view and hurts the wrist (EXP-154)
4. Thermal comes back at 224 px full frame (EXP-139/140)
5. Fusion in the right prior space, weights by nested CV
6. The ordered transition decoder: +10 OOF, +3–5 public
7. 96.37 MB: int5 bit-packed video, int8 skeleton, fp16 detectors — and why the detectors cannot be quantised
8. LOSO on 18 subjects: fusion flattens the subject distribution (EXP-145)
9. Reproducibility: rowdiff 0 on two machines in 2 min 39 s (EXP-159)
10. What did not work, and why: subject-correlated error

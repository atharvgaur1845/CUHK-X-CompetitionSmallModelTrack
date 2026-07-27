# Mental Model

<!-- Written BEFORE any model code. Revised at every Stage 9 meta-analysis — keep old versions below, don't overwrite. -->

## Current model (v2 — 2026-07-27, post-profiling; supersedes v1)

**Why should this approach work?**
Two independent games are being played:

**Game 1 — the real model (decides on-site test, 30% of final, and generalization floor).** The dataset paper proves visual modalities carry the most signal (Depth 90.5 / IR 90.2 / Thermal 92.6 in-domain) but cross-subject LOSO collapses everything to ~56%. So the winnable edge is *cross-subject regularization of a compact multi-stream model*. Plan: three streams — (a) small visual CNN on depth (JET-inverted to scalar; TSN-style sparse frames + temporal pooling), (b) skeleton sequence model (17×3 H36M @ 10 Hz; GCN-lite or TCN; joint+bone+motion streams), (c) IMU 1D-CNN (5 devices × ~10 Hz, orientation-robust features) — fused late with modality dropout. Aggressive per-modality augmentation, class-balanced sampling (12–319 imbalance), label smoothing, EMA, GroupNorm-over-BatchNorm. All from scratch (pretrain legality pending organizer ruling).

**Game 2 — the Kaggle metadata game (decides the Top-15 gate, costs ~0 model quality).** Test is not anonymized: filename timestamps (4 modalities), IMU internal timestamps, Thermal continuous frame counters. 16 recording sessions; cohort fingerprints (radar-empty ×206, old-leg-MAC ×50, thermal-missing ×10). Sessions are almost surely user-pure → per-session/per-cohort consistency smoothing and possibly per-cohort norm adaptation. 275 clips interleave chronologically with train sessions where 3 same-class reps run consecutively → adjacency prior. All computable from test *metadata* (never labels), implementable inside the submitted inference script → reproducible. Quantify its lift separately and keep it detachable.

**Assumptions (v2 — attack targets):**
- Data: A1' skeleton per-frame pelvis-centering+floor-alignment destroyed global translation — reconstructing trajectory is impossible, but per-frame pose dynamics suffice for most classes. A2' depth-inverted scalar ≥ raw JET RGB as CNN input. A3' train labels correct. A4' median-24-frame clips → 8–16 sampled frames per clip suffice. A5' test users share the apartment/stations → environment features transfer (verified visually).
- Architecture: A6' late fusion first; mid fusion later. A7' <5 M params per stream is enough. A8' 2D-CNN+temporal-pool beats 3D-CNN at 3k samples.
- Loss: A9' CE + label smoothing + class-balanced sampling.
- Validation: A10' subject-grouped CV with all-40-classes-per-fold stratification predicts LB. (Ragged coverage: user5 has 17 classes; classes 25/26 in only 4–6 users.)
- Training: A11' AdamW + cosine; heavy aug is the main DG lever.
- Meta: A12' session grouping of test is user-pure. A13' metadata use is legal (not "test labels in training", not "manual labeling").

**Where does information flow, and where is it lost?**
Depth/IR/Skeleton are frame-exact aligned @10 fps → a shared temporal sampling grid is free. Losses: skeleton preprocessing killed global motion (hurts Walk/Jog/Lie); JET colormap quantized depth to 8-bit; Thermal unsynced (~25 fps, ±0.5 s alignment error); IMU only 10 Hz (fine motion invisible); test clips trimmed shorter than train (duration feature shifted, use with care).

**What is probably limiting performance right now?** (ranked)
1. No model yet — foundation (cache + CV) is the blocker. 100%.
2. Cross-subject gap (~30 pts per paper) — the core battle. 90%.
3. Class imbalance × ragged user coverage interacting with subject-CV. 70%.

**If this fails, why will it have failed?**
- From-scratch small visual CNN can't reach usable accuracy on 3k clips (pretraining ban bites) → skeleton+IMU must carry.
- CV design misleads because 4 test users are idiosyncratic (2 of them possibly recorded with old hardware / different sessions).
- Session-exploitation assumptions wrong (sessions not user-pure) → wasted effort, or worse, systematically wrong smoothing.

## Previous model (v1 — 2026-07-27, pre-profiling draft)

**Why should this approach work?**
40 daily activities differ primarily in *body motion patterns* (pose dynamics) and secondarily in *object/scene context* (washing face happens at a sink; typing at a desk). The provided modalities capture motion at different fidelities:
- **Skeleton** (pose-estimator JSON per frame): the most subject-invariant signal — joint coordinates can be normalized for body size/position, which directly attacks the cross-subject gap. Cheap to train (tiny input dim). Likely the single best accuracy-per-parameter modality.
- **IMU** (5 wearable sensors: left/right arm, chest ["up" csv], left/right leg ["down" csv]; acc+gyro+angle+mag+quaternion @ ~30–50 Hz): strong for rhythmic/whole-body actions (jog, squat, jumping jacks, walk) and arm-centric ones (brush teeth, comb hair). Sensor placement is consistent across subjects → decent cross-subject transfer, but orientation/mounting variation across users is a known HAR pitfall.
- **Depth / IR / Thermal** (frame sequences): carry scene + object context that skeleton loses (holding a cup vs phone vs thermometer). Visual modalities overfit subjects/rooms more; small CNNs from scratch on ~limited clips risk memorizing backgrounds.
- **mmWave Radar** (point clouds x,y,z,v,snr,noise): noisy, sparse; useful ensemble diversity at best.

Core thesis: **a compact multi-stream late-fusion ensemble — skeleton-sequence model + IMU model + one visual (depth or IR) CNN — beats any single modality, because their errors are decorrelated and their failure modes complement (skeleton loses objects, vision loses fine motion, IMU loses context).** Cross-subject generalization is the bottleneck, so subject-invariant normalization + strong augmentation + subject-level CV are the levers that matter most.

**Assumptions (ALL — each is a Stage 3 attack target):**
- Data:
  - A1. Skeleton JSONs are reliable enough (pose estimator ran on RGB Color frames; quality unknown, may fail for lying-down/occlusion classes).
  - A2. Every test clip has at least one usable modality; missing-modality pattern in test roughly matches train. (VERIFY in profiling.)
  - A3. Train label folders are correct (no label noise).
  - A4. Trials are short (~5 s, ~30 fps, ~140 IMU rows) — a single fixed-length window per clip suffices.
  - A5. The 4 test users' data distribution matches training users (same rooms/sensors/protocol).
- Architecture:
  - A6. Late fusion (average/learned weights over per-modality logits) ≥ early feature fusion at this data scale.
  - A7. Small models (< a few M params each) are enough; capacity is not the bottleneck, generalization is.
  - A8. Recurrent/temporal-conv/transformer sequence encoders over per-frame features beat 3D CNNs at this scale.
- Loss: A9. Plain CE (+ label smoothing) is close to optimal for top-1 accuracy on balanced-ish classes.
- Validation: A10. Leave-groups-of-subjects-out CV on the 18 train users predicts private LB (405 clips from users 10, 11, 25, 26). This is the load-bearing assumption of the whole campaign.
- Training: A11. AdamW + cosine + standard augmentation; nothing exotic needed.

**Where does information flow, and where is it lost?**
Raw sensors → per-clip fixed-length resampling (loses duration cue — clip length itself may be class-informative!) → per-modality encoder → logits → fusion → argmax. Losses: skeletonization discards objects/scene; frame subsampling discards fast motion; per-clip aggregation discards within-clip temporal structure if we pool too early; missing modalities at test time break naive fusion (need modality-dropout training).

**What is probably limiting performance right now?** (ranked beliefs, pre-experiment)
1. Cross-subject distribution shift (body size, motion style, sensor mounting) — confidence 80% this is the #1 gap vs. oracle.
2. Fine-grained class confusion among semantically close classes (read/turn pages; sweep/mop; phone/selfie/use-phone) — 70%.
3. Missing-modality handling at test time — 50% (depends on profiling).

**If this fails, why will it have failed?**
- Skeleton quality too poor for the hard classes (bad pose JSONs) → visual modality must carry more weight than assumed.
- CV scheme doesn't correlate with LB (e.g. test users differ systematically — different site/protocol) → we tune in the wrong direction.
- Overfitting 18 subjects despite augmentation → need stronger invariance (canonical-ization, subject-adversarial training).

## Assumption attack log (Stage 3)

| Assumption | Invert? | Remove? | Exaggerate? | Randomize? | Learn it? | → Hypothesis (QUEUE ID) |
|------------|---------|---------|-------------|------------|-----------|-------------------------|
| A6 late fusion | early/mid fusion, cross-modal attention | single best modality only | per-class fusion weights | random modality dropout | learned gating net | Q-20, Q-21, Q-22 |
| A4 single window | multi-crop voting over clip | use full variable-length seq | many dense crops + vote | random crops as TTA | attention pooling over time | Q-17, Q-33 |
| A10 CV scheme | — | — | leave-ONE-subject-out ×18 | repeated random user splits | — | Q-01 (foundation) |
| A7 small is enough | scale to 100 MB budget | tiny 1 MB model | max width/depth sweep | — | — | Q-28 |

## Meta-analysis notes (Stage 9, every ~20 experiments)
(none yet)

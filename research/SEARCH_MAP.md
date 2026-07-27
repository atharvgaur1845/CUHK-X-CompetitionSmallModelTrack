# Search Map

<!--
Legend: ✓ tested (see LOG) · ✗ tested and eliminated · ? untested · ~ partially explored
-->

**Estimated search space remaining: 74%** (baselines + DG diagnostics done: skeleton=DG-bottlenecked (53 subj / 73 rand), depth=recipe-bottlenecked (24 / 32), IMU weak (27), fusion 54.3 +Hungarian 55.4; submission pair prepared)

```
Data understanding
 ├── Train profiling                                      ✓ (EXP-000b)
 ├── Test profiling (modality availability)               ✓ (EXP-000b: Depth/IR/Skel/IMU ≥404/405; radar 51% empty; thermal -10)
 ├── Test session/timeline structure                      ✓ (EXP-000c: 144 recordings, user-pure, class-DISTINCT within group)
 ├── Test cohort fingerprinting                           ✓ (EXP-000c: 2-way provable E199/L206; 4-way ≈99/100/105/101 moderate)
 ├── Adjacency label prior                                ✗ (EXP-000c: 0/405 same-class neighbors — DEAD)
 ├── Station prior via background matching                ✗ (EXP-000c: recovery ~50% → 0.09 bits — DEAD)
 ├── Depth JET scale (fixed vs per-frame)                 ✓ (EXP-000c: FIXED absolute mapping; scalar depth is metric)
 ├── Label sanity / duplicate detection                   ?
 └── Clip-duration-as-feature                             ~ (7× class spread; test trimmed ~20%, hard-capped 10 s — recalibrate)
Data / preprocessing
 ├── Preprocessing cache (npz per sample)                 ? (Q-00 — required first)
 ├── Skeleton: re-normalization (torso scale, seq-level)  ? (per-frame pelvis-center+floor-align already baked in!)
 ├── Skeleton: multi-person disambiguation policy         ? (pick-first vs largest vs temporal-consistent)
 ├── Skeleton: multi-person-rate as feature               ? (84.6% Comb_hair — discriminative)
 ├── IMU: device pivot + resample + dual-header parsing   ?
 ├── IMU: orientation-invariant features                  ?
 ├── Visual: depth inversion to scalar                    ✓ (in cache; Q-63 proved absolute scale)
 ├── Visual: person-ROI crop via depth foreground         ? (Q-70 — TOP visual lever after EXP-006)
 ├── Visual: resolution / frame sampling / epochs scaling ? (Q-71)
 ├── Thermal: alignment + ironbow decoding                ? (low priority — misfit modality)
 └── Missing-modality handling                            ? (benign in test for big-4)
Augmentation
 ├── Skeleton: bundled 5-aug suite                        ✗ (EXP-001b: −1.6% vs no-aug — component ablation needed)
 ├── Skeleton: per-component aug ablation                 ? (Q-11b/c — mild magnitudes)
 ├── IMU: jitter/scale/rotation/time-warp/channel-drop    ?
 ├── Visual: crop/flip/erase/temporal-crop/RandAugment    ?
 ├── Mixup / CutMix / SDMix                               ?
 └── Modality dropout (fusion robustness)                 ?
Architecture (per modality)
 ├── Skeleton: GRU/TCN                                    ?
 ├── Skeleton: ST-GCN / CTR-GCN-lite (1.5M params)        ?
 ├── Skeleton: multi-stream (joint/bone/motion)           ?
 ├── IMU: 1D-CNN / DeepConvLSTM / transformer             ?
 ├── Visual: 2D-CNN + temporal pool (TSN-style)           ?
 ├── Visual: R(2+1)D / X3D-XS-style factorized 3D         ?
 ├── Visual: dynamic images / depth motion maps           ?
 ├── Radar: any model stream                              ✗ (EXP-000b: 51% empty by cohort, 6.6 pts/frame — B-008)
 ├── Radar: summary stats as aux features                 ? (low priority)
 └── Capacity sweep within 100 MB single-file             ?
Fusion
 ├── Late (logit avg / weighted)                          ?
 ├── Learned gating / attention                           ?
 ├── Mid-level feature concat                             ?
 └── Per-class fusion weights                             ?
Loss
 ├── CE + label smoothing                                 ?
 ├── Class-balanced sampling/loss (12-319 imbalance!)     ?
 ├── Subject-adversarial (DANN)                           ?
 └── Supervised contrastive (paper used it for LOSO)      ?
Training
 ├── Optimizer / LR schedule                              ?
 ├── EMA / SWA                                            ?
 ├── InstanceNorm/GroupNorm vs BatchNorm (LOSO-proven)    ?
 └── Regularization sweep                                 ?
Validation
 ├── Subject-grouped CV handling ragged coverage          ? (Q-01 — foundation)
 └── CV↔LB correlation                                    ? (needs 2+ submissions)
Inference
 ├── Temporal multi-crop TTA                              ?
 ├── Fold/seed ensembling (within single .pth)            ?
 ├── Recording-group Hungarian distinctness assignment    ? (Q-68 — THE metadata lever; simulate on train CV first)
 ├── Per-cohort BN/stat adaptation (2-way E/L split)      ? (Q-54)
 ├── Test-time BN adaptation (TENT)                       ?
 └── Pseudo-label self-training on test                   ? (not in forbidden list; reproduction-risk)
Rules/meta
 ├── ImageNet-init legality ruling                        ? (Q-65 email draft)
 ├── Kaggle submission limit / test balance check         ? (Q-62b)
 └── Efficiency score optimization (10% of final)         ? (later: distill/quantize)
Unknown ideas (literature)
 ├── CUHK-X paper recon                                   ✓ (EXP-000b)
 ├── Compact skeleton/IMU/visual HAR prior art            ✓ (EXP-000b: CTR-GCN, DeepConvLSTM, TSN, Um-aug)
 └── Missing-modality + DG literature                     ✓ (EXP-000b: modality dropout, ShaSpec, SDMix, DANN)
```

## Frontier
1. SUBMIT the pair (Atharv, manual) — calibrates CV↔LB and the Hungarian lever with one Kaggle day.
2. Q-70 person-ROI depth pipeline — the visual recipe fix with the largest expected step (32%→? in-domain).
3. Skeleton DG block: Q-11b aug ablation, Q-10 seq normalization, Q-34 mixup — attacks the measured 20-pt gap.

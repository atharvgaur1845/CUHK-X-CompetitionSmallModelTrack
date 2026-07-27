# Experiment Queue

<!-- Re-ranked 2026-07-27 after EXP-000b profiling. EV/hour = expected gain × P(works) ÷ hours. -->

## Budget ledger

| Tier | Target share | Hours spent | Actual share |
|------|--------------|-------------|--------------|
| exploit | 60% | 0 | — |
| explore | 30% | 0 | — |
| crazy | 10% | 0 | — |

## Active queue (ranked by EV/hour within tier)

### Foundation — ALL DONE (see LOG EXP-000b/c)
| ID | Idea | Outcome |
|----|------|---------|
| Q-60 | Timeline merge audit | DONE: test clean; adjacency prior DEAD (0/405); 144 recording groups found |
| Q-61 | Test cohort clustering | DONE: 2-way provable (E199/L206); 4-way ≈99/100/105/101; test_cohorts.csv |
| Q-62 | Census fixes | DONE: test median 1.9 s, 10 s hard cap, ~20% trimmed; date_map.csv |
| Q-00 | Preprocessing cache | DONE: 2933+405 npz, 1.4 GB |
| Q-01 | Subject CV harness | DONE: cv_folds.json, 4 folds, 39-40/40 coverage |
| Q-63 | Depth JET scale probe | DONE: FIXED absolute mapping — scalar depth is metric |
| Q-64 | Station prior | DEAD: background matching ~50% → 0.09 bits effective |
| Q-65 | Organizer email draft | DONE: research/organizer_email_draft.md awaiting send |

### New this round
| ID | Tier | Idea | Hypothesis (one line) | Beliefs tested | Expected gain | P(works) | Hours | Status |
|----|------|------|-----------------------|----------------|---------------|----------|-------|--------|
| Q-68 | exploit | Recording-group Hungarian assignment | DONE (EXP-004/005): +0.6% skel, +1.1% on fusion, NEGATIVE on weak streams — apply to strong fused probs only | B-007, B-011 | measured | — | — | done |
| Q-69 | exploit | Group-key extraction in inference code | DONE: predict_test.py radar-ts + camera-anchor fallback; found exactly 115 multi-groups on test | B-011 | enabler | — | — | done |
| Q-70 | exploit | Person-ROI crop via depth foreground | Cropping a person-centered ROI (depth blob nearest/largest) at ~112-128px before the CNN lifts depth random-split 32%→55%+ and subject-CV proportionally | B-006 | +10% on depth stream | 0.6 | 4 | queued |
| Q-71 | exploit | Depth recipe scale-up | Longer (100+ ep), wider CNN, 12-16 frames, 160×214 res each add; combined with Q-70 targets depth ≥45% subject-CV | B-006 | +8% on depth | 0.6 | 4 | queued |
| Q-11b | exploit | Skeleton aug component ablation | One-at-a-time (yaw, scale, jitter, joint-drop, temporal-jitter) identifies which component(s) hurt EXP-001b | B-001 | +2% | 0.7 | 2 | queued |
| Q-11c | exploit | Mild skeleton aug | yaw ±10°, scale ±7%, no joint-drop recovers aug benefit | B-001 | +2% | 0.5 | 1 | queued |

### Exploit (60%)
| ID | Tier | Idea | Hypothesis (one line) | Beliefs tested | Expected gain | P(works) | Hours | Status |
|----|------|------|-----------------------|----------------|---------------|----------|-------|--------|
| Q-02 | exploit | Skeleton TCN/GRU baseline | ≥45% subject-CV from normalized 17×3 @10 Hz sequences | B-002 | baseline | 0.8 | 3 | queued |
| Q-04 | exploit | Depth-scalar CNN baseline (TSN-style, 8 frames, small 2D-CNN + temporal pool) | ≥50% subject-CV from scratch | B-006 | baseline | 0.7 | 4 | queued |
| Q-03 | exploit | IMU 1D-CNN baseline | ≥35% subject-CV from 5-device 10 Hz tensor | B-009 | baseline | 0.7 | 3 | queued |
| Q-05 | exploit | IR CNN baseline | IR ≈ Depth accuracy; check subject-identity overfit via CV gap | B-006 | baseline | 0.6 | 2 | queued |
| Q-11 | exploit | Skeleton augmentation suite | rotate/scale/shear/joint-drop/time-warp +3-8% | B-001 | +5% | 0.8 | 2 | queued |
| Q-12 | exploit | IMU augmentation suite (Um et al.: 3D rotation critical) | +3-6% | B-001 | +4% | 0.7 | 2 | queued |
| Q-14 | exploit | Visual augmentation suite (crop/flip/erase/RandAugment-lite) | +3-6% on depth CNN | B-001, B-006 | +4% | 0.7 | 2 | queued |
| Q-13 | exploit | Class-balanced sampling + label smoothing + EMA | +2-4% given 27× imbalance | B-010 | +3% | 0.7 | 1 | queued |
| Q-20 | exploit | Late fusion big-4 (skel+depth+IR+IMU) | ≥ +5% over best single | B-003 | +6% | 0.7 | 2 | queued |
| Q-21 | exploit | Modality dropout in fusion training | robustness + regularization +1-2% | B-003 | +1.5% | 0.7 | 1 | queued |
| Q-17 | exploit | Temporal multi-crop TTA | +1-2% | — | +1.5% | 0.8 | 1 | queued |
| Q-18 | exploit | Fold/seed weight-space or logit ensembling within one .pth | +1-3% | B-005 | +2% | 0.8 | 1 | queued |
| Q-15 | exploit | GroupNorm/InstanceNorm vs BatchNorm | GN beats BN cross-subject +1-3% | B-001 | +2% | 0.6 | 1 | queued |
| Q-10 | exploit | Skeleton seq-level re-normalization (torso-scale; velocity channels to recover motion cues) | +2-5% (esp. classes hurt by per-frame floor-align) | B-002 | +3% | 0.6 | 2 | queued |
| Q-28 | exploit | Capacity sweep of winning arch | +1-2% | B-005 | +1.5% | 0.6 | 3 | queued |

### Explore (30%)
| ID | Tier | Idea | Hypothesis (one line) | Beliefs tested | Expected gain | P(works) | Hours | Status |
|----|------|------|-----------------------|----------------|---------------|----------|-------|--------|
| Q-66 | — | ~~Session same-class smoothing~~ | KILLED by EXP-000c: recordings are class-DISTINCT, not class-pure — pooling would be systematically harmful; superseded by Q-68 Hungarian | B-011 | — | — | — | dead |
| Q-67 | — | ~~Session-adjacency prior~~ | KILLED by EXP-000c: 0/405 same-class nearest train neighbors | B-007 | — | — | — | dead |
| Q-30 | explore | CTR-GCN-lite skeleton (vs TCN) | graph conv +2-4% over Q-02 | B-002 | +3% | 0.6 | 4 | queued |
| Q-30b | explore | Skeleton multi-stream (joint/bone/motion) ensemble | +2-4% (NTU-proven) | B-002 | +3% | 0.7 | 3 | queued |
| Q-34 | explore | Mixup / SDMix per modality | +1-3% cross-subject | B-001 | +2% | 0.5 | 2 | queued |
| Q-32 | explore | Subject-adversarial DANN head | +2-4% cross-subject | B-001 | +3% | 0.4 | 4 | queued |
| Q-35 | explore | SupCon pretrain (paper's own LOSO trick) | +1-3% | B-001 | +2% | 0.5 | 4 | queued |
| Q-31 | explore | Cross-modal attention mid-fusion | +2% over late fusion | B-003 | +2% | 0.4 | 5 | queued |
| Q-38 | explore | Per-class fusion weights | +1-2% | B-003 | +1.5% | 0.5 | 2 | queued |
| Q-33 | explore | Attention temporal pooling | +1-2% | A4' | +1.5% | 0.5 | 2 | queued |
| Q-37 | explore | Duration + multi-person-rate + station-prior aux features | +1-2% (recalibrate for trimmed test) | B-007, B-010 | +1.5% | 0.5 | 2 | queued |
| Q-39 | explore | Confusable-cluster specialist heads | +1-2% | — | +1.5% | 0.4 | 3 | queued |
| Q-06 | explore | Thermal CNN stream (ironbow hue, ~25 fps unsynced) | worth a 4th stream? | B-006 | +1% | 0.4 | 3 | queued |

### Crazy (10%)
| ID | Tier | Idea | Hypothesis (one line) | Beliefs tested | Expected gain | P(works) | Hours | Status |
|----|------|------|-----------------------|----------------|---------------|----------|-------|--------|
| Q-50 | crazy | Test-time self-training (pseudo-label confident test clips) | +2-5% LB; not in forbidden list; must survive reproduction | B-007 | +3% | 0.4 | 3 | queued |
| Q-54 | crazy | Per-cohort BN/statistics adaptation (TENT-style, using Q-61 cohorts) | +1-3% | B-001, B-007 | +2% | 0.4 | 2 | queued |
| Q-55 | crazy | Skeleton limb-length retargeting (synthetic subjects) | +1-3% cross-subject | B-001 | +2% | 0.3 | 3 | queued |
| Q-56 | crazy | Person-silhouette stream from depth (segment person by depth band, model silhouette dynamics) | subject-invariant shape signal +1-3% | B-006 | +2% | 0.3 | 4 | queued |
| Q-57 | crazy | Knowledge distillation of full fusion into one compact student (efficiency score play) | keeps accuracy, wins efficiency 10% | B-005 | finals-only | 0.5 | 4 | queued |

## Done (moved after LOG entry written)

| ID | Idea | Expected gain | Actual gain | LOG ref |
|----|------|---------------|-------------|---------|
| — | Profiling workflow (8 agents) | — | strategy reshaped | EXP-000b |

## Parked (blocked / needs compute / needs data)

| ID | Idea | Blocked on |
|----|------|-----------|
| Q-36 | Radar aux features | B-008 says defer; revisit only if fusion plateaus |
| Q-52 | Depth colormap inversion | folded into Q-00 cache design (decided: invert) |

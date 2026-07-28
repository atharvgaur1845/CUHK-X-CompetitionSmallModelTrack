# Experiment Queue

<!-- REWRITTEN after DA-001 (2026-07-28). Metric = MACRO OOF accuracy (B-004/B-012).
Governing belief: B-013 — stream quality is the campaign; oracle of current streams = 63.4%.
Target >0.85 requires: strong visual stream + upgraded skeleton + calibrated fusion + transduction endgame. -->

## Budget ledger

| Tier | Target share | Hours spent (est) | Actual share |
|------|--------------|-------------------|--------------|
| exploit | 60% | ~10 | 77% |
| explore | 30% | ~2 | 15% |
| crazy | 10% | ~1 | 8% |

## BLOCKED ON ATHARV
| ID | Action | Why |
|----|--------|-----|
| SUB-003/004 | Submit sub_fuse3_prioradj.csv then sub_fuse3_sinkhorn.csv | Tests B-012 (balanced test) — highest-EV pending measurement |
| Q-65b | SEND the organizer email (draft ready) | Legality of pretrained init gates the visual strategy |

## Active queue (ranked, macro-metric era)

### Exploit — stream quality (60%)
| ID | Idea | Hypothesis | Beliefs | Exp gain (macro) | P | Hrs | Status |
|----|------|-----------|---------|------------------|---|-----|--------|
| Q-80 | IR-ROI stream (EXP-009) | Paper's best modality + person crop ≥35% macro | B-013, B-006 | new stream | 0.6 | — | RUNNING |
| Q-81 | Depth-ROI stream (EXP-010) | ROI fixes tiny-person → ≥30% macro (vs 17.7 full-frame) | B-013, B-006 | +12 on stream | 0.6 | — | RUNNING |
| Q-82 | Balanced skeleton reference (EXP-008) | balanced sampling + macro selection ≥47% macro | B-012 | +1-2 | 0.7 | — | RUNNING |
| Q-83 | Skeleton jvb adoption + capacity/epoch scale-up | bones (+1.8 measured) + width256/depth6 + 150ep + balanced ≥50% macro | B-013 | +3-4 | 0.7 | 3 | queued |
| Q-84 | Pretrained ResNet-18 probe on IR-ROI (DIAGNOSTIC ONLY until ruling) | ImageNet init random-split ≥70% ⇒ representation is the gap, from-scratch recipe must mimic transfer | B-006, B-013 | information | 0.7 | 2 | queued |
| Q-85 | Label-smoothing ablation (0 vs 0.1) | LS=0 improves macro + calibration for fusion/assignment | DA-001 #3 | +1 | 0.5 | 1 | queued |
| Q-86 | Person-selection policy (max-motion vs person-0) | fixes mirror-person in bathroom classes (+Comb_hair/Brush_teeth) | DA-001 #4 | +1-2 | 0.6 | 2 | queued |
| Q-87 | No-stretch temporal (pad+mask) + duration input + valid-mask channel | removes duration erasure + velocity-unit corruption + invalid/near conflation | DA-001 #5/#7 | +1-2 | 0.5 | 3 | queued |
| Q-88 | Temporal multi-crop TTA + nested fusion weights | honest fusion + eval; removes triple-dipping | DA-001 | +1 honest | 0.8 | 2 | queued |
| Q-89 | Visual capacity/epochs scale-up (width 48-64, 100ep, 12 frames) | visual streams are under-trained | B-013 | +3-5 on stream | 0.6 | 4 | queued |
| Q-28b | Skeleton capacity sweep (0.66M → 5-10M) | 0.2% of budget used; capacity is free | B-013 | +2 | 0.5 | 3 | queued |

### Explore — transduction + DG (30%)
| ID | Idea | Hypothesis | Beliefs | Exp gain | P | Hrs | Status |
|----|------|-----------|---------|----------|---|-----|--------|
| Q-90 | Global balanced assignment (Sinkhorn) at inference | if B-012 true, matching test marginal to uniform adds +2-5 | B-012 | +3 | 0.6 | done-file | awaiting LB |
| Q-91 | Per-user (cohort) self-training | pseudo-label confident clips per cohort, retrain; 4 users × ~100 clips | B-007, B-013 | +3-6 | 0.5 | 4 | queued |
| Q-92 | Per-cohort feature alignment (mean/var matching, GN-compatible) | reduce subject shift at test time without BN | B-001 | +2 | 0.4 | 3 | queued |
| Q-34 | Mixup (skeleton + visual) | +1-3 macro cross-subject | B-001 | +2 | 0.5 | 2 | queued |
| Q-32 | Subject-adversarial DANN head | +2-4 macro | B-001 | +2.5 | 0.4 | 4 | queued |
| Q-30 | ST-GCN-lite skeleton | graph conv beats TCN | B-002 | +2 | 0.5 | 4 | queued |
| Q-93 | R(2+1)D-lite / temporal conv visual (vs frame-pool) | motion modeling beats mean-pool on ROI clips | B-006 | +3 on stream | 0.5 | 4 | queued |
| Q-35 | SupCon pretrain | +1-3 | B-001 | +2 | 0.4 | 4 | queued |

### Crazy (10%)
| ID | Idea | Hypothesis | Beliefs | Exp gain | P | Hrs | Status |
|----|------|-----------|---------|----------|---|-----|--------|
| Q-94 | Self-supervised pretrain on ALL clips (train+test, no labels) — masked reconstruction / temporal order | legal transfer without external weights; closes the init gap | B-013 | +5 | 0.35 | 8 | queued |
| Q-95 | Cross-modal distillation (skeleton teacher → visual student on shared clips) | visual student learns pose-invariances it can't find alone | B-013 | +3 | 0.3 | 5 | queued |
| Q-55 | Skeleton limb-length retargeting aug | synthetic subjects | B-001 | +1.5 | 0.3 | 3 | queued |
| Q-96 | Hungarian/distinctness — REVISIT only after distinctness-on-test verified and base ≥55 macro | H2 test from DA Q5 (accuracy-matched null) | B-011 | +1 | 0.4 | 2 | parked |

## Dead (do not resurrect)
| ID | Idea | Killed by |
|----|------|-----------|
| Q-66 | Same-class group pooling | EXP-000c: groups are class-DISTINCT |
| Q-67 | Adjacency prior | EXP-000c: 0/405 |
| Q-64 | Station prior via backgrounds | EXP-000c: 50% recovery → 0.09 bits |
| Q-54 | BN adaptation | DA-001: models use GroupNorm — no BN exists (superseded by Q-92) |
| Q-11-family | Geometric skeleton aug at v1 magnitudes | EXP-007: every component hurts; rot worst (−2.0) |
| — | jv+tn torso-norm | EXP-007b: 44.1 vs 44.9 reference |

## Done
| ID | Idea | Outcome | LOG |
|----|------|---------|-----|
| Q-00/01/60-65 | Foundation + audits | see LOG EXP-000b/c | |
| Q-02/03/04/05 partial | Baselines skel/imu/depth | 53.0/26.8/26.7 micro | EXP-001..003 |
| Q-68/69 | Hungarian + group keys | +1.1 CV, −1.0 LB → parked as Q-96 | EXP-004/005, SUB-001 |
| Q-70/71 partial | ROI cache built | 2910+405 crops | — |
| Q-11b | Aug component ablation | all negative | EXP-007 |
| — | jvb bones | +1.8 macro | EXP-007b |

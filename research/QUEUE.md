# Experiment Queue

<!-- Current queue v2: 2026-07-30. Primary metric = paired MICRO subject-OOF;
LEADERBOARD.md owns external scores. The historical 2026-07-28 queue is
preserved below for decision provenance but is superseded. -->

## Current decision gates

1. **DA-005 and RESET-002 completed before EXP-047:** counters reset to 1/10
   and 1/25. The new wave must follow the audited sequence→all-user
   validation→visual/reranking order.
2. A member is adopted only if it beats an exact scalar-weight control with a
   material paired OOF delta or replicated evidence.
3. Every candidate logs model bytes, four-fold inference bytes, test argmax
   changes, and a deterministic manifest.
4. Every candidate must fit into a verified ≤100 MB serialized artifact and a
   defensible live-weight interpretation. World25 is 85.218 MB and now streams
   one member at a time with at most 10.135 MB persistent fp32 parameters;
   preserve exact organizer wording for transient accounting.
5. Public changes of one clip are directional evidence only; unsupported scores
   remain unverified in `LEADERBOARD.md`.

## Current queue (ranked)

### P0 — legal champion and evidence hygiene

| ID | Action | Success gate | Why now | Status |
|---|---|---|---|---|
| P-02 | Legal architecture-diverse champion package | ≤100 MB verified artifact; preserve full accepted inventory | world25 int8 is 85,217,859 bytes and its exact output scored 0.54228 | **done** |
| P-03 | Package audit from one `model.pth` with packaged inference | byte count, payload verification, and all-405 argmax audit | `a20`: 82.696 MB/1 change; world25: 85.218 MB/3 changes | **done for both candidates** |
| P-11 | Stream packaged members one at a time | bit-exact probabilities; no all-member fp32 residency | 338.0 MB → 10.1 MB persistent fp32 params; exact parity | **done** |
| P-07 | Measure exact legal world25 output | verified Kaggle score bound to exact CSV hash | `sub_astgcn_world25_int8.csv` = 0.54228 = 109/201 | **done; new champion** |
| P-08 | Upload ordered-transition candidate | score exact SHA `ed678466…e346`; compare with 109/201 base | scored 0.55721 = 112/201 | **done; champion** |
| P-10 | Upload repeat-consensus candidate after P-08 | score exact SHA `1a1dbda…c377`; compare with transition-only | scored 0.55223 = 111/201 | **done; marginal rejected** |
| P-09 | All-user outer protocol | every user and clip held out exactly once; outer fold scored once | `cv_folds_all18.json` covers 18/18 users and 2933/2933 clips | **protocol done; fold-2 matched baseline chained (EXP-050); full 4-fold rerun deferred to adoption time** |
| P-04 | Validated full-data training protocol using fold-derived best-epoch distribution, not last epoch | simulated fold refit does not repeat SUB-007 harm | could improve efficiency even though int8 already clears the cap | queued |
| P-05 | Reconcile the historical `sub_block10=0.52736` claim and harvest ranks 10/15/20/30 | screenshot or mark permanently unsupported; measured Top-15 bar | objective and strategy cannot rely on an unverified score/bar | blocked on Atharv |
| P-06 | Reproducible cache→train→package→infer command + README/manifests | clean rerun, exact CSV hash, no hard-coded machine path | packager/inference exist; clean-room orchestration/report remain | parallel |

### P1 — new information, not more soup

| ID | Experiment | Hypothesis | Adoption gate | Status |
|---|---|---|---|---|
| X-01b | Orthogonal graph view only with a new premise | Adaptive ST-GCN proved one graph structure transfers, while CTR-GCN and dual-frame views failed their screens | must beat the accepted Adaptive control on fold 0 before replication; no adaptive/CTR soup | parked-low |
| X-02 | Nested low-parameter pair specialists for Read↔Turn pages, Pour↔Stir, Sweep↔Mop, Jog↔Walk/Lunge | targeted boundaries recover hard mass without global prior distortion | nested gain on held-out users; ≤2 parameters or a separately validated tiny head per pair | queued |
| X-03 | Materially different competition-data pretext/teacher objective | legal representation learning may unlock absent information, but transformer SSL and cross-user SupCon both failed final gates | one-fold gain ≥2 points before any full sweep | queued-low after reset |
| X-04 | Early-cohort radar summary/stream screen | radar adds a genuinely new view on the 199 usable test clips | paired E-cohort simulation gain and graceful empty-radar fallback | queued-low |
| X-05 | Class-conditional shift audit for classes 16/28/35 and object classes | zero test argmaxes reveal a correctable representation/calibration failure | diagnostic only; no forced test marginal | queued |
| X-06 | Repeated-recording probability consensus | repeated passes of one routine share label order; pool only metadata/probability-matched recordings | nested OOF gain beyond transition-only and deployable test clustering | **closed: −1 public clip vs transition** |
| X-07 | Thermal/IR/Depth object branch reset | 90.7% of current errors are object/context-driven; current ROI extraction loses object resolution | all-user outer-once gain, explicit missing masks, and positive fused marginal on ≥2 hard folds | **cache implementation running** |
| X-08 | Guarded exact recording templates | exact routines may correct Markov paths when source coverage is strong | must beat repeat+Markov under strict nested selection | parked: fixed 71.741, nested 70.333 |

### P2 — compactness research

| ID | Experiment | Gate | Status |
|---|---|---|---|
| C-02 | Logit/feature distillation from oversized block into one MultiTCN+graph student | compact student beats plain MultiTCN by ≥1 paired OOF point | queued |
| C-03 | Quantization sensitivity by member | serialized total <100 MB and test argmax parity measured | done at ensemble level: `a20` 1/405 change; world25 3/405 |
| C-04 | Pareto search over accuracy, diversity, and bytes | report complete frontier, not a single OOF-maximized subset | queued |

## Newly closed (2026-07-30)

| ID | Result | Decision |
|---|---|---|
| P-02/P-03 int8 packaging | `a20` 82.696 MB/44 members/1 test change; world25 85.218 MB/48 members/3 changes; payload audits pass | legal artifact gate passed; do not infer LB scores |
| EXP-040 Adaptive ST-GCN | 59.56 solo; assembly +1.00 OOF on all folds; LB +3 clips | adopted core family; full stack now legally int8-packaged |
| EXP-041 Identity | fold0 56.08 vs 59.20 | rejected |
| EXP-042 cross-user SupCon | fold0 56.38 vs 59.20 | rejected; other pretexts require a new premise |
| EXP-043 CTR-GCN | fold0 58.16 vs Adaptive 58.90; fold2 stopped at 52.29 vs 56.57 | rejected |
| EXP-044 dual-frame skeleton | fold0 57.27 vs Adaptive 58.90 under outer-once protocol | rejected |
| EXP-045 IR motion maps | solo 44.36/37.67 on folds 0/2; 10% fusion +0.30/−0.89 | fusion member rejected; cache/solo finding retained |
| EXP-046 world-frame IMU | solo 32.07 vs 29.63; +0.48 nested stack marginal; fixed 25% positive all folds; legal int8 LB 0.54228 | adopted; +1 verified public clip |
| EXP-047 ordered transitions | 61.778→66.963 strict nested OOF; fixed λ=.5 67.815; public 0.55721 = 112/201 | adopted champion; +3 public clips |
| EXP-048 repetition consensus | 61.778→70.667 strict nested OOF; fixed 70.889; public 0.55223 = 111/201 | reject marginal; −1 public clip vs EXP-047 |
| EXP-049 exact templates | fixed guarded hybrid 71.741, but strict nested selection 70.333 vs 70.889 fallback | implementation retained; no CSV |
| EXP-034 dynamic truncation | solo +0.59, block marginal +0.04 | loader fix adopted; extra member rejected |
| EXP-035 MultiTCN | 56.60 solo; +0.59 over weight control; LB +1 clip | representation adopted and included in legal full-stack package |
| EXP-036 pad64 mask TCN | 55.34, flat vs dynamic TCN | rejected |
| EXP-037 DANN | −2.97/−4.01 on paired fold 0 | rejected |
| EXP-038 mixup α=0.2 | +0.15 across folds 0/2 with split signs | inconclusive; rejected by gate |
| EXP-039 BiGRU | 53.34; TTA further negative | rejected |
| blanket temporal-jitter TTA | negative for base TCN, MultiTCN, BiGRU, and slightly negative for Adaptive ST-GCN; negligible elsewhere | use only after held-out per-member gate |
| same-family soup expansion | block12 tied block8; added members flat/negative fixed OOF | stop unless exact marginal gate passes |

---

## Historical queue (2026-07-28; superseded)

The tables below are retained to show what was believed and queued at the time.
Statuses and macro-metric language are not current.

### Historical budget ledger

| Tier | Target share | Hours spent (est) | Actual share |
|------|--------------|-------------------|--------------|
| exploit | 60% | ~10 | 77% |
| explore | 30% | ~2 | 15% |
| crazy | 10% | ~1 | 8% |

### Historical blocked items (resolved or superseded)
| ID | Action | Why |
|----|--------|-----|
| SUB-003/004 | Submit sub_fuse3_prioradj.csv then sub_fuse3_sinkhorn.csv | Tests B-012 (balanced test) — highest-EV pending measurement |
| Q-65b | SEND the organizer email (draft ready) | Legality of pretrained init gates the visual strategy |

### Historical active queue (macro-metric era)

#### Exploit — stream quality (60%)
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

#### Explore — transduction + DG (30%)
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

#### Crazy (10%)
| ID | Idea | Hypothesis | Beliefs | Exp gain | P | Hrs | Status |
|----|------|-----------|---------|----------|---|-----|--------|
| Q-94 | Self-supervised pretrain on ALL clips (train+test, no labels) — masked reconstruction / temporal order | legal transfer without external weights; closes the init gap | B-013 | +5 | 0.35 | 8 | queued |
| Q-95 | Cross-modal distillation (skeleton teacher → visual student on shared clips) | visual student learns pose-invariances it can't find alone | B-013 | +3 | 0.3 | 5 | queued |
| Q-55 | Skeleton limb-length retargeting aug | synthetic subjects | B-001 | +1.5 | 0.3 | 3 | queued |
| Q-96 | Hungarian/distinctness — REVISIT only after distinctness-on-test verified and base ≥55 macro | H2 test from DA Q5 (accuracy-matched null) | B-011 | +1 | 0.4 | 2 | parked |

### Historical dead list (do not resurrect)
| ID | Idea | Killed by |
|----|------|-----------|
| Q-66 | Same-class group pooling | EXP-000c: groups are class-DISTINCT |
| Q-67 | Adjacency prior | EXP-000c: 0/405 |
| Q-64 | Station prior via backgrounds | EXP-000c: 50% recovery → 0.09 bits |
| Q-54 | BN adaptation | DA-001: models use GroupNorm — no BN exists (superseded by Q-92) |
| Q-11-family | Geometric skeleton aug at v1 magnitudes | EXP-007: every component hurts; rot worst (−2.0) |
| — | jv+tn torso-norm | EXP-007b: 44.1 vs 44.9 reference |

### Historical done list
| ID | Idea | Outcome | LOG |
|----|------|---------|-----|
| Q-00/01/60-65 | Foundation + audits | see LOG EXP-000b/c | |
| Q-02/03/04/05 partial | Baselines skel/imu/depth | 53.0/26.8/26.7 micro | EXP-001..003 |
| Q-68/69 | Hungarian + group keys | +1.1 CV, −1.0 LB → parked as Q-96 | EXP-004/005, SUB-001 |
| Q-70/71 partial | ROI cache built | 2910+405 crops | — |
| Q-11b | Aug component ablation | all negative | EXP-007 |
| — | jvb bones | +1.8 macro | EXP-007b |

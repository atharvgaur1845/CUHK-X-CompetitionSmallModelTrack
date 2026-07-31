# Mental Model

<!-- Written BEFORE any model code. Revised at every Stage 9 meta-analysis — keep old versions below, don't overwrite. -->

## Current model (v8 — 2026-07-30, after EXP-047/048 leaderboard results)

1. The verified champion is transition-only:
   `sub_astgcn_world25_int8_trans05.csv` = **0.55721 = 112/201**, three clips
   above the exact package base.
2. Repetition consensus is rejected as a marginal. It scored
   **0.55223 = 111/201**, one clip below transition-only despite reaching
   70.667% strict nested OOF. This is direct evidence that local metadata gains
   can overstate hidden-user transfer.
3. Exact templates already failed their strict nested gate and must not be
   uploaded. Sequence postprocessing is closed unless a new mechanism has
   independent evidence.
4. A score above 0.83 still requires 167/201, another **55 public clips**.
   Postprocessing produced three; the remaining problem is representation.
5. The dominant actionable bottleneck is object/context appearance: loose
   crops, low-resolution caches, motion summaries that discard appearance, and
   a nearly unused Thermal modality.
6. The next experiment is the frozen all-user high-resolution
   IR/depth/Thermal candidate-conditioned visual MIL screen. Fold 2 is the
   untouched hard gate; if it fails, stop rather than tune on it.
7. Deployment is materially hardened: the 85.218 MB package now streams one
   member at a time, reducing persistent fp32 model parameters from 338.0 MB to
   at most 10.1 MB with bit-exact probability parity. The exact organizer reply
   is still needed for transient-dequantization accounting.

**Operating order:** finish and validate the high-resolution cache; run the
predeclared all-user fold-2 visual screen; replicate on fold 0 only if it
passes; generate another upload only after paired visual+sequence evidence.

## Previous model (v7 — 2026-07-30, post-EXP-048/049; superseded by v8)

**The evidence-backed picture now:**

1. The verified public best is still **0.54228 = 109/201**. No sequence-aware
   leaderboard score has been observed yet.
2. Directional transitions remain the clean first upload: strict nested OOF
   reaches 66.963% and the fixed tie-safe candidate reaches 67.815%.
3. Deployable repeated-recording consensus is the stronger local mechanism.
   Consensus plus transitions reaches **70.667% under strict nested selection**
   (70.889% fixed), with every fold positive. Its exact CSV changes 104/405
   rows versus the scored base and 41 rows versus transition-only.
4. Exact sequence templates are not robust. A fixed guarded hybrid reached
   71.741%, but strict nested selection fell to 70.333%, below the
   repeat+Markov fallback. It is not an upload candidate.
5. The remaining model bottleneck is object appearance. Current IR motion maps
   score about 27.1% on object/context clips in their audited folds; 34% of
   train and 43% of test ROI crops span full image height, and summary maps
   destroy instantaneous appearance. Thermal is almost fully available but
   unused.
6. The next representation must retain high-resolution frames, absolute depth,
   validity, and Thermal, then rerank a strong model-derived candidate set.
   The accepted-member top-two union oracle is about 89.41%, which makes
   candidate-conditioned visual evidence a plausible regime change.
7. Historical 2700-row OOF remains descriptive because it omits users 5 and
   21 and reuses selected base checkpoints. New visual work must use
   `cv_folds_all18.json`, a frozen epoch recipe, and outer-once scoring.
8. The serialized package is 85.218 MB but the current all-resident loader
   expands fp32 member weights to 338.14 MB. Streamed loading or a lazy compact
   artifact remains a separate compliance task.

**Operating order:**

- Upload EXP-047 and bind the score to SHA-256 `ed678466…e346`.
- Only then upload EXP-048 and bind it to SHA-256 `1a1dbda…c377`.
- Complete the deterministic high-resolution visual cache, then hard-screen
  the frozen compact visual MIL recipe on all-user fold 2 and replicate on
  fold 0 only if the predeclared gates pass.
- Do not upload the template hybrid or resume same-family skeleton/IMU soups.
- Preserve the exact organizer reply and resolve serialized-versus-live size.

## Previous model (v6 — 2026-07-30, post-RESET-002 and EXP-047; superseded by v7)

**The evidence-backed picture now:**

1. The verified public best remains **0.54228 = 109/201**. The next exact
   candidate is `sub_astgcn_world25_int8_trans05.csv`, not a new backbone.
2. Ordered recording context is the first regime-scale local gain:
   complementary-user transitions raise world25 OOF
   **61.778% → 66.963%** under strict nested transition-label selection
   (67.815% at fixed `lambda=0.5`), with all four folds positive and
   reversed/shuffled controls negative. The tie-safe candidate changes 93/405
   exact-package predictions and is pending leaderboard measurement.
3. This does not rehabilitate generic metadata forcing. Hungarian
   distinctness, Sinkhorn, and class-prior adjustment remain public failures.
   Directional action order is the supported mechanism.
4. The dominant remaining model failure is missing object/context evidence:
   gross-motion accuracy is 88.34%, object/context accuracy 50.13%, and
   object/context classes contribute 90.7% of OOF errors. Skeleton+IMU cannot
   be the only final information source.
5. Current top-1 predictor outputs have a label-aware oracle of only
   77.6--82.0%, below the target. Their any-member top-2 oracle is 89.41%, so
   sequence and visual context should be used as a candidate-conditioned
   reranker rather than another scalar ensemble weight.
6. Historical validation is descriptive, not unbiased. It omits two users,
   covers 2700/2933 clips, and outer-fold checkpoint selection contributes
   about 1.2--1.3 optimistic points. The new all-user protocol covers
   18/18 users and 2933/2933 clips; future models use fixed recipes and
   outer-once scoring.
7. Serialized size and live weights are separate gates. World25 is 85.218 MB
   on disk but expands to 338.14 MB in the current loader. Preserve the package
   while building an interpretation-independent streamed/quantized/distilled
   path.
8. A score above 0.83 still requires 58 additional public clips. EXP-047 is a
   plausible step, not proof of target completion.

**Operating order:**

- Upload the exact EXP-047 candidate and bind its score to its SHA-256.
- Finish the deployable repeated-recording consensus gate without spending a
  submission unless it adds nested OOF beyond transition-only.
- Use `cv_folds_all18.json` for the compact Thermal/IR/Depth object reset.
- Fuse visual evidence as a top-2/context reranker; do not resume broad
  skeleton/IMU soup searches.
- Preserve the exact organizer reply and resolve serialized-versus-live model
  size before Selection Stage.

**Explicitly closed:**

- scalar reweighting as a path to 0.83;
- treating hard distinctness and directional order as the same hypothesis;
- claims that the current package is unconditionally compliant;
- sub-point adoption decisions from the historical selected/incomplete OOF.

## Previous model (v5 — 2026-07-30, post-EXP-046 + serialized packaging; superseded by v6)

**The evidence-backed picture now:**

1. The verified public best is **0.54228 = 109/201**, from the exact legal
   world25 int8 package. The
   `sub_block10 = 0.52736` line is unverified and excluded. `LEADERBOARD.md` is
   authoritative.
2. The useful verified ladder is `97 → 98 → 103 → 104 → 105 → 108 → 109`. Invariant IMU
   supplied cheap diversity; the first ST-GCN block supplied the largest step;
   MultiTCN supplied one more clip; Adaptive ST-GCN supplied three. Adding more
   same-family seeds/widths and broad TTA has saturated.
3. Adaptive ST-GCN is the best single skeleton stream at 59.56% subject-OOF.
   It raises the controlled assembly by +1.00 point with all four fold deltas
   positive, and its three-clip LB gain is the strongest recent transfer
   evidence.
4. Physically grounded quaternion normalization is the first post-Adaptive
   cross-modal gain: world-frame IMU improved solo 29.63% → 32.07% and added
   +0.48 point under leave-one-fold-out fusion. The fixed 25% candidate is
   positive on all folds. Its legal int8 output scored 0.54228, adding one
   public correct clip.
5. Public resolution is extremely coarse: one clip is 0.4975 point. Equal
   scores can conceal different private predictions; small LB deltas cannot
   substitute for paired OOF evidence.
6. The newest different-view screens set useful boundaries. CTR-GCN and
   dual-frame skeleton failed same-fold Adaptive controls. IR motion maps
   reached 44.36%/37.67% solo on folds 0/2, but their 10% fusion marginal
   flipped from +0.30 to −0.89 point; stronger solo accuracy is not enough.
7. A score above 0.83 requires 167/201, another 58 public clips. No measured
   soup-scale lever has arithmetic remotely near that gap; only new information
   or representation can change the regime.
8. **The current model-size gate is solved, with a parity caveat.** The exact
   fp32 `a20` inventory is oversized, but deterministic int8 packaging stores
   all 44 members in 82,696,132 bytes. Its packaged CSV differs from the
   leaderboard-verified fp32 CSV on one argmax and is unscored. The 48-member
   world candidate also fits at 85,217,859 bytes and changes three argmaxes
   under quantization.

**Current operating plan:**

- **P0 — audit the remaining gap:** world25 int8 is now verified at 0.54228.
  Use its one-clip transfer to calibrate new mechanisms and do not transfer
  that score to the different fp32 CSV.
- **P1 — preserve the legal champion:** retain the verified `a20` fp32 CSV and
  its 82.696 MB int8 package together. Keep exact manifests, hashes, and the
  one-row quantization difference explicit.
- **P2 — seek new information:** prioritize legal competition-data
  representation learning, object-aware visual features, or another physically
  grounded sensor view. Same-family seed/width/graph expansion is no longer
  exploration.
- **P3 — class-specific diagnosis:** study hard pairs and classes absent from
  test argmaxes with nested specialists or diagnostics; never force global
  uniform priors.
- **Parallel — reproducibility:** the deterministic packager and packaged
  inference path exist; finish clean-room cache→train→package→infer commands,
  README, and report while preserving exact artifact/CSV hashes.

**Closed or parked unless new evidence changes the premise:**

- uniform prior adjustment, Sinkhorn, and current Hungarian assignment;
- naive pseudo-label self-training and TENT-GN;
- learned logit-gate/transformer fusion at this data scale;
- current ROI/upper-body visual CNN recipes and the unreplicated IR motion-map
  fusion member;
- pad64 masked TCN, tested DANN, mixup α=0.2, BiGRU, Identity, and the current
  cross-user SupCon recipe;
- the tested CTR-GCN and dual-frame skeleton implementations;
- last-epoch full-data refit and unvalidated blanket temporal TTA;
- additional same-family seeds/widths without a controlled marginal gain.

**Assumptions still standing:**

- A18. Architecture diversity can be quantized below 100 MB. This is now
  measured: full `a20` is 82.696 MB and differs on 1/405 test argmaxes. The
  remaining uncertainty is the unknown label of that row and end-to-end
  clean-room reproduction, not serialized size.
- A19. Competition-data-only representation learning may still reveal
  information unavailable to supervised streams, but transformer SSL failed
  its final gate and cross-user SupCon lost 2.82 points on fold 0. Any next
  pretext needs a materially different mechanism and a one-fold hard gate.
- A20. Test prior is train-like in aggregate, but several class-conditional
  shifts may remain. This permits diagnosis, not manual or global prior forcing.
- A21. Micro subject-CV selects large moves, while effects below roughly one
  point require replication or stronger paired evidence.
- A22. World-frame IMU's +0.48 nested OOF point transferred directionally but
  modestly: the exact int8 package gained one public clip. This supports the
  representation while limiting expectations for further scalar reweighting.

## Previous model (v4 — 2026-07-30, post-EXP-042 + 11 verified LB results; superseded by v5)

**The evidence-backed picture then:**

1. The verified public best was **0.53731 = 108/201**, from a 20% Adaptive
   ST-GCN addition to the MultiTCN architecture-diverse block. The
   `sub_block10 = 0.52736` line was unverified and excluded.
2. The useful verified ladder was `97 → 98 → 103 → 104 → 105 → 108`. Invariant
   IMU supplied cheap diversity; the first ST-GCN block supplied the largest
   step; MultiTCN supplied one more clip; Adaptive ST-GCN supplied three.
   Adding more same-family seeds/widths and broad TTA had saturated.
3. Adaptive ST-GCN was the best single skeleton stream at 59.56% subject-OOF.
   It raised the controlled assembly by +1.00 point with all four fold deltas
   positive, and its three-clip LB gain was the strongest recent transfer
   evidence.
4. The missing mass had not been accessible through the generic invariance
   recipes tested to that point: pad+mask was flat, two DANN strengths lost
   3-4 points, mixup was split-sign noise, and BiGRU was weaker. Learned fusion,
   naive transduction, hard metadata assignment, current visual crops,
   Identity, and the current cross-user SupCon recipe had also failed.
5. Public resolution was extremely coarse: one clip is 0.4975 point. Equal
   scores could conceal different private predictions; small LB deltas could
   not substitute for paired OOF evidence.
6. Model size appeared to be a live modeling constraint. The exact
   verified-best four-fold ensemble was ~328.7 MB fp32 / 164.3 MB fp16, above
   the 100 MB total cap; int8 packaging had not yet been measured.
7. A score above 0.83 required 167/201, another 59 public clips. No measured
   soup-scale lever had arithmetic remotely near that gap.

**Operating plan at v4:**

- **P0 — legal champion:** greedily/nested-select a compact
  architecture-diverse subset, validate a single/full-data checkpoint strategy,
  and test fp16/int8 packaging or distillation.
- **P1 — preserve proven diversity:** retain MultiTCN, Adaptive ST-GCN, at most
  one complementary legacy graph/TCN family, and invariant IMU only when each
  survived leave-one-member-out paired OOF and bytes-per-gain accounting.
- **P2 — seek new information:** prioritize legal competition-data
  representation learning, object-aware visual features, or a genuinely new
  sensor/motion view.
- **P3 — class-specific diagnosis:** study hard pairs and classes absent from
  test argmaxes with nested specialists or diagnostics; never force global
  uniform priors.
- **Parallel — reproducibility:** build one command for
  cache→train→package→infer, deterministic manifests, exact checkpoint
  inventory, CSV hash, README, and report.

**Closed or parked at v4:**

- uniform prior adjustment, Sinkhorn, and current Hungarian assignment;
- naive pseudo-label self-training and TENT-GN;
- learned logit-gate/transformer fusion at this data scale;
- then-current ROI/upper-body visual CNN recipes;
- pad64 masked TCN, tested DANN, mixup α=0.2, BiGRU, Identity, and the
  cross-user SupCon recipe;
- last-epoch full-data refit and unvalidated blanket temporal TTA;
- additional same-family seeds/widths without a controlled marginal gain.

**Assumptions standing at v4:**

- A18. Architecture diversity could be compacted below 100 MB without losing
  more than about one verified public clip. This was untested then.
- A19. Competition-data-only representation learning might still reveal
  information unavailable to supervised streams, but transformer SSL and
  cross-user SupCon had failed their gates.
- A20. Test prior was train-like in aggregate, but several class-conditional
  shifts might remain.
- A21. Micro subject-CV selected large moves, while effects below roughly one
  point required replication or stronger paired evidence.

## Previous model (v3 — 2026-07-28, post-DA-001 + 15 experiments + 4 submissions; superseded by v4)

**The evidence-backed picture of this competition:**
1. Test = 4 unseen users, ~train-like class prior (B-012 dead), clips trimmed ~20% (Δ≈1.7 pts), genuine subject shift ≈5-6 pts. CV(micro) − 9 ≈ LB.
2. No current stream is close to sufficient: skel 53 micro subj (73 in-domain), visual ≤27 macro subj even ImageNet-pretrained (46 random-split), IMU 27 micro. Oracle over streams = 63. (B-013)
3. The visual DG cliff (−20 pts even pretrained) is the largest unsolved structure; skeleton's cliff is −20 too but from a higher base with a body-normalizable representation.
4. The metadata game is small: Hungarian ≈ ±1 (parked), cohorts/groups useful mainly for per-user adaptation.
5. Leader = 0.836; target 0.85. Nothing in our measured space sums there yet — the missing mass must come from compounding: stronger streams × fusion × transduction.

**The plan (each stage gated by measurement):**
- **S1 Skeleton max-out** (reliable stream): jvb bones (+1.8 measured) + capacity + long training + trunc-aug (running); then multi-stream (joint/bone/motion logits), ST-GCN-lite, mixup, person-selection fix, no-stretch+duration. Goal: 58-62 micro subj-CV.
- **S2 Visual via legal transfer**: SSL pretraining (masked/temporal pretext) on ALL unlabeled clips (train+test ROI crops — rules-legal, no external weights), then fine-tune; 224 px ROI cache; longer schedules. Goal: 40-50 subj on depth+IR each. If organizers rule ImageNet-init legal, swap in pretrained small backbones immediately (email pending!).
- **S3 Fusion**: nested-tuned weighted late fusion + modality dropout + TTA. Diversity now includes IMU. Goal: fusion ≥ best+6-8.
- **S4 Transduction multiplier** (the leader-gap hypothesis): per-user/cohort self-training + feature alignment on the 405 test clips (4 users × ~100). MEASURE FIRST on CV (simulate: adapt on each val user's unlabeled clips). If the simulated multiplier is ≥+5, this is the biggest single lever we have. Must run inside submitted inference code (reproducible; also fine for on-site since it's per-batch).
- **S5 Package**: distill/ensemble into one ≤100 MB model.pth; efficiency score.

**Assumptions still standing (all previously-attacked ones logged in v2/DA-001):**
- A14. SSL on 3.3k clips can recover a useful fraction of the ImageNet-init gain (untested — Q-94).
- A15. Per-user adaptation transfers from CV simulation to LB (untested — Q-91 sim first).
- A16. Trimming countermeasure (trunc-aug) recovers most of the 1.7-pt trim loss (in wave 3).
- A17. Skeleton per-frame preprocessing (pelvis-center+floor-align) is survivable; global-motion loss is priced into the 73% in-domain ceiling. Raising that ceiling may require sequence-level re-normalization or multi-person cues — partially explored (tn failed).

**What is probably limiting performance right now?**
1. Visual stream quality (recipe + DG) — 85%.
2. Skeleton ceiling (73 in-domain) — 70%.
3. No transduction — unknown multiplier, potentially decisive — 60%.

**If this fails, why will it have failed?**
- SSL on 3k clips too weak to matter (small-data SSL is hard); visual stuck ≤35 → fusion caps ~60 LB.
- Transduction sim doesn't transfer (4 real users ≠ simulated val users).
- 0.85 target simply exceeds what this data supports cross-subject without pretrained weights; leader may be using something we've ruled out (or breaking rules that reproduction will catch).

## Previous model (v2 — 2026-07-27, post-profiling)

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

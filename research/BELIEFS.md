# Belief Ledger

## B-028 — Feature-space adaptation transfers where probability-space fitting does not
- **Confidence:** 75% (new, 2026-08-20) — OOF-measured, **awaiting public verification**
- **Importance:** High
- **Claim:** the cross-subject shift lives in the *features*, and correcting it there is
  cheap and safe, whereas every attempt to correct it in *probability space* has failed.
- **Evidence for (EXP-099):** re-estimating BatchNorm running statistics from the
  unlabeled held-out clips of `vid_ig65m_f2` moves micro **0.67638 → 0.69172**
  (+11 object clips), monotone in the blend weight with the optimum at the **endpoint**
  (w = 0.25/0.50/0.75/1.00 → 0.67945/0.68558/0.69018/0.69172). Adapting per *subject*
  instead of pooled reaches **0.70092** and raises motion accuracy too (0.89595 vs the
  deployed 0.87861), so the residual shift really is subject-specific.
- **Why this is a different class from the graveyard:** six fitted-combination levers
  (GBDT stacker, structure decoder, cohort weights, learned gate, temperature
  calibration ×2) all posted large OOF gains and landed ≤0 on public. Those fit
  *parameters on held-out probabilities*. AdaBN fits **nothing** — the optimum is the
  endpoint, so there is no parameter to overfit and no train/test recipe drift.
- **Falsification test:** `sub_n1.csv` (single change: AdaBN, rowdiff 14 vs the 162
  champion) scores at or below 162 on public. If it does, this belief drops to ~35% and
  the fitted-lever graveyard grows by one.
- **Caveat, measured:** purity does not beat sample size. Adapting per timestamp-block
  (100% subject-pure but median 10 clips) gives only 0.67945 — **worse than pooled**.
  Groups must be both pure and large, which is why per-subject recovery (clustering
  blocks) is the follow-up rather than per-block.
- **Bonus:** parameter-free and adds no packaging bytes, and it adapts to whoever shows
  up — so it should also help the **on-site 8-new-subject stage (30% of the grade)**.

## B-001 — Cross-subject generalization (not capacity) is the primary bottleneck
- **Confidence:** 70% (↓ from 90%) — **the headline evidence was misread; see EXP-055**
- **Importance:** High
- **RETRACTED EVIDENCE (EXP-055):** the "~30-point cliff" compared incomparable numbers. The
  paper's LOSO **56.38% is RGB-only, best-case with contrastive learning, and excludes
  cross-domain and long-tail classes**. Our 56.90% is 40 classes, no RGB, all classes. **No
  published all-modality cross-subject number exists for this dataset**, so the cliff's size —
  and whether we sit at it — has never actually been measured.
- **Evidence that survives:** EXP-006's own subject-split vs random-split diagnostics measured a
  real domain gap on our data. The gap is real; its *magnitude* and the claim that we have
  reached its floor are not established.
- **Original (now non-load-bearing) note:** EXP-000b recorded random-split avg 76.5% / visual ~90%.
  The random-split half of that is verified in the paper's Table 3; the LOSO half is not comparable.
- **Evidence against / failed levers:** aggressive geometric augmentation hurt
  (EXP-007); DANN hurt −3 to −4 points (EXP-037); mixup was split-sign noise
  (EXP-038); pad+mask was flat (EXP-036); Identity and cross-user SupCon lost
  3.12 and 2.82 points on fold 0 (EXP-041/042); dual-frame and CTR-GCN did not
  beat Adaptive ST-GCN (EXP-043/044). These reject implementations, not the
  measured domain gap. World-frame quaternion normalization is a counterexample
  showing that a physically grounded invariance can help a weak modality.
- **Remaining uncertainty:** which new representation or data view closes the gap; generic “add invariance” is no longer a sufficient hypothesis.
- **Experiments that would settle it:** repeated harsh user splits plus genuinely different motion/object representations.
- **Last updated:** EXP-046

## B-002 — Skeleton is the best single modality per unit compute
- **Confidence:** 90% (↑; operational result under the no-pretrained rule)
- **Importance:** High
- **Evidence for:** TCN 53-55%, MultiTCN 56.60%, and Adaptive ST-GCN **59.56%** all dominate current depth/IR/IMU streams while remaining cheap to train. Adaptive ST-GCN produced the 108/201 base to which world IMU added the current 109th public clip.
- **Evidence against / ceiling:** preprocessing destroys global translation; hard object classes remain structurally weak; single skeleton streams have saturated in the mid-50s.
- **Remaining uncertainty:** whether a smaller subset can improve efficiency
  without losing the verified signal; full-stack size legality itself is now
  solved by int8 packaging.
- **Last updated:** EXP-040/SUB-013 and P-02/P-03

## B-003 — ~~Multi-modal fusion beats the best single modality by ≥5%~~ REWRITTEN
- **Current belief:** simple scalar fusion of genuinely decorrelated members helps, but weak modalities and learned combiners cannot manufacture missing information.
- **Confidence:** 90%
- **Evidence:** initial three-stream fusion added only +1.26 OOF; invariant IMU added one verified public clip; ST-GCN/MultiTCN architecture diversity supplied the useful steps. The learned logit gate lost 6.8 nested points (EXP-032).
- **New evidence:** world-frame IMU improved solo 29.63% → 32.07% and its
  additive leave-one-fold-out marginal was +0.48 point. In contrast, the IR
  motion-map member was +0.30 on fold 0 but −0.89 on fold 2 at the same 10%
  weight, so a stronger solo stream still needs replicated ensemble evidence.
- **Constraint:** member value must still be priced per byte, but symmetric-int8
  packaging now fits the full `a20` stack in 82.696 MB and the expanded
  world-IMU candidate in 85.218 MB.
- **External result:** the exact legal world25 int8 output scored
  **0.54228 = 109/201**, one clip above the fp32 `a20` baseline. The different
  fp32 world25 CSV remains unscored.
- **Last updated:** EXP-045/046 and P-02/P-03

## B-004 — Micro subject-CV is useful for large moves, not fine LB ranking (REWRITTEN)
- **Confidence:** 80% for model-representation direction; low for
  metadata/transductive effect magnitude or marginal ordering
- **Importance:** High — load-bearing metric choice
- **Evidence for:** micro-OOF correctly selected the broad 97→103→105→108→109 improvement ladder. Adaptive ST-GCN added +1.00 OOF point with positive deltas on all folds and transferred as +3 public clips; world IMU's +0.48 nested marginal transferred as +1. The test prior is train-like (SUB-003/004), so micro remains the primary metric.
- **Limits:** block12 raised nested OOF yet tied block8 at 104/201; one public clip is 0.4975%; equal public scores hid 18/405 different predictions. The `0.52736` block10 claim is unverified, so the prior “offset is narrowing” inference is withdrawn.
- **New limit:** transition decoding gained 163/2700 OOF clips but only 3/201
  public clips. Repeat consensus then gained another 100 strict-nested OOF
  clips over transition-only while losing one public clip. Even careful nested
  source exclusion does not make train recording construction identical to
  hidden-test construction.
- **Policy:** require paired OOF deltas, isolated changes, and a material effect size; use LB as sparse external calibration, not a sub-point hyperparameter oracle.
- **Last updated:** EXP-047/048 leaderboard pair

## B-012 — ~~Test set is class-balanced~~ DEAD (SUB-003/004)
- **Killed:** prior-adjusted submissions dropped −6.5/−6.0 pts (0.393/0.398 vs 0.458 raw) → test prior ≈ TRAIN prior. Metric policy back to micro-OOF. The DA-001 macro↔LB match was coincidental overlap of two effects (subject shift ≈ macro-micro gap in magnitude). Kept here (not just the table) as a warning: a single corroborating coincidence is not confirmation — the submission PAIR was what falsified it.

## B-013 — Current top-1 inventory cannot reach >0.83 by scalar fusion alone
- **Confidence:** 90%
- **Importance:** Highest — governs compute allocation
- **Evidence for:** a label-aware oracle over the current predictor families
  reaches only about 77.6--82.0% using their top-1 outputs;
  visual ROI/object routes, learned fusion, naive transduction, DANN, pad,
  mixup, BiGRU, Identity, current SupCon, CTR-GCN, and dual-frame skeleton all
  failed their gates. Motion-map IR improved visual solo accuracy but failed
  replicated fusion. World-frame IMU added only +0.48 nested OOF point. The
  pre-sequence verified base was 109/201.
- **Important counterevidence:** an any-member top-2 oracle reaches 89.41%, so
  useful candidate-ranking information remains. EXP-047 then raised the
  complete stack from 61.778% to 66.963% under strict nested transition-label
  selection (67.815% at the fixed test lambda) using ordered-recording context.
  The ceiling applies to scalar/top-1 reuse, not to context-conditioned
  reranking.
- **External update:** ordered decoding transferred to 112/201, while repeated
  consensus fell back to 111/201. The top-1 family still cannot bridge the
  regime.
- **Target arithmetic:** >0.83 requires 167/201, another 55 correct public clips.
- **Remaining uncertainty:** legal competition-data SSL, object-aware
  representations, and new sensor views; same-family seeds/widths and more
  sequence postprocessing are no longer ceiling-breaking hypotheses.
- **Last updated:** EXP-047/048 leaderboard pair

## B-005 — Serialized quantization and streamed fp32 execution clear the main size risks
- **Confidence:** 99% for serialized size and one-member-at-a-time execution;
  70% for organizer interpretation until the exact reply is preserved
- **Relayed ruling (2026-07-28):** ensembles are legal only when total weights
  used at inference are ≤100 MB; strict no-pretrained; test-time transduction
  legal. The exact organizer response is not preserved in the repository.
- **Measured state:** block8's four-fold checkpoints are ~284.8 MB fp32 /
  142.4 MB fp16, and the Adaptive ST-GCN stack is ~328.7 MB fp32 / 164.3 MB
  fp16. Deterministic symmetric-int8 packaging now stores the full 44-member
  `a20` inventory in **82,696,132 bytes** and the 48-member world-IMU candidate
  in **85,217,859 bytes**; both payload audits pass.
- **Streamed runtime:** `infer_packaged.py` now iterates the existing
  per-tensor ZIP in manifest order, dequantizes/runs/releases one member, and
  accumulates probabilities. Persistent fp32 parameter bytes fall from
  **337,973,856 to 10,134,688**; the largest packed member is 2,533,672 bytes
  and the conservative model-plus-dequantized-state transient bound is
  20,269,376 bytes excluding conversion scratch.
- **Parity:** all 16,200 float64 probabilities, 405 argmaxes, and CSV bytes are
  bit-identical to eager inference. The package hash remains unchanged.
- **Boundary:** `a20` int8 changes 1/405 test argmaxes versus its verified fp32
  CSV; world25 int8 changes 3/405 versus fp32. The exact world25 int8 output is
  now independently verified at 0.54228; the fp32 CSV does not inherit it.
- **Remaining caveat:** peak process RSS is much larger because it includes
  Python, Torch, input caches, and conversion scratch; the public rule does not
  define whether any of that counts as “model size.” Preserve the exact ruling.
- **Last updated:** streamed-package audit

## B-016 — Rules-clean pipeline is a Selection-Stage asset (NEW)
- **Confidence:** 70%
- **Importance:** High — changes what score ADVANCES
- **Evidence for:** RULING confirms strict no-pretrained; DA-002 put ~85% on the 0.73-0.77 cluster using pretrained visual recipes → they fail reproduction; effective private-LB bar for top-15 advancement is likely well below the public cluster.
- **Evidence against:** cluster teams may also be from-scratch and enforcement
  rigor is unknown. Exact `a20` int8 parity is still unmeasured, although the
  legal world25 int8 package now has its own verified 0.54228 score.
- **Remaining uncertainty:** rank-15 public score (Atharv harvest pending); how
  organizers audit training provenance; clean-room reproducibility beyond
  payload reconstruction.
- **Last updated:** P-02/P-03

## B-006 — Visual modalities are the accuracy backbone (UPGRADED by EXP-055; re-diagnosed by EXP-056)
- **EXP-056 — the bottleneck moved.** The "recipe-gated" qualifier is **retired**. The MIL branch
  reaches `final_train_losses.cross_entropy = 0.5763` (chance = ln 40 = 3.689), so it fits its
  training data, then scores 0.3328 on held-out subjects. The visual branch is now
  **DG-bottlenecked**, like skeleton — not recipe-bottlenecked as EXP-006 found for the old
  120x160 recipe. EXP-006's "DG work on depth is premature" verdict is **expired** and had been
  silently gating this branch for 50 experiments.
- **Soft spot:** that CE is last-epoch loss on augmented training data, not a clean in-domain
  validation number, so "fits" is not fully separated from "memorizes". A random-split run would
  separate them but costs 9.3h (measured), and both readings imply the same remedy family.
- **Confidence:** **92%** (↑ from 80%) that the missing object information is
  visual; **50%** (↑ from 25%) that a high-resolution from-scratch visual recipe
  unlocks a material cross-user marginal
- **EXP-055 evidence:** the paper's own random-split table ranks
  **Thermal 92.57 > RGB 90.89 > Depth 90.46 > IR 90.22 > Skeleton 79.08 >
  mmWave 46.63 > IMU 45.52**. Our champion is built on the 4th-best and the
  worst *for the skeleton/IMU members*. Rank-1's 90.05% sits exactly at the
  Depth/IR/Thermal random-split level.
- **CORRECTION (same day):** an earlier draft of this entry claimed Thermal was
  never used. **False.** `visual_mil_cache.py:81` sets
  `MODALITIES = ("ir", "depth", "thermal")` at 192x256 with frame/modality masks;
  the branch has been deployed since EXP-050 and is in the champion at w=0.225.
  The RESET-002 audit note that said "Thermal unused" was correct when written
  and was superseded by EXP-050.
- **The sharpened question:** that branch scores **0.3328 solo** while its three
  modalities score **90-92% each** under the paper's random split. Even allowing
  for cross-subject degradation, the shortfall is large and unexplained. The
  bottleneck is the *recipe*, not modality availability -> OUT-011/OUT-012.
- **Direct corroboration:** the first serious visual member (EXP-053) produced
  the campaign's largest single jump, **+11 public clips (112→123)**, from a
  branch that is weak solo (0.3328). Visual information is additive and mostly
  untapped.
- **Key lever:** only a genuinely new legal representation/pretraining objective, not another crop/width sweep
- **Importance:** High — decides where most compute goes
- **Evidence for:** EXP-000b — paper: Depth 90.5/IR 90.2/Thermal 92.6 vs IMU 45.5/mmWave 46.6 (random split); test has Depth+IR in all 405 clips; same apartment/stations in test → environment transfers.
- **Evidence against:** depth/IR ROI streams remained ~17-30%; pretrained
  diagnostic still collapsed cross-subject; upper-body crop scored zero on its
  target object classes; IR as a fourth scalar stream was flat. The newer IR
  motion-map stream reached 44.36% on fold 0 and 37.67% on fold 2, but a 10%
  ensemble addition changed the accepted stack by +0.30 then −0.89 point.
- **Audit update:** motion maps score only about 27.1% on object/context clips
  in their measured folds. Existing ROI construction spans full image height
  for 34.0% of train and 43.0% of test clips, while Thermal is unused despite
  2,788/2,933 canonical train and 395/405 test coverage. The earlier 2,891
  directory count included 103 Thermal-only records outside canonical
  metadata. These are concrete preprocessing omissions rather than evidence
  of a visual ceiling.
- **Remaining uncertainty:** whether competition-data-only VICReg/temporal
  order training plus spatial MIL transfers to unseen users.
- **Last updated:** RESET-002 visual audit

## B-007 — Directional recording order is a high-value transductive signal
- **Confidence:** 99% locally; leaderboard transfer verified directionally
- **Importance:** Highest immediate inference lever
- **Evidence for:** EXP-000c — 405 test clips = 144 recordings (radar-ts key); train ground truth: recording groups 100% user-pure AND 100% all-classes-distinct (741/741) → per-group Hungarian assignment covers 93% of test clips; 2-way cohort split provable (radar-usable E n=199 vs empty L n=206); 4-way ≈ 99/100/105/101.
- **New evidence:** EXP-047 learns ordered transitions only from complementary
  users and improves every fold, **61.778% → 66.963%** under strict nested
  transition-label selection; fixed `lambda=0.5` reaches 67.815%. Reversed,
  shuffled, and label-permuted controls collapse. This is different from hard
  class distinctness or global marginal forcing.
- **Follow-up:** EXP-048 pools only label-free, metadata/probability-matched
  repeated passes and reaches 70.667% strict nested OOF (70.889% fixed), above
  transition-only on every fold relative to the original per-clip base.
- **Leaderboard evidence:** transition-only scored **0.55721 = 112/201**, +3
  clips over the exact package base. Repeat consensus scored 111/201, so
  directional order transfers but extra repeated-pass pooling does not.
- **Evidence against careless use:** Hungarian still lost two public clips;
  prior adjustment lost 13; Sinkhorn lost 12; naive self-training and TENT-GN
  remain negative. Only the directional transition mechanism is reopened.
- **Remaining uncertainty:** private transfer and magnitude on unseen hidden
  users; the public gain is much smaller than local OOF suggested.
- **Last updated:** EXP-047/048 leaderboard results

## B-008 — Radar is not worth a model stream (softened)
- **Confidence:** 70% (↓ from 80)
- **Importance:** Medium (frees compute)
- **Evidence for:** EXP-000b — empty for 100% of users 16-24, 51% of test; 6.6 pts/frame; Doppler clipped ±0.97 m/s; paper mmWave 46.6% even with full coverage.
- **Evidence against:** EXP-000c — radar is usable for ALL 198 early-cohort test clips and its train coverage (users 1-9) includes both batch A+B → an E-cohort-only radar stream could contribute on exactly half of test; its filename timestamp is the recording-group KEY regardless.
- **Remaining uncertainty:** marginal value on the E-half after visual+skel+IMU fusion.
- **Experiments that would settle it:** Q-36 (still deprioritized until fusion exists).
- **Last updated:** EXP-000c

## B-011 — Recording groups are ordered, user-pure activity sequences
- **Confidence:** 95% (train side is exhaustive ground truth: 741/741)
- **Importance:** Highest as an inference lever
- **Evidence for:** EXP-000c purity check; group key (radar filename ts) exists for 404/405 test clips; camera-anchor clustering is the backup key.
- **Evidence against overgeneralization:** hard distinctness improved train OOF
  but lost two public clips; test class distinctness is not labeled. Do not
  equate distinctness with directional order.
- **Positive transfer candidate:** the fold-safe first-order decoder reaches
  66.963% strict nested OOF (67.815% fixed-lambda) and changes 93/405
  exact-package predictions. Repetition consensus before decoding raises the
  strict nested result to 70.667% and changes 104/405 exact-package
  predictions.
- **Leaderboard result:** transition-only gained three public clips; repetition
  consensus lost one of those three. Tied/missing timestamp groups remain
  skipped rather than ordered by label-bearing train IDs.
- **Last updated:** EXP-047/048 leaderboard results

## B-020 — Repeated passes can be pooled only with deployable safeguards
- **Confidence:** 95% that the local repetition structure is real; 20% that
  pooling it improves hidden-test accuracy beyond transitions
- **Importance:** High inference lever
- **Evidence for:** repeated same-user `(trial prefix, equal length)` routines
  are label-sequence consistent about 93% of the time. A deployable rule using
  only owner/slot, chronology, equal length, gap, and probability cosine raises
  fixed OOF from 67.815% transition-only to 70.889%; strict nested parameter
  selection reaches 70.667%.
- **Guardrails:** no label-bearing SID tie breaks; ambiguous groups are no-ops;
  outer+inner users are excluded from nested transition fits; exact template
  forcing is rejected because its nested gate regressed.
- **Leaderboard evidence against adoption:** repeat consensus scored
  111/201 versus 112/201 for transition-only. The locally stronger mechanism
  failed its marginal transfer test.
- **Decision:** retain for analysis only; do not tune or submit another
  repetition/distinctness variant.
- **Last updated:** EXP-048 leaderboard result

## B-009 — IMU is weak alone but cheap ensemble diversity (NEW)
- **Confidence:** 95% that IMU supplies cheap diversity; 95% that the tested
  world-frame IMU stack transfers directionally
- **Evidence for:** invariant features raised the original solo stream about
  four OOF points and added one verified public correct clip. EXP-046's
  quaternion world-frame representation then improved solo
  **29.63% → 32.07%** and added **+0.48 nested OOF point** to the complete stack;
  the fixed 25% marginal was positive on all four folds. Its exact legal int8
  output scored **0.54228 = 109/201**, +1 public clip.
- **Evidence against / ceiling:** only ~10 Hz and ≈22 samples/trial; ConvLSTM
  was weaker; world-frame fold 3 was solo-flat and same-budget replacement of
  `imu_inv4` was slightly negative.
- **Remaining uncertainty:** the different fp32 world25 CSV is unscored, and
  the one-clip gain is too small to establish the effect size on private users.
- **Last updated:** EXP-046

## B-022 — Distinctness coupling pays only above a base-accuracy threshold (NEW)
- **Confidence:** 85%
- **Importance:** High — it is predictive, not just descriptive
- **Evidence for:** EXP-052 ran hard distinctness on all four folds and the
  delta ordered monotonically with base accuracy: fold 1 (base 0.664) +6,
  fold 0 (0.631) +5, fold 2 (0.591) −9, fold 3 (0.586) −3. Soft penalty 1.0
  was the best trade-off at +10 clips overall and still fold-2 negative.
- **Mechanism:** a distinctness constraint couples clips inside a recording.
  With a reliable posterior it propagates correct information; with an
  unreliable one a single wrong clip evicts a neighbour from its correct label,
  so it propagates errors symmetrically.
- **Retrodiction:** this explains EXP-004/005, where Hungarian assignment lost
  public clips at a 0.458 base — the regime where coupling should hurt most.
  Three separate results now share one account instead of three.
- **Prediction (falsifiable):** the sign flips as base accuracy rises. Re-test
  distinctness after any material base improvement; do not re-test before one.
- **Remaining uncertainty:** the crossover point, and whether test recordings
  are as class-distinct as the 741/741 train groups — that is assumed, never
  verified on hidden users.
- **Last updated:** EXP-052

## B-021 — Object information is weakly represented in current inputs (DOWNGRADED by DA-006-A)
- **Confidence:** 55%, down from 90%. **Half its evidence was retracted.**
- **Importance:** Highest — but it no longer closes the family it appeared to
- **Retraction:** this belief was written from two screens that DA-006-A then
  showed were underpowered. The seed-variance floor on this exact partition is
  **2.80 points**, and EXP-051's −0.0107 is −0.38 sigma: indistinguishable from
  zero. The claim that narrowing the label space fails is **not supported**.
  The EXP-050b fusion delta (+0.33 sigma) is likewise noise.
- **Evidence that survives:** EXP-050b's *solo* visual result. 0.3328 against a
  0.5445 seed mean is about −21 points (~7.6 sigma), with object classes at
  21.9% versus gross motion at 64.7%. A from-scratch supervised visual branch is
  genuinely weak on object classes at this data scale.
- **Both reopened questions are now SETTLED, in opposite directions.**
  (a) *Visual branch in fusion:* **helps** — EXP-053 measured +1.00 point paired,
  std 0.15, 4/4 seeds positive, and it delivered **+11 public clips** (112→123).
  (b) *Cluster-conditioned routing:* **does not help** — EXP-054's paired 4-seed
  re-run gives mean −0.0096 (−6.2 clips of 652), std 0.0076, −1.27 sigma, 3/4
  seeds negative. The surviving mechanism is that the base posterior is already
  well-ordered inside a cluster (top-2 0.6641), so an equal-authority specialist
  overwrites more than it repairs.
- **The 44 visual-only rescues (35 object, union oracle 63.65% vs 56.90%) remain
  unexplained** by any refuted mechanism, and are now the largest unexplained
  error structure in the campaign → OUT-004.
- **Lesson recorded:** a belief assembled from two same-direction results is
  only as strong as the noise floor of the weaker one. The floor was never
  measured until DA-006-A.
- **Boundary:** the visual branch did rescue 44 clips (35 object) with a union
  oracle of 63.65% versus 56.90%. Complementary signal exists; scalar fusion
  and label-space narrowing are both unable to extract it.
- **Implication (REVISED by EXP-055):** stop proposing new predictors over
  **skeleton-derived** features — that half stands, and EXP-054 confirmed it with
  power. But the stronger reading, that the model side as a whole is closed, is
  **withdrawn**: it was conditioned on sitting at a ~56.4% published ceiling that
  EXP-055 showed is an RGB-only, class-subsetted number measuring a different
  problem. A genuinely different input is exactly what is called for — and the
  paper's three strongest modalities (Thermal 92.57, Depth 90.46, IR 90.22
  random-split) are unused, partially used, and reduced to motion maps
  respectively, while our stack rests on Skeleton (79.08) and IMU (45.52).
- **Remaining uncertainty:** whether a confidence-gated pair-level route using
  the visual embedding as *input* behaves differently from either failure.
- **Last updated:** EXP-051

## B-010 — Hard/rare classes need targeted handling, not global prior forcing
- **Confidence:** 90% for the "not global forcing" half; the "targeted
  handling works" half is **falsified at cluster granularity with proper power**
  by EXP-054 (paired 4-seed mean −0.0096, −6.2 clips of 652, std 0.0076,
  −1.27 sigma, 3/4 seeds negative). EXP-051 reached the same verdict from one
  seed; DA-006-A's withdrawal of it used an unpaired denominator and is itself
  withdrawn. **Untested at finer granularity:** `MAX_CLUSTER=2` true-pair routing
  touching tens of clips rather than 614 is a different hypothesis, not a retest.
- **Evidence for:** EXP-000b — 12 vs 319 samples/class (27×); classes 25/26 present for only 4-6 users; duration varies 7× by class and test is trimmed shorter.
- **Evidence against global correction:** balanced-prior adjustment and Sinkhorn catastrophically failed; test prior is approximately train-like. Recent ensembles predict no test examples for classes 16/28/35, but blindly forcing them would repeat the same error.
- **Policy:** use nested class-specific diagnostics or constrained specialists only when they improve held-out users; never impose a uniform marginal.
- **Last updated:** SUB-003/004 and SUB-012 audit

## B-017 — Architecture diversity beats same-family accumulation
- **Confidence:** 97%
- **Evidence for:** the first ST-GCN block moved 98→103 public correct clips;
  MultiTCN moved 104→105; Adaptive ST-GCN then moved 105→108 with +1.00 OOF
  point and positive deltas on every fold. World-frame IMU supplied +0.48
  nested OOF point across modalities and moved the exact legal stack 108→109.
  In contrast, TCN soup6→7 and GCN soup3→4
  were flat/slightly negative, block12 tied block8 publicly, and CTR-GCN plus
  dual-frame skeleton failed their same-fold architecture screens.
- **Implication:** a new member must beat an exact scalar-weight control with paired evidence and justify its package bytes. Seeds and widths are variance reducers, not ceiling breakers.
- **Last updated:** EXP-046

## B-018 — Public-LB resolution is too coarse for marginal search
- **Confidence:** 99%
- **Evidence:** 201 public clips means one clip = 0.4975 point; block8 and block12 tie despite 18/405 different test predictions; MultiTCN gained one clip, Adaptive ST-GCN gained three, and world IMU gained one.
- **Policy:** record every result in LEADERBOARD.md, preserve exact CSV hashes, and label unsupported scores unverified. Never infer a smooth CV→LB offset from one-clip movements.
- **Last updated:** SUB-011/013

## B-019 — Adaptive graph structure is the strongest live representation axis
- **Confidence:** 90%
- **Evidence:** Adaptive ST-GCN is the best solo stream at 59.56%, adds +1.00 OOF point over the accepted MultiTCN assembly with all four folds positive, and adds three verified public clips. It uses only 0.842M parameters per checkpoint.
- **Boundary:** CTR-GCN lost 0.74 point on fold 0 and was 4.28 points behind on
  the stopped fold-2 screen; dual-frame skeleton also lost 1.63 on fold 0. The
  evidence supports this Adaptive ST-GCN implementation, not arbitrary graph
  expressivity or more graph soup.
- **Last updated:** EXP-043/044

---

## Dead beliefs (killed by evidence — keep, so we don't resurrect them)

| ID | Belief | Killed by | Why |
|----|--------|-----------|-----|
| — | "Test set is anonymized" (implicit) | EXP-000b | Timestamps intact in 4 modalities + IMU internal timestamps + Thermal frame counters |

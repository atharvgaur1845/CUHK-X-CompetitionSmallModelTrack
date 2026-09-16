# Expert Q&A bank — fact / interpretation / limitation

Every answer leads with a measured fact (experiment id in `research/LOG.md`), then the reading, then
the limitation. Do not bluff past the third line.

**Why this architecture (MViTv2-S)?** Fact: at 224 px it beat the IG-65M R(2+1)D by +2.42 pooled OOF
(p = 0.0016, EXP-103) and alone reached 166 public. Reading: a 224-native transformer initialised on
Kinetics is the right size for 2.3 k clips; the stem widens to 4 channels without losing the init.
Limitation: no 3D-CNN was ablated at 384 px.

**Why not one bigger model / one giant multimodal transformer?** Fact: 86 M-parameter backbones
overfit — VideoMAE-B fine-tuning lost 0/4 folds (EXP-115); the full-frame IR+Depth member was a
correlated copy of the crop (EXP-143). Reading: capacity is not the bottleneck; independent views
are. Limitation: we never trained a joint early-fusion multimodal transformer; the 100 MB rule made
it a poor bet, not an impossible one.

**Why thermal?** Fact: 0.544 and zero fusion gain at 128 px cropped; 0.627 and +46/2700 fused at
224 px full frame (EXP-139/140), +5 public. Reading: object heat and scene context are complementary
to the IR crop; the earlier failure was recipe, not sensor. Limitation: weak alone; 10 Kaggle clips
have no thermal frames and receive its zero-input posterior.

**Why skeleton?** Fact: removing it costs 16 clips of 2 700 (EXP-140), 27 in the frozen fusion
(EXP-162). Reading: a motion prior that helps gross-motion classes. Limitation: it transfers worst
of all members (−7.6 OOF→public, EXP-063) and its pruning step is unrecorded (PROVENANCE).

**Why keep the IMU forest?** Fact: removing it costs 2 clips (EXP-140); paired LOSO delta −0.25
points (EXP-145); the unselected IMU-off submission scored 180/204 private vs 179. Reading: neutral;
kept because the paired test did not justify removing it. Limitation: it is the weakest member.

**Why this fusion?** Fact: wrong prior direction costs 4 points (0.587 vs 0.627); nested-CV
coefficients; every coefficient ±50 % stays within ±1 % OOF (EXP-162) and moves ≤ 8 test rows.
Reading: a calibration in the right space, on a plateau. Limitation: coefficients are fitted on 16
training subjects; they are coefficients, not a model.

**Why this decoder — aren't you exploiting recording order?** Fact: +10.4 OOF, +3–5 public
(EXP-125/155); shuffled order +9, uniform transitions +55, reversed −54 (EXP-162). It is label-free
and fitted on training metadata only (organiser ruling R-7 permits deterministic label-free
preprocessing). Reading: half the gain is "distinct activities within a session", half the script.
Limitation: neutral without the script; harmful if a script were reversed.

**How do you prevent subject leakage?** Fact: folds and LOSO are subject-grouped; test subjects are
unseen; the reported public/private ranks agree. Limitation: OOF is biased high by 8–11 points and
was never an adoption criterion.

**How do you know it generalises?** Fact: 18 LOSO draws, sd 0.054, worst 0.667, best 0.876
(EXP-145); private rank = public rank. Reading: the ensemble narrows the subject spread from 28 to
21 points. Limitation: LOSO on 18 subjects is not a new site; 4- and 8-subject draws carry ±5–7 clips.

**External data? Test labels? Manual test inspection?** Facts: none, none, none. A test-label leak
was reported on the forum on 2026-06-27; we recorded it as closed and never used it (LOG L-1). Only
counts and timestamps of test files are read, as features, at inference.

**How were ensemble members selected and pruned?** Fact: leave-one-out clips and bytes per clip
(EXP-140/142); the 48→20 prune to fit 100 MB cost 2 rows on the Kaggle test. Limitation: the
prune's invocation was not logged (disclosed).

**Your primary CSV is reproduced in 403/405 rows — why?** Fact: the primary was fused from the
48-member skeleton output two hours before the pruned package; the two rows are decoder near-ties
(margins 0.012/0.006); ~900 coefficient settings with frozen tensors come no closer than one row.
Reading: we ship the principled coefficients and disclose; the second Selected submission is
byte-identical. Limitation: it is a 0.5 % residual and we asked the organisers how they prefer it handled.

**Missing modalities / a completely new subject?** Fact: ≤ 48/2700 loss for any absent sensor
(EXP-162); zero-input behaviour identical to the Kaggle run; LOSO worst subject 0.667. Limitation:
a subject whose scene scripts differ gets no decoder gain.

**How deterministic is inference?** Fact: byte-identical repeats on one machine; RTX 4060 vs A100
max posterior Δ 1.6e-3, 0/405 prediction changes (EXP-159). Reading: fp16 autocast + fixed batches,
no RNG. Limitation: detector windows are not bit-reproducible across GPUs (22 rows) — hence the
fingerprint-gated shipped windows.

**Why should we trust the package?** Fact: hashes of every artifact, build-time offline self-checks,
byte-identical reproduction on two machines and in a network-isolated container, a hostile synthetic
suite, and an independent comparison script that shares no code with the package.

**Computational footprint?** Fact: 96.37 MB; 2 min 39 s for 405 clips on an A100, 4 min 53 s on a
laptop RTX 4060; peak GPU memory 5.4 GB.

**Weakest component? One more month?** Facts: the IMU forest (0.377 alone); the residual error is
per-subject and correlated (EXP-132/156/157) and the subject learning curve is flat past 6 subjects
(EXP-136). Reading: more subjects, not more model. Limitation: that is a data request, not a method.

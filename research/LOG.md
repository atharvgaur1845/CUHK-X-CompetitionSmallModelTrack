# Evidence Log

**Counters:** experiments since last devil's-advocate pass: **4 / 10** · since last reset: **4 / 25**

**Leaderboard source of truth:** [LEADERBOARD.md](LEADERBOARD.md). Scores without
user-supplied Kaggle evidence are unverified even if an older entry called them
results.

---

## EXP-050b — Visual MIL fold-2 result: BOTH GATES FAILED, BRANCH REJECTED
**Date:** 2026-07-31 · `oof_visual_mil_v1_f2.npz` · **Tier:** explore

Outer-once evaluation on all-18 fold 2 (652 clips, users 5/6/22/24), scored
exactly once after the final EMA update:

```
micro 0.33282 · macro 0.33046
object       0.21921 (105/479)
gross_motion 0.64740 (112/173)
```

Partition-matched control (`astgcn_all18_f2`, EXP-040 recipe, identical
partition): **0.5690** micro (`_best`, epoch 61), 0.5521 (`_last`).

| Predeclared gate | Required | Actual | |
|---|---|---|---|
| object accuracy | ≥ 0.32 | 0.21921 | fail |
| fusion delta vs `_best` | ≥ +0.02 | +0.0092 | fail |

The +0.0092 is itself optimistic: `w=0.20` was chosen by sweeping on the same
fold being scored. An honestly nested weight would sit at or below +0.005.

The branch fails in the most informative direction: it is weakest exactly
where it was supposed to help. Object/context 21.9% against gross-motion
64.7%, while the skeleton it was meant to complement already reaches 88.34% on
gross motion. It is also below the IR motion maps it replaced (~27.1% on
object/context). The recipe ran as designed — loss 7.70 → 0.58, LR landing on
2e-6 at epoch 60, 10 AMP skips — so this is a real negative, not a broken run.

### One positive finding

Complementarity is genuine even though fusion is not:

```
both right 173 · baseline only 198 · VISUAL ONLY 44 · both wrong 237
union oracle 0.6365 vs baseline 0.5690  (+6.75 points of headroom)
```

**35 of the 44 visual-only rescues are object classes.** The cache and
representation carry object information the skeleton lacks; the *classifier*
is too weak for scalar probability averaging to extract it. Any follow-up must
use the visual branch as a feature/evidence source, not as another scalar
probability member.

### Failure analysis

**Three reasons it failed.** (1) 2,281 clips is far too little to learn
object appearance from scratch at 192×256×16 across three modalities — 2.31M
parameters reached training loss 0.58, so it fit the training users and did
not transfer. (2) Class-specific top-k MIL pooling has to *discover* where the
object is with no localization supervision; with 40 classes and no pretrained
features that search is underdetermined. (3) The three modalities are
appearance-poor for object identity — IR and depth resolve shape and distance,
not the cup-versus-phone distinction the object classes hinge on.

**Three alternative explanations.** (1) The experiment is valid but the
recipe is one of many — a different architecture might clear the gate, though
B-006 already logs repeated visual recipe failures. (2) Fold 2 is the hardest
fold (baseline 0.5690 here versus 0.618 stack OOF elsewhere), so the gate may
be harsher than average; the predeclared protocol chose fold 2 deliberately as
the hard screen. (3) 60 epochs at effective batch 8 may be under-trained, but
loss 0.58 and a flattening tail argue the opposite.

**Three follow-ups.** (1) Use the visual branch only as a gated evidence
source on the 44-rescue population rather than a fusion member. (2) Attack the
confusion clusters directly with low-parameter specialists (X-02). (3) Settle
the rank-15 bar before spending more compute on ceiling-chasing (P-05).

**Decision:** reject the visual member per the predeclared rule. Per
`MENTAL_MODEL.md` v8 item 6, fold 2 was the hard gate and it failed, so there
is no fold-0 replication, no threshold revision, and no recipe retuning.
`B-006` drops to low confidence for from-scratch supervised visual recipes.

---

## EXP-050 — Visual MIL fold-2 screen: HARNESS DEFECTS FIXED, RUN LAUNCHED
**Date:** 2026-07-30 · `code/train_visual_mil.py` · **Tier:** explore · **status: running**

The X-07 cache completed at 2933 train and 405 test shards. Before spending
GPU hours, the training harness was audited and **two defects were found that
would have invalidated the screen**. Both are now fixed.

1. **The LR schedule was sized for accumulation that never happened.**
   `updates_per_epoch` divided by `GRAD_ACCUMULATION_STEPS = 4`, but the inner
   loop called `optimizer.zero_grad`/`scaler.step` on *every* mini-batch. The
   cosine would therefore have reached `MIN_LEARNING_RATE = 2e-6` after roughly
   15 of 60 epochs, training the remaining 45 epochs at a frozen LR.
2. **`scheduler` was never defined.** The loop called `scheduler.step()` inside
   the `stepped` branch, so the first successful optimizer update would have
   raised `NameError`. The OOM at `--batch-size 2` masked this by failing
   earlier.

Real accumulation is now implemented against a declared
`EFFECTIVE_BATCH_SIZE = 8`: micro-batch size is a memory decision only, loss is
scaled by `1/accumulation_steps`, the trailing partial window still steps, and
the LR closure is driven by the successful-update counter. A simulation of the
repaired schedule gives warmup to 3e-4 by epoch 5 and 2e-6 at epoch 60; the
observed epoch-1 LR of 5.87e-05 matches the predicted 6.00e-05.

This matters beyond one run: a frozen-LR screen would most likely have failed
its gate and been recorded as evidence against the visual/object hypothesis —
the direction holding 90.7% of current error mass. A false negative here was
the most expensive available outcome.

**Configuration:** fold 2, `--batch-size 1 --workers 6`, `expandable_segments`,
2,314,518 parameters, 9.258 MB fp32. Batch 2 does not fit in 8 GB VRAM.
Measured pace is about 13 minutes/epoch, so the fold-2 screen is roughly a
13-hour run.

### Blocker found for the predeclared fusion gate

`GATE_MIN_SHARED_COVERAGE = 1.00` cannot be satisfied on fold 2 with the
existing baseline. All-18 fold 2 holds out users 5, 6, 22, and 24 (652 clips),
but `oof_astgcn_world25.npz` covers only 16 users — **user5's 100 clips are
absent**, so shared coverage is at most 552/652 = 84.7%.

Consequences: the solo and object-accuracy gates remain evaluable, the fusion
delta is only measurable on the 552-clip intersection, and P-09's pending
model rerun is now a hard prerequisite rather than hygiene. An all-18
outer-once baseline OOF for the skeleton/IMU core must be produced before any
visual member can be adopted on paired evidence.

**Correction to the first framing of this blocker.** Missing user5 coverage is
the lesser problem. On the clips that *are* shared, the old OOF's predictions
come from a different subject partition: old fold 2 trained on user6, old
fold 3 trained on users 22 and 24, and user5 was an `always_train` user in
every old split. No clip was predicted by a model that saw its own user, so
per-clip cross-subject validity holds — but the baseline is systematically
advantaged on exactly the clips the marginal would be measured on, which
biases the visual member's apparent contribution **downward**. That is the same
false-negative failure mode as the LR defect, reached by another route.

**Resolution — partition-matched control (predeclared).**
`code/run_matched_baseline_f2.sh` waits on the visual training PID, confirms
the GPU is free, then trains the Adaptive ST-GCN core on exactly the all-18
fold-2 training partition using the EXP-040 recipe verbatim (adaptive skelg,
jv, width 64, 100 epochs, bs 64, lr 1e-3, ls 0.1, trunc augmentation, seed 0);
only the fold protocol differs. `code/baseline_all18_probs.py` then dumps
validation probabilities on the 652 held-out clips. The partition was verified
before launch: 652 val clips over users 5/6/22/24, 2281 train clips over 14
users, zero id overlap. `har_data.FOLDS` now honours `CUHKX_FOLD_FILE`, so the
historical protocol remains the default for every existing caller.

Two baselines are dumped deliberately, because each carries opposite bias:

- `*_best` selects its epoch on fold-2 validation accuracy. It is optimistic,
  and therefore the **conservative** control for adopting a visual member.
- `*_last` applies no selection at all, symmetric with the visual model's
  final-EMA outer-once evaluation, but risks flattering the visual marginal.

**Predeclared adoption rule:** the visual member is adopted only on a positive
fusion delta against the optimistic `*_best` baseline. Reporting a delta only
against `*_last` is not sufficient.

**Decision:** let the repaired fold-2 screen run; the matched baseline is
chained behind it. Do not weaken the coverage gate to make a number
reportable. A full 4-fold all-18 rerun is not required now — only at adoption
time, when a complete OOF matters.

---

## EXP-049 — Exact recording templates: POSITIVE FIXED RESULT, NESTED GATE REJECTED
**Date:** 2026-07-30 · `code/recording_template_decoder.py` · **Tier:** explore

Complete, timestamp-unambiguous `(user, trial)` label sequences from
complementary users were tested as exact same-length templates. Unguarded
template replacement collapsed OOF to roughly 45%, because exact routines do
not transfer reliably between users. A guarded template correction on top of
repeat consensus plus Markov reached **71.741% (1937/2700)** at one fixed gate,
23 clips above its 70.889% fallback. Reverse and shuffled controls fell to
70.481% and 69.519%, respectively.

The adoption test failed: strict outer-isolated nested gate selection scored
**70.333% (1899/2700)**, 15 clips below the 70.889% fallback, with regressions
on the two harder folds. The fixed result is therefore exploratory evidence,
not a submission recipe. The implementation remains report-only by default
and did not write a CSV.

**Decision:** retain the auditable backend, but do not package or upload an
exact-template candidate until a new predeclared gate replicates.

---

## EXP-048 — Repeated-recording consensus: PUBLIC MARGINAL REJECTED
**Date:** 2026-07-30 · `sub_astgcn_world25_int8_repeat_trans05.csv` · **Tier:** exploit

Within a user, consecutive equal-length recording passes frequently repeat the
same action sequence. A deployable, label-free cluster links only consecutive
recordings with an end-to-start gap at most 300 seconds and mean aligned
probability cosine at least 0.6. Per-position probabilities are averaged
inside each cluster before the fixed `lambda=0.5` transition decoder.

All ordering and selection leaks found in audit were removed:

- tied or missing clip timestamps make a recording ambiguous;
- ambiguous source recordings are excluded from transition fitting;
- ambiguous target recordings are neither pooled nor decoded;
- no sample ID is used to break a tie because train IDs contain class prefixes;
- for outer-fold parameter selection, every inner transition fit excludes the
  union of outer and inner users.

Fixed OOF moved **61.778% (1668/2700) → 64.889% consensus → 70.889%
(1914/2700)**. This is +9.111 points over the clip base, with 321 rescues and
75 harms. Fold finals were 72.552%, 77.563%, 67.061%, and 66.420%, all above
their per-clip baselines. A strict 16-candidate nested grid selected the same
gap/cosine/consensus settings and `lambda=0.5` on three folds (`0.4` on fold
1), scoring **70.667% (1908/2700)**.

The exact packaged-probability test run found 42 repeat clusters covering 266
clips. It left one ambiguous five-clip recording untouched, excluded 30
ambiguous train recordings, and changed 104/405 predictions versus the
verified 0.54228 base. The staged CSV:

- is in exact official order with 405 unique rows and labels 0--39;
- is 16,514 bytes;
- has SHA-256
  `1a1dbdaeba41814cf0b459b78c2ae95eb9d80389ae62a842970cac76cd6dc377`;
- differs from transition-only EXP-047 on 41/405 rows;
- scored **0.55223 = 111/201**.

EXP-048 stayed two clips above the 109/201 package base, but lost one clip
versus the cleaner transition-only result. Its much larger local OOF gain did
not transfer proportionally.

**Decision:** reject repetition consensus as a submission marginal and stop
consensus/distinctness tuning. Preserve the diagnostic implementation, but
return to transition-only as champion.

---

## EXP-047 — Ordered-recording transition decoding: VERIFIED NEW BEST
**Date:** 2026-07-30 · `sub_astgcn_world25_int8_trans05.csv` · **Tier:** crazy→exploit

The previous metadata experiments tested class marginals or hard within-group
distinctness. This experiment instead uses the actual chronological order of
clips inside each recording. A 40×40 add-one-smoothed transition matrix is fit
from ordered `(user, trial)` sequences. For outer fold `f`, every clip from
fold-`f` users is excluded from transition fitting. For inner lambda selection,
transition fitting excludes both the outer and inner users. Groups with tied
or missing timestamps are skipped: train IDs encode the class prefix and must
never be used to break an ordering tie.

The strictly nested transition-label selection chose lambda
`0.4 / 0.5 / 0.6 / 0.6`:

| Fold | Per-clip base | Transition decode | Delta |
|---:|---:|---:|---:|
| 0 | 63.056% | 69.288% | +6.231 |
| 1 | 66.419% | 73.551% | +7.132 |
| 2 | 59.084% | 62.629% | +3.545 |
| 3 | 58.580% | 62.426% | +3.846 |

Aggregate nested OOF rose
**61.778% (1668/2700) → 66.963% (1808/2700)**: **+5.185 points**, with
230 rescues, 90 harms, and all four folds positive. The independently useful
fixed `lambda=0.5` robustness result is
**67.815% (1831/2700), +6.037 points**, with 241 rescues and 78 harms.

Negative controls establish that the gain comes from directional order:
reversed source order scored 55.852%, reversed target order 56.185%,
shuffled target order 59.741%, label-permuted transitions 59.333%, and a
uniform transition matrix reproduced the 61.778% base exactly. Shuffling
source order retained only +1.148 points versus +6.037 for real order. This is
not the failed Hungarian or prior-adjustment mechanism.

`code/infer_packaged.py` was extended to retain the exact probabilities from
the scored int8 package. A deterministic rerun reproduced the verified base
CSV byte-for-byte (SHA-256
`879417469e0c4de3d52862465126c26e9813dd35678fa9725d61d935f89a2e45`).
Applying the frozen `lambda=0.5` decoder to those exact probabilities produced:

- `submissions/sub_astgcn_world25_int8_trans05.csv`;
- 405 unique rows in official sample-submission order;
- 30 ambiguous train groups excluded from fitting and one five-clip tied test
  group left at its base predictions;
- **93/405** changes versus the verified 0.54228 base;
- SHA-256
  `ed6784665f7aacf5832ca10d7b7a0adc8fd333977cd25effedc8e4efe168e346`.

**Leaderboard result:** the exact CSV scored **0.55721 = 112/201**, three
public clips above the 109/201 package base and the new verified best. This
confirms directional-order transfer while also showing the public effect is
far smaller than the +5.2 to +6.0 OOF-point estimate.

Do not infer a private score from the public result.
Because the organizer ruling relayed by Atharv permits test-time transductive
processing, joint decoding of unlabeled test probabilities is treated as
allowed; the exact organizer response still needs to be preserved.

### Validation repair completed alongside EXP-047

`code/cv_protocol_all18.py` generated a deterministic outer protocol covering
**18/18 users and 2933/2933 clips exactly once**. Fold sizes are
814/814/652/653, with class coverage 40/40/40/39; class 25 exists for only
three users, so four disjoint folds cannot all contain it. This protocol fixes
the historical always-train omission of users 5 and 21. It does not
retroactively make the current 2700-row model OOF unbiased; new model runs must
use fixed epoch recipes and score each outer fold once.

**Decision:** adopt EXP-047 as champion. Retain the all-user protocol for the
visual/object reset; another postprocessor cannot close the remaining
55-public-clip gap.

---

## DA-005 / RESET-002 — Bottleneck and evidence audit
**Date:** 2026-07-30 · **Tier:** mandatory reset

The complete worktree, validation protocol, probability inventory, modality
coverage, package loader, leaderboard ledger, and official-rule evidence were
re-audited before opening another training wave.

Load-bearing findings:

1. Current world25 OOF is 88.34% on gross-motion classes but only 50.13% on
   object/context classes. Those classes contribute 936/1032 errors, while the
   accepted package contains no visual modality.
2. A label-aware oracle over existing top-1 predictor outputs reaches only
   about 77.6--82.0%, so scalar reuse cannot reach the target. An any-member
   top-2 oracle reaches 89.41%, leaving a route for context-conditioned
   reranking.
3. Historical OOF excludes users 5 and 21 and covers only 2700/2933 clips.
   Outer-fold checkpoint selection adds roughly 1.2--1.3 optimistic points to
   complete stacks. Fine sub-point conclusions are not reliable.
4. World25 local 61.778% versus public 54.228% is a 7.55-point transfer gap.
   Broad OOF level tracks the leaderboard, but recent incremental changes do
   not.
5. The 85.218 MB archive expands to 338.14 MB of live fp32 weights. Compliance
   depends on organizer size semantics until streaming/true quantization or
   distillation removes the ambiguity.
6. The exact organizer reply is absent; local documents preserve only the
   paraphrased ruling and the unsent draft.
7. Existing visual failures are recipe failures, not a modality ceiling:
   roughly 21% of train and 28% of test ROIs are effectively full-frame, and
   Thermal remains nearly untouched.

**Reset decision:** stop same-family skeleton/IMU soup work. First exploit
ordered recording context with strict fold exclusion, then rebuild all-user
outer-once evaluation and pursue a compact object/visual branch plus
candidate-conditioned reranking.

---

## EXP-046 — Quaternion world-frame IMU: VERIFIED LEGAL NEW BEST
**Date:** 2026-07-30 · `sub_astgcn_world25_int8.csv` · **Tier:** explore→adopt

The recorded quaternion was used in the empirically verified local-to-world
direction `R(q)`. Each of the five devices contributes gravity-removed
world-frame acceleration, world-frame angular velocity, magnitudes, and
availability features; absolute quaternion components are not exposed to the
classifier.

The four-fold world-frame TCN scored **32.074% solo**, versus **29.630%** for
`imu_inv4`. Fold results were:

| Fold | `imu_inv4` | World IMU | Delta |
|---:|---:|---:|---:|
| 0 | 29.82 | 34.57 | +4.75 |
| 1 | 33.88 | 36.40 | +2.52 |
| 2 | 28.06 | 30.58 | +2.52 |
| 3 | 26.78 | 26.78 | +0.00 |

Replacing the existing IMU at the same 17.5% budget was flat-to-negative:
61.04% → 60.96%. The useful result is additive diversity. Adding 25% world IMU
to the complete accepted stack raised descriptive OOF **61.037% → 61.778%**
(56 rescues, 36 harms), with positive fold deltas
**+1.19 / +0.45 / +1.03 / +0.30** points. Leakage-safe leave-one-fold-out
weight selection chose 17.5% or 25% and raised aggregate OOF to **61.519%**,
**+0.481 point**, with 44 rescues, 31 harms, and paired SE 0.32 point.

**Leaderboard result:** the exact legal-package output
`sub_astgcn_world25_int8.csv` scored **0.54228 = 109/201**, a new verified best
and one public clip above `sub_astgcn_a20.csv` (108/201). The fp32 world25 CSV
remains unscored because it differs from the int8 output on 3/405 rows.

**Decision:** externally adopt the legal int8 world25 stack. The positive
nested marginal transferred directionally, but only as one public clip; this
is evidence against expecting scalar fusion refinements to close the remaining
58-clip target gap.

The legal package is
`research/artifacts/model_astgcn_world25_int8.pth`: **85,217,859 bytes**, 48
members, SHA-256
`bccd32dc997839dd085717f42fd600192d5b692fc962075fa55565600ae95ee0`;
payload verification passes. Its derived
`sub_astgcn_world25_int8.csv` is **16,516 bytes**, SHA-256
`879417469e0c4de3d52862465126c26e9813dd35678fa9725d61d935f89a2e45`,
and differs from the fp32 candidate on 3/405 argmaxes. This exact int8 CSV is
the verified 0.54228 submission. The fp32 candidate CSV
is also 16,516 bytes with SHA-256
`bc5305a6631bb4f9e036af70c6773bc092efb38c2730fb98244cb39ba1f1f3ed`.
It remains unscored.

---

## EXP-045 — Dual IR motion maps: SOLO PROGRESS, FUSION REJECTED
**Date:** 2026-07-30 · **Tier:** crazy

A compact 1.339M-parameter dual CNN consumed full-frame and person-ROI
five-channel summaries: mean, standard deviation, dynamic rank, positive
motion, and negative motion. Training used fixed 40 epochs and evaluated each
outer subject fold exactly once.

- fold 0 solo: **44.362%** (macro 36.399%);
- fold 2 solo: **37.666%** (macro 30.267%).

This is meaningful solo progress over prior from-scratch visual streams, but
the ensemble control did not replicate. On fold 0, 10% motion-map probability
raised the accepted stack **61.869% → 62.166%** (+0.297 point; 10 rescues,
8 harms). The identical 10% addition on fold 2 reduced it
**58.050% → 57.164%** (−0.886 point; 3 rescues, 9 harms); even 5% was
−0.443 point on fold 2.

**Decision:** stop before folds 1/3 and reject this member from fusion. Preserve
the cached summary implementation and solo result as evidence that visual
motion maps are better than the earlier supervised visual recipe, but do not
spend package bytes or a submission on an unreplicated marginal.

---

## EXP-044 — Dual-frame skeleton view: REJECTED
**Date:** 2026-07-30 · **Tier:** explore

The 2.40M-parameter model combined the recorded station frame with a canonical
body frame under a fixed-100-epoch, outer-validation-once protocol. Fold 0
scored **57.270%**, below the paired Adaptive ST-GCN result **58.90%**
(−1.63 points).

**Decision:** stop after fold 0. The additional global/station frame did not
clear the same-fold architecture gate and does not justify more folds or
package bytes.

---

## EXP-043 — CTR-GCN screen: REJECTED
**Date:** 2026-07-30 · **Tier:** explore

The channel-wise topology-refinement graph model reached **58.16% on fold 0**,
below Adaptive ST-GCN's paired **58.90%**. The fold-2 screen was only
**52.29% at epoch 40**, versus Adaptive ST-GCN's **56.57%**, so training was
stopped rather than completing an already-negative replication.

**Decision:** reject this CTR-GCN implementation. Adaptive adjacency remains
the strongest graph family, but “more expressive graph topology” is not itself
a supported improvement mechanism.

---

## P-02/P-03 — Legal int8 package for the verified `a20` stack: PASSED
**Date:** 2026-07-30 · **Tier:** deployment

`code/package_ensemble.py` deterministically stores the complete 44-member
`astgcn_a20` inventory with per-tensor symmetric int8 weights. The resulting
`research/artifacts/model_astgcn_a20_int8.pth` is **82,696,132 bytes**
(82.696 MB), SHA-256
`d5a3ee6ee46df5e48e0eff2d06bff66dd9e2ee7c3dc2cace1b63d4d14142c286`;
sidecar and tensor-payload verification both pass.

`code/infer_packaged.py` regenerated
`submissions/sub_astgcn_a20_int8.csv` (**16,517 bytes**, SHA-256
`b9e419f3b4a904731ce36a659bbb50714aeee730855cdcd046ead4fe88cffd36`).
It differs from the verified fp32 CSV on exactly one row:
`SM_test_0303`, fp32 class 28 versus int8 class 12.

**Decision:** the 100 MB package gate is passed for the full accepted
architecture-diverse inventory. Do not attach the verified 0.53731 score to the
int8 CSV: the one changed public/private-unknown row makes its leaderboard
score unverified until uploaded.

---

## EXP-040/SUB-013 — Adaptive ST-GCN: **0.53731 VERIFIED NEW BEST** (108/201)
**Date:** 2026-07-30 · `sub_astgcn_a20.csv` · user-verified Kaggle result · **Tier:** explore→adopt

The adaptive multi-partition graph stream (`astgcn_v1`, 0.842M params) scored
**59.556% four-fold micro**: 58.90 / 64.78 / 56.57 / 57.84. It is the strongest
single skeleton stream so far, +2.96 points over MultiTCN. Three-pass temporal
jitter slightly hurt, 59.556 → 59.519 (91 changed; 20 rescues, 21 harms), so
center inference was retained.

The accepted `a20` assembly adds 20% Adaptive ST-GCN inside the existing
MultiTCN candidate:
- OOF: **61.037%**, versus 60.037% without the adaptive member;
- fold OOF: **61.869 / 65.973 / 58.050 / 58.284**;
- marginal gain: exactly **+1.00 OOF point**, positive on all four folds
  (+0.74 / +1.63 / +1.48 / +0.15);
- test argmax changes versus MultiTCN: 27/405.

The verified LB rose `105/201 → 108/201`, a **three-clip / +1.493-point**
improvement. This is the strongest recent evidence that a genuinely new
inductive bias transfers better than same-family accumulation.

**Decision:** Adaptive ST-GCN is adopted as a core diversity family. The exact
fp32 score-producing stack is about **328.7 MB fp32 / 164.3 MB fp16**; the
later P-02/P-03 audit encoded its full 44-member inventory in a legal
82,696,132-byte int8 artifact, with one test argmax differing from fp32.

The `>0.83` target remains 167/201; the verified gap is now **+59 clips**.

---

## EXP-042 — Cross-user supervised contrastive (SupCon) screen: REJECTED
**Date:** 2026-07-30 · **Tier:** crazy

The fold-0 SupCon variant scored **56.38%**, versus **59.20%** for the paired
plain MultiTCN: **−2.82 points**. The hoped-for cross-user representation gain
did not survive supervised fine-tuning at this recipe.

**Conclusion:** stop after fold 0. This implementation of SupCon is closed; it
does not invalidate every competition-data pretext, but “contrastive
pretraining” is no longer an unqualified high-priority claim.

---

## EXP-041 — Identity screen: REJECTED
**Date:** 2026-07-30 · **Tier:** explore

The Identity variant scored **56.08% on fold 0**, versus the paired MultiTCN
reference **59.20%**: **−3.12 points**.

**Conclusion:** decisive same-fold harm; stop after the screen and spend no
additional folds or ensemble bytes.

---

## SUB-012 result — **0.52238 VERIFIED NEW BEST** (105/201) · MultiTCN transfers, narrowly
**Date:** 2026-07-30 · `sub_multitcn_g50.csv` · user-verified Kaggle result

The MultiTCN candidate improved the verified public score by exactly one clip:
`104/201 → 105/201` (+0.4975 points) over block8/block12. Its OOF assembly was
60.04% versus 58.74% for reconstructed block8; however, most of that apparent
gain came from scalar weight changes. Against the exact `g50/i175` weight
control (59.44%), adding 7.5% MultiTCN contributed **+0.59 OOF point**. This is
directionally consistent with the one-clip LB gain, but still only one public
observation.

**Decision:** MultiTCN is the best new representation member and remains active.
Do not interpret a one-clip gain as a calibrated +0.5 expectation. The exact
score-producing stack is about **315.0 MB fp32 / 157.5 MB fp16** because
inference averages four checkpoints per tag, so it is **not package-legal**
under the 100 MB total cap. Compact subset selection, validated full-data
refits, quantization, or distillation is now part of the accuracy path—not a
later packaging chore.

The standing `>0.83` target requires at least 167/201, still **+62 correct
public clips** from the verified best.

---

## EXP-039/039b — BiGRU diversity stream: REJECTED
**Date:** 2026-07-30 · **Tier:** explore

`skel_bigru` (0.654M params) scored **53.34% four-fold micro**:
54.75 / 56.46 / 50.81 / 51.33. This is −3.26 points below MultiTCN and below
the established TCN block members. Three-pass temporal jitter further reduced
OOF 53.33 → 53.07 (4 rescues, 11 harms).

**Conclusion:** recurrent order sensitivity did not create useful accuracy or
TTA diversity at this sample size. BiGRU is rejected; no fusion/submission
spend.

---

## EXP-038 — MultiTCN mixup α=0.2: INCONCLUSIVE, REJECTED by adoption gate
**Date:** 2026-07-30 · **Tier:** explore

On screening folds 0 and 2, mixup scored **55.74%** versus the paired MultiTCN
reference **55.60%**: fold deltas −0.15 and +0.44 point. The +0.15 mean is far
below the paired-noise/adoption threshold and the signs disagree.

**Conclusion:** there is no evidence that convex interpolation of pose
sequences improves subject generalization. Stop after the two-fold screen; do
not spend two more folds or add the member.

---

## EXP-037 — Subject-adversarial MultiTCN (DANN): REJECTED
**Date:** 2026-07-30 · **Tier:** explore

Two gradient-reversal strengths were screened on fold 0. Re-evaluation of their
saved best checkpoints scored **56.23%** and **55.19%**, versus **59.20%** for
the paired plain MultiTCN: −2.97 and −4.01 points. The uneven user×class support
makes subject removal conflict with class learning; stronger invariance is not
automatically useful.

**Conclusion:** decisive same-fold harm. DANN is closed without spending the
remaining folds.

---

## EXP-036 — Pad+mask TCN at T=64: REJECTED
**Date:** 2026-07-30 · **Tier:** exploit

Validity-aware, no-stretch padding scored **55.34% four-fold micro**
(55.79 / 58.25 / 53.03 / 54.29), equal to the corrected dynamic-truncation TCN
and −1.26 points below MultiTCN. Temporal-jitter OOF changed only 11/2700
predictions and added 0.04 point.

**Conclusion:** preserving raw length with this padded TCN does not recover the
measured trim/domain gap. Reject this implementation. Duration-aware designs
would need a genuinely different mechanism, not another wider padded TCN.

---

## EXP-035 — Joint/bone/motion MultiTCN: ADOPTED as a diversity member
**Date:** 2026-07-30 · **Tier:** exploit

`skel_multitcn` separates joint position, velocity, and bone vectors into three
TCN branches before feature fusion. It scored **56.60% four-fold micro**
(59.20 / 60.92 / 51.99 / 54.29) with 1.884M parameters—the best single
from-scratch skeleton stream so far.

Assembly audit:
- reconstructed verified block8: 58.741% OOF;
- scalar control (`g50`, IMU 0.175): 59.444%;
- same control + 7.5% MultiTCN: **60.037%**;
- isolated MultiTCN contribution over its control: **+0.593 point**;
- test argmax changes versus block8: 21/405.

Three-pass temporal jitter hurt MultiTCN solo, 56.59 → 56.41, so the submitted
candidate used center probabilities. SUB-012 records the resulting LB outcome.

---

## EXP-034 — Corrected dynamic truncation augmentation: SOLO WIN, ensemble-flat
**Date:** 2026-07-30 · **Tier:** exploit / infrastructure correction

The training loader previously used persistent workers while mutating
`epoch_seed`; worker copies therefore reused a frozen augmented view. After
workers were restarted each epoch, truncation-only `skel_dynaug` scored
**55.34% four-fold micro** (56.38 / 58.84 / 52.88 / 53.25), +0.59 over the
original `skel_jvb_big`.

Held-out temporal jitter modestly improved this member 55.33 → 55.48, but adding
it to the `g40` block changed OOF only 59.593 → 59.630 (+0.04) and produced the
same clean/TTA submission argmax. **Adopt the loader fix; do not adopt this
extra member into the package on current evidence.**

---

## EXP-033/SUB-011 — 12-member + test-TTA assembly: NESTED 59.33
**Date:** 2026-07-29 · stgcn_w96_s1 53.79 (best GCN member) · skel_w192 54.30. Blocks: 7-TCN soup + 4-GCN soup (grid still rising at 0.4; shipped 0.65/0.35) + IMU pair + depth 0.1. Test side regenerated with 3-pass jitter-TTA on every member. **NESTED = all-tuned = 59.33, w=0.7/0.2/0.1 unanimous across folds.**

**Verified result update:** `sub_block12_tta.csv` scored **0.51741 = 104/201**,
exactly tying block8 and missing the ~0.533 projection by about three public
clips. The CSVs differ on 18/405 test argmaxes, so equal public scores do not
imply equal private behavior.

Post-hoc member audit: TCN soup6→7 fell 56.259→56.185; GCN soup3→4 fell
55.481→55.444; the fixed skeleton block fell 57.852→57.704; and the fixed final
stack remained exactly 1602/2700. The nested increase did not correspond to a
fixed-stack top-1 gain. Because new members and TTA were bundled, this
submission cannot assign causality. Same-family member accumulation is now
gated by isolated paired evidence.

---

---

## SUB-010 score claim — **0.52736 UNVERIFIED** · EXP-032b IR-4th-stream: no gain
**Date:** 2026-07-29 · **Data-integrity correction 2026-07-30:** the supplied
Kaggle screenshots do not contain `sub_block10.csv`, and no independent score
artifact supports 0.52736. Preserve the historical claim, but exclude it from
the verified ladder and all “current best” statements until Atharv confirms it.
The prior inference that the CV→LB offset was narrowing is therefore withdrawn.
IR as a fourth fusion stream was flat at all tested weights and remains closed.

---

---

## EXP-031/SUB-010 — 10-member assembly: NESTED 58.70
**Date:** 2026-07-29 · mildaug (4-fold ≈54.97) joined TCN soup (now 6); imu_lstm (25.05 solo) added +0.3 to the IMU block (29.93); depth re-entered fusion at w=0.1 (0.7/0.2/0.1). NESTED 58.70 (+0.26), all-tuned 59.33. **SUB-010 staged: sub_block10.csv, projected ~0.520.**

---

---

## EXP-032 — W-01 logit-gate fusion: FAIL (−6.8 nested)
**Date:** 2026-07-29 · **Tier:** explore

2-layer MLP over concatenated stream log-probs, trained per-fold nested: **51.96 vs 58.74 scalar**. Third confirmation of the data-scale law (with EXP-020/021): learned combiners overfit at 2.3k clips; scalar late fusion is the operating point. Fusion-vs-oracle gap is NOT gate-mineable — remaining fusion upside is only better/more members.
**Beliefs updated:** fusion axis CLOSED except member addition; W-02 pairwise specialists demoted (same overfit risk, unless implemented as pure calibration with ≤2 params/pair).

---

---

## SUB-009 result — **0.51741 NEW BEST** (104/201) — 5th consecutive accurate projection
**Date:** 2026-07-29 · Ladder: 0.458 → 0.483 → 0.488 → 0.512 → 0.517. Offset stable ≈ −6.7.

---

## DA-004 — Fourth devil's-advocate pass (inline)
**Date:** 2026-07-29 · **After:** EXP-030/SUB-009

**1. Why probably wrong?** The diversity grind yields +0.5/wave and is visibly saturating; extrapolation plateaus ~0.53-0.55 LB. Without a NEW mechanism the 80+ directive is unreachable, and whether 0.53 even makes top-15 is STILL unknown — the LB harvest has now been flagged in three consecutive DA passes (only Atharv can do it).
**2. Never-questioned (current round):** (a) fusion is still SCALAR-weighted — a small per-clip logit-gate MLP (2 layers over concatenated stream logits, nested-evaluated) has never been tried; part of the fusion-vs-oracle gap is mineable; (b) the 40-class space is treated flat — pairwise specialist recalibration for the top confusion pairs (Read↔Turn_pages, Sweep↔Mop, Pour↔Stir) untried; (c) multi-window voting per clip beyond jitter-TTA untried; (d) ST-GCN member seed-variance unmeasured.
**3. Gold-medalist critique:** submission cadence now good; the big hole is REPRODUCIBILITY PACKAGING — Selection Stage requires organizers to rerun us, and our pipeline is scattered scripts with hardcoded paths. Report + package are 30% of the final score and cost nothing in GPU time. Start now, not at deadline.
**4. Steelman (partially adopted):** "CV-mineable mass left is ~2-4 pts; the rest is representation/data-locked. Accept ~0.53-0.55 terminal, pivot fully to packaging/report." — Adopted as a PARALLEL track, not a pivot: improvement continues per the standing directive while the deliverable package gets built alongside.
**Queue produced:** W-01 logit-gate fusion (nested); W-02 pairwise specialists; W-03 imu-LSTM member + mildaug folds 1,3 + stgcn seed variance; P-01 packaging artifact (single model.pth + inference.py + README, fp16, ~42 MB).
**Counter reset:** 0/10.

---

---

## EXP-029/030/SUB-009 — Ladder 2: GCN soup + reassembly → NESTED OOF 58.44
**Date:** 2026-07-29 · **Tier:** exploit

stgcn_s1 51.90 · stgcn_w96 52.71 (best GCN member) · skel_mildaug 53.89 folds 0,2 (+0.3, in-noise; parked as member candidate). **GCN-soup3 = 55.48 alone** (≈ TCN-soup5 55.96 — two near-equal decorrelated blocks). Block grid still rising at 0.4 (57.93); shipped conservative 0.7/0.3 → **NESTED fusion 58.44** (+0.5; all-tuned 58.74). SUB-009 staged: sub_block8_gcnsoup.csv, projected ~0.517-0.52.
**Next:** DA-004 (due next experiment); block-mix nested exploration; second imu arch member.

---

---

## EXP-028 — S-04 TENT-GN cohort adaptation sim: FAIL (gated, 0 submissions spent)
**Date:** 2026-07-29 · **Tier:** explore

**Result:** per-user entropy-min on GN affines (3 ep, lr 1e-3): Δ = exactly 0.000/16 users (params moved too little to flip anything). Sensitivity at lr 1e-2/10 ep: params move (L2 0.69), 9/149 flips, accuracy DROPS 0.456→0.423, entropy oscillates (2.25→2.29) — the objective doesn't descend usefully on this tiny affine set.
**Conclusion:** TENT-GN dead on this architecture. Transduction family 0-for-2 (with EXP-016). Last reserved variant: head-only + τ≥0.9 pseudo-labels on the FUSED probs at higher base.
**Beliefs updated:** B-007/transduction: adaptation magic is not hiding in easy variants; the leader (if adapting) does something else or from a much higher base.
**Next:** diversity ladder continues (stgcn seed/width members, mild-aug retry Q-11c); DA-004 in 3.

---

---

## EXP-027 — Hands/upper-body crop stream (C-01): FAILED TO BREAK THE CEILING
**Date:** 2026-07-29 · **Tier:** crazy

**Result:** hroi solo 24.51 (folds 0,2); ensemble +0.10 weight → +0.23 (58.17→58.40) — under the 2×SE adoption bar. **Decisive evidence: 0.00 accuracy on the target object classes themselves** (Watch_TV, Play_games, Phone_call, Write all 0.00 solo) despite the crop.
**Conclusion:** The object-in-hand error mass is REPRESENTATION-locked, not framing-locked: from-scratch small CNNs at this data scale cannot learn object identity at any crop, and pretraining is banned. C-01 parked (its +0.2 can be reclaimed at final packaging if it survives full-fold verification).
### Failure analysis
**3 reasons:** (1) object recognition needs texture priors that 2.3k clips can't teach; (2) upper-crop of a loose person-ROI often misses the actual hands region; (3) IR contrast on small objects is poor at 112px.
**3 alternatives:** (1) crop quality (skeleton wrists unavailable in pixel space — can't verify); (2) 60-ep budget; (3) folds 0,2 screen noise.
**3 follow-ups:** (1) SSL-init the visual STEM (EXP-021's +3 might transfer to CNNs — last visual play); (2) accept the cap: Watch_TV/Play_games are also data-poor (12-40 train clips) — mine non-object classes instead; (3) at packaging, test hroi's +0.2 on full folds.
**Beliefs updated:** B-006 final form: visual object-route is CLOSED under the no-pretrained rule at this data scale. The remaining routes to higher accuracy: skeleton-block depth, cohort/transductive adaptation (S-04), calibration/fusion refinement, and per-class specialists on non-object confusions.

---

---

## SUB-008 result — **0.51243 NEW BEST** (103/201; +2.5 over SUB-006)
**Date:** 2026-07-29
Beat the 0.500 projection: offset narrowed to −6.7 (was −7.9) — ensemble diversity transfers BETTER to test users than CV predicted (plausible: averaging washes subject-specific errors hardest exactly where subject shift is largest). Ladder: 0.458 → 0.483 → 0.488 → 0.512.
**Beliefs:** B-004 ↑ (4th accurate-or-better prediction); B-013's diversity corollary now LB-proven twice.
**Next:** C-01 hands-crop stream from roi224 upper-crop (zero re-extraction), S-04 cohort sim, stgcn-weight nested exploration.

---

---

## EXP-026/SUB-008 — 5-seed soup + ST-GCN block + fusion: NESTED OOF 57.93 (+1.8)
**Date:** 2026-07-29 · **Tier:** exploit (ship)

Seeds s3 54.75 / s4 54.86 (5-seed σ 0.15). Ladder: single 54.74 → soup5 55.96 → +0.2×stgcn 57.04 (grid still rising at 0.3: 57.37 — explore under nested next) → nested fusion with imu_inv **57.93** (all-tuned 58.22; optimism gap 0.3 healthy). Depth weight remains 0.
**SUB-008 staged:** sub_block5_stgcn.csv — projected LB ≈ 0.500 (offset −7.9).
**Beliefs:** B-002/B-013: diversity ladder works — each decorrelated member adds; skeleton BLOCK now ~57 vs single-model wall 55.
**Next:** C-01 hands-crop stream (ceiling-breaker #1), S-04 cohort adaptation sim, stgcn weight/nested exploration.

---

---

## EXP-025 — ST-GCN stream: individually weaker, ensemble WINNER (+1.6)
**Date:** 2026-07-29 · **Queue ID:** Q-30 · **Tier:** explore

**Result (folds 0,2):** ST-GCN alone 52.04 (1.06M) vs TCN 53.59. BUT ensemble: TCN-3-seed-soup 54.77 → **56.40 at soup+0.2×stgcn (+1.6)**; argmax agreement only 61% — strong decorrelation. Graph-conv inductive bias sees different errors than temporal conv.
**Conclusion:** Architecture diversity > capacity for ensemble gains (as B-013 implied). ST-GCN adopted as a skeleton-block member; folds 1,3 queued to complete OOF; full fusion re-tune after seeds s3/s4.
**Beliefs updated:** B-002 refined: the skeleton BLOCK (multi-arch ensemble) has headroom beyond the single-model ~55 wall.

---

---

## SUB-006/007 — LB pair: imu-inv CONFIRMED (+0.5, new best 0.48756); refit-18+TTA REFUTED (−1.5)
**Date:** 2026-07-29
- sub_soup3_imuinv → **0.48756** (98/201; predicted 0.483-0.487 ✓ — CV→LB calibration holding at −7.9±1).
- sub_ship1_refit_tta → 0.47263. **Refit-on-18 with last-epoch checkpoints transfers WORSE than val-selected fold models.** Causes (ranked): last-epoch ≠ val-best (OneCycle end-state); 150 ep no-val overfit; TTA-jitter mismatch; 0.5/0.5 dilution of proven soup. Shelved — retry only with early-stop at CV-derived epoch budget.
**Beliefs updated:** B-004 ↑ 85% (third consecutive accurate LB prediction); "more data always helps" nuance — checkpoint SELECTION matters more than +2 users.
**Next:** architecture-diversity wave (ST-GCN stream + 5-seed soup) per Atharv's use-the-headroom directive; rules re-confirmed: transformer ARCH legal (was dropped on merit), no pretrained, no LLMs anywhere.

---

---

## S-01/S-02/S-03 — Ship-mode assembly: refit-18 + TTA + package audit → SUB-007 staged
**Date:** 2026-07-29 · **Tier:** exploit (ship)

- **S-01 refit-on-18:** 3 skeleton seeds (150 ep) + IMU-inv (60 ep) retrained on ALL 18 users, no val (fixed hyperparams from CV era). No OOF possible — gain rides on +22% more training users (esp. always-train users 5/21 now contributing to every member).
- **S-02 TTA:** 3-sample temporal-jitter averaging on refit members at inference.
- **SUB-007:** sub_ship1_refit_tta.csv = skel(0.5 fold-soup + 0.5 refit-soup) 0.8 + imu(same mix) 0.2. Prediction histogram sane (Walk 59 ≈ train prior).
- **S-03 package audit (historical estimate):** 7 skel + 5 imu members = **41.8 MB fp16** (83.6 fp32) when counted as one checkpoint per member. **Correction after SUB-012:** score-producing inference averages four fold checkpoints per tag, so this estimate did not describe the actual ensemble package and must not be used for compliance.
**Expected LB:** 0.49-0.51 (refit +1-2 hypothesis on top of 0.483 baseline — SUB-006/007 pair measures it).
**ATHARV:** submit sub_soup3_imuinv.csv AND sub_ship1_refit_tta.csv — the pair isolates the refit+TTA effect on LB.

---

---

## S-05 — Label sanity check (A3, flagged by three DA passes): PASSED
**Date:** 2026-07-29 · **Tier:** diagnostic (CPU)

**Setup:** best fused OOF (3-seed soup 0.8 + imu_inv 0.2); Signature-B = confident (>0.7) wrong prediction that equals a recording-group-mate's label (folder-shift errors would concentrate here).
**Result:** 14/2700 = **0.52%** hits; inspection shows adjacent-class confusions (6↔7 drink/eat, 9↔10 pour/stir, 22↔23) not systematic mislabels; 2 of 14 are the census's known 2-modality oddballs (07_user22_3-1-3, 37_user23_7-1-2). High-conf-wrong tail tops out at 0.79 conf — no smoking guns.
**Conclusion:** Train labels are clean (noise ≤0.5%); label-cleaning has no meaningful upside. A3 settled after 27 experiments.
**Beliefs updated:** A3 verified — removed from every future DA checklist.

---

---

## DA-003 — Third devil's-advocate pass (inline)
**Date:** 2026-07-29 · **After:** EXP-024

**1. Why is the current direction probably wrong?** We are grinding +0.5-4 levers (soup +0.8, imu-inv +4 on a 0.2-weight stream ≈ +0.8 fused) while the diagnosed 34.7% object-in-hand error mass has NO live experiment against it — Q-97 (hands/upper-body crop stream) was queued at EXP-017 and never built. Best-case grind arithmetic: fused ~57-58 OOF → LB ~0.50; if rank-15 sits above 0.50 we lose while "winning" every screening.
**2. Never-questioned assumptions:** (a) full-length training clips are the right training view when test is trimmed — trim-matched training (trim as the DEFAULT view, not an aug) never tried; (b) we predict test with 4 fold-models (each trained on 14 users) and have never refit on ALL 18 users (+1-2 free, flagged in DA-002 Q3, still undone); (c) label sanity — THIRD consecutive DA flag, still never checked.
**3. Gold-medalist critique:** (a) still flying blind on the actual Top-15 bar — LB harvest pending since DA-002 (only Atharv can); (b) provable E/L cohort split never used at inference (per-cohort GN-affine adaptation sim never run); (c) efficiency score (10% of final) unexamined — our ~5 MB total is likely near-max points, worth confirming and PROTECTING in design decisions; (d) submission cadence wasteful: ~245 available, ~6 used — every honest candidate is cheap calibration.
**4. Steelman (adopted):** "The campaign is now over-optimized for process, under-optimized for shipping. With ~7 weeks left the highest-EV block is boring: refit best configs on all 18 users, add TTA, soup everything into one ≤100 MB model.pth, submit variants daily, and lock the reproducible Selection-Stage package early. The marginal screening experiment returns <0.5 pts; packaging/refit returns 1-3 guaranteed." Verdict: correct — this week pivots to SHIP MODE, exploration continues only in the crazy tier (hands-crop stream is the one sanctioned moonshot).
**Queue produced:** S-01 refit-on-18 (all adopted configs); S-02 temporal multi-crop TTA; S-03 soup-everything packaging + efficiency audit; S-04 E/L cohort GN-adaptation sim; S-05 label sanity (finally); C-01 hands/upper-body crop stream (crazy tier).
**Counter reset:** 0/10.

---

---

## EXP-022/023/024 — Three-lever screening (folds 0,2; paired vs jvb_big 53.59 / imu_aug 24.8)
**Date:** 2026-07-29 · **Tier:** exploit

- **EXP-022 LS-off:** 52.93 (−0.7) — keep LS=0.1; calibration-blame theory unsupported.
- **EXP-023 motion-person:** 52.56 (−1.0) — person-0 stays. Mirror reflections move WITH the person → motion energy can't separate them; policy adds switching noise elsewhere. Mirror fix would need depth-consistency cues (parked).
- **EXP-024 IMU-inv:** **28.94 (+4.1, clear win)** — dropping absolute angle+mag channels (subject/mounting leak), adding |acc|/|gyro|, fixing quat hemisphere. ADOPTED; 4-fold retrain (imu_inv4) launched for the fusion stack. Confirms EXP-003's failure-analysis hypothesis #3 two weeks late — the leak channels were fighting DG the whole time.
**Beliefs updated:** B-009 ↑ (IMU ceiling higher than 27; orientation-invariance was the blocker); skeleton input-space levers exhausted (B-002 note: remaining skeleton gains = architecture diversity + ensembling).

---

---

## EXP-021 — SSL→fine-tune transformer: GATE FAILED, transformer line closed
**Date:** 2026-07-29 · **Queue ID:** M-03/Q-94 · **Tier:** explore/crazy

**Result:** cross-modal masked pretrain (40 ep, 3,338 clips incl. test; recon loss 0.30→0.0146) → fine-tune = **48.19%** folds 0,2. vs from-scratch transformer 45.23 (+3.0 from SSL) but vs GATE 56.15 → **−8. Transformer fusion line CLOSED per RESET-001 gate. Late fusion is the permanent architecture.**
**What survives:** (1) cross-modal SSL demonstrably works at this scale (+3.0) — retained as an option for pretraining per-modality STEMS/encoders later (e.g. init the visual CNN from an SSL objective); (2) M-01 aligned dataset reusable.

### Failure analysis
**3 reasons:** (1) 2.3k labeled clips cannot polish a 3.8M-token-mixer even with SSL init — the late-fusion streams each exploit stronger inductive bias per parameter; (2) vis stem tokens weak (visual stream itself only ~30%) — transformer can't select good vis information that isn't there; (3) reconstruction pretext may reward low-level smoothness, not class-discriminative structure (loss 0.0146 ≈ trivially predictable signals dominate).
**3 alternatives:** (1) 60 fine-tune epochs too few for a pretrained trunk (no LLRD/warmup tuning); (2) aux-head weight 0.3 may dominate gradients; (3) fold-0,2 screen ±1.4 SE — but an 8-pt gap is far outside noise.
**3 follow-ups (parked, low priority):** LLRD fine-tune; contrastive (not reconstructive) pretext; SSL-init only the visual stem inside the LATE fusion stack.
**Beliefs updated:** B-013 stands; architecture question settled — the campaign is now: better per-modality streams × late fusion × soups (+ gated transduction).
**Next:** cheap untested exploits — Q-85 LS-off, Q-86 person-selection, IMU orientation fix — paired on screening folds.

---

---

## EXP-020 — Fusion transformer from scratch: BELOW GATE
**Date:** 2026-07-29 · **Queue ID:** M-02 · **Tier:** explore

**Result:** mmfuse_v1 (3.78M, d256 L4, aligned tokens, modality dropout, aux heads) = **45.23%** folds 0,2 — vs gate 56.15 (nested late fusion) and ~52 for plain skeleton on those folds.
**Conclusion:** From-scratch transformer fusion loses badly at 2.3k clips, as RESET-001's gate anticipated. NOT dead yet: the design's bet is SSL+transformer; EXP-021 (cross-modal masked pretrain 40 ep on 3,338 clips → fine-tune) is the deciding run. If EXP-021 also < 56.15 → transformer line dies, fall back to late fusion permanently.
**3 reasons if it fails for good:** (1) data scale below transformer viability even with SSL; (2) vis stem too weak to give the trunk useful tokens (garbage in); (3) aligned-grid pad/mask regime needs longer training than 60 ep.

---

---

## SUB-005 — soup3+imu+droi_big fusion → LB 0.48258 (best yet)
**Date:** 2026-07-28 · 97/201 public clips (prev best 92). Predicted 0.48-0.49 ✓.
**Offset calibrated: LB ≈ honest-nested-OOF − 7.9** (56.15 → 48.26). trunc-aug reclaimed ~1 of the original −9. CV→LB tracking is now trustworthy for planning (B-004 ↑ 80%).
**Beliefs updated:** B-004 ↑; ensemble/soup mechanics verified end-to-end on LB (+2.5 over SUB-001 with same modality set).
**Next:** M-02 fusion transformer (mine the 56→63 oracle gap), M-03 cross-modal SSL (attack the 35% no-stream-right mass), ST-GCN, person-fix, LS-off.

---

---

## EXP-013b/018/019 — Depth folds, seed replication, soup + nested fusion, SUB-005 prepared
**Date:** 2026-07-28 · **Tier:** exploit

- droi_big folds 1-3: 28.81 ± 1.6 micro (4-fold ≈ 29.6 with fold0 31.94; +3 over 40-ep recipe). Depth stream ≈ 30.
- **Seed variance (EXP-018): skel_jvb_big seeds 0/1/2 = 54.75 / 55.08 / 54.67 → seed σ ≈ 0.18%** — training is stable; fold σ (2-3%) dominates; paired deltas ≥1% on 4 folds are meaningful. (Resolves DA-002 Q5's top concern for skeleton; visual seed σ still unmeasured.)
- 3-seed soup: 54.74 → **55.56** (+0.8, free, ensemble-legal).
- **Nested fusion OOF: 56.15% micro** (all-tuned 56.63, w=0.6/0.2/0.2 — optimism gap only 0.5). New honest best.
- SUB-005 file: sub_soup3_fusion.csv (soup + imu + droi_big, all-tuned weights on test). Predicted LB ≈ 0.48-0.49 (offset −8±2 with trunc-aug partially reclaiming trim).

---

## RESET-001 — From-scratch review (scheduled at 25 experiments)
**Date:** 2026-07-28 · **After:** EXP-019

**"If we started from scratch today, knowing everything, what would we build?"** — answered in full by DA-002 Q3 and ratified here: per-modality stems (keep current TCN-jvb + ROI CNNs) feeding a small temporal fusion transformer on the shared 10 Hz grid (modality-token dropout, duration token, pad+mask not stretch), trained in 2 stages: (A) **cross-modal masked pretraining on all 3,441 unlabeled clips incl. test** — the only legal transfer under the no-pretrained ruling, subsumes distillation; (B) supervised fine-tune with mixup/trunc-aug/modality-dropout. Multi-seed soup (now proven +0.8). Gates: fusion-transformer must beat late fusion by week 3 or fall back; transduction only at ≥60 CV.
**Gap vs current solution:** current = scalar late fusion, no SSL, stretch-resampling. The 56.15-vs-63.4 oracle gap (per-clip stream selection) and the 34.7% no-stream-right error mass (object-in-hand classes, EXP-017) are exactly what stages A+B target.
**Migration queued:** M-01 aligned multimodal Dataset (pad+mask+duration), M-02 fusion transformer, M-03 cross-modal masked pretraining, M-04 refit-on-18-users + packaging. Est. ~2 weeks alongside continued stream work.
**Verified:** metric = micro-OOF with nested tuning (CV↔LB tracking continues); crazy-tier spend ≈ 8% (target 10) — acceptable.

---

---

## RULING — Organizer clarifications received (via Atharv, 2026-07-28)
1. **NO pretrained weights at all** — strict from-scratch. Consequences: (a) our pipeline already compliant; (b) DA-002's cluster hypothesis (0.73-0.77 ≈ pretrained visual recipes, ~85%) now implies those teams are DISQUALIFIABLE at reproduction → effective private-LB Top-15 bar drops for rules-clean teams; (c) SSL pretraining on competition data (Q-94, no external weights) is THE only visual transfer path — promoted to top priority; (d) pretrained probes (EXP-015/T1) retain only diagnostic value; T1 rerun cancelled.
2. **Ensembles legal if total ≤100 MB** — multi-stream × multi-seed soups are fair game. The contemporary “~5 MB” estimate described an early single-model state; it was later invalidated for fold ensembles by the SUB-012 package audit. Multi-seed averaging was promoted at this point in history.
3. **Test-time transductive processing legal** — TENT-style/statistics adaptation inside inference code officially allowed. Gated behind base ≥60 CV per EXP-016 discipline.
**Also:** previous session's background chain died with the process (T1 probe at ep15, droi_big after fold0). droi_big folds 1-3 + skel seeds 1-2 relaunched (chain5).
**Beliefs updated:** B-005 SETTLED (ensemble counting known); B-006 path fixed to SSL-only; NEW B-016: rules-clean pipeline is a Selection-Stage asset — the private-LB bar for ADVANCING is likely below the public cluster's scores.

---

---

## EXP-017 — First confusion/oracle analysis (DA-002 Q2 item #1)
**Date:** 2026-07-28 · **Tier:** diagnostic (CPU)

**Setup:** OOF fusion skel_jvb_big(0.6) + imu_aug(0.2) + depth_aug(0.2), fixed weights (no grid → honest-ish). New best fusion: **56.15% micro OOF** (jvb_big OOF alone 54.74).
**Findings:**
1. **Worst classes are object-interaction classes** — Watch_TV 0.00 (n=12), Play_games 0.05, Wipe_bowls 0.11, Make_a_phone_call 0.16, Take_and_use_tableware 0.21, Use_a_mobile_phone 0.23, Read_documents 0.23↔Turn_pages 0.26 (mutual confusion), Write 0.28→Tap_keyboard. Pattern: classes separated by WHAT THE HANDS HOLD. Skeleton is structurally blind to it; current visual too weak to supply it. Motion classes excellent: Walk 0.97, Lie_down 0.85, Jumping_jacks 0.83, Jog 0.81.
2. **Error decomposition:** total err 43.9% = 9.1% fusion-recoverable (some stream right) + **34.7% needs better streams**. Better fusion alone caps at ~65%.
3. Low-support × hard: Watch_TV/Play_games also have the least train data (12-40) — even their ORACLE coverage is 0-7%.
**Implication:** The visual stream's specific job is hands+object resolution — favors IR (texture) over depth, favors ROI/resolution/pretraining (T1 tonight), and possibly an upper-body/hands sub-crop stream. This is where the missing ~35 points live.
**Beliefs updated:** B-013 sharpened: "stream quality" = specifically object-discriminative visual capability.
**Next:** T1 outcome routes everything; hands-crop stream added to queue (Q-97, explore).

---

---

## DA-002 — Second devil's-advocate pass (5 agents) + immediate follow-ups
**Date:** 2026-07-28 · **After:** EXP-016

**Q1 — Leader mechanism (KEY DERIVED FACT: public split = 201 clips; every score is k/201; leader = 168/201; private = 204 clips → 1 public clip = 0.4975%, shakeup ±2-3%).** Mechanism posteriors for 0.836: M1 strong transfer-visual recipe (ImageNet-init, possibly rule-gray) 30-40% — our EXP-015 probe was too crippled to refute it (112px/1-ch/25ep); M5 MANUAL TEST LABELING ~25% (IR is a face-visible photo stream; 405 clips = one afternoon; 83.6% sits mid-band of human accuracy; unenforceable on public LB, dies at Selection Stage); M2 public-LB probing ~20% (score granularity reveals exact correct-count; ≤190 subs available; leader needs only +14 clips over cluster base); M3 metadata ~5%; rest ~10%. The 0.73-0.77 CLUSTER is ~85% M1 — likely pretrained visual recipes. **Strategic corollary: if leader = M5/M2, the real competitive bar is ~0.76-0.77, and if pretrained is ruled illegal at reproduction, the effective top-15 bar drops further — our rules-clean pipeline gains free places at Selection Stage.**
**Kill/confirm tests:** T1 full-recipe transfer probe (224px, 3-ch, ResNet-18, 60-80ep, IR-ROI, random+subject splits; decision: subj ≥55 ⇒ cliff additive, M1 viable AND our own roadmap; subj ≤40 with rand ≥70 ⇒ cliff proportional ⇒ red alert on all visual plans). T2 LB forensics (leader submission count + staircase; GitHub mirror hides counts — ATHARV: read Kaggle LB directly: leader's entry count, score-vs-time shape, and scores at ranks 10/15/20/30 — the rank-15 number converts the campaign target from fantasy to measured bar).
**Q2 — Still-untested, ranked:** (1) per-class confusion analysis of best fusion — 16 experiments, 4 submissions, NOBODY has looked at a confusion matrix; prices everything else; (2) seed-variance measurement (3 seeds × skel_noaug) — protects all ±1-2 decisions; (3) label sanity via group-distinctness signature (free detector: OOF prediction matching a group-mate's label); (4) LS-off ablation (gates calibration → Hungarian/transduction/fusion); (5) person-0-vs-mirror; (6) IMU orientation/quat fixes (+leak removal); (7) no-stretch+duration (with trim-sim eval); (8) mid-fusion gated on 1+4.
**Q3 — Reset preview ("what would we build from scratch"):** per-modality stems (keep) + small temporal fusion TRANSFORMER on the shared 10 Hz grid (replace scalar late fusion — the 54.3-vs-63.4 oracle gap is per-clip stream selection that scalars can't express) + cross-modal masked pretraining on ALL 3,441 clips incl. test (legal SSL that subsumes distillation; the only transfer we own) + multi-seed soup + gated TENT-style adaptation. Prediction: 0.58 central, 90% CI 0.50-0.66; P(top-15) ≈ 0.4-0.6. Migration ~2 wks, keeps most code. HARD GATE: fusion transformer must hit ≥60 CV by week 3 or revert to late fusion.
**Q4 — Retarget steelman: CONFIRMED. P(0.85) < 2-5%.** 0.85 LB needs ~94% subject-CV (offset math) — no published result on this dataset is within 25 pts. OBJECTIVE.md's real win condition was always Top-15 → Selection Stage; the campaign now explicitly optimizes P(top-15), and the rules-clean pipeline is a Selection-Stage asset (cluster disqualification scenario). 245 submissions remain — abundant; GPU-hours are the scarce resource.
**Q5 — Hygiene audit: most adopted/killed decisions are within noise.** Only "ROI +3.4" clearly survives (2.8σ, 4/4 folds, corroborated). bones +1.8 ≈ 0.6-0.9σ; tn-death <0.5σ (reclassified parked-unmeasured); balanced +1.5 driven by one fold; trunc NEVER measured alone (bundling sin, third offense); EXP-012 +1.7 = p≈0.17 (2/4 folds). NEW DECISION PROTOCOL: paired-delta on same OOF clips (SE≈0.6-0.9%), adopt only if |Δ| > 2×paired-SE or replicated across ≥2 seeds; nested tuning for fusion weights; no bundled changes ever.
**Follow-ups executed immediately:** (a) leaderboard mirror fetched — top-6 only, counts hidden ('—'), leader last-submission Jul 26, cluster ties at 0.75124 (151/201 — /201 grid makes ties common); (b) M3 within-group class-order check on train (radar-ts grouping, 45 signatures with ≥3 groups): order-consistent only 2/45 → NO fixed script; M3 dead properly (first attempt with mangled keys gave a false 61/62 — re-verification caught it); (c) 224px ROI cache rebuild launched for T1.
**Counter reset:** DA at 0/10; from-scratch reset review due in 4 experiments (will adopt Q3's design as its base).

---

---

## EXP-016 — Per-user self-training simulation (Q-91 sim): NEGATIVE
**Date:** 2026-07-28 · **Queue ID:** Q-91 · **Tier:** explore

**Hypothesis:** Per-user pseudo-label fine-tuning on the user's unlabeled clips adds ≥+3% (the "leader's transduction" hypothesis, DA-001 Q1).
**Setup:** transduction_sim.py — skel_noaug fold checkpoints; per val-user: pseudo-label (τ=0.60), fine-tune copy 15 ep @2e-4, re-evaluate. 16 users.
**Result:** **mean Δ −3.2% · median −2.5% · helped 2/16 users** · pseudo-label acc 60-93%.
**Conclusion:** Naive self-training is DESTRUCTIVE at ~50% base accuracy. Do not spend submissions on it. The transduction family is not dead — but the entry bar is a stronger base model and/or gentler adaptation.

### Failure analysis
**3 reasons:** (1) pseudo-label noise (7-40% wrong) compounds through 15 epochs; (2) confident clips are the already-easy ones — fine-tuning on them shifts decision boundaries away from hard clips (distribution narrowing); (3) full-model fine-tuning on 40-80 clips overfits instantly.
**3 alternatives:** (1) τ=0.60 too low; (2) lr/epochs too aggressive; (3) skeleton-only sim understates fusion-level transduction (better-calibrated probs).
**3 follow-ups (parked until base ≥60%):** (1) head-only fine-tune + τ=0.9 + 3 epochs; (2) entropy-minimization/consistency (TENT-style on GN affine params) instead of hard labels; (3) class-balanced pseudo-label selection per user.
**Beliefs updated:** B-013 reinforced (base model quality gates EVERYTHING downstream: Hungarian, transduction, fusion); "leader uses naive transduction" demoted — if the leader adapts, it's not this way, or their base is already ≥70.
**Next ideas:** DA-002 due; fusion re-tune with skel_jvb_big when droi_big lands → new submission candidate.

---

---

## EXP-012 — Skeleton scale-up: jvb + width 256 + 150 ep + trunc-aug
**Date:** 2026-07-28 · **Queue ID:** Q-83 · **Tier:** exploit

**Result:** **54.75% ± 2.89% micro** (4-fold, micro-selected), 2.53M params — new best single stream (+1.7 over skel_noaug 53.04).
**Conclusion:** Positive but sublinear: 4× params + 2.5× epochs + bones + trunc bought +1.7. The TCN family saturates ≈55 micro subject-CV; the in-domain ceiling (73) is not being approached via capacity. Next skeleton levers must change the inductive bias or the data view: ST-GCN-lite (Q-30), joint/bone/motion multi-stream logit ensemble, person-selection fix (Q-86), mixup (Q-34).
**Beliefs updated:** B-002: skeleton solid but its DG gap (~18 pts) is not capacity-limited.

---

---

## EXP-015 — Pretrained ResNet-18 probes (Q-84, DIAGNOSTIC — legality pending)
**Date:** 2026-07-28 · **Tier:** exploit (diagnostic)

**Setup:** probe_pretrained.py — ImageNet ResNet-18, 1-ch conv1 (summed RGB filters), 112×112 ROI crops, 8-frame TSN mean-pool, 25 ep, macro metric.
**Results (macro):**
| stream | random-split | subject-fold0 | from-scratch subject ref |
|---|---|---|---|
| depth-ROI | 45.0% | 25.9% | ~21 (droi_bal) |
| IR-ROI | 46.8% | 27.3% | ~17 (iroi_bal) |
**Conclusions:** (1) ImageNet init ≈ doubles from-scratch visual accuracy → representation matters, and legal SSL pretraining (Q-94) inherits a real prize. (2) BUT even pretrained, cross-subject collapses visual by ~20 pts (46→27) — the visual DG gap is far larger than skeleton's; init does not fix DG. (3) Paper's 90% not approached at 112 px/25 ep/macro — resolution (224+/full-frame context?), 3-ch input, longer training, and micro metric all separate us; a stronger transfer recipe would land higher, but the DG cliff pattern will remain. (4) Strategic: visual streams are worth pushing to the ~40-50 subject range (SSL + res + capacity) as FUSION diversity, not as a backbone that alone reaches 0.85.
**Caveats:** probe recipe deliberately cheap; treat absolute numbers as lower bounds, the random-vs-subject GAP as the robust finding.
**Beliefs updated:** B-006 finalized: visual = mid-strength fusion stream, ceiling recipe-and-DG-bound; B-013 reinforced — no single stream reaches target; the road is (max skeleton) + (SSL visual ~45+) + (fusion) + (transduction/self-training multiplier).
**Next ideas:** Q-94 SSL pretrain (masked/temporal pretext on ALL unlabeled clips incl. test — rules-legal, no external weights); ROI cache at 224 px; Q-91 transduction SIMULATION on CV folds (adapt-per-val-user with pseudo-labels — measures the multiplier without submissions).

---

---

## EXP-008/009/010/011 — Wave 2: balanced skeleton + first visual ROI/IR streams (macro-selected era)
**Date:** 2026-07-28 · **Queue IDs:** Q-80/81/82 · **Tier:** exploit

**Results (MACRO, 4-fold unless noted):**
- skel_bal (balanced sampling): **47.6% macro** vs 46.1 skel_noaug macro reference → +1.5 macro. (Micro effect unknown; balanced sampling on hold after B-012 died — test ≈ train prior.)
- iroi_bal (IR person-ROI, first IR training ever): **17.3% macro** — paper's 90% modality nearly useless from scratch at this recipe.
- droi_bal (depth person-ROI): **21.1% macro** vs 17.7 full-frame depth macro → ROI +3.4 on depth.
- ir_bal (full-frame IR, folds 0,2): killed at fold0 ep30 (~12-17%, info value spent — worse than IR-ROI as expected).
- Feature ablation completions (folds 0,2 macro; jv reference 44.9): **jvb (bones) 46.6 (+1.8)** — adopt; jv+tn (torso-norm) 44.1 (−0.7) — dead.
**Conclusion:** (1) ROI cropping helps but the from-scratch visual recipe remains the wall — three independent data points (depth-rand 32, iroi 17, droi 21) say representation/training, not modality or crop, is the blocker → pretrained probe (Q-84) is the decisive next diagnostic, self-supervised pretraining (Q-94) the likely legal remedy. (2) Bones adopted into skeleton. (3) Never edit files a running chain imports (crashed 3 ablation runs; re-run cost ~20 min).
**Beliefs updated:** B-006 unchanged (recipe-gated); B-013 reinforced.

---

## EXP-014 — E1 trim-shift simulation (inference-only)
**Date:** 2026-07-28 · **Queue ID:** DA-001 E1 · **Tier:** diagnostic

**Setup:** trim_sim.py — quantile-map val skeleton clips onto the TEST frame-count distribution (med 20, p90 41), re-score skel_noaug fold checkpoints on CPU.
**Result:** OOF 53.04% full → 51.33% trimmed → **Δ_trim = 1.7 pts**.
**Conclusion:** Offset decomposition: ≈1.7 trimming + ≈1-2 selection/tuning optimism (DA E3, partially fixed) + ≈5-6 genuine subject shift + public-subset noise. Countermeasure shipped: random-truncation augmentation (aug component "trunc") — in all wave-3 runs.
**Beliefs updated:** B-004: offset now decomposed and mostly attributed to subject shift → DG (B-001) remains the true battle; CV↔LB tracking continues on micro + offset.

---

---

## SUB-003/004 — Prior-corrected submissions: REFUTED B-012
**Date:** 2026-07-28

**Results:** sub_fuse3_prioradj → **0.39303** (−6.5 vs raw 0.45771); sub_fuse3_sinkhorn → **0.39800** (−6.0).
**Conclusion:** Test class prior ≈ TRAIN prior (imbalanced), NOT uniform. Both corrections pushed mass toward rare classes (25/26 got ~13% of predictions) and paid for it. The macro-OOF ≈ LB match in DA-001 was coincidence; the −9pt CV→LB offset is therefore GENUINE domain shift (test users harder — per-user OOF spread 42.8-63.4% supports; plus ~20% trim + 10 s cap), not prior mismatch.
**Metric policy reverted:** micro-OOF is again the primary selection/headline metric (test ≈ train prior); log macro alongside. Balanced sampling: re-evaluate on micro (EXP-008's +1.5 was macro; micro effect unknown — measure OOF).
**What the pair bought:** B-012 killed with certainty for 2 submissions; without the pair we might have balanced-trained the whole campaign in the wrong direction.
**Beliefs updated:** B-012 → DEAD (as "uniform test"); B-004 revised again — micro-OOF −9±2 offset with domain-shift attribution; next diagnostic = trim simulation (E1).
**Next:** E1 trim simulation (inference-only); continue stream-quality wave (B-013 unaffected — governs under any prior).

---

---

## DA-001 — Devil's-advocate pass (5-agent workflow) + zero-GPU diagnostics
**Date:** 2026-07-28 · **After:** EXP-006 + SUB-001/002

**1. Why is our current direction probably wrong?** The plan's own EV arithmetic tops out ~27 pts short of 0.85: skeleton in-domain ceiling is 73% (EXP-006) — DG cannot exceed it; depth fix + fusion + DG are +1-3% levers on a 46% base. The 0.836 leader implies a categorically different mechanism: pretrained visual backbone (if legal) and/or per-user transduction (4 users × ~100 clips, balanced test). **VERIFIED by oracle probe: pick-best-of-3-streams oracle = 63.4% — current streams structurally cannot reach 0.85. Stream replacement/upgrade is the campaign.** (Any-stream top-5 oracle 87.8% — labels are near the top; calibration/transduction can mine it later.)
**2. Never-questioned assumptions (code audit):** (top) train-prior≈test-prior — FALSE, see diagnostics below; val-best checkpoint = order-statistic optimism; label smoothing 0.1 distorts Hungarian costs; person-0 skeleton may be the MIRROR in bathroom classes; T=32 stretch-resampling erases duration + gives per-sample velocity units; CV structurally easier than test (always-train users, coverage-maximized folds); depth invalid=0 conflates with near after /255; IMU quaternion double-cover + absolute angle channels = subject/mounting leak; train/eval temporal sampling mismatch; `hash(sid)` seeding not reproducible (PYTHONHASHSEED) — Selection-Stage risk; MENTAL_MODEL claimed class-balanced sampling + EMA that train.py never implemented; Q-54 BN-adaptation impossible — models use GroupNorm.
**3. Gold-medalist critique:** single seeds; 60-ep budgets; bundled augs (already bitten twice); triple-dipping on OOF (checkpoint selection + fusion weights + Hungarian tuning); no TTA; email drafted but never sent ("DONE" self-deception); 0.66M skeleton = 0.2% of the 100 MB budget — capacity massively underused.
**4. Steelman verdict:** "The CV is broken and every decision made against it is suspect" — CONFIRMED within hours (see diagnostics).
**Diagnostics run (da_analysis.py):**
- **Prior mismatch is THE CV→LB offset:** balanced-mean (macro) OOF = 46.7% vs LB 45.8% — near-exact match; micro-CV 54.3% was the wrong metric. LB 0.458 is at the 0.1th percentile of all C(16,4) user-subset micro accuracies → subject variance CANNOT explain it; prior mismatch CAN. Test behaves ~class-balanced (405 ≈ 40×10). Predicted-class histogram was Walk=60/405 (~15%) vs balanced-expected ~10.
- Per-user OOF accuracy spread: 42.8% (user6) to 63.4% (user19) — huge subject variance confirms DG remains real too.
- Actions taken: macro-accuracy is now the checkpoint-selection AND headline metric; --balanced sampler added; last-epoch checkpoints saved; crc32 seeding; prior-adjusted + Sinkhorn submissions written (sub_fuse3_prioradj.csv, sub_fuse3_sinkhorn.csv) for Atharv.
**Queue entries produced:** pretrained ResNet-18 probe (diagnostic; legality pending email), IR stream (running, EXP-009), transduction tier promoted (balanced-assignment/self-training once base ≥55% macro), label-smoothing ablation, person-selection policy, no-stretch+duration-feature variant, valid-mask channel, LOSO harshness measurement, nested fusion-weight tuning, TTA. Killed: Q-54 as stated (no BN to adapt).
**Counter reset:** next DA after 10 more experiments; from-scratch reset review still due at 25.

---

## EXP-007 — Skeleton aug component ablation (folds 0,2; micro-selected, pre-macro-change)
**Date:** 2026-07-28 · **Queue ID:** Q-11b · **Tier:** exploit

**Setup:** one component at a time, magnitudes from EXP-001b; screening folds 0,2 (no-aug reference on same folds = 52.26%).
**Results:** rot 50.3 (−2.0) · scale 51.1 (−1.2) · jit 50.9 (−1.4) · jdrop 50.7 (−1.5) · tjit 51.0 (−1.2) · rot+jit+tjit 51.9 (−0.3). Feature configs (jvb, jv+tn) crashed due to my mid-chain train.py edit (race — never edit files a running chain imports); re-running in wave 2.
**Conclusion:** EVERY component hurts individually at these magnitudes; rot worst — consistent with station-anchored orientation being a genuine, transferable prior (don't rotate it away). This aug family is not the skeleton DG lever. Alternatives: normalization/features (jvb/tn — wave 2), mixup, capacity+longer training, subject-adversarial.
**Confidence:** 70% (screening folds only, single seed, micro metric).
**Beliefs updated:** B-001 refined — "aggressive geometric aug" is NOT the DG lever for skeleton; station-anchored orientation is signal, not noise.

---

---

## SUB-001/002 — First LB results (public)
**Date:** 2026-07-28 · **Tier:** foundation (calibration)

**Results:** sub_fuse3_nohung.csv → **0.45771** · sub_fuse3_hung.csv → **0.44776** (public LB; leader 0.836, target >0.85 per Atharv).
**Finding 1 — CV→LB offset ≈ −9 pts:** OOF 54.3% → LB 45.8%. Subject-CV overestimates. Candidate causes: (a) 4 test users idiosyncratic (2 possibly different hardware/session era); (b) test clips trimmed ~20% shorter + 10 s hard cap (train has longer clips; our temporal sampling sees different length distribution); (c) public LB is a subset — noise ±2-3%; (d) same-user val leakage in our CV? No — folds are user-grouped; but always-train users 5/21 make CV slightly easier than a true 4-unseen-user test.
**Finding 2 — Hungarian NEGATIVE on LB (−1.0%)** vs +1.1% on CV fusion. Hypotheses: (H1) at 46% base accuracy test probs are worse-calibrated than CV probs → assignment noise dominates (consistent with depth-alone −0.9% at 27%); (H2) test recording groups do NOT obey all-distinct (test clips are trimmed segments — organizers may have sliced multiple clips from one activity); (H3) public-subset noise (~2-4 clips). Action: do NOT apply Hungarian until base model ≥60% AND H2 is tested (check: do same-group test clips get near-identical embeddings/predictions more often than distinct-class train groups would?).
**Beliefs updated:** B-004 ↓ 40% (CV predicts LB direction but with −9pt offset; must track deltas not absolutes and fix CV harshness); B-007 ↓ 50% (Hungarian flipped sign on LB); B-011 mechanism still 95% on train, but its TEST transfer now uncertain (H2).
**Next:** devil's-advocate pass NOW; then improvement wave: depth ROI recipe (Q-70/71), skeleton DG block (Q-11b/Q-10/Q-34), CV redesign to include harder folds.

---

---

## EXP-006 — DG-gap diagnostics: subject-split vs random-split per modality
**Date:** 2026-07-27 · **Queue ID:** follow-up of EXP-002 failure analysis · **Tier:** exploit (diagnostic)

**Hypothesis:** If depth's 27% is a cross-subject problem, random-split accuracy will be high (≥60%); if it's a recipe problem, random-split will be low too.
**Setup:** (a) depth no-aug subject fold0; (b) depth no-aug random 25% split; (c) skel no-aug random split. 40/40/60 ep. NOTE: random split leaks same-recording/session clips across train/val → random numbers are OPTIMISTIC upper bounds.
**Results:**
| modality | subject-CV | random-split | DG gap |
|---|---|---|---|
| depth (no-aug) | 23.7% (fold0) | **32.4%** | ~9 pts |
| skel (no-aug) | 53.0% (4-fold) | **73.2%** | ~20 pts |
**Conclusion:** Diagnosis is bimodal and actionable: (1) **Depth is recipe-bottlenecked** — it can't even fit in-domain (32% vs paper's 90% with pretrained ResNet-50 at full res). Root-cause candidates: person small/off-center at 120×160 full-frame; too few epochs; too small CNN; temporal pooling. DG work on depth is premature. (2) **Skeleton is DG-bottlenecked** — solid recipe (73% in-domain), loses 20 pts to unseen subjects → normalization/mixup/adversarial/aug-ablation is where skeleton gains live. Depth aug wasn't the problem (no-aug fold0 23.7 ≤ with-aug).
**Confidence in conclusion:** 85%.
**Unexpected observation:** Even with same-session leakage inflating it, depth random-split is only 32% — the recipe deficit is worse than any DG effect.
**Beliefs updated:** B-006 revised (see ledger): visual weakness = recipe not modality; B-002 ↑ 65%: skeleton confirmed backbone for now; B-001 refined: the cross-subject cliff is modality-dependent (20 pts skel vs 9 pts depth at current depth quality).
**Next ideas:** Q-70 person-ROI cropping via depth foreground (top priority for visual); Q-71 depth recipe scale-up (epochs/width/frames/res); skeleton DG block (Q-10, Q-11b/c, Q-34, Q-32).

---

## SUB-001/002 — First submission pair prepared (NOT yet submitted)
**Date:** 2026-07-27 · files: submissions/sub_fuse3_hung.csv, submissions/sub_fuse3_nohung.csv
Fusion skel_noaug:0.6 + imu_aug:0.2 + depth_aug:0.2, 4-fold checkpoint prob-averaging, temporal-center eval. Hungarian version changed 55/405 predictions across 115 multi-groups (de-duplicated Walk 60→52, Stir 40→25). OOF reference: 54.30% argmax / 55.44% Hungarian. Purpose of the pair: (1) calibrate CV↔LB (B-004), (2) measure the Hungarian lever on real LB (B-007/B-011). ATHARV: submit BOTH on the same day; record public LB scores in this log.

---

---

## EXP-002 — Depth-scalar CNN baseline (TSN-style, with aug)
**Date:** 2026-07-27 · **Queue ID:** Q-04 · **Tier:** exploit

**Hypothesis:** 8 uniform frames of JET-inverted scalar depth (120×160) + small 2D-CNN + temporal pool ≥50% subject-CV from scratch.
**Setup:** train.py --modality depth --aug, FrameCNN 1.19M params, 40 ep, bs 48.
**Expected gain:** ≥50%
**Actual gain:** **26.71% ± 2.21% — hypothesis strongly REFUTED (−23 pts)**
**Conclusion:** From-scratch small visual CNN massively underperforms the paper's 90% (pretrained ResNet-50, random split). Which factor dominates (data scarcity vs cross-subject appearance overfit vs training recipe vs aug harm) is UNKNOWN → EXP-006 diagnostics launched (subject vs random split, no-aug).
**Confidence in conclusion:** 90% that depth@27% is real under this exact recipe; 30% that it's depth's ceiling.

### Failure analysis
**3 reasons:** (1) ~2.3k training clips is tiny for visual texture learning from scratch; (2) CNN may key on subject appearance/body shape (nearest depth blob) — cross-subject collapse; (3) 40 epochs + OneCycle may underfit; no train-acc logging to tell (harness gap — fixed in EXP-006 interpretation via rand-split).
**3 alternatives:** (1) bundled visual aug harmful (crop+erase together); (2) depth scalar input representation poor (invalid=0 conflates with near; no normalization per scene); (3) temporal pooling too lossy (mean+max over 8 frames).
**3 follow-ups (launched as EXP-006):** (a) depth no-aug subject fold0; (b) depth no-aug RANDOM split — the DG-gap probe; (c) skel no-aug random split for the same gap on skeleton.
**Beliefs updated:** B-006 ↓ 70%→40% (visual backbone thesis in doubt pending EXP-006); B-002 ↑ (skeleton back as the backbone candidate).
**Next ideas:** if rand-split depth is HIGH (≥60%): invest in visual DG (mixup, subject-adversarial, stronger aug, IR/depth ensembling). If LOW (≤40%): visual needs capacity/epochs/recipe work first.

---

## EXP-005 — 3-stream late fusion (OOF) + Hungarian
**Date:** 2026-07-27 · **Queue ID:** Q-20 (OOF version) · **Tier:** exploit

**Hypothesis:** Late fusion of skel(53.0)+imu(26.8)+depth(26.7) ≥ +5% over best single.
**Setup:** fuse_oof.py weight grid 0.1 steps on OOF probs (weights tuned on OOF — mild optimism).
**Actual gain:** best weights 0.6/0.2/0.2 → **54.30%** argmax (+1.26 over skeleton), **55.44%** with Hungarian (+2.4 total).
**Conclusion:** Fusion helps but below +5% hypothesis — the weak streams (27%) contribute little; B-003 partially supported. Hungarian on fused probs +1.14% (vs +0.63 on skeleton alone, and NEGATIVE −0.93 on weak depth alone → apply Hungarian only to strong/calibrated probs).
**Confidence:** 80%.
**Unexpected observation:** Hungarian can HURT on badly-calibrated probs — assignment noise displaces correct predictions (depth alone: 26.7→25.8).
**Beliefs updated:** B-003 ↓ 75%→65% (fusion gain scales with stream quality, not stream count); B-007 note: Hungarian gain requires calibrated fused probs.
**Next ideas:** improve streams before widening fusion; re-tune fusion after every stream upgrade.

---

---

## EXP-003 — IMU 1D-CNN baseline (with aug)
**Date:** 2026-07-27 · **Queue ID:** Q-03 · **Tier:** exploit

**Hypothesis:** 5-device 10 Hz IMU tensor (85ch × 64) + SeqTCN ≥35% subject-CV.
**Setup:** train.py --modality imu --aug; 0.65M params, 60 ep.
**Expected gain:** ≥35%
**Actual gain:** **26.75% ± 3.82%** — below hypothesis.
**Conclusion:** IMU is weak cross-subject on this data: ~2 s clips @10 Hz ≈ 20 samples/device is very little; paper's 45.5% was random-split. High fold σ (3.8%) suggests user-dependent sensor signal. Keep only as fusion diversity; do not invest in IMU architecture search yet.
**Confidence in conclusion:** 75% (aug suite untested for harm here, like EXP-001b; orientation-invariant features unexplored — Q-12 could still add a few %).

### Failure analysis
**3 reasons:** (1) 10 Hz × 2 s gives ~20 timesteps — most gesture dynamics are invisible; (2) my aug suite may hurt (same bundling flaw as EXP-001b); (3) angle/mag channels are device-orientation-specific → subject/mounting overfit.
**3 alternatives:** (1) resampling to fixed 64 steps stretches variable durations — destroys absolute rhythm cues; (2) per-channel normalization constants (500°/s etc.) may be poorly scaled; (3) 4% device dropouts unhandled beyond zero-fill.
**3 follow-ups:** (1) no-aug IMU run; (2) acc-magnitude + gyro-magnitude orientation-invariant channels; (3) absolute-time positional channel instead of stretch-resampling.
**Beliefs updated:** B-009 confirmed (65%→80%): weak alone, cheap diversity.
**Next ideas:** defer IMU tuning until after fusion baseline exists.

---

---

## EXP-004 — Hungarian distinctness assignment simulated on skeleton OOF
**Date:** 2026-07-27 · **Queue ID:** Q-68 · **Tier:** exploit

**Hypothesis:** Per-recording-group distinct-label assignment (train-proven constraint, B-011) adds +2-5% over argmax.
**Setup:** code/hungarian_sim.py; OOF probs from skel_noaug 4-fold checkpoints (n=2700 val clips); groups = radar filename ts; scipy linear_sum_assignment on −log p.
**Expected gain:** +2-5%
**Actual gain:** **+0.63%** (53.04% → 53.67%); 98% of val clips sit in multi-clip groups; 246 groups had argmax collisions; on multi-group clips 53.38% → 54.03%.
**Conclusion:** Constraint is genuinely valid (never hurts; every changed prediction was in a collided group) but the lift is bounded by P(true label is the runner-up) — at 53% base accuracy, most collision fixes still miss. Keep as a free, safe post-process; expect the absolute lift to change (direction uncertain: fewer collisions vs better fixes) as base accuracy rises. Re-measure at each milestone.
**Confidence in conclusion:** 85%
**Unexpected observation:** Train val groups reach size 10 (test max 8) — simulation slightly pessimistic vs test's smaller groups? Unclear; group-size-stratified analysis later if it matters.
**Beliefs updated:** B-007 confidence stays 80% but expected magnitude ↓ (+0.5-1.5% not +2-5%); B-011 mechanism confirmed on real predictions.
**Next ideas:** soft-constraint variant (Sinkhorn / duplicate penalty instead of hard assignment); re-run on fusion OOF.

---

---

## EXP-000c — Round-2 audits (Q-60 timeline, Q-61 cohorts, Q-62 fixes, Q-63 depth scale, Q-64 station) + recording-group purity check
**Date:** 2026-07-27 · **Queue IDs:** Q-60,Q-61,Q-62,Q-63,Q-64 · **Tier:** foundation
**Setup:** Workflow wf_2c584dd3-05f (5 agents) + my direct train-radar-timestamp purity check. Artifacts: timeline_merged.csv, test_adjacency_prior.csv, test_cohorts.csv, date_map.csv, depth_jet_scale_probe.csv, station_class_prior.csv.

**Q-60 timeline:** Test clips are CLEAN held-out recordings: 0 timestamp overlaps with train, 0 shared camera recordings (anchor = t0 − frame_idx/10 clustering), min gap to any train trial 284 s, block-level alternation only. **Adjacency prior DEAD: same_class(nearest before, after) = 0/405.** Rules-consistent with users 10/11 (recorded between batch-B slots, May31-Jun2) and 25/26 (Jun12-17). BONUS: 405 test clips = **144 continuous recordings** (29 singletons, 376 clips in 115 multi-clip groups, sizes 2-8; radar filename timestamp = authoritative key, present even on empty radar files).
**Recording-group purity (my check on ALL 2914 train radar files):** 741 multi-trial train groups → **user-pure 741/741, class-pure 0/741, all-classes-DISTINCT 741/741, station-pure 741/741, trial-name-pure 741/741**. So trial A-B-C = one continuous recording pass in which the user performs a SEQUENCE OF DIFFERENT activities; C=1..3 are repeated passes. Implication for test: within each multi-clip recording group, labels are (a) same user, (b) same station, (c) almost surely pairwise DISTINCT → constrained-assignment (Hungarian) post-processing lever covering 93% of test clips. The "same-recording ⇒ same-class pooling" idea is DEAD (checked before building — saved us from a systematically harmful smoother).
**Q-61 cohorts:** 2-way split PROVABLE: E (radar-usable, May31-Jun2, n=199) vs L (radar-empty, Jun12-17, n=206). 4-way E1 99 / E2 100 / L1 105 / L2 101 ≈ 4×101 (moderate confidence; 21 slots and the 143-144 radar recording groups are the provably-pure units; E1/E2 boundary heuristic; S16 has NEW→OLD→NEW leg-MAC blocks—could be two people turn-taking). SM_test_0054 = fully sensor-less outlier (no radar, no IMU rows).
**Q-62 fixes:** True test durations: median 1.90 s, max 10.00 s (hard cap; train max 26.6 s) — test trimmed ~20% shorter. Date map: batch A u1-5 (May7-8), B u6-9 (May30-Jun2), C u16-20 (Jun9-11), D u21-24 (Jun11-13). Radar in TEST is usable for ALL early-cohort clips (198) and empty for ALL late ones → radar could serve the E-half of test (trained on users 1-9). IMU near-complete for all users.
**Q-63 depth scale:** JET mapping is FIXED/ABSOLUTE (inversion lossless, median NN-dist 0.0; background patches constant ±0.5 idx across frames/users/days; per-frame rescale ruled out). Depth index = absolute depth proxy comparable everywhere; guard: physical camera repositioning between dates; mask black invalid (9-21%/frame, corners always dead).
**Q-64 station prior:** DEAD END: I(class;station)=0.711 bits but station-recovery from IR/depth backgrounds tops out ~50-53% (within-station corr 0.186 vs between 0.175) → effective 0.092 bits. Station-only classifier = 12.3%. Revisit only if another station channel appears.

**Conclusion:** Metadata game reshaped: adjacency and station priors dead; the live levers are (1) recording-group distinctness assignment (93% coverage), (2) user-pure grouping → per-user/cohort adaptation, (3) 2-way cohort split for BN adaptation. Test is clean cross-subject — the real model carries the campaign.
**Confidence:** 90% (train-side purity facts are exhaustive, not sampled).
**Beliefs updated:** B-007 revised (see ledger), B-008 softened (radar usable for E-half of test), NEW B-011 (recording groups).
**Next ideas:** Q-68 Hungarian distinctness assignment; Q-69 verify group-key recoverability end-to-end in inference code; per-cohort adaptation stays Q-54.

---

## EXP-001b — Skeleton TCN + augmentation suite
**Date:** 2026-07-27 · **Queue ID:** Q-11 (partial) · **Tier:** exploit

**Hypothesis:** Yaw-rotation ±30° + scale ±15% + jitter + joint-dropout adds +3-8% over EXP-001a.
**Setup:** identical to EXP-001a but --aug (also temporal jitter sampling).
**Expected gain:** +3-8% → ~56-61%
**Actual gain:** **51.49% ± 2.15% — NEGATIVE (−1.6% vs no-aug 53.04%)**
**Conclusion:** Hypothesis REFUTED at these aug strengths. The full suite hurts.
**Confidence in conclusion:** 70% (single seed; σ across folds 2.2% — the deficit is within ~1σ, but direction is clear enough to redesign).

### Failure analysis (mandatory)
**3 reasons why it failed:**
1. Yaw rotation may be harmful: H36M-lifted poses are camera-relative and activities are station-anchored (facing the sink/desk); rotating breaks a genuine, test-transferable orientation prior.
2. Scale ±15% destroys absolute body-size/height in meters, which the floor-aligned z uses to distinguish e.g. Lie_down/Sit vs Stand classes.
3. Joint dropout (2 joints, 20% of clips) may be too destructive with only 17 joints when wrist/head joints carry most class signal.
**3 alternative explanations:**
1. Experiment invalid as a test of augmentation per se — 5 augs bundled; one bad component can mask 4 good ones.
2. Temporal jitter sampling (train-time) changed the effective frame sampling vs eval — train/eval distribution mismatch.
3. Seed noise: ±2.2% fold σ; deficit could be ~luck (needs seed repeat).
**3 follow-up experiments:**
1. Q-11b: one-aug-at-a-time ablation (rotation-only, scale-only, jitter-only, dropout-only, temporal-jitter-only) — 5 cheap runs.
2. Q-11c: milder magnitudes (yaw ±10°, scale ±7%, no joint dropout).
3. Q-10: seq-level normalization (torso-scale) instead of scale-aug — normalization may beat augmentation for size invariance.
**Beliefs updated:** B-001 nuance: not all invariance helps; augmentation must respect station-anchored orientation priors.
**Next ideas:** queued Q-11b/c.

---

---

## EXP-001a — Skeleton TCN baseline, no augmentation
**Date:** 2026-07-27 · **Queue ID:** Q-02 · **Tier:** exploit

**Hypothesis:** Normalized 17×3 H36M skeleton sequences (person-0, T=32, coords+velocity) + small TCN ≥45% subject-CV.
**Setup:** code/train.py --modality skel --tag skel_noaug; SeqTCN 0.66M params, GroupNorm, AdamW 1e-3 OneCycle, 60 ep, label smoothing 0.1, batch 64, seed 0; 4-fold subject CV (cv_folds.json).
**Expected gain:** ≥45% (baseline floor)
**Actual gain:** **53.04% ± 1.32%** (folds in results.csv)
**Conclusion:** Hypothesis confirmed with margin. A 0.66M no-aug skeleton model nearly matches the dataset paper's ~56.4% LOSO (which used contrastive learning). Skeleton stream is real; fold σ is low (1.3%) → CV is stable.
**Confidence in conclusion:** 85%
**Unexpected observation:** —
**Beliefs updated:** B-002 ↑ (skeleton competitive after all); B-004 ↑ (fold σ low → protocol usable).
**Next ideas:** person-selection policy ablation; torso-scale normalization (Q-10).

---

---

## EXP-000 — Environment + data reconnaissance (not a model experiment)
**Date:** 2026-07-27 · **Queue ID:** — · **Tier:** foundation

**Setup:** Local: RTX 4060 8 GB, 16 cores, 15 GB RAM, torch 2.12.0+cu130 OK. Train 43 GB extracted; test zip extracted.

**Findings:**
- Layout: `HAR/data/<modality>/<id>_<name>/user<N>/<s>-<r>-<t>/files`. Modalities: Depth_Color, IMU, IR, Radar, Skeleton, Thermal — all 40 classes each.
- Trial folders look like `1-1-1`, `1-1-2`, `1-1-3` (scene?-?-repeat). Class 0 Depth has 11 users, NOT all 18 → per-class user coverage is partial; must profile the full grid.
- Depth_Color: color-mapped PNG frames (~30 fps, timestamped filenames). IR: PNG frames. Thermal: JPG frames (`frame_000226.jpg`, no timestamps in name).
- IMU: two CSVs per trial — `up(LA+RA+C).csv` (left arm, right arm, chest) and `down(LL+RL).csv` (legs); Chinese headers; per row: device name + acc(3) gyro(3) angle(3) mag(3) quaternion(4) temp battery. Multiple devices interleaved by row → must pivot by device. ~137 rows / ~4.5 s trial → ~10 Hz per device per file. Timestamps present.
- Radar: single CSV per trial, header `timestamp,frame,DetObj#,x,y,z,v,snr,noise` — sparse point cloud per frame.
- Skeleton: `predictions/*.json` per RGB frame (pose-estimator output; RGB itself withheld). Schema TBD.
- Test: 405 clips `SM_test_XXXX/`; sample_submission predicts 34 for all.

**Conclusion:** Data is rich but heterogeneous; profiling grid (modality × class × user × trial counts, frame counts, missing patterns, test availability) is the required next step.
**Beliefs updated:** none yet (pre-experiment).
**Next ideas:** profiling workflow; skeleton JSON schema inspection; test-set modality census.

---

## EXP-000b — Deep profiling workflow (8 agents: censuses, 4 modality deep-dives, literature, critic)
**Date:** 2026-07-27 · **Queue ID:** — · **Tier:** foundation
**Setup:** Workflow wf_411e5e03-190; full reports in workflow transcript; artifacts in research/artifacts/ (train_index.csv, test_index.csv, skeleton_quality_sample200.csv, skeleton_frames_per_trial.csv, imu_duration_by_class.csv, imu_train/test_device_scan.psv, visual_trial_profile.csv, radar_file_sizes.csv, radar_profile_sample300.csv).

**Train census:** 18 users (1-9, 16-24), 3036 unique samples (class,user,trial), ~2900 trials/modality. NO user has all 40 classes (user5 worst: 17; user18 best: 35). Classes 25/26 (Watch_TV/Play_games) missing for 14/12 of 18 users. Class imbalance 12 (cls 25) to 319 (cls 36 Walk) samples. Trial name A-B-C: A=camera station 1-7, C=repetition 1-3. Missing-modality: 2748/3036 have all 6; 103 Thermal-only samples; Thermal is the misfit (coverage differs, ~25 fps unsynced, no timestamps).
**Test census:** 405 clips; Depth/IMU/IR/Skeleton present in all 405; Radar missing 1 + EMPTY in 206 (51%); Thermal missing 10. Timestamps NOT anonymized (Depth/IR/Skeleton/Radar filenames + IMU internal): dates 05-31(62) 06-01(57) 06-02(80) 06-12(23) 06-13(53) 06-16(29) 06-17(101); >30-min-gap clustering → 16 sessions. 275 clips fall on train recording days (interleaved with train users' sessions); 130 clips (06-16/17) on days with zero train coverage. Test clips slightly shorter than train (frames med 20 vs 24-26).
**Skeleton:** H36M 17-joint **3D** (meters), 10 Hz, per-frame JSON list of persons; keypoint_scores always exactly 1.0 (useless); pelvis root-centered x,y and floor-aligned min-z=0 PER FRAME (global translation destroyed by preprocessing). Multi-person 5.6% of frames overall but 84.6% in Comb_hair, 53.5% Brush_teeth (mirror reflections at bathroom station → itself class-discriminative). Frames/trial: min 1 / med 24 / max 236. Test identical format.
**IMU:** 5 devices (chest WTC, arms WTLA/WTRA, legs WTLL/WTRL) at **~10 Hz only**, 21 Chinese-header cols (acc g, gyro °/s, angle °, mag uT, quat, temp, battery). Only 7 physical sensors in whole dataset → MACs shared across users, NO per-user leak; but legs have OLD pair (users 1-5) vs NEW pair epochs, and 50 TEST clips use the OLD leg pair (cohort fingerprint). 4% trials missing ≥1 device; 40 train trials fully empty; SM_test_0054 IMU fully empty; SM_test_0001 down-csv has English 23-col header (parse both). Trial duration med 2.22 s train / 1.80 s test. Duration is class-informative (7× spread: Stand_up 0.87 s vs Sweep 6.22 s) but test is systematically shorter → trimmed.
**Visual:** Depth_Color 640×480 PNG = JET-colormapped depth, exactly invertible to 8-bit scalar (low=near); ~10% black = invalid. IR 640×480 gray PNG = near-IR photo (full texture, faces visible). Thermal 320×240 JPG ironbow-ish colormap, NOT cleanly invertible, ~25 fps, unsynced, no timestamps but CONTINUOUS session frame counter (second session-grouping key for test!). Depth/IR/Skeleton are frame-exact aligned at 10 fps (same timestamps+indices). ONE apartment, ~7 fixed camera stations; station↔activity correlated (bathroom→wash face); same rooms in test; different users share rooms → environment transfers, station prior is exploitable.
**Radar:** sparse (med 6.6 pts/frame, cap ~20), Doppler clipped ±0.97 m/s, med 2.5 s recorded. **Header-only EMPTY for 100% of users 16-24** (51.6% of train files); test 51% empty → radar-empty is a test cohort fingerprint; as a signal, DEFER (weak aux at best).
**Literature/rules:** Kaggle deadline **Sep 15, 2026**; code upload Sep 22; top-15 verification = live Zoom inference on fresh samples + offline reproduction within ≤10% of Kaggle score. Final score: Kaggle private 20%, on-site private test (8 NEW subjects) 30%, reproducibility 10%, report 20%, presentation 10%, efficiency 10%. Submission package = code + SINGLE checkpoints/model.pth ≤100 MB (implicitly caps ensembles). ImageNet-init of small CNNs AMBIGUOUS (organizers' own paper used pretrained ResNet-50) → email cuhkx.competition@gmail.com for ruling. Dataset paper (arXiv 2512.07136): random-split baselines Depth 90.5 / IR 90.2 / Thermal 92.6 (ResNet-50) / Skeleton 79.1 (MotionBERT) / mmWave 46.6 / IMU 45.5; **cross-subject LOSO ≈ 56.4%**. Public LB (135 teams): #1 0.8358, #2 0.7662, cluster 0.73-0.77. No public baseline code.
**Critic (contradictions to resolve):** (1) test-user attribution vs "users 10/11/25/26" rules statement — timestamp overlap means interleaved recording, NOT train-user identity; needs timeline merge audit. (2) Test-duration median 10.3 s figure is likely a bug (all other evidence says ~2 s) — recompute. (3) Train census counted files not contents → its Radar/IMU coverage overstates usable data. (4) Underweighted signals: session→class adjacency leakage for 275 clips; cohort fingerprint = radar-empty × leg-MAC epoch × thermal-missing × session; station prior P(class|station) + scene classifier; multi-person rate as feature; Thermal frame-counter session linking.

**Conclusion:** Strategy reshaped: (1) visual modalities are the accuracy backbone, not skeleton; cross-subject regularization of a small depth/IR CNN is the core battle; (2) skeleton is still the cheapest strong stream but its ceiling is lower than assumed (79% with pretrained MotionBERT on random split); (3) IMU is weak-but-cheap ensemble diversity; radar defer; (4) session/cohort structure of test is a legitimate, reproducible post-processing lever for the Kaggle LB (worth several %), but the on-site test (30% of final) has none of it — core model quality still dominates; (5) 0.75 public ≈ top-5; 0.76+ realistic target from fusion + DG; leader 0.836 suggests either strong DG or session exploitation.
**Confidence in conclusion:** 80% (pending critic follow-ups).
**Beliefs updated:** B-002 ↓45%, B-003 ↑75%, B-005 confirmed shape (single model.pth), NEW B-006..B-010 (see BELIEFS.md).
**Next ideas:** Q-60 timeline merge audit, Q-61 cohort clustering, Q-62 duration recompute, Q-63 depth-scale probe, Q-64 station prior, Q-65 organizer email draft — queued as follow-up profiling round.

---

# Evidence Log

**Counters:** experiments since last devil's-advocate pass: **2 / 10** (DA-006 was the last pass) · since last contradiction search: **n/a — never run** · since last reset: **9 / 25**
**Coverage:** mechanism axes probed **6/10 + 4 partial** (see OUTLIERS.md) · **kill rate:** 1 hypothesis killed in last 10 experiments (EXP-054)

**Standing target (2026-07-31):** 0.89+ = 179/201. Measured bars from the
verified top-20 snapshot: rank 1 = 181/201, rank 15 = 137/201, rank 20 =
131/201, ours = 112/201.

**Leaderboard source of truth:** [LEADERBOARD.md](LEADERBOARD.md). Scores without
user-supplied Kaggle evidence are unverified even if an older entry called them
results.

---

## EXP-112 — Layer-wise LR decay is null. TWO recipe changes now say the member is not recipe-limited.
**Date:** 2026-08-25/26 · `--llrd 0.8` · **Tier:** explore · **Purpose:** SCORE

| fold | base micro | LLRD micro | Δ | base object | LLRD object | Δ |
|---|---|---|---|---|---|---|
| 0 | 70.516 | 71.130 | +0.61 | 59.034 | 60.286 | +1.25 |
| 1 | 69.287 | 69.042 | −0.25 | 62.735 | 62.051 | −0.68 |
| 2 | 71.472 | 69.632 | −1.84 | 64.927 | 61.169 | −3.76 |
| 3 | 73.966 | 76.417 | **+2.45** | 65.837 | 69.683 | +3.85 |

**Mean +0.245, sd 1.79, 2 of 4 folds positive.** Fails the adoption bar (≥3 folds, or one
fold >2.80). Note fold 3 gave +2.45 — the *same* headline number fold 2 gave for jitter,
from a different change. That is what σ=2.80 noise looks like when you only run one fold.

### The pattern across two independent recipe changes

| change | mean Δ | sd | positive folds |
|---|---|---|---|
| temporal jitter (EXP-110/111) | +0.14 | 1.57 | 1/4 |
| layer-wise LR decay | +0.245 | 1.79 | 2/4 |

Both land at ~+0.2 with fold-scatter of ±1.8, i.e. indistinguishable from seed noise, from
changes on opposite sides of the recipe (data augmentation vs optimizer schedule).

**Conclusion: MViTv2-S at 224 px on 2,281 clips is not recipe-limited.** The bottleneck is
*information* — data volume, initialisation domain, or architecture — not how we train.
This is direct evidence for strategy candidates 1 (distil from a large teacher) and 2
(in-domain depth/IR pretraining), and against further recipe work. **Stop tuning the
recipe.**

### Side finding: the fused output is severely under-confident

`sub_r2` scores 166/201 = 82.6% and yet its max-probability is **median 0.306**, with
**zero** clips above 0.90 and only 86 of 405 above 0.50.

Consequences:
- **Confidence-thresholded pseudo-labelling is unworkable on the fused probabilities** —
  a 0.70 gate retains 22 of 405 clips. Any self-training here would have to use rank
  selection or a member's own softmax, not the fusion output.
- It does not affect accuracy directly (argmax is invariant to monotone rescaling), and
  sharpening is equivalent to a weight change in log space, which the 5,227-point sweep
  in EXP-108 already covered. So this is a *calibration* fact, not a lever.

### On pseudo-labelling (R-4), which the low confidence forced a look at

Permitted, but the organisers attach a caveat that outweighs the permission here: Stage 2
and Stage 3 use unseen subjects, and over-fitting to the current test distribution
generalises poorly. **The on-site test is 30% of the final grade against Kaggle private's
20%**, so trading on-site robustness for public clips is negative on weights alone.
`BELIEFS.md` already records naive self-training as negative at a much lower base accuracy.
Not pursued.

---

## EXP-111 — Jitter dead at 4 folds. The person-crop asymmetry is REAL but points the WRONG WAY.
**Date:** 2026-08-25 · **Tier:** explore · **Purpose:** INFORMATION

### Temporal jitter: closed

| fold | baseline | jitter | Δ |
|---|---|---|---|
| 0 | 70.516 | 69.902 | −0.61 |
| 1 | 69.287 | 68.305 | −0.98 |
| 2 | 71.472 | **73.926** | **+2.45** |
| 3 | 73.966 | 73.660 | −0.31 |

**Mean +0.14, sd 1.57, 1 of 4 folds positive.** Fold 2 was an outlier below the seed
spread and I wrote it up as "the largest member gain since 224 px" before the replication
check came back. Delete the `*jit_*` checkpoints; the 32-frame caches stay because
temporal *TTA* is a separate, surviving result.

### Person-crop identity: hypothesis refuted, and the direction is the interesting part

`compute_windows()` takes the highest-confidence person box independently per probe frame
and then the **union** of those boxes, with no identity association. Prediction: this
should damage test more than train, because the repo records test at 2x train's
multi-person rate, and OOF (measured on train subjects) cannot see it.

Measured with `code/probe_crop_identity.py` (YOLO11n, 8 probe frames per clip, CPU):

| | train (2,905) | test (395) |
|---|---|---|
| >1 person in some probe frame | 17.5% | **30.4%** |
| union inflates >2x over median box | **14.9%** | 9.6% |
| union inflates >4x | **2.5%** | 1.0% |
| identity switch (min consecutive-pick IoU < 0.3) | **5.6%** | 3.5% |
| median inflate | 1.28 | 1.19 |

**Test has nearly double the multi-person rate and yet CLEANER crops** — fewer identity
switches, less union inflation, on every measure. The likely mechanism is duration: test
clips are shorter (median 20 raw frames vs train's 24, p90 41 vs 55), so there is less
time for the per-frame pick to drift onto another person.

So the asymmetry exists and runs **opposite** to the hypothesis. Person-crop tracking is
**dead as a test-specific fix**; if anything the crop pipeline is a mild *training* noise
source (5.6% of train clips carry a switch). Filed, not pursued.

**Cost of this probe: ~25 CPU-minutes, no GPU.** It killed candidate 5 of the six-way
strategy list before any of it reached a trainer.

---

## EXP-110 — Temporal TTA is small and real. Temporal jitter training is REFUTED on replication.
**Date:** 2026-08-23/24 · `--frames 32` · **Tier:** explore · **Purpose:** SCORE

50.9% of raw frames are discarded at 16 frames/clip. `cache/crop_224_t32` stores 32
uniform samples instead, built in 90 s by reusing the YOLO windows. Those frames can be
spent at test time or at train time. **Only the first survived.**

### Spending them at TEST time: small, real, consistent

Two interleaved 16-frame views averaged on the **existing** checkpoints — no retraining.

| fold | person 1 view | person 2 views | wrist 1 view | wrist 2 views |
|---|---|---|---|---|
| f0 | 0.69902 | 0.70516 | 0.69410 | 0.69902 |
| f1 | 0.69042 | 0.69656 | 0.71499 | 0.71990 |
| f2 | 0.71319 | 0.71933 | 0.72239 | 0.72393 |
| f3 | 0.74273 | 0.75651 | 0.75038 | 0.77335 |
| **pooled** | 0.70951 | **0.71735** | 0.71872 | **0.72690** |

**All 8 fold-view pairs gain**, which is what makes this credible at a small effect size:
pooled **+23** (person) and **+24** (wrist) clips of 2,933. Through a 0.2925-weight video
slot it dilutes to **+9 clips/2,700 = +0.7 public**, and `sub_s1` is **rowdiff 6** against
`sub_r2` — under the ~20-row readability bar. **Keep it on (free), never spend a
submission proving it.**

### Spending them at TRAIN time: **REFUTED**

`make_dataset` draws a random phase each epoch, so the model sees a different 16-frame
sampling of the same clip every time. Fold 2 looked like the largest member gain since
224 px. Fold 0, run as a deliberate replication check, killed it.

| fold | baseline | jitter | Δ micro | Δ object |
|---|---|---|---|---|
| f2 | 0.71472 | **0.73926** | **+2.45** | +9 clips |
| f0 | 0.70516 | **0.69902** | **−0.61** | −4 clips |
| mean | | | **+0.92** | |

**Recorded seed σ on this partition is 2.80** (`research/Plan.md`, Phase 3 gate), so the
standard error of a 2-fold mean is 2.80/√2 = **1.98**. +0.92 ± 1.98 is indistinguishable
from zero. The fold-2 number was never evidence of anything — a single fold cannot clear
this partition's seed spread, and **+2.45 < σ**.

**This is the campaign's most-repeated failure mode, committed again.** EXP-100
(start-weight, +14 OOF → −8 public) and n7 (+10 fold-2 → −3 public) are the same shape.
The replication check was run *first* here and cost 1.4 h, which is the only part of this
that went right. **Rule, now explicit: a member-level change is not a result until its
effect exceeds 2.80 micro on a single fold, or is positive on ≥3 folds.**

Folds 1 and 3 are still queued and will finish the pooled estimate; nothing will be
adopted on less. The queued `*jit_all` models are being trained on an unconfirmed change
and should be treated as disposable until the pooled number exists.

### What this does not change

`sub_r2` = 166/201 from an 83.82 MB legal package still stands, and temporal TTA rides
along free inside it.

---

## EXP-109 — THE LEGAL PACKAGE MATCHES THE ILLEGAL CHAMPION. 83.82 MB = 166 = 309 MB.
**Date:** 2026-08-23 · **Tier:** exploit · **Purpose:** SCORE

| submission | package | legal? | score | clips |
|---|---|---|---|---|
| `sub_n8` — MViT x4, full branches | ~309 MB | no | 0.82587 | 166 |
| `sub_r1` — + wrist 4-fold | ~446 MB | no | 0.82587 | 166 |
| **`sub_r2` — MViT person+wrist int6, pruned skel, shrunk imu** | **83.82 MB** | **YES** | **0.82587** | **166** |

**The 100 MB constraint now costs exactly nothing.** Legal best went 160 -> 166 in one
step, and the package is 16.18 MB under budget. Item ① of the handover is closed: a
legal Stage-2 package exists, is measured on public, and matches the best pipeline we
have ever built at any size.

### The video slot saturates — this is the important finding

- Adding the wrist view to a **1-model** video slot: `q4` 160 -> `r2` 166 = **+6 clips**.
- Adding the wrist view to a **4-fold** video slot: `n8` 166 -> `r1` 166 = **+0 clips**.

Four person folds and one person + one wrist reach the *same* place. The wrist view is
not adding information the person view lacks; both are substituting for the variance
reduction the other provides. Together with EXP-098's oracle collapse (15.4 -> 7.85 pts)
this says the **video branch is information-saturated at ~166**, and further members —
more folds, more views, more crops — will not move it.

**Calibration:** pooled OOF predicted **+3.3** for `r1` and public delivered **0**. Inside
the ±6 noise floor, so not a refutation of the estimator, but it is the first miss after
three hits and it lands exactly where saturation predicts one. Recorded, not explained
away.

### What this closes and what it opens

Closed: video-bag composition (again, now including views), fusion weights (EXP-108 §3),
packaging (this entry). **Adding members to the video slot is now a graveyard axis.**

Open, and the only axis with genuinely unused information: **50.9% of raw frames are
discarded**. Median clip holds 24 frames, 32.4% hold >32, and the cache samples 16.
`cache/crop_224_t32` now stores 32 uniform samples per clip (built in 90 s by reusing the
YOLO windows), so two interleaved 16-frame views can be averaged **on the existing
checkpoints** — no retraining, and critically **no package bytes**, which is what
EXP-108's byte economics say to optimise for.

---

## EXP-108 — int6 is free, the wrist view holds at 4 folds, and a legal 83.82 MB package exists.
**Date:** 2026-08-23 · `code/quant_probe.py`, `code/quantize_checkpoint.py` · **Tier:** exploit · **Purpose:** SCORE

### 1. Quantization: int6 is accuracy-identical to int8; int4 is the cliff

MViTv2-S, weight-only, symmetric, per-output-channel, on the same 652 fold-2 held-out
clips the member was scored on. fp32 reproduces its logged 0.71472 exactly, so the
harness is sound.

| bits | MB/model | micro | object | argmax agreement w/ fp32 |
|---|---|---|---|---|
| 32 | 137.10 | 0.71472 | 0.64927 | 1.0000 |
| 8 | 34.55 | 0.71319 | 0.64718 | 0.9954 |
| **6** | **26.01** | **0.71319** | 0.64509 | 0.9724 |
| 4 | 17.46 | 0.70399 | 0.62630 | 0.9141 |

**int6 costs nothing and saves 25% of the bytes.** int4 costs 7 clips of 652 and is the
first width where the model stops being the same model. Norm scales, biases and
positional tables are left fp32 throughout — a fraction of a percent of the parameters,
disproportionately rounding-sensitive.

### 2. The wrist view holds at 4 folds — pooled, not fold-2

| video slot (pooled 2,933) | micro | object | clips |
|---|---|---|---|
| person 4-fold | 0.71156 | 0.62906 | 2087 |
| wrist 4-fold | 0.71565 | 0.63487 | 2099 |
| **person + wrist, equal** | **0.74122** | **0.66586** | **2174** |

**+87 clips** at member level; agreement 74.97%, wrist rescues 242 and breaks 230.
Through the full fusion on 2,700 pooled clips the gain is **+44 clips → +3.3 public**.
Note the wrist view alone now slightly *beats* the person view (2099 vs 2087), which the
fold-2 measurement had backwards — another entry for B-029.

### 3. The fusion-weight axis stays closed, re-checked against the new member

It was worth re-testing since the video member gained ~7 points solo since the weights
were fitted. Deployed (V 0.2925, B 0.35, I 0.3575) = 2012 clips/2700; the best of
**5,227 grid points** = 2018, i.e. **+6 clips = +0.4 public**, and that is the
selection-biased maximum. Closed again, now with current evidence.

### 4. Branch economics — what a megabyte buys

| branch | MB | marginal value | clips/MB |
|---|---|---|---|
| skeleton (`world25` p4) | 22.80 | **+6.0 public** | 0.26 |
| IMU (`imu_stats` 150/10) | 4.61 | +0.6 public | 0.13 |
| video, 1 → 4 MViT @ int4 | +51.5 | +5.0 public | 0.10 |

The IMU branch carries the **largest** fusion weight (0.3575) for +0.6 public clips, and
reweighting still does not help (§3) — it moves many rows and nets almost nothing.
Adding video *models* is the least byte-efficient move available.

### 5. **A legal package exists: 83.82 MB**

| component | MB |
|---|---|
| MViT person all-train, int6 | 26.01 |
| MViT wrist all-train, int6 | 26.01 |
| `world25` pruned, 5 archs x 4 folds | 22.80 |
| `imu_stats` ExtraTrees 200 trees / depth 12 | 9.00 |
| **total** | **83.82** (16.18 MB headroom) |

`sub_r2` is that package end to end — rowdiff **23** against `sub_q4`, the 160-clip
legal baseline. `sub_r1` is the illegal ceiling (person 4-fold + wrist 4-fold, full
branches), rowdiff **13** against the 166 champion, isolating the wrist view as a single
change.

### 6. Untouched: **50.9% of all frames are discarded**

Median clip holds 24 raw frames, 32.4% hold more than 32, and the cache samples 16.
This is the largest unexploited information source left in the video path and it costs
**no package bytes** — temporal TTA over multiple 16-frame windows needs only a deeper
cache, not a bigger model. Next after the current submissions land.

---

## EXP-107 — THE CHAMPION IS NOT A LEGAL SOLUTION. Legal best is 160, exactly the top-15 bar.
**Date:** 2026-08-23 · **Tier:** exploit · **Purpose:** SCORE

### Public results

| submission | score | clips | vs `n8` (166) |
|---|---|---|---|
| `sub_p1` — wrist f2 at half the video slot | 0.82089 | **165** | −1 |
| `sub_q1` — all-train MViT alone as the video slot | 0.80099 | **161** | −5 |
| `sub_q4` — the 66.1 MB package pipeline | 0.79601 | **160** | −6 |

### 1. The packaging prunes are confirmed nearly free — and pooled OOF called it exactly

`q4 − q1 = −1 clip`. That one clip is the **entire** cost of pruning `world25`
84.54 → 22.80 MB, shrinking `imu_stats` 87.64 → 9.00 MB, *and* dropping MotionBERT.
Pooled OOF predicted ≈ −1 (2 rows + 6 clips/2,700 + 9 rows). **Third consecutive
confirmation** that the 2,700-clip pooled estimate tracks public. B-031 holds at 90%.

### 2. The whole loss is in the video slot: 4 folds → 1 all-train model costs 5 clips

`k224_mvit_all` is trained on 29% more data and all 18 users, and still loses to the
4-fold bag by 5 clips. Ensembling the video slot is worth more than the extra data.
This is the number OOF structurally cannot produce (pooled OOF scores each clip with
the single model that held it out, so it estimates a member, never a bag).

### 3. **R-6 re-read: the 100 MB limit applies to the submitted solution, not just Stage 2**

> "package all weights that need to be loaded at inference — **including every model in
> an ensemble** — into a single checkpoint file, and that file must be under 100 MB on
> disk… Quantization (fp16 / int8 or lower) is allowed and encouraged"
> — organiser, topic 729056

The 166 champion needs MViT x4 (137 MB int8) + `world25` (84.5) + `imu_stats` (87.6)
≈ **309 MB**. **It is not a legal solution and never was.** Our legal best is
`sub_q4` = **160 clips — exactly the measured top-15 bar, with zero margin**, not the
+6 the handover claimed.

**Correction to the ledgers:** the "Rules §2.8.b >10% Kaggle-vs-package gap" line in
`CLAUDE.md` and the handover is **unsourced** — it appears in neither
`RULES_VERIFIED.md` nor `OBJECTIVE.md`, both of which state a flat ≤100 MB limit with
disqualification at the reproduction stage. Until someone produces the rule text, 100 MB
is treated as hard with **no** accuracy-gap allowance. Assuming otherwise is the more
expensive error.

### 4. What this makes urgent

The budget after the branches that cannot be cut (`world25` p4 22.80 + `imu_stats`
150/10 4.61 = 27.4 MB) leaves **72.6 MB for video = two MViT models at int8**, and the
measured penalty for one is −5. So the live question is **how far below int8 MViT
survives**, since R-6 explicitly permits "or lower": at int4 the entire 4-model champion
slot is 68.6 MB and fits. `code/quant_probe.py` is measuring it — weight-only,
symmetric, per-output-channel, on the same 652 fold-2 clips the members were scored on.

### 5. Wrist at one fold is neutral (−1, rowdiff 15)

Consistent with n7: a single-fold member given half the video slot is over-weighted
relative to its evidence. The 4-fold wrist is in flight and is the real test.

---

## EXP-106 — all-train MViT landed; the 66.1 MB package pipeline is built and unscored. A pgrep self-match cost 7 GPU-hours.
**Date:** 2026-08-23 · `code/build_video_slot.py` · **Tier:** exploit · **Purpose:** SCORE

### The overnight loss, and the bug that caused it

`run_wrist_queue.sh` waited on the in-flight job with

```bash
while pgrep -f "cuhkx_224_kaggle.py --stage train --all-train --tag k224_mvit_all"; do sleep 60; done
```

**`pgrep -f` matches the full command line of every process, including the queue's own
parent shell**, whose command line contains that string verbatim because the script was
written by a heredoc in the same invocation. The loop therefore waited on itself and
could never exit. `k224_mvit_all` finished at 23:39; the GPU then sat idle for
**seven hours** with the queue alive, its log empty, and no error anywhere.

Worse, the stuck queue was still live the next morning and would have launched a
**duplicate** fold-0 run the moment its parent died. Killed, and the wait loop removed
rather than fixed — the new queue simply runs its jobs in sequence.

**This is EXP-064's lesson in a new costume: a crash announces itself, a deadlock does
not.** Add to it: *never let a waiter's predicate match the waiter.*

### `k224_mvit_all` — no honest local estimate exists, by construction

It reports fold-2 micro **0.97546**, which is memorisation: `--all-train` trains on all
18 users, so fold 2 is training data. **Do not record this as a result.** The same holds
for any `--all-train` member — its only honest measurement is public.

Nor can OOF settle the k-fold-bag-vs-single-model question in general: pooled OOF scores
each clip with the one model that held it out, so it estimates a *single* model, never
the bag. The 4-fold bag's advantage was only ever measured on public (n8 = 166).

### Candidates built

| tag | video slot | skel | imu | MotionBERT | rowdiff vs 166 |
|---|---|---|---|---|---|
| `sub_q1` | `k224_mvit_all` alone | full | full | yes | **22** |
| `q2` | 4 folds + all-train | full | full | yes | 4 |
| **`sub_q4`** | **`k224_mvit_all` alone** | **p4 22.80 MB** | **200/12 9.00 MB** | **dropped** | **36** |
| `sub_p1` | person 4-fold + wrist f2, 50:50 | full | full | yes | 15 |

**`sub_q4` is the 66.1 MB Stage-2 package pipeline itself**, end to end. Submitting it
is the point: Rules §2.8.b makes a >10% Kaggle-vs-package gap a disqualification, and
if the package *is* the submission the gap is zero by construction.

`q2` at rowdiff 4 shows the all-train member is nearly redundant *inside* the 4-fold bag
(1/5 weight, highly correlated) — it earns its place by being shippable alone, not by
adding information.

**Tooling note.** `build_video_slot.py` now carries `--skel`, `--imu` and
`--no-motionbert`, and reproduces both the 166 champion and the package pipeline to
**0.0**. A first attempt at that patch silently failed to match and was caught only
because the reproduction check printed 0.168 instead of 0 — the check is the reason the
error lasted one minute instead of shipping.

---

## EXP-105 — The Stage-2 packaging blocker is SOLVED (4x cut, ~2 rows). Wrist crop pays only in fusion.
**Date:** 2026-08-22 · `code/prune_world25.py`, `code/build_video_slot.py`,
`code/imu_stats_size_sweep.py` · **Tier:** exploit · **Purpose:** SCORE + INFORMATION

### 1. The champion recipe is a script again, not a lost heredoc

`code/build_video_slot.py` reproduces `testprobs_n8.npz` (the 166 champion) to
**max abs diff 0.0**. The video slot is now a weighted log-mean over *views*, each view
an equal log-mean over its folds -- which separates the two things `n7` confounded:
how much weight a view earns, and how many folds back it.

### 2. Wrist crop -- the mechanism holds, but only in fusion (B-030 refined)

| fold-2 video slot | micro | object | motion |
|---|---|---|---|
| `k224_mvit_f2` (person) | 0.71472 | 311/479 | 0.89595 |
| `k224_mvitwrist_f2` | 0.71166 | 304/479 | 0.92486 |
| **person + wrist, equal weight** | **0.75153** | **330/479** | 0.92486 |

The pre-registered prediction was that the wrist view would raise *object* accuracy on
its own. **It did not (-7 clips).** What it does instead is fail on different clips:
argmax agreement **74.7%**, oracle-any 522/652. Fused at equal weight the pair gains
**+19 object clips and +3.68 micro** over the best member of the campaign. The optimum
sits at exactly w=0.5, the prior-free choice, so this is not a fitted peak.

`sub_p1` (wrist view at half the video slot) is rowdiff **15** vs the 166 champion --
under the ~20-row readability bar, because the wrist view is half of a video slot that
is itself 0.2925 of the fusion. Folds 0/1/3 are queued to give the wrist view the same
4-fold backing the person view has, which also removes the single-fold asymmetry that
produced n7's -3.

### 3. Swin3D-T stays refuted; MotionBERT is nearly free to drop

### 4. **PACKAGING: measured, not estimated**

Per-member probabilities were dumped once from the int8 package
(`--per-member-output`, reproduces the deployed branch to **2.2e-08**), after which
every prune is offline arithmetic.

| component | deployed | pruned | cost |
|---|---|---|---|
| `world25` | 84.54 MB | **22.80 MB** (5 archs x 4 folds) | **2 of 405 rows** |
| `imu_stats` | **87.64 MB** (1000 trees, no cap) | **9.00 MB** (200 trees, depth 12) | **-6 clips / 2,700 pooled OOF** |
| `skel_mb_f2` (MotionBERT) | 241 MB fp32 | dropped | 9 of 405 rows |
| video slot | 137 MB (MViT x4) | 34.3 MB (all-train x1) | *pending* |

**`imu_stats` was the larger blocker and nobody had seen it**, because the member is
refit at inference and never written to disk -- its size existed only in RAM. It is
also the one component that **cannot** be dropped: removing it moves **43 of 405 rows**.

**MB per unit fusion weight** is the metric that makes the prune obvious:

| tag | weight | MB | MB/weight |
|---|---|---|---|
| `imu_world` | 0.2500 | 2.50 | **10** |
| `astgcn_v1` | 0.1237 | 3.39 | 27 |
| `skel_jvb_big` x5 seeds | 0.2290 | 50.65 | **221** |

**Candidate legal package (all measured except the video row):**
`world25` p4 22.80 + `imu_stats` 200/12 9.00 + MViT person all-train 34.3 +
MViT wrist all-train 34.3 = **100.4 MB**; swapping `imu_stats` to 150/10 (4.61 MB,
-9 clips/2,700) gives **96.0 MB**. Without the wrist view it is **66.1 MB**.
This is the first legal Stage-2 package of the campaign, and Stage 2 is a **gate**.

**Calibration note:** rowdiff measures the size of a perturbation, not its sign. Every
sign above comes from the 2,700-clip pooled OOF, which EXP-104 established tracks
public ~1:1; fold-2 OOF is not used for any adoption decision here.

---

## EXP-104 — MViT alone = 166 public (new champion). Swin3D-T refuted. Wrist crop built.
**Date:** 2026-08-22 · **Tier:** exploit/explore · **Purpose:** SCORE

**`sub_n8` (MViT x4 alone in the video slot) and `sub_n9` (CNN bag + MViT, half each)
both scored 0.82587 = 166/201**, +2 over the 164 champion. **Identical scores from a
34 MB video branch and a 657 MB one** — the K400 and IG-65M families are fully
redundant now, which is what makes a legal Stage-2 package possible.

**Calibration result worth banking:** pooled 4-fold OOF predicted +2.8 public clips
and public delivered +2. After a campaign of OOF inversions, the **2,700-clip pooled**
estimate tracks roughly 1:1. Fold-2 OOF never did (it produced n7's −3). Screen on
pooled, never on a single fold.

### Swin3D-T — REFUTED as a second 224-native family

| member | micro | object | motion |
|---|---|---|---|
| `k224_mvit_f2` | **0.71472** | 311/479 | 0.89595 |
| `vid_ig65m_f2` | 0.67638 | 289/479 | 0.87861 |
| `k224_swin_f2` | **0.61656** | 277/479 | **0.72254** |

Below even K400 (0.63957), with motion collapsing 0.90 → 0.72. Swin3D's K400 weights
are trained on **32 frames** and its 3D attention windows assume that depth; feeding 16
mismatches the temporal window. Not retried — the failure is structural, not a
hyperparameter.

### Wrist crop — built, training

57.7% of residual error is fine-grained hand-object confusion (EXP-103). `yolo11n-pose`
gives wrists in image space, which the skeleton cannot: the skeleton is 3D world
coordinates, pelvis-centred and floor-aligned, so it cannot drive an image crop without
calibration we do not have.

Measured over all 3,338 clips: **wrist window found for 3,321 (99.5%)**, only 2.6%
falling back to the person box. Median wrist side **147 px** against the person crop's
416, so the hands render at **224 px instead of 79** — 2.8x linear, 8x the pixels, on
exactly the region that carries the error.

### Also set aside, on existing evidence rather than new work

- **Deep IMU model.** `imu_stats_member.py` already records that neural members were the
  wrong model class for a 10.8 Hz stream with a **median 23 samples per device per
  clip**, and that dropping angle/magnetometer/quaternion *improved* accuracy
  (0.3856 → 0.4030) because they encode room heading, not activity.
- **Thermal early fusion (was item ③).** Thermal is a *different camera* with no
  calibration to IR/depth, so a 7-channel tensor would not be pixel-aligned. Early
  fusion is not the cheap experiment the handover implied.

---

## EXP-103 — MViT 4-fold pooled: +2.42 over IG-65M (p=0.0016), and ALONE it beats every bag we own.
**Date:** 2026-08-22 · `code/run_mvit_folds.sh` · **Tier:** exploit · **Purpose:** SCORE + packaging

Folds 0/1/3 trained to give MViT the same coverage the CNN bag has. Pooled over all
**2,933** clips — not the 552 that misled EXP-102's deployment choice:

| family | micro | object clips | motion |
|---|---|---|---|
| K400 | 0.64371 | 1124/2065 | 0.88018 |
| IG-65M | 0.68735 | 1233/2065 | 0.90207 |
| **MViT-224** | **0.71156** | **1299/2065** | **0.90783** |

Per-fold micro 0.70516 / 0.69287 / 0.71472 / 0.73966 — positive against IG-65M on
every fold. Paired: MViT-only-right **282**, IG-only-right **211**, net **+71 clips**,
**McNemar chi2 = 9.94 (p ~ 0.0016)**. Agreement 0.7317, so decorrelated as well as
stronger.

**The finding that matters: the older families are now redundant.** Pooled OOF
through the fold-safe decoder, video slot =

| video slot | decoded |
|---|---|
| K400 + IG-65M (deployed) | 2095/2700 |
| IG-65M + MViT | 2121/2700 |
| K400 + IG-65M + MViT | 2124/2700 |
| **MViT alone (4 folds)** | **2133/2700 = 0.79000** |

**MViT alone is the best video slot measured, +38 clips over the deployed pair.** This
is the first time in the campaign that *removing* members helped — every previous
replacement was flat (`g4only` 157 = `bag4_prior25` 157). The difference is that
IG-65M merely matched K400's strength, while MViT is +2.42 pooled over IG-65M.

**Packaging is solvable for the first time.** MViT is 34.3 MB int8:

| package | size | |
|---|---|---|
| deployed video bag (13 CNN + 4 MViT) | 657 MB | hopeless |
| MViT x4 | 137 MB | over |
| **MViT x2** | **68.6 MB** | **fits** |
| **MViT x1 (all-train)** | **34.3 MB** | **fits with room** |

`world25` at 85.2 MB is the remaining blocker, but **32 of its 48 members are seed
replicas** — pruning to distinct architectures should land near 10-20 MB, leaving
MViT x2 + pruned skeleton around 84 MB. That is the first credible route to a legal
Stage-2 package, and Stage 2 is a **gate**: `OBJECTIVE.md` records top-15 advancing to
a Selection Stage where the organizers reproduce the solution, so failing it forfeits
every remaining mark regardless of rank.

**Candidates:** `sub_n8` (MViT alone, rowdiff 26), `sub_n9` (CNN/MViT half each, 15),
`sub_n10` (all 17 equal, 12), all against the 164 champion.

---

## EXP-102 — B-030 CONFIRMED. 224px + a 224-native backbone is the strongest member of the campaign.
**Date:** 2026-08-21 · `kaggle/cuhkx_224_kaggle.py` · **Tier:** explore · **Purpose:** SCORE

`k224_mvit_f2` — MViTv2-S, Kinetics-400, 224px JPEG person crops, fold 2:

| member | micro | object | motion | int8 |
|---|---|---|---|---|
| `vid_r2p1d_f2` (K400) | 0.63957 | 263/479 | 0.89017 | 31 MB |
| `vid_ig65m_f2` (IG-65M) | 0.67638 | 289/479 | 0.87861 | 63 MB |
| **`k224_mvit_f2`** | **0.71472** | **311/479** | **0.89595** | **34 MB** |

**+3.83 micro and +22 object clips over the best member we owned**, at half its size.
The jump is the same size as the K400 -> IG-65M jump (+3.68 on this fold), and it
**lands where the mechanism predicted**: OBJECT, the hand-object detail a 3.1x
downsample destroys. Motion rose too, so nothing was traded away.

**Bag composition, fold 2:**

| bag | micro | object clips |
|---|---|---|
| 7-member CNN bag | 0.70859 | 306 |
| **mvit alone** | **0.71472** | **311** |
| 7-member CNN bag + mvit | 0.71779 | 312 |
| **mvit + ig + ig32 + igU** | **0.72393** | **316** |

**One member now beats the entire seven-member bag.** Agreement with `ig` is 0.7163,
so it is decorrelated *and* stronger. Note `ig+mvit` alone scores 0.70552, *below* mvit
by itself — equal-weighting a much weaker member drags it down, which is why the
deployed candidate gives mvit half the video slot rather than 1/14 of it.

**The enabling trick was storage, not compute.** A 224px cache is 10.7 GB as raw uint8
— the reason 224 was written off as impossible on a 15 GB laptop. As JPEG q90 it is
**1.13 GB train + 0.15 GB test**, measured **8.1x** smaller, decoding in 10.9 ms/clip
(~6 s per epoch across 4 workers against ~230 s of GPU). Training ran on the local
8 GB card at batch 4, 5.18 GB peak, 246 s/epoch. **The hardware was never the
constraint; the storage format was.**

**Also retired:** the plan to rent Kaggle GPU. The competition page hosts only
`sample_submission.csv` and `test.csv` — the ~50 GB of frames is not there, so that
plan could not have worked as written.
**Beliefs updated:** B-030 -> 90% CONFIRMED. EXP-093's res160 null formally retracted
as a step-size + backbone-mismatch artifact.
**Candidates:** `sub_n7` (mvit = half the video slot, rowdiff 21) and `sub_n6`
(mvit as one of 14 equal members, rowdiff 11), both against the 164 champion.

**Note on the run log:** epoch 10 reported 16878 s against 246 s for every other
epoch. The machine stalled, not the recipe; the loss curve is continuous across it.

---

## EXP-101 — The binding constraint is RESOLUTION, and it is a hardware constraint. 0.89 is real: four teams are there.
**Date:** 2026-08-21 · **Tier:** explore · **Purpose:** INFORMATION

**Live leaderboard (222 teams, pulled via the Kaggle API):**

| rank | score | clips | note |
|---|---|---|---|
| 1 | 0.98009 | 197 | +13 clips clear of rank 2 — outlier |
| 2–5 | 0.91542 / 0.91044 / 0.90049 / 0.89552 | 184 / 183 / 181 / 180 | **tight four-team cluster** |
| 6–7 | 0.86567 / 0.85572 | 174 / 172 | |
| 8 | 0.82587 | 166 | |
| **9–10** | **0.81592** | **164** | **us** |

**Four independent teams clustered at 180–184 is the signature of a reproducible
method, not a leak** — a leak yields near-perfect scores (rank 1) or scatter, not a
band. **0.89 = 179 is therefore established as achievable.** Our own failure is not
the ceiling; the standing directive applies.

**What we are giving away, measured** (`crop_window` on 40 sampled clips):

| | |
|---|---|
| source frames | 640×480 |
| person-crop side | median **396 px**, p25 225, p75 464 |
| crops ≥ 224 px | **100%** |
| crops already ≤ 128 px | **0%** |
| downsample to reach the 128 cache | **3.1× median** (≈9.6× fewer pixels) |

`crop_window` caps the square at `min(W,H)=480` and floors it at `0.35·max(W,H)=224`,
so **every** crop is between 224 and 480 px and every one is downsampled. 75% of our
error mass is OBJECT classes — hand-object interactions — which is precisely the fine
detail destroyed by a 3.1× downsample.

**This RETRACTS the standing reading of the res160 null.** 160 px is a **1.25× step**
against a 3.1× loss, and R(2+1)D's native pretrain resolution is **112×112**, so 160
moved the input *further* off the backbone's distribution than the extra pixels were
worth. Resolution was tested at the wrong step size with the wrong backbone. It is
**not** a closed axis.

**Why we cannot fix it on this machine:** a 224 cache is **10.7 GB** against 8 GB of
available RAM (15 GB total, 6 GB used); 192 px is 7.9 GB and already recorded as
thrashing. Disk is fine (37 GB free). GPU is an 8 GB RTX 4060 running batch 2 + accum 8.
**The constraint is hardware, not method.**

**Implication:** Kaggle supplies 30 GPU-hours/week free (T4×2 / P100 16 GB) and already
hosts this dataset. A 224-px cache with a **224-native** backbone (VideoMAE-V2,
Video Swin, MViTv2, X3D-L) is the untested experiment with by far the largest expected
gain — and a single strong model also **fixes the Stage-2 packaging risk**, which our
12-member ~700 MB bag cannot.
**Beliefs updated:** B-030 NEW (resolution is the binding constraint; the res160 null was
a step-size and backbone-mismatch artifact, not evidence against the axis).

---

## EXP-100 — AdaBN CONFIRMED on public (+2, new champion 164). Start-weight REFUTED (-8): test groups are fragments.
**Date:** 2026-08-21 · **Tier:** exploit · **Purpose:** SCORE + INFORMATION

| submission | change vs the 162 champion | public | clips |
|---|---|---|---|
| `sub_h8all` (old champion) | — | 0.80597 | 162 |
| **`sub_n1`** | **AdaBN alone** | **0.81592** | **164** |
| `sub_n2` | AdaBN + start-weight 0.5 | 0.78109 | 157 |
| `sub_n3` | AdaBN + start-weight + 12th member | 0.78606 | 158 |
| `sub_h8all_sw05` | start-weight 0.5 alone | 0.76616 | 154 |

**Three effects cleanly separated** because the ladder was built as single changes:
- **AdaBN = +2** (162 -> 164). OOF predicted +1.53 micro on the member; public
  delivered +2 clips. **B-028 confirmed.**
- **start-weight 0.5 = -8** (162 -> 154, and -7 on top of AdaBN: 164 -> 157).
- **12th member (`vid_ig65m_f32_f2`) = +1** (n2 157 -> n3 158), matching its +0.15
  fold-2 micro.

**Why start-weight failed, measured — this is NOT ordinary overfitting.** The OOF said
+14 clips on 2,700 and public said -8 on 201: opposite sign, large magnitude. The cause
is a **structural mismatch in the thing the lever depends on**:

| | train | test |
|---|---|---|
| mean group size | 3.72 | **2.83** |
| singleton groups | 42/783 = **5.4%** | 28/143 = **19.6%** |
| max group size | 10 | 8 |
| clips that are group-first | 26.9% | **35.4%** |

**Test recording groups are fragments of passes.** The first clip of a test group is
frequently *not* the first clip of the real recording pass — merely the earliest
surviving one. EXP-097's prior ("every pass opens with class 36 `Walk`; classes 21, 22,
5, 33 never open one") is a true statement about *complete* passes and a false one about
*truncated* ones, so it is applied wrongly to 35.4% of test clips. The mechanism analysis
was right about train and irrelevant to test.

**The lesson generalises beyond this lever:** OOF validates a *parameter* against a
*distribution*, but it cannot validate a **structural assumption** that train satisfies
and test does not. Any future lever keyed on group position, group length, or group
completeness inherits this defect and must be checked against the test group-size
histogram before it is measured, not after.
**Beliefs updated:** B-028 -> 90% (confirmed on public). B-029 NEW.
**Retracted:** EXP-097's "+~1 public clip" estimate for `--start-weight`. The direction
was wrong, not just the size.

**Next candidate:** `sub_n5.csv` = AdaBN + 12 members, **start-weight 0.0** — the two
confirmed-positive changes with the refuted one removed. Expected 165. rowdiff 7 vs `n1`.

---

## EXP-099 — AdaBN: parameter-free test-time BN re-estimation is worth +1.53 micro, all of it in OBJECT.
**Date:** 2026-08-20 · scratchpad `adabn_probe.py` · **Tier:** explore
**Purpose:** SCORE + robustness. Targets cross-subject shift, which is worth 2.5x the public LB.

Video models carry BatchNorm running statistics estimated on the 18 training
subjects and apply them unchanged to unseen subjects. Re-estimating those buffers
from the **unlabeled** held-out clips (labels never touched) on `vid_ig65m_f2`:

| | micro | object | motion |
|---|---|---|---|
| as deployed | 0.67638 | 0.60334 (289/479) | 0.87861 |
| AdaBN w=0.50 | 0.68558 | 0.61587 (295/479) | 0.87861 |
| AdaBN w=0.75 | 0.69018 | 0.62213 (298/479) | 0.87861 |
| **AdaBN w=1.00** | **0.69172** | **0.62630 (300/479)** | 0.87283 |

**Monotone in w, optimum at the endpoint.** This is the property that matters: the
best setting is "replace the statistics", not an interior value, so there is no
fitted parameter and none of the failure mode that killed six previous fitted
levers. +11 object clips, -1 motion clip: the gain sits exactly on the 75% of the
error mass.

Costs one extra forward pass, no labels, no training, no packaging bytes. It should
also help the on-site 8-new-subject stage by construction.
**Beliefs updated:** B-028 NEW (feature-space adaptation transfers where
probability-space fitting does not).

---

## EXP-098 — Weight axis closed on the CURRENT members; the oracle gap has collapsed to 7.85 pts.
**Date:** 2026-08-20 · **Tier:** exploit · **Purpose:** INFORMATION

Re-measured on pooled 2,700-clip OOF with the IG-65M-era members, because the
deployed weights were fitted when the video branch was 4.4 points weaker.

| branch | solo OOF | cost to drop |
|---|---|---|
| `world25` skeleton (48 members) | 0.5919 | -37 clips |
| `imu_stats` ExtraTrees | 0.4033 | -22 clips |
| **video bag8** | **0.6981** | **-313 clips** |

Full 3-way grid optimum = 1982 vs champion 1969 = **+13 clips on 2,700 = +1 public
clip**, and that is the selection-biased number. The axis is flat *for the new
members too* — this is now measured, not inherited.

**Oracle: any-branch-correct 0.8078 vs fused 0.7293 = 7.85 pts**, down from the
15.4 pts measured when video was weak. As the video branch improved, the headroom
inside the existing members disappeared. **"Extract more from the members we have"
is no longer a large prize.**

Post-decoder, base+imu are worth **137 clips (5.1 pts)**, more than their 2.2 pts
pre-decoder — the decoder amplifies them, so they cannot be dropped for packaging
cheaply.

---

## EXP-097 — Session-structure audit: distinctness dead at session scale, cross-group chaining refuted, start-weight found unused.
**Date:** 2026-08-20 · **Tier:** explore · **Purpose:** INFORMATION

The radar key is a timestamp (`2025-06-17_14-32-57.263`), so sessions are
recoverable. 404/405 test clips carry one; **7 distinct days**, 262 of 397
same-day consecutive gaps under one minute.

Three results, two of them kills:

1. **Distinctness at session scale is FALSE.** Clustering train by (user, day,
   gap<=5min) gives 147 sessions with **1,937 duplicate-label clips in 125 of
   them** — trials `1-1-1/2/3` repeat each activity. Killed in minutes.
2. **The chain does not continue across group boundaries.** Cross-group transition
   top-1 = 0.340 against a **0.392** unigram baseline: the transition model is
   *worse* than knowing "this clip is first in its group". Within-group is
   0.351 vs 0.068, a 5x lift. Do not merge groups.
3. **`--start-weight` defaults to 0.0** ("the fold-safe gate selected zero" — gated
   when public was ~125). P(class | first-in-group) vs global: **KL 0.428 nats**,
   H 3.492 -> 2.708, class 36 `Walk` enriched 3.4x (38.8% vs 11.4%), and four
   classes (21 `Read_documents`, 22 `Turn_pages`, 5 `Put_on_clothes`, 33 `Lie_down`)
   are **hard zeros in 783 groups**. Physically grounded: every pass begins with the
   subject walking into the scene. 143 of 405 test clips are group-first.

Re-measured on the current fusion: sw=0.5 gives **+14 clips on 2,700** (unimodal,
3/4 folds positive) = **~+1 public clip**. Real but small; free, so it rides along
with the next member change rather than spending a submission.

Also verified: train `radar:` groups (783, sizes 1-10) and test groups (143, sizes
1-8) are the same shape, and `trial:` ~= `radar:` (792 vs 783). **The old plan's
"train and test group by different partitions" concern is wrong.** Self-transitions
are 0/2141.

---

## EXP-090..096 — The IG-65M campaign: 156 -> 162.
**Date:** 2026-08-19/20 · `code/ig65m_model.py`, `code/train_video_crop.py` · **Tier:** exploit

**EXP-090 IG-65M R(2+1)D-34 adopted.** Per-fold vs the K400 r2plus1d_18 member:
+5.16 / +5.65 / +3.68 / +2.45, positive 4/4. Pooled 0.64371 -> **0.68735**, paired
**p=1.04e-08**, net +128 clips on 2,933.
*A midplane bug was caught before execution:* torchvision reuses conv1's midplanes
for conv2 (230/460/921) where the checkpoint uses (planes,planes) (288/576/1152).
18 of 416 tensors would have silently failed to load, randomising every
downsampling path and producing a **false negative on the largest lever of the
campaign**. `build_ig65m` now asserts a complete load.

**EXP-091 additions, not replacements — confirmed 3x.** `g4only` (IG-65M *replacing*
K400) = **157**, identical to `bag4_prior25` (K400 alone) = 157. `g5` (IG-65M
*added*) = **161**. A provably stronger member swapped in is worth +0.00; bagged in
it is worth +4.

**EXP-092 MotionBERT skeleton member.** The load-bearing detail was the coordinate
system: our skeleton is world-frame with **z up**, so the image-plane feed is
(x, -z), not (x, y). Measured on 400 clips: head-minus-foot z +1.131 vs y -0.119.
Feeding (x,y) would have looked like "MotionBERT does not transfer".

**EXP-093 variants.** `f32` / `res160` / `upper` are individually null or marginal
but bag-positive. `ig65m_upper` is the sharpest case: **identical object accuracy
solo (the same 289/479 clips)** yet the best bag member tested —
ig65m+k400+ig65m-upper+k400-upper = 0.70399 vs ig65m alone 0.67638. Solo null,
bag +2.76. Do not screen bag members on solo score.

**EXP-094 prior tilt.** alpha=0.25 -> **157**; alpha=0.50 -> 152. Adopted at 0.25.

**EXP-095 composition ladder** (public, all with the champion decoder):
`gonly` 151 · `bag4_prior25` 157 · `g4only` 157 · `h8` 160 · `g5`/`g5mb`/`h8mb` 161
· **`h8all` 162**. Among >=5-member bags the whole axis spans 157-162, i.e. about
one SD. **The composition axis is saturated; a 12th member buys ~0-1 clip.**

**EXP-096 dilution.** `m15` (adds 2 all-18 seeds + mc3_18 + r3d_18 = 15 members) =
**160, -2 clips**. The video slot is an equal-weight log-mean, so weak members
dilute strong ones. `m16` (+thermal) and `m11e` (13 members) remain unscored.

---

## EXP-088 — Thermal through the video recipe: strong member, decorrelated, and it adds NOTHING.
**Date:** 2026-08-16 · `code/build_thermal_cache.py`, `code/train_video_thermal.py` · **Tier:** explore
**Purpose:** SCORE. Thermal is the paper's top-ranked sensor (92.57) and both public notebooks discard it.

**Setup:** identical to EXP-086 except the input tensor — YOLO person-crop on thermal's
own frames, 3-channel (ironbow JPG kept as RGB, so the Kinetics stem needs no surgery
at all), 2165 train clips after dropping 116 with no thermal, same fold-2 outer set.

**Result:** micro **0.54448**, object 0.46764 (224/479), motion 0.75723. Against the 2D
ImageNet thermal member's 0.34783 that is **+19.7 points** — the crop+video recipe
transfers to a third modality, so EXP-086 was not an IR/depth-specific fluke.

**And yet it contributes zero.** Added to the champion fusion at every weight from 0.05
to 0.35, the best result is +0.00 (flat at w=0.15) and it degrades from w=0.20 up. This
despite video/thermal argmax agreement of only **0.5435** — as decorrelated a pair as we
have ever measured.

**Why (measured, not assumed):** thermal is uniquely correct on **34 of 552 clips the
champion gets wrong** — real headroom, ~12 public clips. But it is wrong on 247, and its
mean top-1 confidence is **0.511 on its unique-correct clips vs 0.413 on its errors**. The
separation is far too small for any global weight or confidence gate to harvest the 34
without importing a larger share of the 247.

**Conclusion:** decorrelation is NOT sufficient for fusion gain. A member must be either
accurate or *calibrated* — thermal is neither enough of the first nor remotely the second.
This kills the "add more decorrelated modalities" line as a general strategy and explains
retrospectively why IMU worked (its errors are confined to classes where it is reliably
unconfident) while thermal does not.
**Beliefs updated:** B-027 NEW (decorrelation is insufficient; calibration separation is
the binding requirement for a fusion member).
**Not doing:** thermal folds 0/1/3. 8 GPU-hours for a member with no extractable signal.

---

## EXP-086b/089 — Four video folds complete; all-18 members training.
**Date:** 2026-08-16 · `code/run_vid_folds.sh`, `code/run_all18.sh` · **Tier:** exploit

| fold | micro | object | motion |
|---|---|---|---|
| 0 | 0.62776 | 0.49732 (278/559) | 0.91373 |
| 1 | 0.61916 | 0.54188 (317/585) | 0.81659 |
| 2 | 0.63957 | 0.54906 (263/479) | 0.89017 |
| 3 | 0.69832 | 0.60181 (266/442) | 0.90047 |

Mean 0.6462, spread 0.079 — the recipe replicates; fold 2 was not a lucky draw. Pairwise
test-set argmax agreement between folds is only **0.625–0.686**, so the 4-fold bag is
genuinely additive rather than four copies of one model.

**Bug (mine, cost ~1 GPU-hour):** `build_model()` unconditionally rebuilt the stem as
4-channel, so the first thermal run failed every forward pass and the watchdog burned its
8 restarts. `build_model` now takes `in_channels` (default 4, so all existing checkpoints
rebuild identically). Fold 1 was interrupted at epoch 9 to re-prioritize thermal and
resumed from its per-epoch state, costing one partial epoch.

**Running:** `vid_all18_s*` — two seeds trained on all 2,933 clips with no held-out fold.
Each fold member sees only ~2,200; the published notebook trains on everything and its
single model scores 143/201 against our best single fold's 121 solo, so part of that gap
is simply data volume.

---

## EXP-086 — **Person-crop + Kinetics video backbone: +23.9 micro over the best prior member.**
**Date:** 2026-08-15 · `code/build_crop_cache.py`, `code/train_video_crop.py`, `code/infer_video_crop.py` · **Tier:** exploit
**Purpose:** SCORE. Ports the published LB 0.711 recipe (143/201 vs our 131).

**Setup:** YOLO11n person crop (union of 8 per-probe top-confidence boxes, margin 1.4,
floored at 0.35·max(W,H), ONE fixed window for all 16 frames) → 128×128×4ch
(Depth_Color RGB + IR) → torchvision `r2plus1d_18` Kinetics-400, 4-ch stem with the IR
kernel initialized to the mean of the RGB kernels. lr 5e-5, EMA 0.99, 30 epochs, last
epoch kept (never best), clip-level augmentation only. Fold 2, identical outer subjects
as EXP-083/084.

**Result — single change (backbone+crop) against every prior visual member, same fold:**

| member | micro | object (/479) | motion |
|---|---|---|---|
| from-scratch MIL trunk | 0.35123 | 0.24843 (119) | 0.63584 |
| ImageNet ResNet18 | 0.34816 | 0.26931 (129) | 0.56647 |
| ImageNet ResNet50 | 0.40031 | 0.34238 (164) | 0.56069 |
| **crop + Kinetics R(2+1)D** | **0.63957** | **0.54906 (263)** | **0.89017** |

**+23.9 micro / +99 object clips / +25.4 motion** over the best prior member. Three times
the largest effect previously measured in this campaign, and unlike every earlier lever it
moves BOTH error masses at once.

**Fusion (fold-2 OOF, 552 clips overlapping the world25 base; 1 clip = 0.18 pts):**

| configuration | micro | object |
|---|---|---|
| base (world25 skeleton stack) alone | 0.56341 | 0.46649 |
| video alone | 0.61957 | 0.50515 |
| imu_stats alone | 0.36051 | 0.23196 |
| base + video, w=0.55 | 0.65399 | 0.55155 |
| base + {video, r50} | 0.61594 | 0.51289 |
| base + {video, mil_v1} | 0.62500 | 0.52577 |
| **base + {video, imu_stats}, w=0.55** | **0.68841** | **0.59794** |
| base .30 / video .25 / imu .45 (grid argmax, fitted) | 0.70833 | 0.62629 |

**Two findings beyond the headline:**
1. **The older visual members are now dead weight.** Adding ResNet50 to the visual slot
   COSTS 3.8 micro (0.65399 → 0.61594); mil_v1 costs 2.9. Every visual member built before
   this one is superseded, not complementary — they should be dropped, not re-weighted.
2. **IMU is strongly complementary**, +3.4 micro / +4.6 object over video alone, and the
   fitted grid wants w_imu ≈ 0.45 despite imu solo being 0.36051. Weak-but-decorrelated is
   exactly the profile that earns high fusion weight.

**Candidates built** (rowdiff vs champion `sub_champ_bounigram` = 131/201): `sub_vidimu_A_trans05`
88, `sub_vidimu_C_trans05` 102, `sub_vidimu_A_raw` 123, `sub_vidcrop_f2_solo` 176. All far
above the ±9–10 clip noise floor. Weights taken from the plateau, NOT the grid argmax:
fitted combinations have failed 4/4 on this competition.

**Packaging constraint (new, load-bearing):** r2plus1d_18 is 31.3M params = 62.6 MB fp16,
31.3 MB int8. A 4-fold bag does NOT fit the 100 MB single-file budget in fp16. Rules §2.8.b
makes a >10% Stage-2/Kaggle gap a disqualification, so the LB configuration and the
shippable configuration must be reconciled before the final submission, not after.

**Conclusion:** The diagnosis was right — our members were weak, and the fusion was
polishing weak components. This is the first member strong enough to carry the ensemble.
**Confidence:** 90% that a public gain lands; unmeasured until scored.
**Beliefs updated:** B-026 NEW (crop+video pretraining dominates). The "visual family is
near its ceiling" belief is falsified — the family was never at a ceiling, only the
from-scratch/ImageNet-2D sub-family was.
**Next:** folds 0/1/3 running (`code/run_vid_folds.sh`, ~8 h); then thermal, which both
public notebooks discard and which the paper rates the best single modality (92.57).

---

## EXP-079/080/081 — **THE CAMPAIGN WAS BUILT ON A FALSE RULE. Pretrained backbones are legal.**
**Date:** 2026-08-10 · **Tier:** foundation · **Purpose:** INFORMATION+SCORE

Atharv issued `research/DIRECTIVE.md` (no premature ceilings; investigate external
evidence via API *before* ideating). Executing STEP 2 took ~10 minutes and overturned
the campaign's foundation. Full quotes in **`research/RULES_VERIFIED.md`**; ledger in
**`research/EVIDENCE.md`**.

**R-1 (topic 711665, 2026-06-25):** *"the restriction 'no large pretrained backbones
permitted' is specifically intended to prohibit … LLMs or large vision-language
foundation models. **Small, standard pretrained CNNs such as ImageNet-pretrained
ResNet18 (~44MB) are perfectly acceptable in the Small Model Track.**"*

Also legal and previously assumed forbidden: external public datasets (**R-2**; NTU
RGB+D, UCI HAR, PAMAP2, WISDM named), knowledge distillation from large teachers
(**R-3**), pseudo-labeling test data (**R-4**), GBDT/SVM in-pipeline (**R-5**),
ensembles bundled to one ≤100 MB file with fp16/int8 encouraged (**R-6**, which also
closes Plan.md Phase 7's open packaging question).

**How the error happened — LOG.md:2712**, a prior session: *"NO pretrained weights at
all — strict from-scratch"* → concluded the 0.73–0.77 teams were *"DISQUALIFIABLE at
reproduction"* → *"pretrained probes (EXP-015/T1) retain only diagnostic value; T1
rerun cancelled."* It explained away the leaderboard and cancelled the one probe that
would have caught it. EXP-015 had already measured ImageNet init at ~2× from-scratch
**here**, and that factor was used to *discount* the paper's baselines as unreachable.

**Live leaderboard (API, 2026-08-10):** 1st 0.91542 · 2nd 0.87064 · 3rd 0.82089 ·
15th 0.74129 · ours 0.62686. The previously stated "0.65–0.73 achievable band" is
**externally falsified**. The 07-31 snapshot in this file was stale (top-15 was
0.68159; the field gained ~6 points in 10 days).

**L-1 — a real test-label leak existed** (topic 714827): the public CUHK-X repo held
labeled split metadata matchable to test skeleton filenames by timestamp + frame ID.
Organizers confirmed, took it offline 2026-06-28, and said they will **widen Stage 2
selection toward "teams that demonstrate genuine progress."** We do not touch it
(cheating; anti-cheating checks; Stage 3 is on-site with 8 new subjects). Consequence
for calibration: **the LB top is not a clean modelling target**, and Stage 2 selection
rewards a legitimate reproducible pipeline over rank.

### EXP-079 — frozen linear probe (the cheapest discriminating experiment)
Frozen encoder, identical logistic-regression probe, identical fold-2 split, weights
the only variable:

| frozen encoder | micro | object (26 sedentary) | motion |
|---|---:|---:|---:|
| random-init ResNet18 | 0.18558 | 0.09603 (46/479) | 0.43353 |
| **ImageNet ResNet18** | **0.26074** | **0.16284 (78/479)** | 0.53179 |

**+70% relative on the object classes holding 75% of the error**, with no fine-tuning
and no temporal model, for ~4 minutes of GPU. This measures information
*accessibility* and is untouched by every fine-tuning result below.

### EXP-080 — fine-tune, and two failed predictions of mine
| fold 2 | micro | object | motion |
|---|---:|---:|---:|
| from-scratch MIL trunk (EXP-062/064) | 0.35123 | 0.24843 (119/479) | 0.63584 |
| ResNet18, 8 frames, BN live | 0.33282 | 0.24843 (119/479) | 0.56647 |
| ResNet18, 16 frames, BN live | 0.31442 | 0.22547 (108/479) | 0.56069 |
| **ResNet18, 8 frames, BN FROZEN** | 0.34816 | **0.26931 (129/479)** | 0.56647 |

1. I blamed EXP-080 on frame count (EXP-068 measured 16→8 at −4.1). **Refuted:**
   stride 1 scored 0.31442, *below* both. Prediction was ≥0.37. Recorded as failed.
2. BN was the real cause. IR/depth are grayscale-replicated colormaps whose channel
   statistics are nothing like natural images; 2281 clips of BN re-estimation washed
   out the ImageNet features. Freezing BN: **+2.09 object, first member ever to beat
   0.24843 there**, while losing only on motion — which skeleton covers in fusion.

### The confound that invalidated two "failures"
`sub_visual_mil_v1.csv` (0.38805 public solo) matches `testprobs_visual_mil_f0123` at
**405/405** — it is a **4-fold bag**. Both single-fold members were scored against it
and called failures: SSL 0.28358, pretrained **0.24378**. `sub_visual_mil_v1_folds2.csv`
exists on disk but **was never submitted**, so no single-fold control exists; and
LEADERBOARD.md's "bagging is worth zero" was measured on the *fused* submission where
the skeleton stack dominates, which does not transfer to a solo member.
**Neither SSL nor ImageNet has had a fair public test.**

### Target arithmetic, corrected (Atharv: "why are we aiming for >.37, goal is 0.8+")
Member solo scores are intermediate: skeleton 0.542 + visual 0.38805 fuse to 0.62686.
To fuse to ~0.80 against a 0.542 stack, the visual member needs ≈**0.70 solo** — not
0.45. Paper baselines (Thermal 92.57 / Depth 90.5 / IR 90.2, pretrained in-domain)
minus EXP-015's ~20-point cross-subject discount ≈ **0.70**. So the required member
quality is exactly what a properly trained pretrained visual model should reach, and
I had been anchoring targets on beating our own weak baseline instead.

**EXP-081 running:** folds 0/1/3 with BN frozen (`code/run_pre_r18_folds.sh`, ~28
min/fold vs the MIL trunk's 4.5h) → 4-fold bag, then a like-for-like public test
against 0.38805.

---

## EXP-076 — **SSL pretraining works. First lever in the campaign that moves the sedentary error mass.**
**Date:** 2026-08-10 · **Tier:** explore · **Purpose:** SCORE · **VERDICT: gate passed, directional pending folds 0/1/3**

Cross-modal masked pretraining (`code/ssl_pretrain_visual.py`) on all 3,338 clips
including unlabelled test, trunk identical to `train_visual_mil.py`. Masked-L1
**0.43761 → 0.24666** over 40 epochs, converged (0.24710 at ep33 → 0.24666 at ep40,
so more SSL *epochs* are not the lever). Checkpoint `checkpoints/ssl_trunk_v2.pt`,
139 tensors, geometry [32, 96, 128].

**The gate — two arms, fold 2, identical recipe/data/epochs, only `--init-trunk` differs:**

| fold 2 | scratch | SSL-init | Δ |
|---|---:|---:|---:|
| micro | 0.33282 | **0.39417** | **+6.14** |
| macro | 0.36135 | 0.39799 | +3.66 |
| **object (26 sedentary)** | 0.23591 (113/479) | **0.32359 (155/479)** | **+8.77** |
| gross_motion | 0.60116 | 0.58960 | −1.16 |

Predeclared gate was +2.0 micro; it cleared 3×. **The reason this is read as mechanism
rather than a seed draw** (σ = 2.80 on this partition) is not the magnitude but the
*shape*: the entire gain lands on the 26 object classes and motion stays flat. A seed
fluctuation does not respect that partition. This is precisely the capability EXP-068
proved the supervised trunk never acquired — the visual branch was a pure motion model —
and B-021's error mass is 75% inside those classes.

**Cache v2 geometry decided on evidence, not preference.** v2-scratch vs v1-reg is
−1.84 micro but only −1.25 object (113 vs 119 of 479, six clips, inside noise); the loss
is almost entirely motion (−5.78), which skeleton already covers in fusion. So the
96×128 downscale costs nothing on the axis that matters, and re-running SSL at v1
geometry (~9h) was **not** worth it.

**OOF fusion says the visual member does not help — and that is expected here.** On the
552 fold-2 clips shared with the world25 OOF, base alone is 0.5942/0.4923 (all/object)
and every fusion weight is ≤ that, including SSL (best 0.5906 at w=0.25). But on public,
visual geometric fusion is what *built* the champion. The two are reconciled by the
measured transfer asymmetry: skeleton loses 7.6 pts OOF→public, visual loses 0.1, so
OOF systematically **understates** the visual member's fusion value. Corrected for
transfer the members sit at ≈0.518 (base) vs ≈0.408 (SSL visual), which argues the
public-optimal weight is *above* the champion's 0.35. **This OOF check therefore cannot
veto the direction**, and is logged as reporting-only per the standing rule.

---

## EXP-077 — Champion fusion recipe recovered and committed; SSL member staged for public
**Date:** 2026-08-10 · **Tier:** exploit · **Purpose:** SCORE · **STAGED, unscored**

`sub_priorB_trans05.csv` (0.62686 = **126/201, new best**) was built ad hoc and was not
reproducible from any committed script. Recovered as
`code/fuse_prior_visual.py`: `geometric_mean(base / train_prior, visual; w=0.35)` then
`ordered_transition_decoder --transition-weight 0.5 --transition-score conditional
--distinctness none`. **Verified at 404/405 argmax parity**; the single differing clip
is a 1.6e-4 top-2 tie, i.e. float32 storage precision.

Also settled the prior question posed in EXP-075: **priorB (base → uniform space) 0.62686
vs priorA (visual → train space) 0.58706**. Same mis-specification, opposite directions,
20 clips apart — the visual member's balanced-softmax logits are the correct target space.

Fold-2 SSL test probabilities inferred (`testprobs_v2ssl_f2.npz`); it disagrees with the
deployed 4-fold visual member on **229/405** clips and is better calibrated (mean max
prob 0.3875 vs 0.3119). Staged, single-change from the 126 champion:

| submission | visual member | w | rowdiff vs champion |
|---|---|---:|---:|
| `sub_ssl035_trans05.csv` | SSL f2 | 0.35 | 79/405 |
| `sub_ssl050_trans05.csv` | SSL f2 | 0.50 | 106/405 |
| `sub_sslmix045_trans05.csv` | SSL f2 ⊗ v1 f0123 | 0.45 | 60/405 |

**Handicap to keep in mind when reading these:** the SSL member is a *single* fold-2
model (14 users) against a deployed *4-fold bag*. Any score it reaches is a floor on
what the completed member does.

**Running:** folds 0/1/3 SSL (`code/run_ssl_deploy.sh`, ~13.5h) — simultaneously the
replication the σ=2.80 floor demands and the deployment artifact.

**Defect fixed:** `model_config()` folds the dropout/augmentation constants into the
checkpoint guard, so any infer/eval call that omits the `CUHKX_VMIL_*` recipe env fails
with "model configuration changed" despite correct weights. Now one shared
`code/ssl_v2_env.sh` is sourced by every script that touches a v2_ssl checkpoint.

---

## EXP-075 — Three-agent fresh audit; four deployed-pipeline defects found; cache v2 built
**Date:** 2026-08-09 · **Tier:** foundation · **Purpose:** INFORMATION · **RUNNING**

Atharv called a full re-audit ("don't assume the code is correct"). Three parallel
agents: code correctness, ledger/artifact reconciliation, raw-data probes. Findings
that were not in any ledger:

**Defects in the DEPLOYED pipeline (code audit):**
1. **Fusion prior mismatch.** The visual member trains with balanced softmax, so its
   raw logits target a **uniform** prior; skeleton members use plain CE, so theirs
   target the **27.9×-skewed train** prior. `fuse_visual_geo.py` fuses them
   geometrically with no reconciliation — the two members disagree about base rates
   by construction. Two principled corrections built and staged.
2. **Transition decoder.** `--distinctness hard` is unused although **0/792 train
   groups contain a duplicate label** while the deployed argmax produces 44
   duplicate collisions on test (hard reduces them to 1, changing 30/405). Separately,
   `pmi` vs `conditional` scoring is a **verified no-op** (0/405 differ) — that audit
   item is closed. The decoder currently flips 93/405 = 23% of predictions.
3. **Ensemble weight misallocated.** world25 = 48 members but ~4 distinct
   architectures (32/48 are seed replicas); **38.1% of weight sits on two IMU
   members at ~0.30 accuracy** while the best member (astgcn_v1, 0.595) gets 12.4%.
   Inherited from EXP-040, never re-optimized.
4. **32/48 packaged members trained with `aug=0`** on a cross-subject problem, and
   no member was ever refit on all 18 users (old `cv_folds.json` holds out user5/21
   permanently, so the 0.6178 OOF never covers them).

**Raw-data gaps (probe agent, all measured):**
- Skeleton stack loads `n_frames=8` → **73.2% of IR/Depth frames discarded**; visual
  cache keeps 16 → 50.9% discarded; **thermal runs at 24.2 Hz** and is decimated 77.8%.
- Multi-person is **21.2% of test vs 10.2% of train** — a 2× shift. No tracking exists;
  slot-0 is the largest person in only 72% of frames and identity switches mid-clip.
  Worst classes are clothing (Put_on 65%, Fold 54%, Take_off 37%) — the garment is
  detected as a second skeleton — not the mirror classes. **`person="motion"` provably
  selects the identity-switching slot**, so EXP-023/057 tested a broken policy.
- Radar is **100.0% empty for users 16–24** and ~98% present for 1–9 → a free exact
  cohort label for every test clip; radar *filenames* collapse 404 test clips into 143
  user-pure sessions even when contents are empty.
- Thermal's colormap is inferno-family and **strictly monotone in luma** → grayscale
  averaging loses contrast only, not a dimension. Kills the "invert the colormap" idea.

**Ceiling correction (ledger audit) — changes the campaign's arithmetic.** The
paper's Depth 90.5 / IR 90.2 / Thermal 92.6 are **pretrained ResNet-50, random-split,
in-domain**. EXP-015 already measured both discounts here: ImageNet init ≈ *doubles*
from-scratch visual accuracy, and cross-subject costs ~20 pts even pretrained. So
"visual reaches 0.90 from scratch" was never supported, and **competition-data SSL is
the legal substitute for the init those numbers depend on** — not an optional extra.
EXP-021 measured a cross-modal masked pretext at +3.0 and it was never re-attached to
a CNN trunk. That re-attachment is now the highest-evidence untested experiment.

**Also:** the prize weighting (Kaggle private 20%, on-site 8-new-subject 30%, repro
10%, report 20%) makes cross-subject robustness worth 2.5× the leaderboard — which is
the same filter the 4/4 transfer failures already impose.

**Built this session:** `code/visual_mil_cache.py` geometry is now env-overridable
(defaults reproduce v1 byte-for-byte; `CACHE_VERSION` carries the geometry so a v2
build cannot be mistaken for v1), plus a **thermal span-reliability gate** — measured
over 572 clips the thermal/IR ratio is median 2.43 but std 0.70 with 23.6% outside
2.0–3.0 (min 0.02, max 7.60), i.e. for a quarter of clips the streams do not span the
same window and normalized-position sampling aligns nothing. Those clips now get
their thermal modality mask cleared rather than being fed misaligned frames.
**cache v2 = 32 frames at 96×128** (`code/run_cache_v2.sh` → `cache/visual_mil_v2`).

**Staged for public measurement (all single-change from the 125 champion):**
`sub_priorA_trans05.csv` (prior→train space, 46/405 rows), `sub_priorB_trans05.csv`
(prior→uniform, 64/405), `sub_champ_conditional_disthard.csv` (distinctness hard,
30/405), plus the four never-scored EXP-074 members. `app_all` (all-18) disagrees with
the 4-fold member on 36.5% of test clips, so that submission is a real measurement.

**Next:** SSL pretrain on cache v2 (trunk identity with `train_visual_mil.py`
mandatory), frame-level aux CE head to break the motion-only shortcut, skeleton
multi-person tracking (OOF structurally cannot measure it — public item).

---

## EXP-073 — GBDT stacking works on OOF, fails on public. **FOURTH consecutive transfer failure — the pattern is the finding.**
**Date:** 2026-08-08 · **Tier:** explore · **Purpose:** SCORE · **VERDICT: rejected; audit triggered**

**Prompted by Atharv:** "linear failure does not imply nonlinear failure" and
"build an expert only for those 26 classes". Both were correct criticisms. The
ledger had **zero** GBDT/stacking attempts — one "learned logit gate" (−6.8) had
been standing in for the whole family — and I had written off +7.1 clips of
measured information after testing only weighted fusion.

**Both ideas worked on OOF:**

| configuration | overall | sedentary | vs w=0.35 |
|---|---:|---:|---:|
| weighted fusion, best (T, w) | 0.6170 | — | +0.3 |
| GBDT pairwise stacker, 3 members | 0.6307 | 0.4982 | +3.1 |
| GBDT pairwise stacker, 8 members | 0.6418 | 0.5149 | +4.7 |
| **+ 26-class sedentary specialist (9)** | **0.6485** | **0.5173** | **+6.1** |

The specialist beat the generalist appearance member on its own classes,
**0.2473 vs 0.2171**, purely from restricting the label space.

**PUBLIC RESULT (user-verified):**

| submission | public | vs champion 125 |
|---|---:|---:|
| `sub_stack9_trans05.csv` | 0.61691 = **124** | **−1** |
| `sub_stack9_classid.csv` | 0.59203 = **119** | −6 |
| `sub_stack9_noclassid.csv` | 0.58706 = **118** | −7 |

**+6.1 OOF → −1 public.** Two useful signals inside it: class-id helped on public
too (119 vs 118), matching its OOF sign; and **the transition decoder is worth
+5 clips** (124 vs 119), not the +3 in the ledger.

**THE PATTERN, now four for four.** Everything that *learns something from the
training users* collapses on the public split:

| lever | OOF | public |
|---|---:|---:|
| skeleton stack | 0.618 | −7.6 transfer |
| ordered-transition decoder | +10.4 | +3…+5 |
| recording-structure decoder | +11.6 | 0 / −2 / −9 |
| GBDT pairwise stacker | +6.1 | −1 |

This is no longer a series of separate disappointments; it is one property of the
problem. **Cross-subject shift destroys fitted combination rules.** Stop proposing
them.

**A specialist bug worth remembering.** The first 26-class run scored exactly
**0.0000** on sedentary clips and 0.1803 on motion — inverted. Balanced softmax
adds log(prior) in training and infers without it; the 14 absent classes got
prior log(1e-9) = −20.7, so training never constrained their logits and raw
inference predicted only them. Masking unsupported classes at inference fixed it.
Read as a headline number it looked like a clean negative, and **EXP-054's
original specialist verdict deserves re-examination for the same reason.**

---

## EXP-074 — Audit pivot: rebuild the family that actually transfers
**Date:** 2026-08-08 · **Tier:** explore · **Purpose:** SCORE · **RUNNING**

**The audit finding, in one table:**

| family | OOF | public | transfer |
|---|---:|---:|---:|
| skeleton / IMU base | 0.618 | 0.542 | **−7.6** |
| visual MIL | 0.389 | 0.388 | **−0.1** |

**Three weeks went into the family that leaks 7.6 points, and almost nothing into
the family that leaks 0.1.** Every point added to a visual member should reach the
leaderboard intact; every point added to skeleton is taxed. The organizers' own
paper reaches 90.2 / 90.5 / 92.6 in-domain on IR / Depth / Thermal against 79.1
for skeleton, so the visual ceiling is far higher than what the 2.78 M-parameter
member reaches.

**Two levers chosen specifically because they are NOT fitted combination rules**
(see EXP-073's pattern):
1. **All-18-user training.** `SEARCH_MAP.md:72` has read
   `Validated full-data checkpoint strategy ?` for the entire campaign while every
   deployed member is a 4-fold ensemble whose models each saw 13–14 users. More
   training subjects generalising better across subjects is a mechanism, not a
   correlation, so it has an argument for surviving the shift.
2. **Capacity**, width 32 → 48 (2.78 M → 6.23 M parameters), on the visual family
   only. Three seeds each for the generalist and the 26-class specialist.

No epoch selection is performed for the all-18 members — there is no honest
held-out set left, so the budget is fixed at the value the fold runs converged on
and the last epoch is kept.

**Also queued (pure inference, no transfer risk in the parameter):** the
transition decoder's λ has only ever been set to 0.5 from an OOF fit, and it is
now measured at +5 public clips. λ ∈ {0.3, 0.7, 1.0} submissions are staged.

---

## EXP-071 — Single-frame appearance member: mechanism confirmed, value not extractable (+0.3 clips)
**Date:** 2026-08-08 · **Tier:** explore · **Purpose:** SCORE → INFORMATION · **VERDICT: not deployable**

**Setup:** `code/train_frame_appearance.py`, a 2.78 M-parameter 2D CNN over single
frames at 96×128 (EXP-068 showed finer detail is unused). One frame carries no
motion, so the model must learn appearance — the capability EXP-068 proved the
visual MIL branch entirely lacks. 4 folds, 40 epochs, ~10 min/fold against the MIL
branch's 9.3 h/fold. Fold accuracies 0.3170/0.3182/0.3206/0.3185 — a very stable
member.

**Result on the full 4-fold OOF (2700 overlapping clips):**

| member | overall | sedentary | unique on sedentary |
|---|---:|---:|---:|
| stack (base + visual, w=0.35) | 0.6156 | 0.4722 | — |
| visual MIL | 0.3893 | **0.2364** | 0.0284 |
| **appearance** | 0.3233 | 0.2183 | **0.0411** |

- The appearance model has the **highest unique contribution of any member** —
  4.11% of sedentary clips right where the stack is wrong, against visual MIL's
  2.84% — confirming the construction argument: denying a model motion makes it
  learn something genuinely different.
- **Oracle stack+appearance = 0.6511, i.e. +7.1 clips of real information.**
- **No fusion rule reaches it.** Log-space fusion at T=1 costs up to −51.7 clips
  because the model is overfit and overconfident. Temperature scaling recovers
  the loss but not the gain: the best cell of a 7×6 (T, w) sweep is
  **T=2, w=0.02 → +0.3 clips**. Arithmetic fusion peaks at +0.7 on one fold and
  does not survive. The 4-fold ensemble did not fix calibration.

**CORRECTION to the fold-2 reading.** On fold 2 alone the appearance model scored
0.2402 sedentary against visual MIL's 0.2039, and I called it the best sedentary
member in the repository. On all four folds it is **0.2183 against 0.2364** — it
is slightly *worse*. That claim was a single-fold artifact, the same error mode as
EXP-060 and EXP-066. Only the unique-contribution advantage survives four folds.

**Conclusion:** the diagnosis chain EXP-067 → EXP-068 → EXP-071 is sound and ends
in a wall. The error mass is fine-grained hand-object discrimination; the visual
branch is motion-only; a model forced onto appearance does learn complementary
information; and that information cannot be extracted by any weighted fusion,
because the member cannot signal when it is right. A learned gate is the obvious
answer and was already rejected (−6.8 nested).
**Kept:** `code/train_frame_appearance.py`, `code/build_frame_memmap.py`, the
1.73 GB frame memmap, and `oof_frame_app_mm.npz`. Cheap to retrain (40 min for
4 folds) if a per-clip gating mechanism is ever found.
**Beliefs updated:** B-024 confirmed; NEW B-025 (complementary information exists
at +7.1 clips but is not linearly extractable).

---

## EXP-070 — **REFUTED ON PUBLIC.** Structure recovers perfectly from timestamps and pays nothing
**Date:** 2026-08-08 · **Tier:** exploit · **Purpose:** SCORE · **VERDICT: REJECTED**

> **PUBLIC RESULT (user-verified, 2026-08-08) — the prediction was wrong.**
>
> | submission | public | vs champion 125 |
> |---|---:|---:|
> | `sub_w25vg035_consensus_trans05.csv` | 0.62189 = **125/201** | **0** |
> | `sub_w25vg035_consensus_trans05_dist.csv` | 0.61194 = **123/201** | **−2** |
> | `sub_w25vg035_struct_both.csv` | 0.57711 = **116/201** | **−9** |
>
> Predicted +5.5 clips (range +4.4…+6.6) at 90% confidence on the sign. Actual:
> **0 to −9.** Distinctness, which validated at **781/781 = 100%** on train, is
> *negative* on public. Three submissions spent.
>
> **This is EXP-063's transfer asymmetry again, and I walked into it having
> written the warning myself.** OOF gains in this repository do not transfer:
> the skeleton stack transfers −7.55, the transition decoder gained +10.4 OOF
> and delivered +3 public, and this gained +11.6 OOF and delivered ≤0. A
> constraint being *combinatorially true on train* is not evidence it pays on
> the public split, because the gain depends on the model's error distribution
> over the group, not on the constraint's validity.
>
> **Standing rule from this failure: no submission is spent on an OOF-validated
> lever again unless the mechanism has an argument for why it survives the
> subject shift.** The only trustworthy instrument is the public LB itself
> (5/day), which is how w=0.35 was found in the first place.
>
> The structure recovery itself is correct and is kept in
> `code/recording_structure.py` — 781/781 runs exact — it simply has no value
> as a decoder. Do not retry it under a new parameterisation.

**Original predeclaration and evidence follow.**

**Setup:** EXP-069 showed the block/repetition constraint is worth up to +8 oracle
clips but needs the grouping, which test does not label. Measured the inter-clip
gap distribution on train (n=2713):

| | median | p25 | p90 |
|---|---:|---:|---:|
| within a repetition | **1.3 s** | 0.6 | 4.2 |
| repetition boundary | **81.9 s** | 65.8 | 199.6 |
| block boundary | **185.1 s** | 126.4 | 606.6 |

The populations separate cleanly. A 20 s cut catches **100% of repetition and
block boundaries with a 0.1% false-positive rate inside repetitions**.

**Validation on train, against the true trial names (`code/recording_structure.py`):**
- runs matching exactly one true (group, repetition): **781/781 = 1.0000**
- runs with pairwise-distinct classes: **781/781 = 1.0000**
- positional triples that are truly one class: **694/734 = 0.9455**, covering 75.7%

Independent corroboration: `ordered_transition_decoder.py` recovers 144 groups /
376 clips in multi-clip groups on test, identical to this module's runs.

**Gain on 2700 OOF clips, fused w=0.35 (baseline 0.6156):**

| | overall | Δ clips/201 |
|---|---:|---:|
| distinctness only | 0.6352 | +3.9 |
| consensus only | 0.6522 | +7.4 |
| consensus + distinctness | **0.6733** | **+11.6** |
| **coverage-matched to test** | 0.6427 ± 0.0043 | **+5.5 (+4.4…+6.6)** |

The coverage-matched row is the honest estimate: test is a released *subset*, so
consensus reaches only 17% of its clips against 75.7% on train, while distinctness
reaches 376/405. The +11.6 figure is not what test will pay.

**Why this should transfer better than the transition decoder.** That decoder gained
+10.4 clips on OOF and delivered +3 on public — it *estimates* transition statistics
from train users. This module fits nothing: distinctness and 3× repetition are
properties of the recording protocol, and the only inputs are frame timestamps.

**Rejected inside this experiment:** extending consensus coverage on test by
matching clips across repetitions instead of by position. Free Hungarian on
Bhattacharyya affinity reached 0.537 clip-weighted purity; monotone
(Needleman-Wunsch) alignment 0.579; adding a positional prior made it *worse*
(0.522), and under simulated test subsampling all variants fell to 0.30–0.46.
The model's own predictions are too noisy to align with. Only the exact-triple
rule is shipped.

**Staged submissions** (from `testprobs_w25vg035_f0123`, the 125/201 member):

| file | pipeline | differs from champion |
|---|---|---:|
| `sub_w25vg035_consensus_trans05_dist.csv` | consensus → transition λ=0.5 → distinctness | 49/405 |
| `sub_w25vg035_struct_both.csv` | consensus + distinctness, no transition | 118/405 |
| `sub_w25vg035_consensus_trans05.csv` | consensus → transition λ=0.5 | 25/405 |
| `sub_w25vg035_struct_distinct.csv` | distinctness only | 106/405 |

**Confidence:** 90% that the sign is positive (constraints validated at 100%/94.6%
without labels); 60% on the +4…+7 magnitude.
**Beliefs updated:** B-022 upgraded — distinctness is exactly recoverable on test,
not merely plausible.
**Next ideas:** single-frame appearance model (EXP-068's diagnosis); the 26-class
sedentary specialist.

---

## EXP-069 — Recording block/repetition structure: real, but a +8 clip ceiling
**Date:** 2026-08-08 · **Tier:** explore · **Purpose:** INFORMATION · **Cost:** ~0 (CPU)

**Setup:** train timestamps parsed from Depth_Color filenames (2910 clips), sessions
split on a 20-minute gap, groups keyed `(user, station, block)` from the `A-B-C`
trial name.

**The structure is exact.** 267 groups. Class multiplicity within a group is
**3× for 926 of 1013 (group, class) pairs**; group sizes are multiples of 3
(6/9/12/15/18); **100% of repetitions contain pairwise-distinct classes**; median
3, mean 3.8 distinct classes per group. A block is an ordered list of K classes
performed three times, e.g. user1 `3-2-*` = `[32,36,13,34,17,20,29,35]` × 3.

**Ceilings measured on 2700 OOF clips over the fused stack (0.6363 baseline):**

| exploitation | overall | sedentary | vs baseline |
|---|---:|---:|---:|
| oracle repetition triples (upper bound) | 0.6767 | 0.5466 | **+8.1 clips** |
| group + rep-position triples (recoverable) | 0.6544 | 0.5308 | +3.6 clips |
| distinctness within each repetition | 0.6556 | 0.5187 | +3.9 clips |
| group + at-most-3-per-class assignment | 0.6411 | 0.5018 | +1.0 clips |

**Conclusion:** the constraint is real and legal (transduction is permitted) but
its *oracle* ceiling is +8 clips, and realistic recovery is +4 to +6. Repetitions
are not independent votes — same user, same station, same activity means errors
correlate, which is why EXP-04x repeat-consensus scored 112→111. Worth banking,
not a path to 0.90.
**Beliefs updated:** B-022 (distinctness) — quantified, ceiling now known.

---

## EXP-068 — **THE VISUAL BRANCH IS A MOTION MODEL WITH NO APPEARANCE.** Spatial detail is unused; temporal detail is everything
**Date:** 2026-08-08 · **Tier:** explore · **Purpose:** INFORMATION · **Cost:** ~15 min GPU

**Why run it:** EXP-067 said the prize is fine-grained hand-object discrimination,
which pointed at "crop to the hands / add resolution". Three iterations of a depth
based person-box detector reached only ~5/12 usable crops, and I was about to
spend 10+ GPU-hours on a cache rebuild. The assumption underneath the whole plan —
*that the model is resolution-starved* — had never been tested. It is testable for
free: hold a trained checkpoint and its tensor shapes fixed, and destroy either
spatial or temporal information in the input.

**Setup:** `visual_mil_reg1_f2`, fold-2 val (652 clips, 440 sedentary).
Spatial: downsample by scale s then upsample back to 192×256.
Temporal: hold each frame for k steps (16 frames → 16/k effective).

**Spatial — accuracy is flat to a quarter resolution:**

| input detail | overall | sedentary | Δ sedentary |
|---|---:|---:|---:|
| 192×256 (as trained) | 0.3512 | 0.2250 | — |
| 96×128 | 0.3528 | 0.2295 | **+0.45** |
| 48×64 | 0.3420 | 0.2159 | −0.91 |
| 28×38 | 0.3206 | 0.1932 | −3.18 |

**Temporal — accuracy collapses immediately:**

| effective frames | overall | sedentary | motion |
|---|---:|---:|---:|
| 16 (as trained) | 0.3512 | 0.2250 | 0.6132 |
| 8 | 0.3098 | 0.2068 | 0.5236 |
| 4 | 0.2485 | 0.1818 | 0.3868 |
| 1 (still image) | 0.0752 | 0.1114 | **0.0000** |

**Conclusion — the model has learned motion and essentially zero appearance.**
Throwing away 94% of the pixels costs nothing; throwing away half the frames costs
4.1 points. At a single frame the motion classes score **exactly 0.000** while
sedentary still scores 0.111, i.e. the entire visual branch is a motion classifier.
The objects *are* in the pixels — a glass, a spoon and a phone screen are plainly
visible in `cache/train_roi224` — but with 2933 clips over 40 classes the network
latched onto motion, which solves the 39% of data that is whole-body activity, and
never learned what is in the hand.

**What this kills:** higher-resolution caches, hand/upper-body crop streams, and
the person-box detector work. They cannot help a model that discards its existing
spatial detail. This retires C-01/Q-97 and the EXP-027 line for good, and it
retroactively explains EXP-027 and the ROI-CNN failures — those were never crop
quality problems.

**What this opens:** the sedentary curve has **not saturated in time** (16→8 still
costs 1.82 points), so frame count is a live lever; `cache/train_roi224` already
stores every frame. And the appearance gap is now a *diagnosed* deficit rather
than a guess, which is the first principled argument for competition-data SSL:
the missing capability is object appearance, and appearance is exactly what an
unlabeled pretext task over 3,338 clips could supply.

**Confidence:** 95%. Two independent ablations on a trained checkpoint, and the
station-prior control (fold-safe P(class|station) scores 0.0115 on sedentary vs
the branch's 0.2281, agreement 9.1%) rules out "it is only recognising the room".
**Beliefs updated:** NEW B-024 (visual branch is motion-only); B-023 sharpened;
kills the crop/resolution family.
**Next ideas:** 32-frame visual member; sedentary-only specialist; SSL for
appearance; bank EXP-069's structural gain.

---

## EXP-067 — **THE ERROR IS ONE SHAPE.** 75% of all error is fine-grained sedentary hand-object confusion
**Date:** 2026-08-08 · **Tier:** explore · **Purpose:** INFORMATION · **Cost:** ~0 (CPU, existing artifacts)

**Why run it:** after EXP-065's fusion marginal came in at +0.96 points (~2 public
clips) for 14 GPU-hours, Atharv asked whether waiting on more folds was worth the
compute. It was not. The campaign had never measured *where* the missing clips
are — only which member scores higher. Every experiment since EXP-050 tuned a
component without knowing which errors that component was supposed to fix.

**Setup:** OOF overlap of `astgcn_world25` (base) and `visual_mil_v1` (visual),
2700 clips with labels, geometric fusion at the OOF-optimal w=0.20. Errors
partitioned by an 11-way semantic clustering of the 40 classes, then by a coarser
2-way split: 26 **sedentary hand-object** classes (person static, hands
manipulating an object) vs 14 **whole-body motion** classes.

**Findings:**

| split | classes | n | % data | base | visual | fused |
|---|---:|---:|---:|---:|---:|---:|
| whole-body motion | 14 | 1046 | 38.7% | 0.822 | 0.631 | **0.849** |
| sedentary hand-object | 26 | 1654 | 61.3% | 0.489 | 0.236 | **0.502** |

- **75.4% of all errors are sedentary-class confused with sedentary-class.**
  That is **55.1 public clips** of error mass in one place.
- The coarse split is nearly solved: only 107 errors cross the
  sedentary/motion boundary. The model knows someone is sitting using their
  hands; it cannot tell *what they are doing with them*.
- **Perfect sedentary discrimination would score 0.9104** — the 0.91 directive,
  almost exactly. The whole target is inside this one cluster.
- Top confusions are all object-identity pairs, not pose pairs: 6 Drink_water ↔
  7 Eat_food (33), 8/9/10/11 tableware/pour/stir/peel (94 within-cluster),
  12 Sweep ↔ 13 Mop (31), 21 Read_documents ↔ 22 Turn_pages (24), 37
  Take_medicine → 6/7 (20). **Class 26 Play_games scores 0.025** (39/40 wrong,
  → 24 Use_a_mobile_phone / 19 Make_a_phone_call).
- Per-member ranking on sedentary classes only: skeleton `world25` **0.489**,
  visual MIL **0.236**, every IMU variant **0.15–0.17**, depth/IR ROI CNNs
  0.11–0.18. Nothing in the repository is good at this.
- Fusion already extracts most of the base/visual complementarity: oracle
  best-of-two on sedentary is 0.556 vs fused 0.502, and visual-right/base-wrong
  is only 6.8%. **The visual branch adds almost no unique information where it
  matters.**

**Conclusion:** the competition is not a 40-class HAR problem. It is a 26-class
fine-grained hand-object problem wearing a 40-class costume, and the skeleton
family is structurally blind to it — hand-to-mouth pose is identical for drink,
eat and take-medicine. This explains three previously separate puzzles: why
three weeks of skeleton tuning bought nothing (B-021's "model side is closed"),
why EXP-065's regularization gains did not survive fusion (it improved classes
that were already solved), and why rank 1 at 181/201 must use a categorically
different signal rather than a better-tuned version of this one.

**What this retires:** any further member-level tuning evaluated on overall
accuracy. Overall accuracy is 39% dominated by a solved sub-problem, so it
cannot resolve progress on the part that matters. **All future experiments
report sedentary-class accuracy as the primary metric.**

**Confidence:** 95% on the decomposition (direct measurement, 2700 clips);
70% that closing it is reachable before 2026-09-15.
**Beliefs updated:** B-021 (model side closed) — REFRAMED, it was closed only
over skeleton-derived features; NEW B-023 (error mass is one fine-grained
cluster); B-013 downgraded further.
**Next ideas:** EXP-068 spatial/temporal information ablations on a trained
checkpoint (is the visual branch even resolution- or motion-limited?);
sedentary-only specialist; class-group-specific fusion weights.

---

## EXP-063 — **THE TRANSFER ASYMMETRY.** Visual transfers, skeleton does not; OOF cannot tune fusion.
**Date:** 2026-08-02 · **Tier:** explore · **Purpose:** INFORMATION · **New best 125/201**

Public results this session (all user-verified):

| submission | public | clips | note |
|---|---:|---:|---|
| `sub_visual_mil_v1` (visual ALONE) | 0.38805 | 78 | first solo visual score in the campaign |
| `sub_world25_visgeo225_f0123_trans05` | 0.61194 | 123 | 4-fold visual at w=0.225 — **null vs champion** |
| `sub_visgeo035_f0123_trans05` | **0.62189** | **125** | **NEW BEST** |
| `sub_visgeo045_f0123_trans05` | 0.61194 | 123 | |
| `sub_visgeo055_f0123_trans05` | 0.58706 | 118 | |

### The finding

| component | OOF | public | delta |
|---|---:|---:|---:|
| **visual MIL alone** | 0.37913 | **0.38805** | **+0.89** |
| skeleton+IMU stack | 0.61778 | 0.54228 | **−7.55** |
| fused w=0.225 + decode | 0.63556 | 0.61194 | −2.36 |

**The visual branch is the only component in the stack that does not degrade on hidden users.**
It gains. The skeleton stack loses 7.5 points (~6.3 after correcting the 2700-OOF's known
optimism). This is the cleanest cross-subject measurement the campaign has.

### Consequence: OOF-based fusion tuning is structurally invalid here

Because the base is inflated on OOF and the visual member is not, **any weight fitted on OOF
under-weights the component that actually transfers.** Measured directly:

| selection method | w chosen | public result |
|---|---|---|
| fold-safe nested OOF, 2700 clips (EXP-061) | **0.15–0.20** | — |
| full-sample OOF argmax (EXP-061) | 0.225 | 123 |
| **public leaderboard** | **0.35** | **125** |

EXP-061's fold-safe refit was methodologically correct *and predicted the wrong weight*, because
it removed selection bias but not **distribution bias**. Unbiased-on-the-wrong-distribution is
still wrong. This retroactively explains B-004's "compressive and unreliable" CV→LB map: it is
not compression, it is a **component-dependent** shift — one member's OOF is honest, the other's
is not, so any mixture fitted on OOF is mis-specified.

### Fusion-scheme search is EXHAUSTED at this level

Fold-safe comparison on 2700 clips, all schemes scored identically:

```
A  global w                        nested 0.63556  +1.778 pts (+48 clips)
B  per-class-group (object/motion) nested 0.63556  +1.778 pts (+48 clips)
C  per-cohort (early/late users)   nested 0.63481  +1.704 pts (+46 clips)
```

Per-class-group and per-cohort weighting buy **nothing** over one global weight, despite larger
search grids. Scheme B also selected *more* visual weight on gross-motion classes (0.25–0.30) and
*less* on object classes (0.05–0.15) — the **opposite** of the predeclared
`VISUAL_WEIGHT_OBJECT=0.35 / VISUAL_WEIGHT_MOTION=0.15` constants sitting unused in
`train_visual_mil.py`. Those constants were never validated and should not be deployed.

**Do not spend further submissions on fusion parameterization.** The axis is closed: global w
tuned on the public LB is the operating point, currently 0.35.

### What this implies for the target

Weight tuning delivered **+2 clips**. The remaining gap to 0.90 is **+56 clips**, and no
re-weighting of a base that loses 7.5 points on unseen users can produce it. The only component
that transfers is the visual branch, at 0.388 solo against constituent modalities the dataset
paper measures at **90.22–92.57%** under random split. **That gap is the entire remaining
opportunity, and it is a cross-subject generalization problem, not a fusion problem.**

**Queued directly from this:** EXP-062 (running) is the first-ever test of the regularization
axis on the visual branch — `SPATIAL_CROP_MIN` was 0.90 (crops spanning 90–100% of the frame,
i.e. almost no spatial augmentation) on a 2.31M-parameter model fitting ~2.1k clips.

---

## EXP-062 — RESULT: regularization helps on every metric; the fitting regime changed
**Date:** 2026-08-03 · **Tier:** explore · **Purpose:** SCORE · fold 2

| metric | fixed recipe | EXP-062 | delta |
|---|---:|---:|---:|
| micro | 0.33282 | **0.35123** | **+1.84** |
| macro | 0.33046 | **0.36083** | **+3.04** |
| object (0-27, 37-39) | 0.21920 | **0.24008** | **+2.09** |
| gross motion (28-36) | 0.64740 | 0.65896 | +1.16 |
| **final train CE** | **0.5763** | **1.1823** | **+0.61** |

**Every metric improved, and object classes improved most in relative terms (+9.5% relative).**

**The train CE is the load-bearing evidence.** It doubled: the model no longer fits its training
set, and held-out accuracy nevertheless *rose*. That is the signature of removing overfitting, not
of noise, and it confirms EXP-056's diagnosis — the branch was DG-limited and the regularization
axis had never been opened in 62 experiments. `SPATIAL_CROP_MIN=0.90` (crops spanning 90-100% of
the frame) was giving essentially no spatial augmentation to a 2.31M-parameter model on 2.1k clips.

**Honest reading of the magnitude.** +1.84 micro is below the ~5-point bar predeclared for this
branch, because the visual seed variance has still never been measured (9.3h per seed). The
*direction* is supported on 4 independent metrics plus a mechanism; the *size* is not established.
Do not quote +1.84 as a measured effect — quote "positive on every metric, magnitude unresolved".

### What it implies, and the follow-up now running

At CE 1.18 after 60 epochs under heavy regularization the model is plausibly **under-trained** —
the fixed 60-epoch budget was tuned for a recipe that overfit by epoch 60. Regularize-then-train-
longer is the standard pairing, and the cosine schedule scales with `EPOCHS`.

---

## EXP-064 — RESULT: 150 epochs = 60 epochs, exactly. Epochs are NOT a lever.
**Date:** 2026-08-07 · fold 2 · tag `visual_mil_reg2_e150`

| | EXP-062 (60 ep) | EXP-064 (150 ep) |
|---|---:|---:|
| **micro** | **0.35123** | **0.35123** |
| macro | 0.36083 | 0.34934 |
| object | 0.24008 | 0.24843 |
| gross motion | 0.65896 | 0.63584 |
| **final train CE** | **1.1823** | **0.8002** |

**Micro is identical to five decimals — 229/652 in both.** 90 extra epochs drove train CE from
1.182 to 0.800 (a large increase in training-set fit) and bought **exactly zero** held-out
accuracy. This is the third branch of the predeclared read: the regularized recipe converges by
60 epochs, and **additional optimization does not transfer**.

Combined with EXP-062, the picture is now specific: the branch's bottleneck is neither
optimization nor capacity. Regularization moved it (0.333 -> 0.351) and saturated; epochs do
nothing. What remains is the representation and the data regime.

**Cost note:** this run lost ~12h to a stall and ~3 days idle to two process-group kills. The
work survived only because of checkpointing. `setsid` never survived teardown — the watchdog died
alongside the trainer both times, which is why no restart was ever logged. Now run under **tmux**
(session teardown) **plus** the watchdog (stalls); the two failures are different and need
different protection. Launcher moved out of `/tmp` (cleared between sessions, which broke one
restart) into `code/run_exp064_reg2.sh`.

---

## EXP-066 — Visual models are strongly DECORRELATED; averaging helps them and HURTS the stack
**Date:** 2026-08-07 · **Tier:** explore · **Purpose:** INFORMATION · **Cost: 0 GPU**

Pairwise agreement between the three fold-2 visual models is only **0.371-0.400**, with a 3-way
union oracle of **0.51534** against individual accuracies of 0.333-0.351.

| combination | standalone | marginal in the full stack (552-clip overlap) |
|---|---:|---:|
| fixed only | 0.33282 | +1.27 |
| **reg1 only** | 0.35123 | **+2.90** |
| reg1 + reg2 | **0.40337** | +1.99 |
| 3-way | **0.40644** | +2.72 |

**Averaging the visual models improves them standalone by +5.2 points and makes them WORSE in
fusion.** The stack does not want a more accurate visual member; it wants a *complementary* one.
Averaging regresses the member toward consensus and destroys the idiosyncratic rescues that gave
it value — consistent with EXP-050b's 44 visual-only rescues and with B-017's repeated finding
that complementary error mechanisms beat better-but-similar members.

**Caveat, stated plainly:** 552 clips with the fusion weight selected on those same clips. The
ranking is suggestive, not established; the +2.90 vs +1.99 difference is ~5 clips.

**Actionable signal:** the regularized member contributes **+2.90** to the stack versus the fixed
recipe's **+1.27** — more than double, on the same clips with the same procedure.

---

## EXP-065 — REGULARIZATION REPLICATES ON FOLD 0, larger than on fold 2
**Date:** 2026-08-08 · fold 0 complete, fold 1 at epoch 36/60, fold 3 pending

| fold | metric | fixed recipe | regularized | delta |
|---:|---|---:|---:|---:|
| **0** | micro | 0.37838 | **0.41032** | **+3.19** |
| 0 | macro | 0.29221 | 0.32216 | +3.00 |
| 0 | object | 0.21467 | **0.25403** | **+3.94** |
| 0 | gross motion | 0.73725 | 0.75294 | +1.57 |
| 2 | micro | 0.33282 | 0.35123 | +1.84 |
| 2 | macro | 0.33046 | 0.36083 | +3.04 |
| 2 | object | 0.21920 | 0.24008 | +2.09 |
| 2 | gross motion | 0.64740 | 0.65896 | +1.16 |

**Eight metric-fold combinations, all positive**, on two independent folds with different user
subsets. EXP-062's caveat ("direction supported on 4 metrics, magnitude unresolved on 1 fold") is
now discharged: this is a replicated effect of roughly **+2.5 micro / +3.0 object**.

**Object classes gain most in both folds** (+3.94, +2.09), which is where B-021's error mass sits
and where every previous attempt failed. The mechanism remains the one EXP-062 identified:
`SPATIAL_CROP_MIN=0.90` gave essentially no spatial augmentation to a 2.31M-parameter model on
~2.1k clips, so the branch was regularization-starved rather than capacity- or optimization-limited
(EXP-064 having ruled out epochs).

**Still unmeasured:** the visual branch's seed variance. Two folds agreeing in sign on 8/8 metrics
is much stronger than one fold, but a same-recipe seed replicate has never been run, so the
effect size carries no error bar. Worth one 9.3h run once the fold sweep finishes.

**Infrastructure:** fold 0 completed cleanly under tmux+watchdog (no restarts). The tmux server
itself disappeared overnight — likely a host reboot rather than a session teardown, since a tmux
server survives shell exit. Fold 1's work was preserved by checkpointing and resumed at epoch 36.

---

## EXP-065 — predeclaration (kept for provenance)
**Date:** 2026-08-07 · `code/run_reg_folds.sh` · tag `visual_mil_reg1`

60 epochs (EXP-064 showed 150 buys nothing). Direct test of whether EXP-066's +2.90 fold-2 signal
survives into a deployed 4-fold member and a public submission. Fold 2 already exists.

**Note the tension this experiment must resolve:** EXP-066 says averaging visual models hurts
fusion, and a 4-fold member *is* an average. The distinction being bet on is that folds average
over *different training users* (coverage) rather than over *the same data* (consensus). If the
4-fold regularized member underperforms the single reg1 fold-2 member in fusion, that bet is wrong
and the deployed member should be a single regularized model.

---

## EXP-064 — predeclaration (kept for provenance)
**Date:** 2026-08-03 · fold 2 · tag `visual_mil_reg2_e150`

EXP-062's overrides, unchanged, plus `CUHKX_VMIL_EPOCHS=150`. Single isolated change from
EXP-062 so the epoch effect is attributable.

**Baselines:** fixed recipe 0.33282 · EXP-062 (same reg, 60 ep) 0.35123.
**Reads as a win** if it clears ~0.38 (i.e. beats EXP-062 by more than EXP-062 beat the baseline);
**reads as a ceiling** if it lands near 0.35, which would say the regularized recipe converges by
60 epochs and the remaining gap is not an optimization problem.

Either outcome is informative, which is why it is worth 23h: it separates "under-trained" from
"the representation cannot do better", and those imply completely different next moves.

---

## EXP-062 — predeclaration (kept for provenance)
**Date:** 2026-08-02 · **Tier:** explore · **Purpose:** SCORE · fold 2, ~9.3h

First variant on an axis that was never opened. `train_visual_mil.py` gained environment
overrides (`CUHKX_VMIL_*`), defaults unchanged, and every override is recorded in the manifest as
`recipe_overrides` so a variant can never be misread as the fixed recipe.

| knob | fixed recipe | EXP-062 |
|---|---:|---:|
| `SPATIAL_CROP_MIN` | 0.90 | **0.60** |
| `DROPOUT` | 0.20 | **0.35** |
| `MODALITY_DROPOUT` | (0.10, 0.10, 0.20) | **(0.25, 0.25, 0.35)** |
| `LABEL_SMOOTHING` | 0.05 | **0.10** |

**Baseline to beat:** fold 2 = 0.33282. **Caveat: the visual branch's seed variance has never
been measured** (one seed per fold, 9.3h each), so only a large move is interpretable. Treat
anything under ~5 points as inconclusive rather than as a result — the DA-006-A lesson.

---

## EXP-061 — All four visual folds complete; fusion weight refit fold-safe on 2700 clips
**Date:** 2026-08-02 · **Tier:** exploit · **Purpose:** SCORE

Folds 1 and 3 finished (fold 1 resumed from epoch 28; fold 3 fresh). Full outer-once table:

| fold | micro | macro | object | gross motion |
|---:|---:|---:|---:|---:|
| 0 | 0.37838 | 0.29221 | 0.21467 | 0.73725 |
| 1 | 0.37838 | 0.32853 | **0.26838** | 0.65939 |
| 2 | 0.33282 | 0.33046 | 0.21920 | 0.64740 |
| 3 | **0.42726** | 0.37369 | 0.26244 | **0.77251** |
| **4-fold OOF** | **0.37913** (2933 rows) | | | |

### CORRECTION — EXP-060's "structural object deficit" claim is WITHDRAWN

EXP-060 claimed object accuracy was stable "to within 0.45 points" across folds and called the
deficit structural. That used folds 0 and 2 only. Folds 1 and 3 give 0.26838 and 0.26244, so the
real spread is **5.4 points (0.215-0.268)**, not 0.45. **I drew a structural conclusion from two
same-direction samples** — precisely the failure mode DA-006-A identified and EXP-054 confirmed.
Withdrawn.

**What survives:** object accuracy (0.215-0.268, mean ~0.241) remains far below gross motion
(0.647-0.773, mean ~0.704) — a ~46-point gap that is large and consistent in *sign* on every fold.
Object also varies less than gross motion (5.4 vs 12.6 points spread). The deficit is real; the
claim that it is invariant to training population is not supported.

### Fusion weight refit — the LEADERBOARD methodology warning, addressed

The champion's w=0.225 came from sweeping ~7 weights on **552 clips**; LEADERBOARD.md flags that
as optimistic, and the tri-member config chosen the same way said +3.99 locally and returned
**−1 public clip**. The base ensemble OOF (`oof_astgcn_world25.npz`, 2700 rows, micro 0.61778) is
a strict subset of the new 2933-row visual OOF, so the weight was refit on **2700 clips** with w
selected per fold using only the other three folds' labels.

```
fold 0: w=0.150 -> held-out 0.63289  (base 0.61233)
fold 1: w=0.225 -> held-out 0.62039  (base 0.60197)
fold 2: w=0.225 -> held-out 0.60688  (base 0.59420)
fold 3: w=0.225 -> held-out 0.68147  (base 0.66309)

nested fused 0.63556 · base 0.61778 · UNBIASED GAIN +1.778 pts (+48 clips of 2700)
full-sample argmax also w=0.225 (0.63741, +1.963) -> selection bias only 0.185 pts
```

**w=0.225 is confirmed**, on 5x more data, by an unbiased procedure, selected by 3 of 4 folds.
The curve is a smooth broad plateau (w=0.15-0.25 within 0.6 points), which is what a structural
optimum looks like — unlike the 36-combination tri-member sweep that produced a fitted spike.

### Staged

`submissions/sub_world25_visgeo225_f0123_trans05.csv`
sha256 `265ef9e45318a5819a382b5bbd0f12d1114d8d66800fb8388771872864436059`
4-fold visual member (`testprobs_visual_mil_f0123.npz`), geometric w=0.225, transition decode
lambda=0.5. **Differs from the champion on 59/405 rows (~29 public).**

**Honest limit on the expected gain.** The change from the champion is *not* "a better visual
member" — it is **one model replaced by an average of four** trained on different user subsets.
The 4-fold OOF of 0.37913 is not comparable to fold 2's 0.33282: they score different clips, and
fold 2 is simply the hardest fold. I have **no unbiased OOF estimate of the 1-model -> 4-model
change**, because the fold-2 model cannot be scored on folds 0/1/3 without training contamination.
The case rests on standard ensembling plus B-017 (user/architecture diversity moved this ladder
repeatedly), not on a measured delta. 59 changed rows is a large edit — the downside is real.

---

## EXP-060 — Visual MIL fold 0 complete: object deficit claim (SUPERSEDED by EXP-061)
**Date:** 2026-08-01 · **Tier:** exploit · **Purpose:** SCORE + INFORMATION

Fold 0 finished before a session teardown killed the driver (fold 1 partial, fold 3 pending).

| metric | fold 0 (13 users) | fold 2 (14 users) |
|---|---:|---:|
| micro | **0.37838** | 0.33282 |
| macro | 0.29221 | 0.33046 |
| **object classes** (0-27, 37-39) | **0.21467** (120/559) | **0.21920** |
| gross motion (28-36) | **0.73725** (188/255) | 0.64740 |

**The finding is the object column.** Across two independent folds with different user subsets and
a 4.6-point spread in micro accuracy, object-class accuracy is **0.2147 vs 0.2192 — a 0.45-point
difference**. Meanwhile gross motion swings **0.647 -> 0.737, nearly 9 points**.

The object deficit is **stable and structural**; the fold-to-fold variance lives almost entirely in
the gross-motion classes. This is not a hard-fold/easy-fold effect and it is not seed noise: two
different training populations converge on the same object accuracy to within half a point.

**Why that matters.** It rules out "fold 2 was unlucky" as an explanation for the visual branch's
weakness on object classes, and it sharpens EXP-056: the branch is DG-limited overall, but the
object classes are limited by something that does *not* vary with which users you train on.
Candidates worth separating: the objects are too small at 192x256 for the 8x-output-stride trunk;
16 sampled frames miss the manipulation moment; or top-k MIL pooling averages the object away.
Each predicts a different fix, and none of them is "more users".

**Consequence for the running job:** folds 1 and 3 will improve *ensemble coverage* and the fusion
weight estimate, but on this evidence they should **not** be expected to move object classes.
Do not price them as a fix for B-021's error mass.

---

## P-10 — Champion's fusion step was UNREPRODUCIBLE; now committed and verified
**Date:** 2026-08-01 · `code/fuse_visual_geo.py` · **Tier:** deployment · **Cost: 0 GPU**

**Defect found.** The verified champion `sub_world25_visgeo225_trans05.csv` (0.61194 = 123/201)
depends on `research/artifacts/testprobs_world25_visgeo225.npz`, and **no script in `code/`
produced it.** `fuse_submit.py` is a hardcoded skeleton-stack script and does not do geometric
fusion; the whole `testprobs_world25_visgeo*` family came from an ad-hoc step that was never
committed. The best submission in the campaign could not be regenerated from the repository.

This is a **Selection-Stage gate**, not a tidiness issue: OBJECTIVE.md records "un-reproducible
pipeline -> eliminated at Selection Stage even with top score." Top-15 advancement requires the
organizers to reproduce the solution.

**Fix.** `code/fuse_visual_geo.py` implements the step and self-verifies:

```
python3 code/fuse_visual_geo.py --weight 0.225 \
    --verify research/artifacts/testprobs_world25_visgeo225.npz
  max abs probability delta : 3.725e-09
  identical argmaxes        : 405/405
  RESULT: REPRODUCED
```

Fusion is computed in log space with a row-max subtraction rather than as a literal product of
powers, so no row underflows; the 3.7e-09 residual is float32 storage rounding.

**Audit implication.** Every remaining artifact should be checked for the same defect — an input
whose producing script is absent. The champion was found by accident while building the runbook,
which means nothing systematically checks this.

---

## EXP-058 / OUT-005 — Public split shows NO evidence of grouping (low-power null)
**Date:** 2026-08-01 · **Tier:** explore · **Purpose:** INFORMATION · **Cost: 0 GPU, 0 submissions**

Method: 24 verified submissions, 276 pairs. For a pair differing on row set S with score delta
d clips, at least |d| of those rows are public. Under a uniform 201/405 = 49.63% sample, the
observed |d|/|S| must sit at or **below** 0.4963 — a pair above that bound would be impossible
under uniform sampling and would prove grouping.

| quantity | value |
|---|---:|
| pairs compared | 276 |
| mean \|d\|/\|S\| | 0.1183 |
| max \|d\|/\|S\| | 0.3333 |
| **pairs above the 0.4963 impossibility bound** | **0** |

**Conclusion: no evidence against a uniform random public split.** This matters because it
removes split structure as the explanation for A-02 (transitions +6.0 OOF -> +1.5 public; repeat
consensus +3.1 OOF -> −1 clip). The CV↔LB compression is a genuine generalization phenomenon,
not a sampling artifact.

**Power, stated honestly.** This test can only *falsify* uniformity, never confirm it: offsetting
errors hide public rows, so a low rate is expected under both hypotheses. A null here is weak
evidence. My "Test 3" (changed-row index histogram) was **uninformative as constructed** — it
shows where predictions differ, not where the score moved, so it cannot discriminate block
structure. Recorded rather than quietly dropped.

**Sharper probe that does not exist yet:** no pair differs by <=3 rows. A deliberately minimal
edit (flip 1-2 clips against the champion) would identify individual clips' public membership
exactly, at one submission each. Not worth the budget now, but it is the tool if split structure
ever becomes load-bearing.

---

## EXP-059 / OUT-007 — Distinctness retest at base 123: CANDIDATE STAGED (needs one submission)
**Date:** 2026-08-01 · **Tier:** exploit · **Purpose:** SCORE

B-022 predicts the distinctness coupling flips sign as base accuracy rises. The measured ladder:

| base | distinctness marginal | evidence |
|---:|---:|---|
| 112 | **−1 clip** | `sub_world25_int8_trans05_dist10` = 111 |
| 121 | **0 clips** | `sub_world25_visgeo15_trans05_dist10` = 121 |
| **123** | **?** | this candidate |

Monotone so far, exactly as B-022 predicted, and not yet positive.

**Staged:** `submissions/sub_world25_visgeo225_trans05_dist10.csv`
SHA-256 `dd57f76acb7499a8d968ffc7439aff756b98b0e6877efe453682c4aefd228d76`
Decoder: `--lambda 0.5 --distinctness penalty --distinctness-penalty 1.0` over
`testprobs_world25_visgeo225.npz` (the champion's own fused probabilities), config otherwise
byte-identical to the champion. Repeat-collisions 39 -> 6.

**Differs from the champion on 12/405 rows (~6 public).** Range of outcomes is roughly −6 to +6
clips; B-022's prediction is a small positive. **This is the cheapest live test of a standing
falsifiable prediction in the campaign** — one submission decides it.

---

## EXP-057 / OUT-003 — RESULT: person selection is a NULL. Plus two process failures worth more than the result.
**Date:** 2026-07-31 · **Tier:** explore · **Purpose:** INFORMATION

**Result: +0.57 ± 1.71 points (+0.34 sigma), 2 of 4 seeds negative. Null.**

| seed | `first` (DA-006-A) | `motion` | delta |
|---:|---:|---:|---:|
| 0 | 0.5690 | 0.5613 | −0.77 |
| 1 | 0.5675 | 0.5690 | +0.15 |
| 2 | 0.5276 | 0.5261 | −0.15 |
| 3 | 0.5138 | 0.5445 | +3.07 |
| **mean** | **0.5445** | **0.5502** | **+0.57** |

Seeds 0-2 are DA-006-B's runs; seed 3 is EXP-057b (`run_OUT-003b_astgcn_motion_aug_s3_2.json`).

**The predeclared power analysis was correct and is the reusable output.** It predicted a
realistic effect of ~+0.5 points and a perfect-repair ceiling of +3.99 against a 3.96-point
detection threshold — i.e. that the question was unresolvable at this power. Observed: +0.57.
The arithmetic held. **OUT-003 is retired: the input is not corrupted in a way that matters**,
and Q-86's 56-experiment tenure in the queue ends here.

### Process failure 1 — the experiment had already been run, and I did not check

`run_DA-006-B_astgcn_motion_s{0,1,2}_2.json` were written at **14:51-14:57 today**, before this
session. DA-006 explicitly queued DA-006-B; it ran, and **its result was never written to LOG.md**.
I read LOG.md, BELIEFS.md, SEARCH_MAP.md and QUEUE.md at session start and none of them recorded
it, so I re-ran it. **The session-start ritual reads the ledgers but never lists `research/artifacts/`.**
A queued item can be complete on disk and invisible in every ledger. Ritual gap, now known.

### Process failure 2 — my first run was misconfigured and its result was wrong

I launched 4 seeds without `--aug-spec trunc` and got 0.4939 / 0.4877 / 0.4801 / 0.4831
(mean 0.4862), and was about to report **−5.8 points** as a decisive negative.

`train.py:26` is `aug = args.aug_spec if args.aug_spec else args.aug`. **`--aug-spec` enables
augmentation on its own; `--aug` is not required.** DA-006-A and DA-006-B both passed
`--aug-spec trunc`, so both had trunc augmentation ON — while their manifests record
**`"aug": false`**, because line 156 logs `args.aug`, the store_true flag, not the effective
setting. Reading the manifest field at face value produces exactly the wrong configuration.

So the invalid run compared **un-augmented motion against augmented first**, and the ~6-point
gap was the augmentation, not the person policy. Those four manifests were deleted; the invalid
numbers are recorded here only so the trap is documented.

**Harness defect to fix:** `train.py` should log the *effective* augmentation
(`args.aug_spec if args.aug_spec else args.aug`) rather than `args.aug`. Until then, every
`aug: false` in an existing manifest is ambiguous and must be read together with `aug_spec`.
This affects the interpretation of any past run reconstructed from manifests.

### Failure analysis (mandatory)

**3 reasons the hypothesis failed.** (1) Multi-person clips are only 63/652 = 9.7% of the fold,
so the mechanism's footprint is small by construction. (2) `first` already selects the subject in
most multi-person clips — person 0 is usually the actor, so the addressable pool was 26 clips.
(3) `motion` picks maximum motion energy, which in bathroom/mirror clips can select the
*reflection or bystander* as readily as the subject; the policy is not obviously better-targeted.

**3 alternative explanations.** (1) Underpowered — true, and predeclared as such; the design could
not have resolved a sub-1-point effect. (2) fold 2 may be unrepresentative; multi-person rates
differ per user. (3) The concentration test (multi vs single subset) could not be run because
`train.py` saves no per-clip OOF for single-fold runs, so only the aggregate was available.

**3 follow-ups.** (1) None on person selection — retire it. (2) If a future run needs the subset
readout, have `train.py` emit OOF npz for single-fold screens. (3) Fix the `aug` logging defect.

**Beliefs updated:** B-021 (the "input describes the wrong person" reframing DA-006 raised is
**refuted**; B-021's downgrade stands on its own evidence, not on this). DA-006 item 2 closed.

---

## EXP-057 / OUT-003 — predeclaration (kept for provenance)
**Date:** 2026-07-31 · `code/train.py` · **Tier:** explore · **Purpose:** INFORMATION
Written before any `motion` model was trained. Gate fixed from DA-006-A data that already existed.

**Hypothesis.** `person="motion"` beats `person="first"`, because 84.6% of Comb_hair clips contain
multiple detected persons (DA-001) and all 56 prior experiments used `first`. If the skeleton
describes the wrong body, that is an input corruption no downstream model can undo.

**Prior art.** `har_data.py:112-132` implements `motion` (highest total motion energy, per-frame
fallback to person 0). Queued as Q-86 by DA-001, re-raised by DA-006 item 2, **never run once**.

### Power analysis FIRST — this is what changed the design

| quantity | value |
|---|---:|
| fold-2 val clips that are multi-person | **63 / 652 = 9.7%** |
| train clips multi-person | 299 / 2933 = 10.2% |
| multi-person clips `first` gets wrong (seed 0) | 26 |
| **perfect-repair ceiling on overall accuracy** | **+3.99 points** |
| 2 SE detection threshold, 4 seeds (sigma 2.80, unpaired) | **3.96 points** |

**The maximum physically possible effect equals the detection threshold.** Overall accuracy
therefore *cannot* resolve this hypothesis at n=4 on this fold, no matter the outcome. Screening
it that way would repeat the DA-006-A error exactly.

**The realistic effect is far smaller.** Multi-person accuracy across the four `first` seeds is
0.5873 / 0.4762 / 0.4603 / 0.4762 (mean **0.5000**) versus single-person mean **0.5492**.
Fully equalizing the two is worth about **+0.5 points overall**. OUT-003's queued "+2-6 points"
was mispriced — it was taken from DA-001's per-class 84.6% figure without checking that
multi-person clips are only 9.7% of the fold. Corrected in OUTLIERS.md.

### Predeclared readout

- **PRIMARY — multi-person subset (n=63), 4-seed mean.** This is where the mechanism lives.
  1 clip = 1.59 points there; across-seed SE of that subset mean is ~2.9 points.
  **Gate: >= +6.0 points on the subset (~4 clips, ~2 SE).**
- **NEGATIVE CONTROL — single-person subset (n=589).** `_pick_person` returns *identical* tensors
  for both policies when `sp.shape[1] == 1`, so these inputs are unchanged. Any movement here is
  pure training-set-change noise and calibrates the primary readout.
- **Overall accuracy: reported, NOT a gate**, for the reason above.

**What the mechanism forbids.** A gain spread evenly across both subsets refutes the stated
mechanism even if overall accuracy rises — that would be a training-perturbation effect, not
person-selection repair.

**Expected effect.** +3 to +8 points on the multi-person subset; ~0 on the control; ~+0.3 to +0.8
overall. A null result retires a 56-experiment-old open item and bounds the input-corruption risk.

---

## EXP-056 / OUT-012 — Visual branch is DG-bottlenecked, not recipe-bottlenecked. **Answered at zero compute.**
**Date:** 2026-07-31 · **Tier:** explore · **Purpose:** INFORMATION · **Cost: 0 GPU-minutes**

OUT-012 was queued as a 1.5h random-split run. Two facts killed that plan and then made it
unnecessary:

1. **Cost estimate was wrong by 6x.** `run_visual_mil_visual_mil_v1_f2.json` records
   `elapsed_minutes = 559.54` — **9.3 hours** per visual MIL run on the 4060, not 1.5h.
2. **The manifest already contains the answer.** `final_train_losses.cross_entropy = 0.5763`.
   Chance on 40 classes is ln(40) = 3.689; CE 0.576 implies ~0.56 average probability on the true
   class. **The branch fits its training data well.** It then scores **0.3328** on held-out
   subjects. That is a generalization gap, not underfitting.

### The diagnosis has flipped since EXP-006

EXP-006 (2026-07-27) classified the visual side as **recipe-bottlenecked**: depth could not fit
even in-domain (random-split 32.4%, and that number was inflated by same-session leakage). Its
verdict was "DG work on depth is premature."

That verdict applied to the **old** recipe — 120x160 full-frame, 40 epochs, small CNN. EXP-050's
MIL branch is a different animal: 192x256, 60 epochs, 2.31M params, masked top-k MIL, EMA,
Balanced Softmax, separate IR/Depth/Thermal adapters. **It fixed the recipe deficit.** The
bottleneck moved from "cannot fit" to "cannot transfer" — the same category skeleton has been in
since EXP-006 measured its 20-point DG gap.

**So EXP-006's "DG work on visual is premature" is now expired, and it has been silently gating
the visual branch's development for 50 experiments.**

### Honest limits of this inference

`final_train_losses` is last-epoch loss on **augmented** training data, not a clean in-domain
validation number. It establishes that the model fits what it is trained on; it does not fully
separate "learns generalizable in-domain features" from "memorizes clips". A random-split
validation run would separate those, and EXP-006 warns that random splits here leak
same-recording clips and read optimistic. For choosing the *remedy family*, both readings point
the same way — regularization/invariance/more data, not more capacity or resolution — so the
9.3h confirmation is not worth its cost right now. Recorded as a known soft spot.

### What this does and does not imply

It does **not** imply easy gains. Skeleton has been DG-bottlenecked throughout, and DANN, mixup,
SupCon, Identity, pad+mask and aggressive augmentation all failed on it (B-001). But **every one
of those was tried on skeleton and none on the visual branch**, whose gap is far larger and whose
data regime is different: 2.31M params trained from scratch on 2,281 clips.

**Beliefs updated:** B-006 (the visual bottleneck is now DG, not recipe — its "recipe-gated"
qualifier is retired), B-001 (the DG gap is modality-dependent and the visual branch's is the
largest measured in the campaign).

**Queued:** competition-data-only SSL for the visual branch (VICReg / temporal-order / masked
reconstruction over 2,933 train + 405 test clips unlabeled = 3,338 clips; transduction is legal
per the organizer ruling), then fine-tune. B-006 has carried this as "remaining uncertainty"
since RESET-002 and it has never been run. EXP-021 ran cross-modal masked pretraining for the
*transformer fusion* line — +3.0 over from-scratch, but the line died for other reasons — so the
approach has never been tested on the branch that is now known to be DG-limited.

---

## EXP-055 / OUT-001 — Rank-1 mechanism probe: **THE 56.4% "CEILING" IS NOT A CEILING**
**Date:** 2026-07-31 · **Tier:** crazy (first crazy-tier spend in the campaign) · **Purpose:** INFORMATION
**Cost:** ~1 hour, zero compute, zero submissions.

Probes run against the source paper (arXiv 2512.07136) and the official project page
(`openaiotlab.github.io/CUHK-X/`). Every number below was fetched, not recalled.

### Result 1 — hypothesis (b) FALSIFIED: test labels are NOT publicly recoverable

Project page, verbatim: **"The full CUHK-X dataset will be released after the competition
concludes."** Only **CUHK-S — a sample subset with 18 users, RGB-free** — is currently public.
18 users is exactly our training set (1-9, 16-24). The organizers withheld users 10/11/25/26.
Rank-1's 181/201 is **not** explained by matching test clips to a public labeled release. Kill it.

### Result 2 — the load-bearing reference number was misread. This is the finding.

The paper's LOSO figure of **56.38%** is, verbatim from the full text, **RGB-only**, best-case
**with contrastive learning**, and measured **excluding cross-domain and long-tail classes**.

We have been comparing it to our 56.90% on **40 classes, no RGB, all classes included**, and
concluding we sit at the published ceiling. **The two numbers measure different problems.**
There is **no published all-modality cross-subject number for this dataset at all** — so no
ceiling has ever been established for the task we are actually solving.

This directly undercuts DA-006 §4's steelmanned argument ("the base sits at the dataset paper's
published cross-subject LOSO of ~56.4%, which we match at 56.90%... each further experiment
samples from a distribution whose mean gain is now approximately zero") and the B-013/B-021
conclusion that the model side is closed. That deadlock was inferred from a non-comparable number.

### Result 3 — we have been optimizing the two weakest modalities

Paper Table 3, random-split accuracy per modality:

| modality | acc | used by our stack? |
|---|---:|---|
| **Thermal** | **92.57%** | **never used** (2,788/2,933 train, 395/405 test coverage) |
| RGB | 90.89% | not provided in this track |
| **Depth** | **90.46%** | ROI CNNs ✗; MIL branch partially (EXP-050b/053) |
| **IR** | **90.22%** | motion maps only |
| Skeleton | 79.08% | **the entire backbone of our stack** |
| mmWave | 46.63% | rejected (B-008) |
| IMU | 45.52% | **a deployed member** (world25) |

The 48-member champion is built on the **4th-best and worst** modalities. The three strongest are
unused, partially used, or reduced to motion maps that discard appearance.

### Result 4 — 90.05% is no longer anomalous

Rank-1's 181/201 = 90.05% sits exactly at the random-split level of Depth (90.46), IR (90.22) and
Thermal (92.57). A strong from-scratch pipeline on the high-accuracy visual modalities reaching
~90% requires **no illegal mechanism** — the "impossible cross-subject score" framing came from
comparing against an RGB-only, class-subsetted LOSO number. **Anomaly A-01 is substantially
dissolved.** Hypothesis (a) (illegal pretrained) drops from ~60% to ~25%; hypothesis (c) (a legal
route we simply have not taken) rises to ~65%.

### Corroboration already in our own evidence

EXP-053's visual MIL member — weak solo (0.3328) but +1.00 point paired — delivered **+11 public
clips (112→123)**, the largest single jump of the campaign, from the *first* serious visual member.
B-006 already held 80% that the missing information is visual. This probe explains why.

**NOT verified:** the Kaggle rules text on pretrained/external data (pages are JS-rendered; needs
Atharv's screenshot), and rank-1's actual method. The organizer ruling remains unpreserved.

**Beliefs updated:** B-001 (the "30-point cliff" framing is invalid), B-013 (the >0.83 ceiling
argument loses its anchor), B-021 (implication "stop proposing predictors" is withdrawn — it was
conditioned on a ceiling that was never established), B-006 (upgraded), B-002 (skeleton is best
per unit compute, but it is the 4th-best modality in absolute terms).

### CORRECTION, same session — Thermal is NOT unused

The first draft of this entry claimed Thermal had never been used and queued a Thermal branch as
the top hypothesis. **That was wrong**, caught by checking the code instead of the belief file.
`visual_mil_cache.py:81` sets `MODALITIES = ("ir", "depth", "thermal")` at 192x256 with frame and
modality masks; `train_visual_mil.py` carries all three with modality dropout (0.10, 0.10, 0.20).
The branch has been deployed since EXP-050 and sits in the champion at w=0.225. B-006's
"Thermal unused" note came from the RESET-002 audit, which EXP-050 superseded — the belief file
was stale, and I quoted it without checking the source. **Lesson: a belief entry is a claim about
the past; the code is the claim about the present.**

**What survives the correction, all verified from the paper text:** the LOSO-56.38% misreading
(Result 2), the per-modality random-split table (Result 3), and the dissolution of A-01 (Result 4).
Only the "unused modality" framing was wrong.

**The sharpened question it leaves is better than the one it replaced.** The visual branch scores
**0.3328 solo** while its three constituent modalities score **90.22-92.57% each** under the
paper's random split. Cross-subject degradation explains part of that, but not a ~57-point gap.
The bottleneck is the **recipe**, not modality availability.

**Next ideas queued:**
- **OUT-012 (diagnostic, do first, 1.5h):** score the existing visual branch under random-split
  vs subject-split — EXP-006's protocol, never applied to this branch. It partitions the 57-point
  gap into "cross-subject collapse" vs "recipe underfits even in-domain", and those two findings
  point at completely different fixes.
- **OUT-011 (4h):** attack whichever half OUT-012 indicts.
- Do **not** queue another modality; all six are now accounted for.

---

## EXP-054 — Cluster specialists re-run with paired 4-seed power: EXP-051's REJECTION STANDS
**Date:** 2026-07-31 · `code/cluster_specialist.py` · **Tier:** exploit · **Purpose:** INFORMATION
**Reopened because** DA-006-A withdrew EXP-051's conclusion as underpowered. That withdrawal was itself wrong.

**Hypothesis under test.** Cluster-conditioned specialist routing improves fold-2 accuracy; the
original −0.0107 was noise against a lucky single-seed control.

**Setup.** The routing delta is a PAIRED quantity — the same base model scored with and without
routing — so base-model seed variance cancels (EXP-053's lesson). Clusters held fixed (derived
leak-free from the 14 outer-train users, `clusters_f2.json`). For each seed 0–3: specialists
trained at that seed, routed against DA-006-A's matching base baseline
(`oof_astgcn_all18_f2{,_s1,_s2,_s3}_best.npz`). Scoring rule byte-identical to the predeclared
EXP-051 rule; only `--seed` and paired reporting were added.

**Reproduction gate.** Seed 0 returned −0.0123 / 9 rescues / 17 harms / 45 changed against
EXP-051's −0.0107 / 9 / 16 / 45 — one clip of 652 apart. The specialist pipeline is **not
bit-deterministic** (cuDNN nondeterminism); same-seed reruns vary by ~1 clip. Gate passed.

| seed | base | new | delta | rescues | harms | net | changed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5690 | 0.5567 | −0.0123 | 9 | 17 | −8 | 45 |
| 1 | 0.5675 | 0.5552 | −0.0123 | 9 | 17 | −8 | 46 |
| 2 | 0.5276 | 0.5291 | **+0.0015** | 14 | 13 | **+1** | 56 |
| 3 | 0.5138 | 0.4985 | −0.0153 | 8 | 18 | −10 | 64 |

**Paired mean −0.0096 (−6.2 clips of 652) · std 0.0076 · −1.27σ · 3 of 4 seeds negative.**

**Conclusion: reject, and the original rejection stands.** Cluster routing does not help; it
leans harmful. Note the effect is *not* clean 4/4 replication either — at −1.27σ this is
"consistently unhelpful", not "decisively harmful". Either reading forecloses adoption.

**The methodological finding is the more valuable output.** DA-006-A scored EXP-051's −0.0107
against the **unpaired** 2.80-point accuracy sigma and got −0.38σ, concluding the experiment was
underpowered. On the correct **paired** scale the same number is −1.27σ with a −6.2-clip mean.
This is the identical denominator error EXP-053 diagnosed, resolving in the opposite direction:
the wrong denominator rescued the visual member (worth +11 public clips) and wrongly reprieved
this one. **A wrong noise model is not biased toward rejection or adoption — it is just wrong,
and it corrupts decisions in both directions.**

**Confidence in conclusion:** 90%. Would change if a *targeted* variant (MAX_CLUSTER=2, routing
tens of clips rather than 614) showed a positive paired mean over ≥3 seeds.

**Unexpected observation → anomaly register.** `changed` grows monotonically as the base weakens
(45, 46, 56, 64) — as expected — but rescues/harms do **not** order with base accuracy
(9/17, 9/17, 14/13, 8/18). Seed 2 is a genuine outlier, not a trend endpoint. After seed 2 landed
I hypothesised base-dependence ("strong bases hurt, weak bases help"); **seed 3 refuted it** — the
weakest base gave the worst result. Recorded because the pattern was tempting and wrong, and
because B-022 established a real base-dependence for a *different* mechanism (distinctness), which
makes the false analogy easy to reach for.

**Beliefs updated:** B-010 (targeted-handling half stays falsified, now with power), B-021
(cluster-conditioned routing moves from "explicitly unknown again" to settled-negative), B-013
(reinforced), B-004 (paired-vs-unpaired denominator is now a named campaign hazard).

### Failure analysis (mandatory)

**3 reasons it failed.** (1) The base 40-way posterior is already well-ordered inside a cluster
(top-2 0.6641, top-3 0.7331), so an equal-authority specialist overwriting it destroys more than
it repairs — the one mechanism that survives all four seeds. (2) Specialists train on 377–670
clips versus the base's 2281; the other 33 classes were acting as regularization. (3) Routing
touched 614/652 clips (94.2%) — this is a near-global re-decision, not a targeted intervention.

**3 alternative explanations.** (1) The clusters are too coarse — `MAX_CLUSTER=8` with connected
components merged kinematically unrelated classes (Jumping_jacks with Brush_teeth); a
`MAX_CLUSTER=2` rule was never tested and is a different hypothesis, not a retest of this one.
(2) The fixed geometric mean grants the specialist equal authority; a confidence-gated weight
might act only where it is strong. (3) Last-epoch no-selection training may leave specialists
under-fit against best-epoch baselines — a real asymmetry in this design.

**3 follow-ups.** (1) **Do not** run another skeleton-feature specialist variant; two powered
negatives now agree. (2) If the pair-level route is tried at all, give it the EXP-050b **visual
embedding as input** — the rescues that motivated it were visual (35 object clips), and EXP-053
proved that member carries additive information. (3) Spend the freed budget on the never-probed
cells instead: OUT-003 (`person="motion"`, unrun in 53 experiments), OUT-001 (rank-1 mechanism).

---

## EXP-053 — Visual fusion re-tested as a PAIRED effect: REAL AND REPLICATED
**Date:** 2026-07-31 · **Tier:** exploit · **reverses the EXP-050b rejection**

DA-006-A reopened the visual fusion question. Re-running it against all four
seed baselines resolves it, and the earlier rejection was wrong for a
methodological reason worth recording.

**Fusion delta is a paired quantity.** It compares the same base model with and
without the visual member on the same 652 clips, so base-model seed variance
cancels. The correct noise scale is the across-seed standard deviation of the
*delta*, not of accuracy. DA-006-A compared +0.0092 against the 2.80-point
accuracy sigma; that was the wrong denominator.

Paired delta by seed, at the fixed weight:

| w | seed 0 | seed 1 | seed 2 | seed 3 | mean | std | all positive |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.15 | +0.15 | +1.07 | +0.61 | +0.46 | +0.58 | 0.38 | yes |
| **0.20** | **+0.92** | **+0.92** | **+0.92** | **+1.23** | **+1.00** | **0.15** | **yes** |
| 0.25 | +0.46 | +0.77 | +1.07 | +1.38 | +0.92 | 0.40 | yes |
| 0.30 | −0.46 | +1.38 | +1.23 | +1.23 | +0.84 | 0.87 | no |

At `w=0.20` the gain is **+1.00 points with a standard deviation of 0.15**,
positive on every seed, across base models spanning 0.5138 to 0.5690. On the
paired scale that is roughly 6.7 sigma. The optimum sits at the same weight for
all four seeds, which is itself evidence the effect is structural rather than
fitted.

**Corrections this forces.**

1. EXP-050b's fusion rejection is **withdrawn**. The visual member does add
   information. The original +0.0092 point estimate for seed 0 was accurate;
   what was wrong was the gate (+2.0 points, far above the achievable effect)
   and later calling it noise.
2. DA-006-A's claim that the fusion delta was "+0.33 sigma" is **withdrawn**.
   It applied an unpaired denominator to a paired statistic.
3. B-021 loses more support: the visual branch is weak *alone* but genuinely
   additive in fusion, which is exactly the pattern its 44 rescues (35 object)
   and 63.65% union oracle predicted.

**Standing lesson for gate design:** set the gate from the achievable effect
size and the correct noise model, not from a round number. A +2.0-point gate
over a mechanism whose true effect is +1.0 point can only ever reject.

**Cost check:** the visual member is 2,314,518 parameters, about 2.4 MB int8
against 14.78 MB of remaining package headroom, so adoption is affordable.

---

## DA-006-A — Seed-variance floor: THE SESSION'S SCREENS WERE UNDERPOWERED
**Date:** 2026-07-31 · **Tier:** validation infrastructure

Four seeds, identical recipe, identical all-18 fold-2 partition, nothing else
changed:

| seed | accuracy | best epoch | clips/652 |
|---:|---:|---:|---:|
| 0 | 0.5690 | 61 | 371 |
| 1 | 0.5675 | 79 | 370 |
| 2 | 0.5276 | 28 | 344 |
| 3 | 0.5138 | 31 | 335 |

**mean 0.5445 · std 2.80 points · range 5.52 points (36 clips).**

Measured against that floor, every gate used this session sits inside the
noise:

| quantity | value | sigma |
|---|---:|---:|
| EXP-051 cluster specialists | −0.0107 | **−0.38** |
| EXP-050b fusion delta | +0.0092 | **+0.33** |
| gate threshold used for both | +0.0200 | +0.71 |

Worse, **the 0.5690 baseline everything was compared against is the maximum of
the four seeds**, 2.45 points above the seed mean. Every method this session was
tested against the luckiest available control, which biases uniformly toward
rejection.

### Consequences — two conclusions withdrawn

- **EXP-051 is not a negative result. It is underpowered.** −0.38 sigma is
  indistinguishable from zero. The claim that narrowing the label space fails
  is not supported by this evidence.
- **EXP-050b's fusion rejection is not supported either.** +0.33 sigma is
  noise. Its *solo* result stands: 0.3328 against a 0.5445 seed mean is −21
  points, roughly 7.6 sigma, and the object-class gap is large. The visual
  branch is genuinely weak on its own; whether it helps in fusion is **unknown**,
  not refuted.
- **EXP-052 is unaffected.** It decodes fixed probability files, so both arms
  share identical inputs and the comparison is paired and deterministic. Its
  +10 clips and the B-022 fold pattern stand.

### Why this was missed

Every recent screen trained one model on one fold and compared it to one
baseline number. The deployed champion is a 48-member ensemble whose averaging
suppresses exactly this variance, which is why it reaches ~61.8% while single
members sit near 54--57%. **We were screening with an instrument roughly five
times noisier than the system the decisions were about.**

A secondary finding: seeds 2 and 3 peaked at epochs 28 and 31 and never
recovered, while seeds 0 and 1 peaked at 61 and 79. The recipe is not merely
noisy, it is unstable — some runs settle into a worse optimum early.

### Required protocol change

No adoption decision from a single seed on a single fold. Screens must report
a mean over >=3 seeds, or use a paired deterministic comparison as EXP-052
did. Any effect below ~5 points needs seed averaging to be visible at all.

---

## DA-006 — Sixth devil's-advocate pass
**Date:** 2026-07-31 · **Tier:** mandatory adversarial audit

Triggered at 7/10 by three consecutive rejections (EXP-050b, EXP-051, EXP-052).

### 1. Why is our current direction probably wrong?

**The measured rate cannot reach the target, and it is decelerating.** The
verified ladder moved 97 → 112 public clips across roughly fifteen experiments:
about one clip each. The standing target needs **+67 more**. The last three
experiments delivered **zero**. Continuing to spend experiments on per-clip
predictors and decode postprocessing extrapolates from a rate that has stalled.

**We keep gating on a metric with a known adverse transfer.** Every mechanism
that transferred did so at a fraction of its local size: transitions +6.0 OOF
points → +1.5 public points; repeat consensus +3.1 nested OOF → **−1 public
clip**. We know the local→public map is compressive and unreliable, and we
still use local OOF as the primary adoption gate.

### 2. What assumption have we never questioned?

**Person selection.** Every skeleton model in this campaign has run with
`person="first"`. `har_data.py` has implemented a `motion` policy the whole
time and **it has never been run once**. Q-86 was raised by DA-001, given
expected value +1--2, and left queued through 52 experiments. DA-001 also
recorded that **84.6% of Comb_hair clips contain multiple detected persons**
(mirrors, bystanders). If the wrong body is selected, that is a systematic
*input* corruption concentrated in exactly the bathroom/object classes that
dominate the error mass — and every experiment since inherited it.

This reframes B-021. We concluded object information is absent from the input.
We never checked that the input describes the right person.

Secondary unexamined assumptions: that the 201/204 public/private split is
random rather than grouped by user or recording; and that per-clip is the right
prediction unit when clips arrive in 144 recordings.

### 3. What would a Kaggle gold medalist criticize?

**No noise floor.** Gates of +0.02 and +2.0 points were set without ever
measuring run-to-run variance. EXP-051's −0.0107 and EXP-050b's +0.0092 may
both sit inside seed noise, making those experiments underpowered rather than
negative. On 652 clips one clip is 0.153 points. *This audit launches the
measurement: three seeds, identical recipe, fold 2.*

**Single-fold adoption decisions.** Recent screens use fold 2 alone (n=652),
the hardest of the four, while an all-18 four-fold protocol exists and is used
for exactly one fold.

**The submission budget is being wasted.** Roughly five per day are available;
about fifteen have been used in the whole campaign and **zero in the last two
days**. The public leaderboard is the only instrument that measures the
quantity being optimized, and it sits idle while we tune a proxy we know is
miscalibrated.

**Known-optimistic validation retained.** The 2700-row OOF omits users 5 and
21 and reuses selected checkpoints, worth ~1.2--1.3 optimistic points, and it
still backs EXP-052's comparisons.

### 4. Strongest argument against the current direction, steelmanned

Model work and transduction are each blocked on the other, and we have not
acknowledged the deadlock.

B-022 establishes that metadata coupling only pays above a base-accuracy
threshold we are below. B-021 establishes that the model side cannot add object
information from current inputs. The base sits at the dataset paper's published
cross-subject LOSO of ~56.4%, which we match at 56.90%. So transduction waits on
the model, the model has no remaining lever over these features, and each
further experiment samples from a distribution whose mean gain is now
approximately zero.

The escape is not a better predictor over the same inputs. It is either a
corrected input (item 2) or a mechanism nobody in the campaign has tested. The
crazy tier was budgeted at 10% and has received roughly 8%, none of it recently.

### Queue entries produced

| ID | Action | Why | Cost |
|---|---|---|---|
| DA-006-A | Seed-variance floor, 3 seeds fold 2 | every gate is uncalibrated; re-reads EXP-050b/051 | ~15 min, **running** |
| DA-006-B | Person-selection `first` vs `motion` | never run in 52 experiments; input-level, hits the dominant error classes | ~10 min |
| DA-006-C | Spend the idle submission budget | the only unbiased instrument is unused | free |
| DA-006-D | Move screens to >=2 folds | single-fold decisions at n=652 | 2x compute |

---

## EXP-052 — Joint distinctness + transition decoding: FAILED GATE
**Date:** 2026-07-31 · `code/ordered_transition_decoder.py` · **Tier:** explore

**Control first:** transition-only at fixed `lambda=0.5`, `distinctness=none`
reproduced **0.67815 (1831/2700)** exactly, matching EXP-047. The comparison is
therefore sound.

Joint results at `lambda=0.5`, full-OOF sweep:

| distinctness | OOF | correct/2700 | vs transition-only |
|---|---:|---:|---:|
| none (control) | 0.67815 | 1831 | — |
| penalty 0.5 | 0.68000 | 1836 | +5 |
| **penalty 1.0** | **0.68185** | **1841** | **+10** |
| penalty 2.0 | 0.67963 | 1835 | +4 |
| penalty 4.0 | 0.67778 | 1830 | −1 |
| hard | 0.67778 | 1830 | −1 |

Per-fold for the best setting (penalty 1.0) against transition-only:

| fold | base | transition-only | joint | delta |
|---|---:|---:|---:|---:|
| 0 | 425 | 477 | 483 | +6 |
| 1 | 447 | 495 | 502 | +7 |
| 2 | 400 | 429 | 424 | **−5** |
| 3 | 396 | 430 | 432 | +2 |

**Gate: fails on both criteria.** The gain is **+0.370 points** against a
required +1.0, and that figure is already optimistic because the penalty was
swept on the scored data; a nested selection can only be lower. Fold 2 is
negative, so the all-folds-non-negative requirement fails outright.

### Mechanism finding (the useful part)

Distinctness is not uniformly good or bad — its sign tracks base quality:

| fold | base accuracy | hard-distinctness delta |
|---|---:|---:|
| 1 | 0.664 | +6 |
| 0 | 0.631 | +5 |
| 2 | 0.591 | −9 |
| 3 | 0.586 | −3 |

It helps on the two strongest folds and hurts on the two weakest, monotonically
in base accuracy. The mechanism is straightforward: a distinctness constraint is
a *coupling*. When the posterior is reliable it propagates correct information
between clips; when it is unreliable one wrong clip evicts a second clip from
its correct label, so errors propagate too. Hard distinctness maximizes the
coupling and is worst overall; penalty 1.0 is the best trade-off found and is
still fold-2 negative.

This also retro-explains EXP-004/005: Hungarian assignment lost public clips at
a 0.458 base, exactly the regime where coupling should hurt most. The mechanism
was never wrong — it was applied where the posterior could not support it.

### Failure analysis

**Three reasons it failed.** (1) The base posterior is not strong enough for
constraint propagation to pay: at 0.59--0.66 per-fold accuracy, roughly one clip
in three is wrong and each error can now damage a neighbour. (2) The gain that
does exist concentrates on folds where transitions already worked, so it is
partly redundant with EXP-047 rather than additive. (3) Distinctness is a
property of *train* recordings (741/741); it is assumed, never verified, for
test recordings, so on hidden users the constraint may be partly false.

**Three alternative explanations.** (1) The penalty could be conditioned on
per-clip confidence, applying coupling only where the posterior is peaked — this
is untested and is the natural next form. (2) The 2700-row historical OOF omits
users 5 and 21 and reuses selected checkpoints, so fold-level deltas of a few
clips sit near its resolution. (3) `lambda` and penalty were optimized
independently; a joint 2-D nested sweep might find a better interior point,
though the +0.370 ceiling makes a +1.0 outcome unlikely.

**Three follow-ups.** (1) Confidence-gated coupling: apply the penalty only to
clips above a posterior threshold. (2) Re-test after any base improvement, since
the mechanism finding predicts the sign flips with base quality. (3) Treat the
+10 clips as real but too small to spend a submission on, per B-018.

**Decision:** reject for submission. Retain the finding that distinctness value
is a monotone function of base accuracy — that is a reusable prediction, and it
converts two historical failures (EXP-004/005) plus this one into a single
consistent account. B-011 keeps its structural claim; B-007's operating rule
narrows to directional order only, as EXP-047 already established.

---

## EXP-052 — predeclaration (kept for provenance)
**Date:** 2026-07-31 · **Tier:** explore

Written before running any joint decode.

**Hypothesis.** Solving distinctness and directional order as one joint
objective per recording beats either alone. EXP-000c proved train recording
groups are 100% class-distinct (741/741); EXP-004/005 applied that as a hard
Hungarian assignment and lost public clips at a 0.458 base; EXP-047 used
directional order alone and gained three public clips. The two constraints have
never been optimized together at the current base.

**Why now.** B-021 closed the model-side branch over skeleton features, and the
per-clip posterior is the input to this decoder rather than something it has to
improve. This spends no new representation capacity.

**Implementation note.** `beam_decode` already supports
`distinctness in {none, hard, penalty}` with a repeated-label penalty; EXP-047
only ever exercised `none`. No new decoder is required, which removes
implementation risk from the test.

**Controls.** The exact per-clip control is `--lambda 0`. The transition-only
reference is the validated fixed `--lambda 0.5` at `distinctness=none`, which
must reproduce **67.815% (1831/2700)** before any joint number is believed.

**Predeclared gate.** Adopt only if the joint decode beats transition-only by
**≥ +1.0 point under strict nested selection** *and* is non-negative on all
four folds. The margin is set deliberately above EXP-048's failure mode, where
+3.1 nested OOF points still lost a public clip: at this resolution a sub-point
OOF gain is not evidence of transfer.

**Beliefs it tests.** B-007 (directional order is the high-value transductive
signal), B-011 (recording groups are ordered, user-pure, class-distinct),
B-020, and B-004's warning that local decode gains overstate public transfer.

---

## EXP-051 — Confusion-cluster specialists (X-02): FAILED, REGRESSED
**Date:** 2026-07-31 · `code/cluster_specialist.py` · **Tier:** exploit

**Result: 0.5583 versus the 0.5690 matched baseline, delta −0.0107.** The
predeclared gate was ≥0.5890. Failed, and in the wrong direction.

Inner CV over the 14 outer-train users reached 0.4673 micro (2281 clips) and
produced six clusters under the fixed rule. The two clusters seen earlier on
fold 2 were recovered independently from training users only, so the derivation
is convergent as well as leak-free:

| size | members |
|---:|---|
| 8 | Wipe_hands, Drink_water, Eat_food, Tableware, Pour, Stir, Peel, Headphones |
| 8 | Keyboard, Write, Phone_call, Check_time, Read_docs, Turn_pages, Mobile, Play_games |
| 8 | Brush_teeth, Comb_hair, Put_on_clothes, Jumping_jacks, Stretching, Take_medicine, Massage, Body_temp |
| 7 | Take_off_clothes, Wipe_windows, Squats, Stand_up, Sit_down, Lunges, Walk |
| 3 | Sweep, Mop, Fold_clothes |
| 2 | Selfie, Lie_down |

532/1215 inner errors (43.8%) fall inside a cluster.

Outcome on fold 2, scored once with the fixed geometric-mean rule:

```
changed 45 of 652 · rescues 9 · harms 16 · net -7 clips
routed to a specialist        614 (94.2% of the split)
  truth inside that cluster   485 (79.0%)
baseline errors among routed  271
  addressable pool            142
  actually fixed                9
```

**The decisive number is 9 of 142.** Given a clean shot at 142 within-cluster
errors, a head trained exclusively on that cluster fixed nine and broke
sixteen. Restricting the label space did not make the boundary easier.

### Failure analysis

**Three reasons it failed.** (1) The discriminating information is absent
from the input, not obscured by the classifier: separating Drink_water from
Take_medicine, or Read_documents from Turn_pages, requires knowing the held
object, and B-002/EXP-050b both show skeleton does not carry it. Narrowing the
output space cannot manufacture an input feature. (2) Specialists trained on
377--670 clips versus the base model's 2281; the other 33 classes were acting
as regularization, and removing them cost more than the narrower decision
boundary gained. (3) The base 40-way posterior is already well ordered inside
a cluster — top-2 0.6641, top-3 0.7331 — so a weaker model overwriting it
destroys more than it repairs.

**Three alternative explanations.** (1) The clusters are too coarse: 94.2% of
the split was routed, so this behaved as a near-global re-decision rather than
a targeted one, and `MAX_CLUSTER=8` with connected components merged
kinematically unrelated classes (Jumping_jacks with Brush_teeth). A stricter
rule might isolate genuine pairs. (2) The fixed geometric mean gives the
specialist equal authority; a confidence-gated or nested weight might only act
where the specialist is strong, though that adds a selection surface this
screen deliberately avoided. (3) Last-epoch, no-selection training was chosen
for honesty and may leave each specialist under-fit relative to the
best-epoch baseline it is compared against.

**Three follow-ups.** (1) Restrict to true pairs (`MAX_CLUSTER=2`,
higher `MIN_PAIR_ERRORS`) so routing touches tens of clips, not 614. (2) Give
the specialist an input the skeleton lacks — the EXP-050b visual embedding
rescued 35 object clips — rather than more capacity on the same features.
(3) Stop adding predictors over these features and hunt the mechanism behind
the rank-1 outlier instead.

**Decision:** reject. Two consecutive negatives (EXP-050b, EXP-051) now agree
on the same boundary: **no rearrangement of skeleton-derived predictors adds
object information.** B-002's ceiling and B-013 are both reinforced. The next
move should not be another predictor over these features.

---

## EXP-051 — predeclaration (kept for provenance)
**Date:** 2026-07-31 · **Tier:** exploit

Written before any cluster was derived and before any specialist was trained.

**Hypothesis.** A low-parameter head that only separates members of one
confusion cluster beats the 40-way base model on clips whose base top-1 falls
in that cluster, because it stops spending capacity on 33 irrelevant classes
and places its decision boundary exactly where the error mass sits.

**Reason.** EXP-050b left 281 baseline errors on all-18 fold 2. They
concentrate in tight groups of object classes that share near-identical
skeleton kinematics (table/kitchen: Drink_water, Eat_food,
Take_and_use_tableware, Pour_drinks, Stir_drinks, Peel_fruits, Take_medicine;
seated screen/paper: Read_documents, Turn_pages, Use_a_mobile_phone,
Play_games). 62 fold-2 clips hold the truth at base rank 2, 52 of them object
classes, and the base top-3 reaches 0.7331 against a top-1 of 0.5690.

**Prior art.** X-02 was queued but never run. It is not EXP-045/050 (whole new
visual streams), not EXP-032 (learned global gate), and not EXP-047/048
(sequence postprocessing). Its novelty is conditioning on the base model's own
candidate set.

**Leakage control — the load-bearing design choice.** The confusion structure
seen while analysing fold 2 must not define the clusters, since fold 2 is the
scored split. Clusters are derived from a 4-way inner cross-validation over the
14 fold-2 *training* users only. Fold-2 validation labels are read exactly once,
at final scoring. The derivation rule is fixed in code before it runs:
`MIN_PAIR_ERRORS=4`, `MIN_CLASS_SUPPORT=10`, `MAX_CLUSTER=8`.

**Predeclared gate.** Adopt only if fold-2 micro accuracy reaches
**≥ 0.5890**, i.e. at least **+2.0 points** over the partition-matched
`astgcn_all18_f2_best` baseline of 0.5690 (+13 clips of 652). Any smaller
movement is treated as noise at this resolution and rejected, consistent with
B-018 and the EXP-048 lesson that local gains need margin to survive transfer.

**Beliefs it tests.** B-010 (hard classes need targeted handling, not global
correction), B-013 (top-1 inventory ceiling), B-002 (skeleton ceiling).

**Expected effect.** +2 to +3.4 points if within-cluster boundaries are
learnable from skeleton features alone; near zero if the discriminating
information is genuinely object appearance, which EXP-050b showed the skeleton
lacks. A null result localizes the ceiling rather than merely repeating it.

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
the rank-15 bar as additional context (P-05); it informs sequencing, not
whether to pursue the target.

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

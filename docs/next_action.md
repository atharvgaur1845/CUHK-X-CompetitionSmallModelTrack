# Next action

**This file is the handover.** Whoever picks this campaign up next — a fresh session, a
different model (Fable, Opus, Sonnet), or a collaborator — starts here rather than
re-deriving it.

**Keeping it current is a rule, not a courtesy.** Any change that alters project state
updates this file *in the same commit*. A stale handover is worse than none: it sends the
next session confidently in the wrong direction.

Read order: `CLAUDE.md` → this file → `research/DIRECTIVE.md` → `research/LOG.md` (newest
first) → `research/RULES_VERIFIED.md` → `research/BELIEFS.md`.

**This section is deliberately first.** Put history *below* the work, never above it.

---

## Do this next

> # ⚡ STATE 2026-09-08 (evening) — the +20 plan was TESTED AND FALSIFIED IN ONE DAY. Read this before re-proposing any of it.
>
> | | |
> |---|---|
> | score | **167/201 = 0.83084** (`sub_r2_dist`), unchanged |
> | legal package | **93.37 MB, verified** (EXP-129), scores **165** |
> | compute | cluster `ssh sharanga` **UP and usable** — see "Cluster" below |
> | Kaggle deadline | **2026-09-15** · code upload 09-22 |
>
> ## ⛔ THE TARGET ARITHMETIC, AND IT IS THE MOST IMPORTANT LINE IN THIS FILE
>
> Anchored on measured, decoder-inclusive numbers (champion decoded OOF 0.81148, public
> 0.83084, so the offset is **+1.94 points**; pre-decoder top-1 0.75630, top-2 **0.86370**):
>
> | capture of the rank-2 pool | decoded OOF | projected public |
> |---|---|---|
> | 0% (today) | 0.7949 | 164 |
> | 30% | 0.8271 | **170** |
> | 50% | 0.8486 | **174** |
> | **100% — a PERFECT rank-2 oracle** | 0.9023 | **185** |
>
> **A perfect resolver of every rank-2 clip lands at ~185, not 188.** The brief's "+21 clips
> from one binary decision" assumed capturing essentially all 285 of them. Realistic capture
> is 20-30%, i.e. **170-175**. **187 is not reachable by re-ranking; it requires a stronger
> MEMBER, which raises top-2 itself.** Do not spend another day on arbitration schemes.
>
> ## What was killed on 2026-09-08, each with a pre-registered bar (EXP-132..137)
>
> | route | measured | verdict |
> |---|---|---|
> | COCO detector on IR as an object channel | handheld AUC **0.502** on held-out subjects, **below** the station-only baseline 0.574 | **DEAD** (EXP-134) |
> | frozen video foundation teacher (V-JEPA 2 ViT-L) | **0.419** pooled vs our MViT **0.712** | **DEAD** (EXP-135) |
> | pseudo-label the test split for +12 subjects | subject learning curve **flat past 6 subjects** (6->14 buys +0.18) | **DEAD** (EXP-136) |
> | per-group logit centering | +0.85 pre-decoder -> **+0.19 after the shipped decoder**, interior alpha | **not adopted** (EXP-137) |
> | subject clustering on test (P0) | 18-cluster purity **0.339** | **FAIL** (EXP-132) |
> | subject feature centering (2a) | +0.63 member pts, 4/4 folds, bar is 1.64 | below bar |
> | within-subject prototype vote (2c) | **+7 clips of 2,700** | negligible |
>
> **The one solid new diagnosis (EXP-132), worth keeping:** 229 of 658 errors repeat the same
> (true -> pred) confusion **inside a single subject**, 55 of 70 subject-pair cells are
> strictly one-directional, and 242 of 290 rank-2 errors have a correctly-predicted clip of
> the true class from the same subject. The error is a subject-constant offset. Every attempt
> to exploit it failed for one measured reason: **the references used to correct the bias are
> produced by the same biased model**, so they inherit it (2c: 17 rescued / 10 broken).
>
> ### ✅ ① T-PKG IS DONE (EXP-138). The 167 champion is now a legal 94.95 MB file.
>
> `research/artifacts/stage2_champion.pth` — **94.95 MB by `ls -la`**, integrity 1,604
> tensors / 0 mismatches, and the IMU member rebuilt **from the package** reproduces
> `sub_r2_dist.csv` (the 167) on **0 of 405 rows**. The sklearn ExtraTrees now ships as
> tensors (7.63 MB: int16 tree-local children, uint8 leaves, leaf index recomputed at load,
> tree blobs deflated 4.8x in the archive). The legal package goes **165 → 167**.
>
>     python3 code/pack_stage2.py --bits 6 --video person=k224_mvit_all \
>       --video wrist=k224_mvitwrist_all --imu-trees imu_trees_t200_d12_int8.npz \
>       --out research/artifacts/stage2_champion.pth
>     python3 code/unpack_stage2.py --package research/artifacts/stage2_champion.pth --check integrity
>
> **Remaining for T-PKG:** `unpack_stage2.py --check weights,infer` does not yet know about
> the `imu` branch (integrity covers it; the round-trip was verified by hand in EXP-138).
> Submit `submissions/sub_pkgchamp.csv` once to bind the package to a measured score — it is
> rowdiff 0 against the champion, so it should score exactly 167.
>
> ### ② The old ① — superseded, kept for the reasoning
>
> The champion (167) is **not packageable**: `imu_stats` is an sklearn ExtraTrees and
> `pack_stage2.py` cannot represent it; dropping it moves 43 of 405 rows, and the legal
> package therefore scores **165**. Serialising the forest as tensors recovers those 2 clips
> **and** closes the largest compliance risk (reproducibility is 10% of the grade).
>
> Budget: person 34.28 + wrist 34.28 + skel 22.80 + trees 9.00 = **100.36 MB — over by 0.36**.
> Two measured ways under: `imu_stats` at 150 trees/depth 10 is **4.61 MB** (costs ~9 clips of
> 2,700 ≈ 0.7 public), or drop one skeleton arch (~4.5 MB, ~2 rows). Bit-packing int6 to
> 6/8 of a byte saves **8.6 MB per view** and is still "not implemented rather than
> implemented and unused" (EXP-129).
>
> ### ② The last untested MEMBER lever — thermal at 224 px through the MViT recipe
>
> Thermal has only ever been run at **128 px through r2plus1d** (EXP-088 0.544, EXP-119
> full-frame 0.558, EXP-120b null at 4 folds). That is the recipe the 224/MViT recipe beat by
> **+7.5 points** on IR+depth (0.64 -> 0.715). Thermal is the dataset paper's **best** modality
> (92.57) and is our most decorrelated view (54% agreement).
> **Sober arithmetic before spending on it:** 0.544 + 7.5 = ~0.62, and EXP-123's accounting
> (19 rescues against 247 errors) says thermal needs roughly **0.68** before any global weight
> can harvest it. So P(pays) ~25%. Gate on **member accuracy first**, fusion only if >= 0.68.
> Raw thermal is syncing to the cluster now.
>
> ### ③ Do NOT re-propose
> Every row of the table above, plus the standing graveyard. In particular: no further
> stacker / gate / router / calibrator over member probabilities (EXP-131 plus five earlier),
> and no re-ranking scheme justified by "+21 clips are sitting at rank 2" — the arithmetic
> above caps that route at 185 with a *perfect* oracle.
>
> ## Cluster — WORKING, and the two things that cost hours today
>
> `ssh sharanga`, user `pabitra`. Code `/home/pabitra/cuhkx`, data on
> `/scratch/pabitra/cuhkx` via symlinks (`cache`, `checkpoints`, `logs`, `submissions`).
>
> - **Use `/home/pabitra/.conda/envs/physmon/bin/python`** — Python 3.11, torch 2.7.1+cu126,
>   torchvision 0.22.1, numpy, sklearn, PIL. Verified against `ClipStore` and the synced
>   cache. The `cuhkx` env in `cluster/env.sh` is **Python 3.8 with no pip**, and
>   `conda env remove` refuses to delete it, so pip falls back to system python and fails.
>   Do not spend time on it; use `physmon` or build a venv on top of it.
> - **The link is 1.6 MB/s.** Measured, not estimated. crop_224 took 11.5 min, the whole
>   priority set ~50 min, and the 11 GB of raw thermal + Testing takes ~2 hours. **Order the
>   sync by what each byte unblocks** — `third_party/` (0.5 GB, IG-65M and MotionBERT, both
>   dropped from the champion) cost 25 minutes before it was noticed.
> - Synced and verified: `cache/crop_224`, `cache/crop_wrist224`, `cache/train`, `cache/test`,
>   `meta_*.csv`, and the 10 MViT checkpoints. Raw thermal + Testing were still transferring.
>
> ## New tools committed today (all reusable, none score-positive by themselves)
>
> `code/exp132_subject_errors.py` · `code/exp133_subject_keys.py` · `code/dump_embeddings.py`
> (768-d penultimate features, all folds honest + test, both views, ~9 min on the laptop) ·
> `code/probe_p0_p2a.py` · `code/probe_2c_prototype.py` · `code/probe_object_channel2.py`
> (the honest, subject-grouped version — `probe_object_channel.py` is the buggy first pass,
> kept only as the worked example of a selection artifact) · `code/probe_subject_curve.py` ·
> `code/teacher_features.py` + `code/teacher_probe.py`.
> `kaggle/cuhkx_224_kaggle.py` gains `--train-users`, `--eval-users`, `--pseudo`,
> `--pseudo-users`, `--pseudo-top-frac` (the 3-way train/pseudo/eval design EXP-136 made
> unnecessary, but they are tested and harmless), and a **fix to `--help`, which was broken
> for the whole campaign** by an unescaped `57.7%` in the `--crop` help string.
>
> ## Two process failures worth more than they cost
>
> 1. **`pkill -f <pattern>` matched my own shell three times** because the pattern appeared in
>    the command line that was running it (exit 144, script never written, old job survived).
>    This is EXP-106's "never let a waiter's predicate match the waiter" in a new costume.
>    Kill by PID, or `pkill -x` on an exact process name, or put the kill in its own file.
> 2. **A probe returned mean pair-AUC 1.000 and it was a selection artifact** (max over 30
>    features on 24 points, plus a station confound). The tell was semantic — `tv` separating
>    Read_documents from Turn_pages. Likewise V-JEPA 2 first scored **0.170**, which was a
>    feature-scaling bug, not the model. **A number at either extreme is a bug until proven
>    otherwise; bug-hunt before it earns a hypothesis.**

> ### ⚠ COMPUTE, 2026-09-03: the cluster is DOWN for 15 days — i.e. past the deadline.
>
> `ssh sharanga` is under storage maintenance until ~2026-09-17. The Kaggle deadline is
> **2026-09-15**. **Plan as though the cluster does not exist.** Everything in
> `cluster/` is written and staged but must not be counted on; the "HIGH JUMP" cluster
> campaign (T1 LOSO at 54 runs, T2 SSL, T4 multimodal) is **not affordable** on what we
> actually have. Do not queue work against it.
>
> **What we actually have:** the 8 GB laptop, and **Kaggle notebooks at ~30 GPU-h/week**
> (~2 h per 224 px MViT fold). That is roughly **50 GPU-hours before the deadline**.
> Split them by what each machine is good at: 224 px MViT work goes to Kaggle, 128 px
> thermal/skeleton work goes to the laptop, and they run concurrently.

> ### ⚠ THE KAGGLE PARITY GATE WAS UNMEETABLE. I wrote it. See EXP-120.
>
> The gate was *"retrain `k224_mvit_f2` and reproduce micro = 0.71472"*. Kaggle returned
> **0.68252** and this looked like a failure. It is not: **`kaggle/cuhkx_224_kaggle.py`
> seeds nothing** — no `manual_seed`, no `np.random.seed` — so 0.71472 is one draw of a
> random process and **the laptop cannot reproduce it either**. Everything checkable
> matched (split 2281/652, params, epoch-1 lr 5.23e-05, 142 steps/epoch under both
> batch/accum pairs, normal loss curve), and the entire deficit is in OBJECT with
> `gross_motion` **identical to five decimals** — the shape of a weaker draw, not of a
> broken pipeline. **The environment is not implicated. Do not go looking for a bug.**
>
> Fixed: the trainer now takes **`--seed`** (default `None` = historical behaviour, so
> no existing artifact or comparison is invalidated). Seeding the parent is sufficient —
> verified under fork — and a per-worker seed is deliberately **not** added, because it
> would change augmentation semantics and make the measured spread describe a pipeline
> we never ran. `cluster/parity.sbatch` and `cluster/README.md` are rewritten to match.
>
> Also falsified and recorded so nobody re-derives it: the augmentation draws from the
> numpy *global* RNG with no `worker_init_fn`, which looks exactly like the classic
> duplicate-stream bug. **It is not present** — torch seeds numpy per worker; measured,
> not assumed. And both arms ran the same `--workers 2` default, so it was never a
> laptop-vs-Kaggle difference either.

> ### ✅ VISUAL SEED σ MEASURED — 1.16 points (n=3). Parity settled. See EXP-120a.
>
> Three seeded replicates of `k224_mvit_f2` on Kaggle: **0.71012 / 0.69172 / 0.71319**,
> mean **0.70501**, sd **1.16 points**. The 2σ band [0.68178, 0.72824] contains both the
> laptop's 0.71472 and Kaggle's unseeded 0.68252. **There is no environment difference —
> Kaggle is a trustworthy second machine and its numbers count.**
>
> **Two things this changes, one of them counter-intuitive:**
>
> 1. **0.71472 is an upper draw, not a baseline.** It is the highest of the five draws we
>    now have of this exact recipe, against a mean of 0.70501. Every delta measured
>    against it was biased ≈ −1 point. Stop quoting it as *the* fold-2 number.
> 2. **The single-fold bar was too LOOSE, not too tight.** A smaller σ does not lower it:
>    a delta is a difference of two runs and carries σ√2 = 1.64, so the 2-SE bar for
>    "one new run vs one old run on one fold" — the design this campaign actually used —
>    is **3.28**, *higher* than the 2.80 it screened against. Three spurious "+2.45"s is
>    exactly what a bar set half a point too low produces.
>
> | design | SE of mean | 2-SE bar |
> |---|---|---|
> | one fold, new vs old | 1.64 | **3.28** |
> | 4 paired folds | 0.82 | **1.64** |
>
> **Operating rule: never screen a visual member-strength change on one fold, at any
> threshold. Use paired multi-fold designs**, where four folds bring the bar to 1.64.
>
> **⚠ Honest limit:** n=3, so the 95% CI on σ is **[0.60, 7.30]** and does **not** exclude
> 2.80. 1.16 is the best point estimate and the first ever measured on this branch, but it
> does not refute 2.80. More seeds are not worth buying — n=6 tightens the CI only to
> ≈[0.72, 2.84]. Use 1.16 as the working estimate and quote the CI.

> ### 📄 NEW: `docs/RESEARCH_PROGRAM.md` — read it before proposing any lever.
>
> A diagnostic pass over artifacts already on disk (EXP-124, no new training) changed the
> shape of the problem:
>
> - **It is a RANKING problem.** top-1 0.7630 but **top-2 0.8685** — **44.5% of all errors
>   have the true label at exactly rank 2**. Resolving only rank-1-vs-rank-2 is +10.55 pts.
> - **The oracle gap IS the rank-2 gap.** Oracle-any-member 0.8774 ≈ champion top-2 0.8685.
>   The headroom is not "which member to trust", it is one binary decision on a pair the
>   fusion already surfaced. This reframes T3.
> - **Subject variance derives the noise floor.** Between-subject sd **5.26 pts** (4.5× seed
>   σ). Public is 4 subjects → SE 2.63 → 2 SE ≈ **±10.6 clips of 201**. The ±9–10 floor is
>   *subject sampling*; no amount of seed or member averaging reduces it. A +2-clip margin
>   is a fifth of one standard error.
> - **Errors spread over 175 pairs** (top-10 = 32.3%), so per-pair specialists (`QUEUE.md`
>   X-02) are the wrong shape; one pair-conditioned discriminator is the right one.
>
> ### ✅ NEW CHAMPION 2026-09-03: `sub_r2_dist.csv` = **0.83084 = 167/201** (EXP-125)
>
> Soft distinctness turned on; everything else identical to `sub_r2` (same
> `testprobs_r2.npz` by SHA-256, one config flag). **+1 clip, and the comparison carries
> no sampling noise** — 399 of 405 rows are identical, so +1 is the exact net of 6 changed
> rows rather than a draw from the ±9–10 floor.
>
> **B-022 is confirmed**: distinctness coupling flips sign as base accuracy rises. Public
> ladder **−1 @112, 0 @121, +1 @166**, monotone across 54 clips of base.
>
> **Recipe change, use it from now on:** add
> `--distinctness penalty --distinctness-penalty 2.0` to every decode.
>
> **Forecasting note worth keeping:** EXP-124 predicted +4/+5 clips; I retracted to
> "~+1, range −3..+3" **before** scoring, after measuring incrementally through the real
> decoder instead of against raw argmax. The outcome was exactly +1. *Measure an add-on
> against the system you ship, not against argmax.*

**Job status, 2026-09-03.**

| where | job | status |
|---|---|---|
| **Kaggle** | EXP-120a — visual seed σ | ✅ **DONE.** σ = 1.16 (n=3); parity settled; Kaggle vindicated as a second machine |
| **laptop** | EXP-120b — paired thermal folds | **running**, 3 of 6 runs done (~2.5 h each). `logs/thermal_pairs_queue.log` |

**EXP-120b interim — full-frame minus cropped thermal, paired:**

| fold | full | cropped | delta |
|---|---|---|---|
| 0 | 0.58354 | 0.58231 | **+0.12** |
| 2 | 0.55828 | 0.54448 | **+1.38** |
| 1 | 0.56511 | *running* | |
| mean (n=2) | | | **+0.75 = 0.65 SE** — null so far |

**Fold 0 does not replicate fold 2**, which is the outcome the design existed to detect:
EXP-119's +1.38 was a single fold, and the 2-SE bar at 4 paired folds is 1.64. Do not
adopt on the interim. Wait for folds 1 and 3. (The SE borrows the MViT seed σ; thermal's
own σ is unmeasured, so treat it as indicative.)

**On the queue's design:** only fold 2 existed for *either* thermal variant, so folds
0/1/3 need **both** arms or the comparison is unpaired and unreadable — hence 6 runs, not
3. They are ordered **fold-major** so killing the queue at any point still leaves complete
**pairs** on disk. An interrupted arm-major queue would be worthless.

**Read the thermal result against this, not against micro alone:** the thermal branch
contributes **exactly zero** to the fusion today, because its confidence when right
(0.511) barely exceeds its confidence when wrong (0.413). A member-accuracy gain on a
branch whose fusion weight is 0 buys nothing. The question that decides adoption is
whether full-frame thermal changes the **fused** score — so the deliverable is
`--stage infer` test probabilities on all four folds, not four more micro numbers.


> # ✅ 2026-09-04: A STAGE-2 PACKAGE NOW EXISTS AS A FILE. EXP-129.
>
> The 2026-09-01 retraction ("no single-file package containing a video member has ever
> been built; every size figure is arithmetic") is **resolved**. A file exists:
>
>     -rw-rw-r-- 1 atharv atharv 93367950 research/artifacts/stage2_package.pth
>
> | branch | contents | MB |
> |---|---|---|
> | skeleton | 20 members, 5 archs × 4 folds, verbatim from `model_astgcn_world25_int8.pth` | 22.80 |
> | video | `distil_oracle_all` + `k224_mvitwrist_all`, symmetric int8 per-output-channel | 69.61 |
> | **file on disk** | | **93.37 / 100** |
>
> **Verified four ways** — size by `ls -la` not arithmetic; 1,598 tensors against manifest
> SHA-256 with **0 mismatches**; dequantised weights **bit-identical** to
> `quantize_checkpoint.py --bits 8`; and models rebuilt **from the package** re-run on the
> test set. The fused, decoded output differs from the configuration that scored **165**
> on **2 of 405 rows**.
>
> Build: `python3 code/pack_stage2.py --bits 8`
> Verify: `python3 code/unpack_stage2.py --check integrity,weights,infer`
> Reproduce the submission: `python3 code/fuse_from_package.py` → decode → `sub_pkg_v2.csv`
>
> **Two corrections this produced.**
> 1. **int6 is strictly dominated and should stop being quoted.** EXP-107/108's int6
>    operating point assumed bit-packing that was never implemented; 6-bit codes in int8
>    containers cost the same bytes as int8 and carry more error (student 394/405 vs
>    **403/405** against fp32; wrist 395 vs **404**). The package ships int8 at the
>    identical 34.28 MB per view.
> 2. **`w25_p4` cannot be regenerated** — `prune_world25.py`'s invocation was never
>    recorded, and the closest reconstruction differs on 14 of 405 rows. The package
>    therefore declares its own `skeleton_spec` in the manifest and
>    `fuse_from_package.py` reads it back out, so the submission is regenerable from the
>    shipped artifact alone. Stage 2 is a reproduction stage; do not create another
>    artifact whose recipe lives only in a shell history.
>
> **Still open:** skeleton members are copied verbatim and were not re-run from the
> package (their source manifest carries per-member argmax verification, so they are
> trustworthy but not re-verified here), and the package's score is **predicted 165 ± 2,
> not measured**, until `sub_pkg_v2.csv` is submitted.

**Best score: `submissions/sub_r2.csv` = 0.82587 = 166/201.** Its *legality is unverified*
— the package has never been built as a file (see the retraction above).

| | |
|---|---|
| Best score (verified on Kaggle) | **0.83084 = 167/201** — `submissions/sub_r2_dist.csv` (EXP-125). **Package NOT built or verified** |
| Target | 0.89 = 179/201 → **+13 clips from the legal 166** |
| Qualification gate | **top-15 on private.** Bar = **162** clips (2026-08-25, 233 teams); legal score 166 — **margin +4** |
| Deadline | Kaggle 2026-09-15; code upload 09-22 |
| Noise floor | **±6 clips.** A change moving <20 of 405 rows cannot be read |
| Screening estimator | **SPLIT — see B-032.** Pooled OOF predicts *combination/inference* changes (2-for-2) and **cannot predict member-strength changes (0-for-2)**. 288 px cleared the bar 4/4, sd 0.79, and still lost 2 clips on public. Member hypotheses cost one submission each to test |
| Champion recipe | `code/build_video_slot.py --tag n8 --view mvit=k224_mvit_f0,k224_mvit_f1,k224_mvit_f2,k224_mvit_f3` then the transition decoder λ=0.5 conditional unigram. **Reproduces the champion to 0.0** |

> ### ⚠ THE PACKAGING BLOCKER IS SOLVED — see EXP-105. Do not re-plan it.
> `world25` 84.54 → **22.80 MB** costs **2 of 405 rows**. `imu_stats` 87.64 → **9.00 MB**
> costs **6 clips of 2,700**. MotionBERT drops for **9 rows**. A legal package is
> **96.0 MB** with both video views, **66.1 MB** with one. What remains is to *build and
> verify* it, not to find it.

> ### ⚠ THREE AXES ARE MEASURED FLAT. DO NOT RE-OPEN THEM.
> Fusion weights, decoder λ, and video-bag composition (157–162 across every ≥5-member
> bag ≈ one SD). Gain must come from a **stronger or genuinely new member**, or from
> inference-time adaptation — not from recombining what we have.

---

### ⓪ THE STRATEGIC QUESTION: what the legitimate top tier is doing that we are not   ← **START HERE**

**Leaderboard, pulled from the Kaggle API 2026-08-25 (233 teams):**

| rank | score | clips | read |
|---|---|---|---|
| 1–3 | 0.98009, 0.98009, 0.97512 | 197, 197, 196 | **almost certainly the L-1 leak.** 196/201 is not a modelling result, and there is a 6.5-point cliff below them |
| 4–7 | 0.91542, 0.91044, 0.90049, 0.89552 | 184, 183, 181, 180 | the real top tier — a tight cluster is the signature of a method |
| 8–10 | 0.86567, 0.85572, 0.83582 | 174, 172, 168 | |
| **11–13 (us)** | **0.82587** | **166** | `sub_r2`; package unverified |
| 15 | 0.80597 | 162 | **the qualification bar moved 160 → 162; margin is +4, not +6** |

Published public notebooks sit at **143**. We reached 166 by adding person crops, a
Kinetics video backbone, 224 px, MViT, skeleton/IMU fusion and the transition decoder.
**Reaching 184 needs +18 — more than everything above added together.** It will not come
from recipe tweaks, and every recombination axis is now measured flat.

**Ranked candidates for the missing mechanism.** The first two are the only ones sized
right for +18; both are explicitly legal and both are untested.

**1. Distil from a large teacher — and note there is a Large Model Track on the same
data with no size limit** (150 teams, same deadline). R-3 permits distillation from
larger models. A team in both tracks builds its best unconstrained model and distils it
into ≤100 MB. **We have never built a large teacher**, and we have direct proof there is
none to distil from: our ~309 MB pipeline and our 83.82 MB package both score exactly 166.
Our whole campaign was size-constrained from day one, which may have been the strategic
error.

**2. In-domain pretraining on external depth/IR data.** R-2 permits external public
datasets. Our backbone is Kinetics-400 — *RGB internet video* — bridged to IR +
colormapped depth by 2,281 clips from 14 subjects. **NTU RGB+D 120** has 114,480 clips,
**IR and depth streams**, 106 subjects, and heavy class overlap (drink, eat, read, write,
type, phone). **The repo's NTU rejection was about skeleton GCN architectures**
(CTR-GCN −0.74, dual-frame −1.63) — video pretraining on NTU's depth/IR streams is a
different experiment and has never been run. Blocker: disk and dataset registration.

**3. A real fine-tuning recipe.** Ours is light for a 34M video transformer on 2,281
clips: 20 epochs, uniform lr 1e-4, crop-scale 0.75–1.0 + hflip, label smoothing 0.1, EMA.
Missing **layer-wise LR decay** (the big one for fine-tuning pretrained transformers on
small sets), mixup/cutmix, RandAugment, repeated augmentation, 30–100 epochs. We also
evaluate with 1 crop + hflip where the standard video protocol is 3 crops × N clips
(+1–2% in the literature). Cheapest real test: ~6 GPU-h at 4 folds, no downloads.

**4. Feature-level thermal.** Thermal is the paper's **best** modality (92.57) and is
uniquely correct on **34 of 552** champion errors — ~12 public clips of *measured*
headroom — but no global weight can harvest it because it is uncalibrated. **B-027
refutes probability-space fusion only**; B-028 says feature-space transfers where
probability-space fitting does not. A two-stream model fusing before the classifier needs
**no pixel alignment**, which was the stated reason for killing this earlier and was wrong.

**5. Person-crop identity.** `compute_windows()` takes the highest-confidence person box
**independently per probe frame** and then the **union** of those boxes — no tracking.
Measured 2026-08-25 on test: **30.4% of clips contain >1 person**, 3.5% show an identity
switch between probe frames (min consecutive-pick IoU < 0.3), 9.6% get a >2× inflated
union crop. The repo records test at 2× train's multi-person rate, so this is a
**train/test asymmetry OOF structurally cannot see** — and it disproportionately affects
the on-site test, which is 30% of the grade against the leaderboard's 20%.

**6. Per-cohort/subject self-training.** Q-91, queued and never run. R-4 explicitly legal.
*Naive* self-training failed; per-subject was never tried.

**Budget reality:** seed σ on this partition is **2.80**, so a single-fold experiment is
uninformative and an honest screen costs ~6 GPU-h. That caps us at a handful of properly
tested hypotheses before 2026-09-15.

### ⓪b Temporal jitter — CLOSED, dead at 4 folds

| fold | 0 | 1 | 2 | 3 | mean |
|---|---|---|---|---|---|
| Δ micro | −0.61 | −0.98 | **+2.45** | −0.31 | **+0.14** |

1 of 4 folds positive. Checkpoints deleted. The 32-frame caches stay — temporal *TTA* is
a separate result and it survived.

> **RULE:** a member-level change is not a result until it **exceeds 2.80 micro on a
> single fold**, or is **positive on ≥3 folds**. Cost so far: EXP-100 start-weight, n7,
> jitter.

### ⓪c Person-crop identity — REFUTED, and the direction matters

Predicted that the untracked union-of-per-frame-boxes crop damages test more than train,
since test has 2× the multi-person rate and OOF cannot see it. Measured
(`code/probe_crop_identity.py`, ~25 CPU-min, no GPU):

| | train (2,905) | test (395) |
|---|---|---|
| >1 person in some probe frame | 17.5% | **30.4%** |
| union inflates >2× | **14.9%** | 9.6% |
| identity switch (IoU < 0.3) | **5.6%** | 3.5% |

**Test has double the multi-person rate and cleaner crops on every measure** — likely
because test clips are shorter (median 20 raw frames vs 24), leaving less time to drift.
Candidate 5 is dead as a test-specific fix. Filed.

### ⓪ THERMAL IS UNDERDEVELOPED — the strongest new lead   ← **START HERE after T0**

`skomuro`, **tied with us at 0.82587 = 166**, published a notebook titled *"From 14th
Place to 0.8+: A Leakage-Safe **Thermal** Baseline"*. Their recipe uses **thermal only,
FULL FRAME, no person crop**, 8 frames at 112 px, a tiny from-scratch 2D CNN with frame
logits averaged.

Ours: `pre_thermal` pooled OOF **0.36277**; EXP-088's video-recipe thermal **0.54448**
(fold 2) — built on a **YOLO person crop**.

**Hypothesis:** in thermal, the discriminative cue for an OBJECT class is the *object's
own heat signature* (kettle, laptop, stove, running tap), not the subject's pose — and
**cropping to the person deletes it**. 75% of our residual error is OBJECT classes, and
thermal is the paper's best modality (92.57) while contributing exactly zero to our fusion.

This re-frames B-027. We measured that no global weight can harvest thermal and concluded
thermal was uninformative. The untested alternative is that **our thermal member is
crippled by preprocessing** and a competent one would fuse fine.

**Experiment:** build a full-frame thermal cache (no YOLO), train the same recipe against
the cropped cache, same folds and seeds. Adopt only on >2.80 micro on one fold or ≥3
positive folds. If it wins, re-test B-027 against the repaired member.

**Caveat:** their notebook has **no test inference** — the "0.8+" is prose, and their LB
is 166, the same as ours. This is evidence thermal can carry a pipeline, not proof that
full-frame beats cropped.

**Also from the mining:** 8 public notebooks exist, the best claims **LB 0.711 = 143**
(already ported and passed in EXP-086). The 176–188 teams have published nothing.

### ① Stage-2 package — **NOT DONE. Retracted 2026-09-01; see the banner above**

`submissions/sub_r2.csv` = **0.82587 = 166/201 from 83.82 MB**, the same score as the
~309 MB pipeline it replaces. 16.18 MB headroom.

| component | MB | measured cost |
|---|---|---|
| MViT person all-train, int6 | 26.01 | int6 == int8 accuracy (0.71319 both) |
| MViT wrist all-train, int6 | 26.01 | |
| `world25` pruned, 5 archs × 4 folds | 22.80 | 2 of 405 rows |
| `imu_stats` ExtraTrees 200 trees / depth 12 | 9.00 | −6 clips / 2,700 pooled |
| **total** | **83.82** | |

Rebuild in one command; verified byte-identical after the 2026-08-25 disk cleanup:
```bash
python3 code/build_video_slot.py --tag r2 --skel w25_p4 --imu imu_stats_t200_d12 \
  --no-motionbert --view person=k224_mvit_all_q6:0.5 --view wrist=k224_mvitwrist_all_q6:0.5
```
(`k224_*_q6` come from `code/quantize_checkpoint.py --bits 6`, then `--stage infer`.)

**`imu_stats` cannot be dropped** — removing it moves 43 of 405 rows. It was invisible as
a packaging cost because it is refit at inference and never written to disk.

### ② Per-subject AdaBN — the sharpest remaining inference-time lever

AdaBN pooled gave +2 public. Per-subject measured **0.70092 vs pooled 0.69172** on
fold 2 — **1.6× the gain**, worth roughly +3 public clips if subjects can be recovered.
Blocker (EXP-099): 3-minute timestamp blocks are 100% subject-pure over 231 train blocks
but hold a median of 10 clips and score 0.67945, *worse than pooled*. Groups must be
pure **and** large — so cluster the blocks into subjects using each block's own mean BN
feature statistics, and validate the clustering against true user labels on train first.

**Note this applies only to the CNN members.** MViTv2-S and Swin3D use LayerNorm and
have no BatchNorm to re-estimate, so as the video slot moves to MViT, AdaBN's reach
shrinks. Quantify what AdaBN is still worth in the current champion before investing.

### ⑤ External data (NTU RGB+D) — ❌ CLOSED, cannot be obtained

NTU RGB+D requires a **supervisor's countersignature** on the access agreement. Dead, not
deferred. For the record, the ROSE listing that is NTU RGB+D is **"Action Recognition
Dataset"** (60: 56,880 samples; 120: 114,480; masked depth 83/147 GB, IR 221/389 GB).

**Do not source it from an unofficial mirror** — signed-agreement licence, plus the
competition requires disclosing external data. Other ungated depth/IR corpora do not
substitute: NTU's value was **106 subjects vs our 18**, and UTD-MHAD (8 subjects) or
CAD-60 cannot move cross-subject generalisation.

Effort redirects to self-supervised pretraining on our own 3,338 clips (train + test,
unlabeled) — in-domain by construction, ungated, legal under R-4/R-7, and EXP-021 already
measured **+3.0** for a cross-modal masked pretext that was never re-attached to a trunk.

### ③ Thermal — the WITHDRAWAL WAS WRONG, see candidate 4 above

Earlier versions of this file killed thermal fusion because thermal is a separate camera
with no calibration to IR/depth, so a 7-channel tensor would not be pixel-aligned. **That
argument only rules out channel-stacking.** A two-stream model — one trunk on the IR/depth
crop, one on the thermal crop, fused at the feature level before the classifier — needs no
pixel alignment at all.

B-027 (thermal contributes zero) is a measured statement about **probability-space**
fusion, where a single global weight cannot separate thermal's 34 unique-correct clips
from its 247 errors because its confidence when right (0.511) barely exceeds its
confidence when wrong (0.413). B-028 says the opposite holds in feature space. Nothing
has tested a jointly-trained thermal + IR/depth model.

`cache/thermal_v1` is retained on disk for this.

---

## Disk hygiene (cleaned 2026-08-25: 25 GB → 72 GB free)

Removed 12 cache directories (35 GB) and 334 checkpoints (12.4 GB) belonging to refuted
or superseded families. **`sub_r2`'s fusion was verified byte-identical (max abs diff 0.0)
after the deletion, and `prune_world25.py` still runs.**

**Kept, and why — do not delete these:**

| path | GB | why |
|---|---|---|
| `cache/train`, `cache/test` | 1.4 | per-clip npz; `imu_stats` refits from them at inference |
| `cache/crop_224`, `cache/crop_wrist224` | 2.3 | the two video views in the package |
| `cache/crop_224_t32`, `cache/crop_wrist224_t32` | 4.5 | 32-frame caches; temporal TTA rides free on these |
| `cache/thermal_v1` | 2.5 | retained for the two-stream thermal experiment (③) |
| `checkpoints/` 58 files | 1.6 | 48 `world25` sources + 2 package models + 8 fold instruments |
| `third_party/` | 0.5 | pretrained weights that may not be re-downloadable |

Everything deleted is rebuildable from `Small-Model-Track/` raw data. The 48 `world25`
source checkpoints are kept **only** so the package can be re-quantized at a different bit
width — pruning itself is offline via `research/artifacts/world25_per_member.npz`.

---

## Do NOT do these

| direction | why it is closed | evidence |
|---|---|---|
| **Exploit the test-label leak** | Confirmed (discussion 714827). Cheating; organisers run anti-cheat; Stage 3 is on-site with 8 new subjects. **Permanently closed.** | `RULES_VERIFIED.md` L-1 |
| **Obtain any subject not already in training** | All 30 accounted for: 18 train (1–9, 16–24), 4 public test (10, 11, 25, 26), 8 private. Any unseen subject **is** a test subject. | `RULES_VERIFIED.md` |
| Re-tune fusion weights | Re-measured on the IG-65M-era members: grid optimum +13 on 2,700 = **+1 public clip**, selection-biased. Flat for the new members too. | EXP-098 |
| Add a 12th video member / re-shuffle the bag | Every ≥5-member bag scores 157–162 ≈ one SD. Saturated. | EXP-095 |
| Add *weak* members to the bag | `m15` (15 members) = 160, **−2**. The slot is an equal-weight log-mean; weak members dilute. | EXP-096 |
| Swap a stronger member in for a weaker one | `g4only` (IG-65M replacing K400) = 157 = `bag4_prior25`. Additions pay, replacements do not. Confirmed 3×. | EXP-091 |
| Screen bag members on solo score | `ig65m_upper` is solo-null (identical 289/479) and the best bag member measured (+2.76). | EXP-093 |
| Session-scale distinctness | **False.** 1,937 duplicate-label clips in 125 of 147 train sessions (trials repeat each activity). | EXP-097 |
| Merge recording groups into longer chains | Cross-group transition top-1 0.340 **< 0.392 unigram baseline**. The chain does not cross boundaries. | EXP-097 |
| **`--start-weight` > 0** | **REFUTED on public: −8 clips** (154 vs 162), despite +14 on 2,700 OOF. Test groups are *fragments* — singletons 19.6% vs train's 5.4%, mean size 2.83 vs 3.72 — so a first-of-pass prior lands on mid-pass clips. Leave it at 0.0. | EXP-100, B-029 |
| Any lever keyed on group position/length without checking the test group-size histogram first | Same defect as above. OOF measures it on train-shaped groups and can be wrong in **sign**. | B-029 |
| Decoder λ > 0.5 | OOF λ=1.0 → +0.006 (231 rescues, 215 harms). `vidimu_C_trans10.csv` **retracted**. | EXP-086 |
| Thermal as a *late-fusion* member | Strong (0.544) and maximally decorrelated, adds **exactly zero** at every weight. Confidence when right ≈ when wrong. (Early fusion is item ③ and is NOT closed.) | EXP-088, B-027 |
| Fitted stackers, learned gates, cohort weights, temperature calibration | Six consecutive fitted-combination levers landed ≤0 on public despite large OOF gains. | LOG, `BELIEFS.md` |
| Sinkhorn / prior-forcing on test | −12 public. Re-derived once by mistake. | `BELIEFS.md` |
| ~~Resolution 160px~~ | **RETRACTED (EXP-101).** 1.25× step against a measured 3.1× downsample, on a backbone pretrained at 112. Resolution is item ⓪, not a closed axis. | EXP-101 |
| 32 frames · all-18 data volume | +1.38 ns · retracted (−3, not significant) | EXP-093 |

---

## Running right now

**`code/run_288_all.sh`** — `k224_mvit288_all`, the shippable 288 px person model
(log `logs/px288_all.log`, ~4.3 h, under the watchdog). When it lands, rebuild the
package with the 288 person view replacing the 224 one and submit.

## RESOLUTION IS THE ONLY LIVE MEMBER AXIS — and it passed

**288 px: mean +1.27 micro, sd 0.79, 4/4 folds positive** (object +1.60, 4/4). The first
change to clear the adoption bar since 224 px itself. Fused: **+36 clips/2,700 = +2.7
public**, projecting `sub_r2` 166 → **~169**.

The `sd` is the tell: jitter sd 1.57 (1/4 positive) and LLRD sd 1.79 (2/4) each threw one
spurious +2.45. 288 moves every fold the same way with half the scatter.

| video slot (pooled 2,700) | micro | Δ public |
|---|---|---|
| 224 person + 224 wrist (`sub_r2` = 166) | 0.75630 | — |
| **288 person + 224 wrist** | **0.76963** | **+2.7** |
| 288 + 224 wrist + 224 person (3 members) | 0.76148 | +1.0 |

Keeping the 224 person as a *third* member is worse than replacing it — EXP-109's
saturation again: the slot rewards a **stronger** member, never an **additional** one.

**Wrist stays at 224.** Person crops are median 416 px so 288 still recovers real pixels;
wrist crops are median 147 px and already **1.52× UPsampled** at 224.

### Next: push resolution further with gradient checkpointing

**320 px at batch 2 OOMs on the 8 GB card** (measured; 288/bs2 peaks at 5.15 GiB,
528 ms/step). So the cap is VRAM, not method. `--grad-checkpoint` is now implemented:
it rebinds the MViT instance's `forward` to recompute each block in backward, ~30% slower
per step. **Verified state_dict keys are unchanged**, so every existing checkpoint loads,
and the eval path is untouched.

**Do this when the GPU frees:** probe memory at 352 / 384 / 416 with `--grad-checkpoint`,
then screen the largest that fits on 4 folds. Person crops are median **416 px**, so 416
is the natural endpoint of this axis — the first resolution that discards nothing.

**Honest projection:** 288 → ~169. Another resolution step of similar size → ~171–172.
Temporal TTA adds +0.7. **0.86 = 173 is reachable but not assured by this path alone.**

**Packaging tools built in EXP-105:**
```bash
python3 code/infer_packaged.py <pkg.pth> out.csv --per-member-output research/artifacts/world25_per_member.npz
python3 code/prune_world25.py --tag w25_p4 --merge "skel_jvb_big=skel_jvb_big_s1,...,s4" \
        --merge "stgcn_v1=stgcn_s1" --drop skel_multitcn,stgcn_w96
python3 code/imu_stats_size_sweep.py     # accuracy vs serialized MB
```


## Numbers, with provenance

| what | value | kind |
|---|---|---|
| Public best (`sub_n8`, MViT x4 alone) | **0.82587 = 166/201** | **external oracle** (Kaggle) |
| Top-15 qualification bar (224 scored teams) | 160 clips — margin **+6** | **external oracle** |
| Legal Stage-2 package | **96.0 MB** both views / 66.1 MB one | measured (EXP-105) |
| `world25` prune 84.54 -> 22.80 MB | 2 of 405 rows | measured |
| `imu_stats` prune 87.64 -> 9.00 MB | -6 clips / 2,700 pooled | measured |
| Public noise floor | ±6 clips | measured |
| Pooled video OOF, K400 4-fold | 0.64371 | first-run |
| Pooled video OOF, IG-65M 4-fold | **0.68735** (p=1.04e-08 vs K400) | first-run |
| Video bag8 solo, pooled OOF | 0.6981 | first-run |
| Champion fusion OOF, pre-decoder | 0.72926 (1969/2700) | first-run |
| Champion fusion OOF, decoded | 0.77593 (2095/2700) | first-run |
| …with `--start-weight 0.5` | 0.78111 (2109/2700) | first-run |
| Any-branch-correct oracle | 0.8078 → gap **+0.0785** | first-run |
| Weight-grid optimum | 1982 vs 1969 = +13/2700 | **fitted** — argmax of ~150 configs |
| AdaBN on `vid_ig65m_f2` | 0.67638 → **0.69172** | first-run, held-out subjects |
| Fusion weights | base .35 / vid .2925 / imu .3575 | **fitted**, confirmed by public ordering |

---

## History

**2026-08-20.** IG-65M 4 folds complete; `h8all` 162. Composition, weight, and decoder
axes all measured flat. Session-structure audit killed session-distinctness and
cross-group chaining, found `--start-weight` unused. **AdaBN found: +1.53 micro, free.**
Ledgers reconciled after a four-day gap.

**2026-08-19.** IG-65M R(2+1)D-34 adopted (+4.4 pooled, p=1e-08); midplane bug caught
before execution. MotionBERT member added (world z-up coordinate fix). 156 → 161.

**2026-08-16.** 4 K400 video folds; bag → 156. Thermal built and refuted as a late-fusion
member. Decoder tuned on pooled 2,700-clip OOF. Weight axis closed (first time).

**2026-08-15.** Person-crop + Kinetics R(2+1)D landed: fold-2 micro 0.63957 vs 0.40031,
**+23.9**. Fusion 131 → 151. `build_model` 4-channel bug cost ~1 GPU-hour.

**2026-08-10.** `RULES_VERIFIED.md` R-1: **pretrained CNNs were legal all along.** A prior
session recorded the opposite as fact and cancelled the pretrained probe.

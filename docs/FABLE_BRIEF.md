# BRIEF FOR FABLE — plan the run to 0.93+ in 7 days

**Written 2026-09-08 by Opus 5.** You (Fable) plan. Opus 5 executes. Atharv decides.
Everything below is *measured* unless marked as a hypothesis, and every number has an
experiment id you can look up in `research/LOG.md`.

---

## 1. The mission, in exact clips

| | |
|---|---|
| now | **167 / 201 = 0.83084**, rank **14** (tied 15th) |
| target | **187–188 / 201 = 0.930–0.935** → **+20 to +21 clips** |
| Kaggle deadline | **2026-09-15 — 7 days** |
| code upload | 2026-09-22 — 14 days |
| submissions | ~5/day, so ~35 remain. Not the binding constraint |
| compute | a **SLURM cluster** just came up (§4). Kaggle's weekly GPU quota is exhausted |

**Grade weighting matters more than the leaderboard:** Kaggle private 20% · **on-site test
with 8 NEW subjects 30%** · reproducibility 10% · report 20% · presentation 10% ·
efficiency 10%. Cross-subject robustness is worth **2.5× the public leaderboard.**

---

## 2. Why 0.93 is provably reachable — do not treat it as a ceiling

Two independent facts, both verified today:

**(a) A legitimate team is already there.** Live board, 2026-09-08: **0.93532 = 188/201**
at rank 5, submitted today; then 184, 184, 181, 180, 176, 173. **Eight teams at 0.90+.**
The 192–198 cluster at the top is the likely L-1 leak, but 173–188 is a populated tier
built by ordinary means.

**(b) Our own ceiling lands on the same number.** The champion's **top-2 accuracy is
0.86850** on 2,700 honest OOF clips. OOF has run **+6.79 points below public** on this
pipeline (0.76296 → 0.83084). That projects our top-2 to **≈0.9364 = 188/201 — exactly the
best legitimate score on the board.**

**Read those together.** The rank-5 team is not using a mechanism we cannot conceive. They
are *resolving the binary decision we already surface and get wrong.* **285 of our 640
errors have the true label at exactly rank 2.** Fixing only that is +10.55 points OOF ≈
+21 public clips — the entire gap, from one decision.

This repo's standing directive is explicit: *"Our own failure is not a ceiling."* A prior
session asserted "0.65–0.73 achievable" and a live leaderboard showed 0.91542. Do not
repeat that error in either direction: the target is reachable, **and** the routes below
are measured dead.

---

## 3. Where the error actually is (EXP-124, measured on 2,700 pooled OOF clips)

| | |
|---|---|
| top-1 | 0.76296 |
| **top-2** | **0.86850** |
| top-3 | 0.90667 |
| top-5 | 0.93926 |
| oracle over all 5 members | 0.87741 |
| **all five members wrong** | **331 clips (12.3%)** — outside the top-2 ceiling entirely |

- **44.5% of errors are at rank 2.** 60.6% within rank 3.
- **The oracle gap IS the rank-2 gap** — oracle-any-member 0.8774 ≈ top-2 0.8685. There is
  no "which member to trust" problem. There is one binary decision.
- **Errors spread over 175 confusion pairs**; top-10 pairs = 32.3%, top-20 = 45.5%. Biggest
  single pair `21 Read_documents ↔ 22 Turn_pages` = 43 errors (6.7%). **Per-pair
  specialists are the wrong shape** — one *pair-conditioned* model sees all 640.
- **All twelve worst classes are OBJECT** (hand–object interaction). OBJECT = 31 classes,
  `set(range(28)) | {37,38,39}`, and holds ~75% of residual error.
- **Cross-subject variance dominates everything:** between-subject sd **5.26 points**
  (best user 0.8172, worst 0.6515) against a seed σ of **1.16**. Public is 4 subjects, so
  SE = 2.63 → the ±9–10 clip noise floor is *subject sampling* and cannot be reduced by
  more seeds or more members.

---

## 4. Compute now available

`ssh sharanga`, `/home` is Lustre with 116 TB free. `cluster/` holds `sync.sh`, `env.sh`,
`parity.sbatch`, `skel_retrain.sbatch` and a bring-up README.

| partition | GPUs/node | time limit |
|---|---|---|
| `gpu_h200_8` | 8× H200 NVL | 1 day |
| `gpu_h100_4` | 4× H100 80GB (×2 nodes) | 3 days |
| `gpu_a100_8` | 8× A100 80GB | 5 days |
| `gpu_rtx_pro_6000_6_c` | 6× RTX PRO 6000 | 2 days |
| `gpu_v100_2` | 2× V100 32GB (×2) | 3 days |

**Parallelism = `sbatch --array` over (tag × fold), one GPU per task.** The repo has zero
`torch.distributed`; skeleton/IMU members are 0.6–2.5M params and the whole 48-member
stack retrains in under a GPU-hour. Do not plan DDP into a 7-day window.

**A 224 px MViT fold is ~2 h on a T4.** On an H100 assume well under an hour, so an
18-fold LOSO × 3 seeds (54 runs) is now a single array job — it was 76 h on the laptop and
is the reason four experiments were decided on noise.

**Known cluster gotchas** (`cluster/README.md`): `research/artifacts/results.csv` has no
file locking and a 48-task array will corrupt it — use the per-run JSON manifests. Keep
`persistent_workers=False` in `code/train.py` (`epoch_seed` mutates per epoch).

---

## 5. THE GRAVEYARD — measured flat. Do not propose these.

Each cost real time. Re-proposing any of them without *new* evidence is the main way this
plan can waste the week.

| axis | evidence |
|---|---|
| fusion weights | 5,227-point sweep, +0.4 total |
| decoder λ | flat |
| video bag composition | 4 person folds = 1 person + 1 wrist = **166**; 4-fold bag (137 MB) = single all-train (34 MB) |
| adding video members | two attempts on 2026-09-04 each **lost** clips |
| thermal late fusion | contributes **exactly 0.00** at every weight 0.05–0.35 |
| full-frame vs cropped thermal | 4 paired folds, **+0.83, 0.96 SE** — not adopted (EXP-120b) |
| temporal jitter | +0.14, 1 of 4 folds |
| LLRD | +0.245 |
| 288 px / 384 px | cleared the local bar 4/4 and **lost 2 clips on public** |
| VideoMAE-B (86.7M) | **lost 0/4 folds** to MViTv2-S (34M). Bigger models overfit at 2,281 clips |
| seed soup | video slot saturated; also 3× the bytes under R-6 |
| **probability-space arbitration** | **five failures**: GBDT stacker, structure decoder, cohort weights, learned gate, and EXP-131 below |

---

## 6. THE FIVE NEGATIVE RESULTS THAT SHAPE THE SOLUTION

**(1) EXP-131, 2026-09-08 — the rank-2 decision is NOT in the posteriors.** A
subject-grouped logistic probe over 16 features (fused p1/p2/margin/log-ratio/entropy, plus
each member's log-odds and vote on the contested pair), on the 2,345 clips whose truth is
in the top-2:

| | |
|---|---|
| base rate (always keep rank-1) | **0.8785** |
| learned probe | **0.8759** — *below base rate* |
| swaps | 42 rescues, 48 broken, **net −6** |

**Any re-ranker built on member outputs will fail.** The posteriors are a compressed
summary that already discarded the discriminating detail. The route to the 285 clips must
consume **raw input conditioned on the candidate pair**. This is B-028: feature space
transfers where probability-space fitting does not.

**(2) EXP-128 — accuracy is not what a fusion slot buys.** Replacing the sklearn
`imu_stats` (accuracy **0.3605**) with the distilled student (**~0.73**) at the same 0.3575
weight **lost 4 clips**. That member earns the heaviest slot through *error placement* —
its errors sit where it is reliably unconfident. **Do not swap it out.**

**(3) EXP-123 — but thermal's problem IS accuracy.** Thermal has the *highest* confidence
separation of any member (+0.0650; the best-fusing member's is −0.0202), yet contributes
zero: 19 rescues against 247 errors, so any weight that harvests the 19 imports from a pool
13× larger. **Fusion value is neither accuracy nor decorrelation — it is where a member's
confidence sits relative to its errors, and it must be measured per member.**

**(4) EXP-127/128 — distillation makes its own sources redundant.** A person-crop-distilled
student made the person view redundant: dropping that view **gained 2 clips**. Leaving both
in is a measured loss.

**(5) EXP-120a — the single-fold bar was too loose.** Visual seed σ = **1.16**, so a paired
delta carries σ√2 = 1.64: the 2-SE bar is **3.28 on one fold**, **1.64 on a 4-fold mean**.
The old *"≥3 of 4 folds positive"* clause is retired (p = 0.31 under a coin-flip null).

---

## 7. STRUCTURAL FACTS — raw material for aggressive ideas

These are measured properties of the data, not proposals. Several are unexploited.

- **Test clips carry timestamps.** `cache/meta_test.csv` has `t0`/`t1` (no user, no
  station). On train, blocks at a **60 s gap are 100% single-user** (740/740), **93.4%**
  have all-distinct labels, and consecutive clips share a class **0.1%** of the time. Test
  splits into ~24 blocks at a 300 s gap against **12 expected test subjects** (4 public +
  8 private). **Recovering test subject groups looks feasible and is legal** — R-4 permits
  unlabeled test use and R-7 rules transduction legal. This attacks the *dominant* variance
  (subject sd 5.26) and has never been tried beyond the transition decoder.
- **50.9% of raw frames are discarded** at 16 frames/clip. `CLAUDE.md` calls this the one
  axis with genuinely unused information. `--frames 32` exists in the trainer.
- **Cross-modal masked pretraining measured +3.0 (EXP-021) and was never re-attached to a
  CNN trunk.** Legal on all 3,338 clips including unlabeled test.
- **A distilled student works:** oracle-target distillation gave **+4.91 over a leak-free
  control** on fold 2 (0.71012 → 0.75920), +21 OBJECT clips. Teacher accuracy is capped at
  **0.87741** (oracle-any-member) and we are already there — the teacher axis is exhausted.
- **The 12.3% where every member fails** (331 clips) cannot be reached by any re-ranking or
  fusion change. A genuinely new input representation is the only route to those.
- **Thermal is the dataset paper's best sensor (92.57) and our member scores 0.544.** Five
  of six differences from a tied team's thermal recipe are untested (112 px, 8 frames,
  from-scratch 2D net with frame-logit averaging, 8 epochs, GroupKFold-5); only the crop
  was tested and it was not the defect.

---

## 8. HARD CONSTRAINTS — a plan that violates these is unusable

- **R-6: every weight loaded at inference, ensemble members included, must fit 100 MB in
  ONE file.** We have a verified **93.37 MB** package (`research/artifacts/stage2_package.pth`,
  EXP-129) whose configuration scored **165**. Build with `code/pack_stage2.py`, verify with
  `code/unpack_stage2.py --check integrity,weights,infer`.
- **L-1: a test-label leak exists and MUST NOT be used.** Permanently closed. All 30
  subjects are accounted for — 18 train, 4 public, 8 private. **Any subject not already in
  training IS a test subject.** Do not propose obtaining one. Do not propose the
  HuggingFace mirror.
- **External data must be disclosed, and NTU RGB+D is closed** — it needs a supervisor's
  countersignature Atharv cannot provide. Do not propose an unofficial mirror.
- **R-1: pretrained CNNs ARE legal** (ImageNet/Kinetics). A prior session recorded the
  opposite and cancelled months of work; check `research/RULES_VERIFIED.md` before assuming
  a restriction.
- **R-3 permits distillation from larger models.** There is a Large Model Track on the same
  data with no size cap.
- **One change per submission.** Adopt on **mean vs 2 SE**, never a fold sign count.
- **A candidate changing <20 of 405 rows cannot be read on public** — build it only if free.

---

## 9. WHAT WE HAVE BUILT (assets you can plan against)

- `kaggle/cuhkx_224_kaggle.py` — the video trainer. Supports `--seed`, `--teacher`,
  `--distill-alpha/-temp`, `--cache-dir`, `--frames`, `--image-size`, `--llrd`,
  `--grad-checkpoint`, arch ∈ {mvit_v2_s, videomae_b, swin3d_t/s, s3d}.
- `code/build_teacher_targets.py` — fused (0.763) and oracle (0.880) soft targets.
- `code/pack_stage2.py` / `unpack_stage2.py` / `fuse_from_package.py` — the R-6 package,
  verified four ways.
- `code/ordered_transition_decoder.py` — Markov beam decode over recording order with
  `--distinctness none|hard|penalty`. **Distinctness is now ON and worth +1 clip (EXP-125).**
- `code/build_video_slot.py` — reproduces the champion to **max abs diff 0.0**.
- Caches: `crop_224`, `crop_wrist224`, `crop_288`, `crop_384`, `thermal_full`, `thermal_v1`,
  plus 32-frame variants. All rebuildable on the cluster's 96-core nodes.
- ~200 member `testprobs_*.npz` / `oof_*.npz` artifacts.

---

## 10. WHAT TO DELIVER

A plan, not a menu. Specifically:

1. **A ranked programme for 7 days** with each track's *mechanism*, its predicted effect in
   **clips**, its falsification test, and its cost in GPU-hours on the partitions in §4.
2. **An explicit answer to the central question:** what consumes raw input conditioned on a
   candidate pair, and how is it trained and validated in 7 days? If you think the rank-2
   framing is wrong, say so and give the alternative that reaches +20 clips.
3. **A parallelisation map** — what runs concurrently on which partition. Serial plans waste
   the cluster.
4. **A measurement plan.** LOSO (18 folds × 3 seeds) is now one array job and would replace
   an inherited σ that has already produced three spurious "+2.45"s. Say whether it earns
   its place on day 1 or whether the 7-day clock forbids it.
5. **A decision rule for the final submission**, weighting private (8 subjects) and on-site
   (8 subjects, 30%) over public (4 subjects, ±9–10 clip noise).
6. **Explicit risk handling for T-PKG.** We have a 93.37 MB package scoring 165. Any plan
   that raises public score must say how the winning configuration gets *into* that file
   before 2026-09-22, or it forfeits everything.

**Be aggressive.** Incrementalism cannot close 20 clips — every incremental axis is in §5
and measured flat. But ground each idea in §3/§6/§7: an idea that contradicts a measured
result needs to say which measurement it disputes and how it would be re-tested.

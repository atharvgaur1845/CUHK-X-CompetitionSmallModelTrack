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

> # ✅ THE PACKAGING PROBLEM IS CLOSED. The legal package matches the best pipeline.
>
> **`submissions/sub_r2.csv` = 0.82587 = 166/201 from an 83.82 MB package** — the same
> score as the ~309 MB `sub_n8`, with 16.18 MB of headroom. R-6 (organiser, topic
> 729056) caps every weight loaded at inference, ensemble members included, at 100 MB in
> one file; we are comfortably inside it and lose nothing.
>
> Still true and still load-bearing: the ">10% Kaggle-vs-package gap" allowance quoted in
> older notes is **unsourced** — it is in neither `RULES_VERIFIED.md` nor `OBJECTIVE.md`.
> Treat 100 MB as hard. We no longer need the slack anyway.

**Best (and legal): `submissions/sub_r2.csv` = 0.82587 = 166/201, 83.82 MB.**

| | |
|---|---|
| Best legal (verified on Kaggle) | **0.82587 = 166/201** — `submissions/sub_r2.csv`, **83.82 MB** |
| Target | 0.89 = 179/201 → **+13 clips from the legal 166** |
| Qualification gate | **top-15 on private.** Bar = **162** clips (2026-08-25, 233 teams); legal score 166 — **margin +4** |
| Deadline | Kaggle 2026-09-15; code upload 09-22 |
| Noise floor | **±6 clips.** A change moving <20 of 405 rows cannot be read |
| Screening estimator | **2,700-clip pooled OOF only.** It tracked public 1:1 (+2.8 predicted, +2 delivered). Fold-2 OOF produced n7's −3 and is not used for adoption |
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
| **11–12 (us)** | **0.82587** | **166** | `sub_r2`, 83.82 MB legal |
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

### ⓪b Temporal jitter — REFUTED at 3 folds, close it out

| fold | baseline | jitter | Δ |
|---|---|---|---|
| 0 | 70.516 | 69.902 | −0.61 |
| 1 | 69.287 | 68.305 | −0.98 |
| 2 | 71.472 | **73.926** | **+2.45** |

Mean **+0.29 ± 1.62**. Only fold 2 was positive and it was an outlier below the seed
spread. Fold 3 is training purely to complete the record. **Delete the `*jit_*`
checkpoints when it lands.**

> **RULE:** a member-level change is not a result until it **exceeds 2.80 micro on a
> single fold**, or is **positive on ≥3 folds**. This failure mode has cost the campaign
> three times (EXP-100 start-weight, n7, jitter).

**Temporal TTA survived and is banked:** two interleaved 16-frame views gain on **all 8**
fold-view pairs, +23/+24 clips per member pooled, but only **+0.7 public** after fusion
dilution (`sub_s1` rowdiff 6 vs `sub_r2`). Free — keep it on; never submit it alone.


### ① Stage-2 package — **DONE and verified on public**

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

`code/run_wrist_queue.sh` holds the GPU (log: `logs/wrist_queue.log`), started
2026-08-23. In order: `k224_mvitwrist_f{0,1,3}` then `k224_mvitwrist_all`. ~5.6 h at
~250 s/epoch × 20 epochs; each stage files its own artifacts as it completes.

`k224_mvit_all` is **done** (`checkpoints/k224_mvit_all.pt`, 34.3 MB int8).

> **⚠ Never give a wait loop a predicate that can match the waiter.** The previous
> version of this queue used `pgrep -f "...--tag k224_mvit_all"`, which matched its own
> parent shell's command line and waited on itself for **7 idle GPU-hours** (EXP-106).
> The loop is gone; jobs now run in sequence.

**Harvest each with:**
```bash
python3 code/build_video_slot.py --tag <NEW> \
  --view person=k224_mvit_f0,k224_mvit_f1,k224_mvit_f2,k224_mvit_f3:0.5 \
  --view wrist=k224_mvitwrist_f0,k224_mvitwrist_f1,k224_mvitwrist_f2,k224_mvitwrist_f3:0.5
python3 code/ordered_transition_decoder.py test --probs research/artifacts/testprobs_<NEW>.npz \
  --transition-weight 0.5 --transition-score conditional --backoff unigram \
  --output submissions/sub_<NEW>.csv
python3 code/rowdiff.py submissions/sub_n8.csv submissions/sub_<NEW>.csv
```

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

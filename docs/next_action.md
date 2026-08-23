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
| Qualification gate | **top-15 on private.** Bar = 160 clips; legal score **166 — margin +6** |
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

### ⓪ Temporal jitter — replicate it, then reship everything   ← **START HERE**

**`k224_mvitjit_f2` = 0.73926 vs `k224_mvit_f2`'s 0.71472: +2.45 micro, +9 object
clips, +4.05 motion, at ZERO package bytes.** Training draws a random 16-frame phase
from the 32-frame cache each epoch, so the model stops discarding 50.9% of frames.
Largest member gain since the 224 px switch, and it changes nothing shippable.

`code/run_jitter_queue.sh` is running (log `logs/jitter_queue.log`), in this order:
fold 0 (replication check) → person all-train → wrist all-train → wrist f2 → folds 1, 3.
~9 h. Harvest as each lands.

**Adopt only on the pooled 4-fold number, never on fold 2 alone** — that is what
produced n7's −3 (B-029). If fold 0 also gains ~+2, ship the two all-train jitter models
in the package and resubmit.

**Temporal TTA is separate and already banked:** averaging two interleaved 16-frame
views on existing checkpoints is +23/+24 clips per member pooled but only **+0.7
public** after fusion dilution (`sub_s1` is rowdiff 6 vs `sub_r2` — unreadable). It is
free, so keep it on in inference; it is not worth a submission by itself.

### ① Build and verify the ≤100 MB Stage-2 package

Every number below is measured (EXP-105); none is an estimate.

| component | MB int8 | measured cost |
|---|---|---|
| `world25` pruned to 5 archs × 4 folds | 22.80 | 2 of 405 rows |
| `imu_stats` ExtraTrees 200 trees / depth 12 | 9.00 | −6 clips / 2,700 |
| MViT person `--all-train` | 34.3 | **built** — `checkpoints/k224_mvit_all.pt`; no honest local estimate exists (it trained on every fold), so `sub_q4` is how we learn its value |
| MViT wrist `--all-train` | 34.3 | *pending item ⓪* |
| **total** | **100.4** | swap `imu_stats` to 150/10 (4.61 MB, −9/2,700) → **96.0** |

`imu_stats` **cannot be dropped** — removing it moves 43 of 405 rows. It was invisible
as a packaging cost because it is refit at inference and never written to disk.

**Then submit the packaged pipeline itself to Kaggle.** Rules §2.8.b makes a >10%
Kaggle-vs-package gap a disqualification; the clean answer is that the package *is* the
submission, so the gap is zero by construction.

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

### ③ ~~Early-fusion thermal~~ — WITHDRAWN, was never cheap

Thermal is a **separate camera with no calibration to the IR/depth pair**, so a
7-channel tensor would not be pixel-aligned and the trunk would be asked to learn a
correspondence that the data does not contain. Registering the two cameras first is a
real project, not the cheap experiment this item claimed. B-027 still stands: thermal
contributes zero in probability space because its confidence when right ≈ when wrong.

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

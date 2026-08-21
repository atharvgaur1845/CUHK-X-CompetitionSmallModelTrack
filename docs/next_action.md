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

**Public best is `submissions/sub_n1.csv` = 0.81592 = 164/201, set 2026-08-21** — AdaBN
applied to the 11-member `h8all` bag. **Top-10 (163) is cleared.** The standing target is
**0.85 = 171/201 → +7 clips.**

| | |
|---|---|
| Current best (verified on Kaggle) | **0.81592 = 164/201** — `submissions/sub_n1.csv` |
| Target | 0.85 = 171/201 → **+7 clips** (top-10 cleared) |
| Deadline | Kaggle 2026-09-15; code upload 09-22 |
| Noise floor | **±6 clips.** A change moving <20 of 405 rows cannot be read |
| Champion recipe | **AdaBN test inference** (`--adabn`), video slot = log-mean of 11 members (K400 f0-3, IG-65M f0-3, f32/res160/upper f2); fusion `0.35·log(B/prior) + 0.2925·log(V̄) + 0.3575·log(I)`, then `0.9·that + 0.10·log(MB)`, then `+0.25·log(prior)`, then the transition decoder λ=0.5 conditional unigram |

> ### ⚠ THREE AXES ARE NOW MEASURED FLAT. DO NOT RE-OPEN THEM.
> Fusion weights (+1 clip at the *selection-biased* grid optimum), decoder λ, and
> **video-bag composition** (157–162 across every ≥5-member bag = about one SD). The
> oracle gap inside the existing members has collapsed from 15.4 pts to **7.85 pts**
> (EXP-098). Gain must come from a *stronger or genuinely new member*, or from
> inference-time adaptation — not from recombining what we have.

> ### ⚠ THE PACKAGING BUDGET IS BLOWN, AND IT IS A DISQUALIFICATION RISK
> One file ≤100 MB, and Rules §2.8.b treats a **>10% gap** between the Stage-2
> verification run and the Kaggle private score as cheating. 11 video members at
> ~63.5 MB int8 each ≈ **700 MB**, plus `world25` 85.2 MB. **Measured 2026-08-20:
> dropping skeleton+IMU costs 137 clips of 2,700 post-decoder (5.1 pts ≈ 10 public
> clips)** — they are expensive *and* load-bearing, so the cheap cut does not exist.
> R-3 distillation from an any-size teacher is the only identified route.

---

### ① AdaBN — DEPLOYED and CONFIRMED (+2). Now sharpen it to per-subject.   ← **the live lever**

**What:** re-estimate BatchNorm running statistics from the **unlabeled** target clips
before predicting. One extra forward pass; no labels, no training, no packaging bytes.
Already in `code/infer_video_crop.py --adabn`; it is what makes the current 164 champion.

**Status: CONFIRMED on public.** `sub_n1` (single change) scored **0.81592 = 164/201**,
+2 over the 162 champion (EXP-100). Measured on held-out fold 2 with `vid_ig65m_f2`
(EXP-099):

| | micro | object | motion |
|---|---|---|---|
| as deployed | 0.67638 | 289/479 | 0.87861 |
| **AdaBN** | **0.69172** | **300/479** | 0.87283 |

**+1.53 micro, +11 object clips**, monotone in the blend weight with the optimum at the
**endpoint** — so it is not a fitted knob, which is why it is not expected to join the six
fitted levers in the graveyard. The gain lands on OBJECT, which is 75% of the error mass.
It should also help the **on-site 8-new-subject stage (30% of the grade)** by construction.

**The remaining headroom: per-subject adaptation measured 0.70092 vs pooled's 0.69172**
— 1.6x the gain — so subject-clustered AdaBN is worth roughly **+3 public clips** if the
subjects can be recovered. The blocker is a chicken-and-egg between purity and sample
size (measured, EXP-099): timestamp blocks at a 3-minute gap are **100% subject-pure**
(validated on train: purity 1.0000 over 231 blocks) but hold a median of 10 clips, which
scores 0.67945 — *worse than pooled*. Groups must be pure **and** large.

**Next step:** cluster the timestamp blocks into subjects, then adapt per cluster.
The natural signature is each block's own mean BN feature statistics — a subject
fingerprint that falls out of the AdaBN pass for free. Validate the clustering against
true user labels on train, where the answer is known, before trusting it on test.
Test side: 404/405 clips carry a timestamp, 7 days, ~12 subjects expected.

**Done when:** a subject-clustered AdaBN submission is scored against 164.

### ② Submit `sub_n5.csv` — AdaBN + the 12th member, start-weight OFF

Built and unscored. The two confirmed-positive changes with the refuted one removed:
AdaBN (+2) and `vid_ig65m_f32_f2` (+1, isolated by n3-vs-n2). Expected **165**.
rowdiff 7 vs the 164 champion.

### ③ Early-fusion thermal — the one genuinely unexploited information source

Thermal is the paper's best modality (92.57), scores 0.544 solo here, is maximally
decorrelated (54% agreement) — and contributes **exactly zero** through late fusion
because its confidence when right ≈ when wrong (B-027). **B-027 is a statement about
probability-space fusion and says nothing about early fusion.** A 7-channel input
(IR 3 + depth 1 + thermal 3) lets the trunk learn the combination that no global weight
can. Cache exists (`cache/thermal_v1`); needs timeline alignment and missing-modality
handling (116 train clips and 10 test clips have no thermal).
**Done when:** a fold-2 micro exists to compare against IG-65M's 0.67638.

### ④ Measure the shippable package

Unchanged and still not deferrable — see the packaging warning above. No GPU needed.

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
| Resolution 160px · 32 frames · all-18 data volume | 4-fold mean −0.19 · +1.38 ns · retracted (−3, not significant) | EXP-093 |

---

## Running right now

**Nothing is running.** The GPU is free — see item ③ for the next job.

`vid_ig65m_f32_f2` finished 2026-08-21 01:19: micro **0.68098**, object 286/479,
motion 0.91329. Solo it is a **null** against the 16-frame IG-65M (0.67638, object 289)
— +0.46 micro is 3 clips on a 652-clip fold — and it *trades* object for motion. As a
bag member it is the strongest pairing we have measured (ig+ig32 = 0.69479, vs ig+igU
0.69018 and ig+k400 0.67791), but at the margin of the existing 6-member bag it is only
+0.15. Another instance of B-029: never screen a bag member on its solo score.

**Unscored submissions, ranked (see LOG EXP-099):**

| file | change vs the 162 champion | rowdiff |
|---|---|---|
| `sub_n3.csv` | AdaBN + start-weight 0.5 + 12th member | **35** |
| `sub_n2.csv` | AdaBN + start-weight 0.5 | 29 |
| `sub_n1.csv` | **AdaBN alone** — isolates the mechanism | 14 |
| `sub_h8all_sw05.csv` | start-weight 0.5 alone | 19 |
| `sub_m16.csv` / `sub_m11e.csv` | dilution variants — **skip**, `m15` already scored 160 | 8 / 4 |

`n3` vs `n2` differ by 8 rows and isolate the 12th member.

**Inference after any member finishes:**
```bash
python3 code/infer_video_crop.py --tag <TAG>                      # 4ch IR+depth
python3 code/infer_video_crop.py --tag <TAG> --cache thermal_v1   # 3ch thermal
python3 code/infer_video_crop.py --tag <TAG> --cache crop_upper   # upper-body crop
```

---

## Numbers, with provenance

| what | value | kind |
|---|---|---|
| Public best (`h8all`) | 0.80597 = 162/201 | **external oracle** (Kaggle) |
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

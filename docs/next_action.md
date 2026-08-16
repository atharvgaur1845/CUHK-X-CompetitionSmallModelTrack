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

**Public best is `sub_vidimu_bag4_trans05.csv` = 0.77611 = 156/201, set 2026-08-16.**
The standing target is **top-7 ≈ 0.80 = 161/201**, so **+5 clips**. The campaign moved
131 → 151 → 156 in one day once pretrained video backbones were adopted; the weight and
decoder axes are now both exhausted, so the remaining gain has to come from stronger
members, not from re-tuning the fusion.

| | |
|---|---|
| Current best (verified on Kaggle) | **0.77611 = 156/201** — `submissions/sub_vidimu_bag4_trans05.csv` |
| Target | top-7 ≈ 0.80 = 161/201 → **+5 clips** |
| Deadline | Kaggle 2026-09-15; code upload 09-22 |
| Noise floor | **±9–10 clips** (SD 4.5–5.0 pts). A change moving <20 rows cannot be read |
| Champion recipe | base `world25` 0.35 / video-bag4 0.2925 / `imu_stats` 0.3575, geometric in log space, then transition decoder λ=0.5 conditional unigram |

> ### ⚠ THE PACKAGING BUDGET IS ALREADY BLOWN, AND IT IS A DISQUALIFICATION RISK
> The Stage-2 package is **one file ≤100 MB**, and Rules §2.8.b treats a **>10% gap**
> between the Stage-2 verification run and the Kaggle private score as cheating →
> disqualification. Current champion, measured:
>
> | component | size |
> |---|---|
> | `world25` skeleton stack, int8 | 85.2 MB |
> | video bag4 | 4 × 31.3 MB int8 = **125 MB** |
> | `imu_stats` ExtraTrees (1000 trees × 545 feat) | **unmeasured — measure before trusting any plan** |
>
> We are several times over. Every member added widens the gap between what scores on
> Kaggle and what can legally ship. **This cannot be deferred to September.** 32 of
> `world25`'s 48 members are seed replicas, so pruning is the obvious first cut.

---

### ① Measure the shippable package, then re-plan around it

**What:** size `imu_stats`, prune `world25` to its distinct architectures, and produce a
≤100 MB package whose CSV is argmax-compared to the champion. Report the clip gap.
**Why now:** it is the only open item that can *disqualify* us, and it constrains every
future member. It is also cheap — no GPU.
**Already tried:** `code/package_ensemble.py` + `code/infer_packaged.py` are bit-parity
verified on `world25` alone; four blockers were noted for visual members and never fixed.
fp16 is measured free (0/405 argmax changes, identical sha256).
**Done when:** a single ≤100 MB artifact exists and its public-equivalent clip count is
known. If the gap exceeds ~10 clips, the LB configuration must change, not the package.

### ② Harvest the GPU queue — two candidates are BUILT AND UNSCORED

| file | members | rowdiff vs 156 champion |
|---|---|---|
| `sub_bag6_all_trans05.csv` | 4 folds + 2 all-18 seeds | 16 |
| `sub_bag2_all18_trans05.csv` | 2 all-18 seeds only | 33 |

**Why now:** already paid for. `bag6` is the safer bet (more members); `bag2_all18`
isolates whether training on all 2,933 clips beats fold training, which is the live
explanation for the 22-clip gap to the published single model (143 vs our 121 solo).
**Already tried:** 4-fold seed-diverse bagging was worth **+5** (151 → 156). Folds agree
pairwise only 0.625–0.686; the two all-18 seeds agree only **0.758** with each other, so
even same-data same-architecture members here are strongly decorrelated.
**Still running:** `mc3_18` then `r3d_18` (`code/run_arch_diverse.sh`). **`mc3_18` is
11.5M params = 23.0 MB fp16 against r2plus1d_18's 62.6 MB** — if it scores comparably it
is worth far more than its accuracy, because item ① is a size problem.
**Done when:** both scored and the better one is the new reference.

### ③ Close the gap to the published single model

**What:** the published notebook's single model = **143** public; our best single fold =
**121**. Two known differences remain after ②: they use **R(2+1)D-34 IG-65M** (63M params)
where we use `r2plus1d_18` Kinetics-400 (31M), and their input may differ in resolution.
**Why now:** the largest identified gap that is not yet explained by anything we control.
**Caution — decide with the user, do not adopt unilaterally:** IG-65M is a much larger
pretrain than R-1's stated example ("ImageNet-pretrained ResNet18 … perfectly
acceptable"). The rule targets LLMs/VLMs, and the 0.711 notebook using it sits publicly on
the competition's own Kaggle page, but a 63M-param video backbone is a judgement call that
interacts with ① and with the Stage-2 review.
**Already tried:** resolution is capped at 128×128 by memory — 192² would be a 6.9 GB
cache against 5 GB of free RAM and would thrash the trainer.

---

## Do NOT do these

| direction | why it is closed | evidence |
|---|---|---|
| **Exploit the test-label leak** | Confirmed leak exists (discussion 714827: public repo had labelled split metadata matchable by timestamp + frame id). Using it is cheating, organisers run anti-cheat, and Stage 3 is on-site with 8 new subjects. **Permanently closed.** | `research/RULES_VERIFIED.md` L-1 |
| **Obtain any subject not already in training** | All 30 participants accounted for: 18 train (1–9, 16–24), 4 public test (10, 11, 25, 26), 8 private. Any unseen subject **is** a test subject. The HuggingFace mirror is byte-identical with no extra subjects. | `research/RULES_VERIFIED.md` |
| Re-tune fusion weights | Surface is flat on 2,700-clip pooled OOF: optimum 0.71000 vs champion 0.70963 = **one clip**. Ordering now matches public (C 156 > D 149 > A 144). | LOG EXP-086 |
| Decoder λ > 0.5 | OOF: λ=0.5 → +0.046; λ=1.0 → +0.006 (231 rescues, 215 harms). `vidimu_C_trans10.csv` is **retracted, do not submit**. | LOG EXP-086 |
| Add the old 2D visual members (`pre_r50`, `pre_r18`, `visual_mil_v1`) to the fusion | They now **hurt**: r50 costs 3.8 micro, mil_v1 costs 2.9. Superseded, not complementary. | LOG EXP-086 |
| Thermal folds 0/1/3 | Thermal member is strong (0.544) and maximally decorrelated (54% agreement) yet adds **exactly zero** at every weight. Uniquely correct on 34 clips but confidence when right (0.511) ≈ when wrong (0.413), so no gate extracts it. | LOG EXP-088, B-027 |
| Skeleton-only base | Worse in the new fusion: 0.68478 vs `world25` 0.69565. | LOG EXP-086 |
| Sinkhorn / prior-forcing on test | −12 public, already in the graveyard. Re-derived once by mistake. | `research/BELIEFS.md` |
| Fitted stackers, learned gates, cohort weights | Four consecutive fitted-combination levers all landed ≤0 on public despite large OOF gains. | LOG, `research/BELIEFS.md` |

---

## Running right now

| job | produces | check with |
|---|---|---|
| `code/run_all18.sh` | `vid_all18_s20260730`, `vid_all18_s20260816` — all-2933-clip members | `tail -2 logs/vid_all18_s*.log` |
| `code/run_arch_diverse.sh` (chained, waits on the above) | `vid_mc3_18_all18`, `vid_r3d_18_all18` | `tail -2 logs/vid_*_all18.log; cat logs/arch_chain.log` |

Both run under `code/run_with_watchdog.sh` (kills and resumes on a stalled log or 15 min
of 0% GPU) and all trainers take `--resume` from per-epoch state.

**Inference after any member finishes:**
```bash
python3 code/infer_video_crop.py --tag <TAG>                 # 4ch IR+depth
python3 code/infer_video_crop.py --tag <TAG> --cache thermal_v1   # 3ch thermal
```

---

## Numbers, with provenance

| what | value | kind |
|---|---|---|
| Public best | 0.77611 = 156/201 | **external oracle** (Kaggle) |
| Public noise floor | ±9–10 clips | measured |
| Pooled video OOF (4 folds, 2933 clips) | micro 0.64371, object 0.54431 | **first-run**, honest |
| Champion fusion OOF (2700 overlap) | 0.70963 | first-run |
| Decoder gain, OOF | +0.046 (212 rescues / 88 harms) | first-run, **both directions** |
| Decoder gain, public | **+26 clips** (125 → 151) | external oracle |
| Best decoder config, OOF | 0.76852 (λ .5, topk 8, uniform, penalty 1.0) | **fitted** — argmax of ~20 configs on 2700 clips |
| Thermal member | 0.54448 solo | first-run |
| Fusion weights | base .35 / vid .2925 / imu .3575 | **fitted**, but confirmed by public ordering |

---

## History

**2026-08-16.** 4 video folds complete (0.62776 / 0.61916 / 0.63957 / 0.69832). Bag → 156.
Thermal built and refuted as a fusion member. Decoder tuned on a pooled 2700-clip OOF;
B-022's pre-registered distinctness prediction confirmed. Weight axis closed.

**2026-08-15.** Person-crop + Kinetics R(2+1)D landed: fold-2 micro 0.63957 vs 0.40031 for
the best prior member, **+23.9**. Fusion 131 → 151. IMU stats member (ExtraTrees on 6
channels) added, +3.4 micro. `build_model` 4-channel bug cost ~1 GPU-hour.

**2026-08-10.** `research/RULES_VERIFIED.md` R-1: **pretrained CNNs were legal all along.**
A prior session had recorded the opposite as fact and cancelled the pretrained probe;
months of SSL work existed as a substitute for a legal initialisation.

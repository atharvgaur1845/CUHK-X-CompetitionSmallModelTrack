# The last 25 hours — protect 174, verify cross-device, and be honest about 0.90

**Written 2026-09-14 15:00 UTC (Fable 5.1). Freeze: 2026-09-15 15:55 UTC — 25 hours.**
Supersedes the 7-day plan below the line, which has been executed in full and logged as
EXP-132 … EXP-158.

---

## Context

**Where we are.** 174/201 = 0.86567, 14th of 307. The package `stage2_ship7.pth` (96.37 MB)
is rules-compliant (both YOLO detectors inside), verified four ways, and `inference.sh`
regenerates its output from raw frames at rowdiff 0. The scored 174 is within 2 rows of that
output. Advancement to Stage 2 is top-15 on the **private** board (204 clips, same four
subjects as public); neighbours are 176 / 175 / **174** / 173 / 172.

**What the target means.** 0.90 = 181/201 = **+7 clips**, in 25 hours, on a package that is
3.6 MB from the cap.

**What the 7-day plan delivered, all measured.** One of five levers paid: thermal at 224
(+5, EXP-139/140). Then resolution on the person view (+3, EXP-154/158). Everything else is a
measured negative or zero, not an assumption: teacher (0.419 vs 0.712), subject alignment
(purity 0.339), object channel (0.502 vs 0.574 control), pair verifier (0.876 vs 0.893 gate,
AUC 0.588 where it matters), seed soup (**−2.39, 4/4 folds**), fold soup on person/wrist
(negative), temporal TTA (−0.15), wrist-288 (**−1.36**), thermal-288 (0 in fusion),
full-frame IR+Depth (correlated copy of the person crop), twin pooling (+1 — twins fail
*together*), scene vocabulary (per-subject), transductive self-training (member +1.38,
**fusion −2**), distinctness and λ (saturated).

**The finding that governs this plan.** Three independent probes (EXP-132, 156, 157) say the
same thing: the remaining errors are **per-subject and correlated**. Twins share the identical
wrong label 68% of the time; the subject's own pseudo-labels reinforce its mistakes; the
scene scripts are unique to each subject. No lever that assumes independent evidence can
reach those clips, and every lever we own assumes exactly that. That is a statement about
the data, and it is why the last five experiments each moved ≤ 1 clip.

## The verdict on 0.90, stated once

**There is no legal path to +7 clips in 25 hours.** The sum of every remaining lever, if
every one of them hit, is ≈ +2. I am not going to pad this plan with things the repo's own
measurements predict will fail.

"By hook or crook" was considered literally. Each hook fails the *actual* objective, not
just the rules:

| hook | why it does not work even on its own terms |
|---|---|
| **Leaderboard probing** (flip rows, read the score) | Extracts *public* labels. Private is different clips and is what decides advancement — so it buys **0** where it counts. A probed CSV cannot be regenerated from the package, so it **fails Stage 2 reproduction**; the organisers said on 09-07 that exactly such scores "won't hold up". It also burns the 5/day submission budget we need for the real candidates. Rejected. |
| **The test-label leak (L-1)** | Closed; the repository is offline; same reproduction failure. Not proposed. |
| **External data** (NTU RGB+D, R-2 legal) | The only mechanism that attacks subject-correlated error is *more subjects*. NTU is ~100 GB+ of depth/IR; download alone exceeds the window. Infeasible in 25 h. |
| **Tighter wrist crop** | I nearly proposed this. The crop-size rule (EXP-154) says resolution pays only where the crop is large relative to the source; the wrist box is already upsampled to 224, and tighter means *more* interpolation — the mechanism that made wrist-288 negative. Withdrawn. |
| **Bigger backbone** (MViTv2-B, 52 M) | Needs 11 MB we do not have (3 skeleton archs, unmeasurable cost), 20 GPU-h on a cluster that was saturated for two days, and EXP-115 measured "bigger overfits" at 86 M. A coin flip that consumes the whole window. Not this week. |

**The honest objective for the next 25 hours is the final grade**, of which public Kaggle is
0%: private 20 · on-site 30 · reproducibility 10 · report 20 · presentation 10 ·
efficiency 10. Clearing the private cut on a package that survives verification is worth
more than any public clip we could still buy.

## Programme, in order

### 1. Protect the result (≈ 1 h, blocking — do first)

1. **Submit `submissions/sub_final_ship7.csv`.** It is the package's own output from raw
   frames; whatever it scores is exactly what Stage 2 reproduces. Expected 174 ± 1.
2. **Select the two private submissions on the Kaggle site**: `sub_pkgship6.csv` (174) and
   `sub_final_ship7.csv`. Do not leave selection to auto-pick.
3. **Tag the artifact of record**: `git tag stage2-final` on the commit holding
   `stage2_ship7.pth`, `inference.sh`, `inference_assets/`. Nothing else is modified after
   this except by a documented reason.
4. Reply to nothing on the forum; there is nothing to gain.

### 2. Cross-device reproduction dry run (≈ 1.5 h, the single most valuable remaining task)

`inference.sh` has only ever run on this laptop. The organisers will run it on theirs. The
crop windows are now shipped (which removes the one cross-device drift we measured), but
the MViT forward pass itself has not been checked across GPUs.

- Rsync the repo (code + `stage2_ship7.pth` + `inference_assets/`, **no caches, no
  training data**) to `/scratch/pabitra/cuhkx/cleanroom/` on the cluster.
- Run `./inference.sh stage2_ship7.pth <test_dir> out.csv` there on an H100 via one sbatch,
  from an empty work dir.
- Gate: `python3 code/rowdiff.py submissions/sub_final_ship7.csv out.csv` → **0**.
  If it is not 0, the diff tells us which branch drifts across GPUs; the fix is to pin
  that branch's inference to fp32 (`predict()` in `kaggle/cuhkx_224_kaggle.py`) and
  re-verify — cheap, and it must be known before 09-17, not discovered by a reviewer.

### 3. The Stage-2 upload bundle (≈ 1 h)

A reviewer with the checkpoint and this repo must be able to run it cold.

- `STAGE2_README.md`: exact command; the environment (Python 3.11, torch 2.7.1+cu126,
  ultralytics 8.4.118 — pin these, the conda env `physmon` on the cluster is the reference);
  expected runtime; the statement that the package contains every weight used at inference
  including the detectors and the crop-window cache; what happens on unseen clips (live
  YOLO); the 2-row difference between the scored 174 and the packaged 20-member skeleton,
  stated up front rather than found.
- `requirements-inference.txt` frozen from `physmon`.
- Confirm `inference_assets/` is tracked and the `.gitignore` excludes `inference_work/`
  and caches (it does).

### 4. One low-EV run, fire-and-forget, only if GPUs are idle (≈ 6 GPU-h, 0 attention)

**Person at 448.** Same windows as 384 (`cache/crop_384/windows.json`), render at 448
(≈ 2 min), sync ≈ 3.5 GB, 4 folds + all-train (`--image-size 448 --batch-size 4 --accum
4`), then the EXP-154 fusion check. The 224→288→384 member curve is +1.5, +0.4 — this is
worth ≤ +1 clip and I would not wait on it. Gate: pooled fusion OOF > 2106 by 2 SE *and*
the all-train model finishes before the freeze; otherwise it does not ship. **The cluster
queue currently holds 7 pending H100 jobs from other users; if the array is not running
within 2 h of submission, cancel it and forget it.**

### 5. Everything else goes to the report and presentation (remaining ~20 h)

This is 30% of the grade and untouched. The material is already measured and is the
strongest story we have for the *on-site* stage specifically:

- **Fusion flattens the subject distribution** (EXP-145): sd 6.89 → 5.42, spread 28 → 21
  points, 8-subject-draw SE 2.44 → 1.92. That is the argument that this ensemble is built
  for unseen subjects, which is what the on-site 30% tests.
- The full LOSO table (18 subjects × 4 members), the recipe-transfer finding on thermal,
  the resolution/crop-size rule, and the honest negative results — a review board values a
  measured negative over an unmeasured claim.
- Compliance and reproducibility as a section, not a footnote: 96.37 MB, detectors inside,
  window cache, rowdiff-0 regeneration.

## What to stop doing

- No more fusion-weight, decoder, TTA, soup, or self-training runs. Each has a logged
  measurement; re-running them spends submissions on noise.
- No submission whose rowdiff against the incumbent is < 20 unless it is the reproduction
  candidate in §1.
- No argparse changes to `kaggle/cuhkx_224_kaggle.py` after the tag; the fingerprint rule.

## Verification

| step | check |
|---|---|
| §1.1 | Kaggle score for `sub_final_ship7.csv` within 174 ± 1; if ≤ 171, re-read the gap (it would mean the 2-row skeleton difference is not the whole story) |
| §2 | cluster `out.csv` vs `sub_final_ship7.csv` **rowdiff 0** |
| §3 | a second person can run `./inference.sh` from the README alone |
| §4 | `code/exp140`-style fusion sweep on the 448 fold OOF; only then the all-train run |
| repo rule | `docs/next_action.md` updated in the same commit as each of the above |

## Files touched

- `STAGE2_README.md`, `requirements-inference.txt` (new)
- `cluster/cleanroom.sbatch` (new, runs `inference.sh` on the cluster)
- optionally `cluster/res448.sbatch` (new)
- `docs/next_action.md`, `research/LOG.md` (EXP-159 clean-room result)
- **No changes** to `inference.sh`, `pack_stage2.py`, `unpack_stage2.py` unless §2 finds drift

---
---

# (Executed) 7-day programme to 187+ — three levers whose own arithmetic reaches ten clips

*Retained for the record. Outcome: L1 failed its gate, L2 failed P0, L3a (thermal)
delivered +5, L3b failed, L4 failed its gate; T-PKG delivered a compliant 96 MB package;
LOSO ran for all 18 subjects; the selection rule was corrected (EXP-145). Final 174/201.*

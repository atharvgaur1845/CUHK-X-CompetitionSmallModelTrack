# Objective

**Competition:** CUHK-X Challenge @ UbiComp 2026 — Small Model Track (Kaggle). 40-class human activity recognition from privacy-preserving sensors.
**Metric (exact formula):** Classification accuracy = (# test clips with predicted action_id == ground truth) / 405. One prediction per clip (integer 0–39).
**What the metric rewards/punishes:** Pure top-1 accuracy. No calibration, no rank, no partial credit. Every clip weighs 1/405 ≈ 0.247%. Confusable class pairs (e.g. Sweep vs Mop floor, Read documents vs Turn pages, Sit down vs Stand up) are where accuracy is won or lost.

## Constraints
- Compute (local): RTX 4060 Laptop 8 GB VRAM, 16 CPU cores, 15 GB RAM, ~117 GB free disk. torch 2.12.0+cu130 working.
- Model size ≤ 100 MB total. Architectures limited to CNN / RNN / Transformer. **ORGANIZER RULING (2026-07-28): NO pretrained weights at all (strict from-scratch); ensembles LEGAL if total ≤100 MB; test-time transductive processing LEGAL.**
- No test labels in training; no manual labeling of test samples.
- Reproducibility matters: Top-15 advance to Selection Stage where organizers REPRODUCE the solution. Keep everything scripted, seeded, and documented from day one.
- Submissions per day: assume 5 (verify on Kaggle).

## Timeline
- Deadline: TBD — verify on Kaggle competition page / official site (https://openaiotlab.github.io/CUHK-X-Challenge/).
- Today: 2026-07-27.

## Win condition (updated 2026-07-28 after DA-002)
- Goal: **Private LB Top 15** (→ Selection Stage → UbiComp finals). Kaggle rank only gates entry; finals decide cash.
- **Public split = 201 clips (derived — all scores are k/201); private = 204.** 1 public clip = 0.4975%; shakeup ±2-3% at 1σ.
- Landscape: leader 0.836 (168/201; mechanism uncertain — 30-40% strong transfer recipe, ~25% manual labeling, ~20% LB probing); cluster #2-#6 0.73-0.77 (likely pretrained visual recipes — possibly disqualifiable at reproduction if the pretrained ban is enforced, which lowers the effective bar for rules-clean teams like us). **Rank-15 score UNKNOWN — Atharv must harvest ranks 10/15/20/30 from Kaggle; this number sets the real bar.**
- **Atharv's standing directive (2026-07-29): keep improving until 80+ accuracy — no convergence before that.** Interpretation per the loop: the exploit grind continues (streams/soups/fusion → low 0.50s), AND the explore/crazy tiers stay funded indefinitely because 80+ requires breaking the measured ceiling (oracle 63.4 on current streams), not polishing it. Every plateau triggers exploration rebalance, never termination. (Calibrated posterior on reaching 0.80+ remains low — but the search does not stop on posterior, it stops on the directive.)
- Current best: 0.458 (92/201), single submission, no post-proc survived LB testing yet.

## What loses
- Overfitting to the 18 training subjects (cross-subject gap is THE challenge — test users 10, 11, 25, 26 are unseen).
- Overfitting public LB (tiny test set).
- Breaking the 100 MB / no-pretrained rule → disqualification at reproduction stage.
- Un-reproducible pipeline → eliminated at Selection Stage even with top score.

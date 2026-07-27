# Objective

**Competition:** CUHK-X Challenge @ UbiComp 2026 — Small Model Track (Kaggle). 40-class human activity recognition from privacy-preserving sensors.
**Metric (exact formula):** Classification accuracy = (# test clips with predicted action_id == ground truth) / 405. One prediction per clip (integer 0–39).
**What the metric rewards/punishes:** Pure top-1 accuracy. No calibration, no rank, no partial credit. Every clip weighs 1/405 ≈ 0.247%. Confusable class pairs (e.g. Sweep vs Mop floor, Read documents vs Turn pages, Sit down vs Stand up) are where accuracy is won or lost.

## Constraints
- Compute (local): RTX 4060 Laptop 8 GB VRAM, 16 CPU cores, 15 GB RAM, ~117 GB free disk. torch 2.12.0+cu130 working.
- Model size ≤ 100 MB total. Architectures limited to CNN / RNN / Transformer. **No large pretrained backbones**, no closed-source APIs, no LLMs for development/labeling. (Interpretation to verify: are small ImageNet-pretrained CNNs like MobileNet allowed? Rule says "no large pretrained backbones" — ambiguous for small ones. Default to training from scratch until clarified.)
- No test labels in training; no manual labeling of test samples. (Unlabeled test-time adaptation / pseudo-labeling legality: check rules — "no using test labels" ≠ "no using test data"; self-training on test data likely allowed but must survive organizer reproduction in Selection Stage.)
- Reproducibility matters: Top-15 advance to Selection Stage where organizers REPRODUCE the solution. Keep everything scripted, seeded, and documented from day one.
- Submissions per day: assume 5 (verify on Kaggle).

## Timeline
- Deadline: TBD — verify on Kaggle competition page / official site (https://openaiotlab.github.io/CUHK-X-Challenge/).
- Today: 2026-07-27.

## Win condition
- Goal: **Private LB Top 15** (→ Selection Stage → UbiComp finals). Kaggle rank only gates entry; finals decide cash.
- Public LB is reference-only; private LB decides. With 405 test clips, public/private splits are tiny — a 1-clip swing = 0.25–0.5%. Expect shakeup; optimize robust CV, not public LB.
- Leaderboard landscape: TBD (check current top scores).

## What loses
- Overfitting to the 18 training subjects (cross-subject gap is THE challenge — test users 10, 11, 25, 26 are unseen).
- Overfitting public LB (tiny test set).
- Breaking the 100 MB / no-pretrained rule → disqualification at reproduction stage.
- Un-reproducible pipeline → eliminated at Selection Stage even with top score.

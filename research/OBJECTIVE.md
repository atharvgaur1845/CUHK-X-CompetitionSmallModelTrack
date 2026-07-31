# Objective

**Competition:** CUHK-X Challenge @ UbiComp 2026 — Small Model Track (Kaggle). 40-class human activity recognition from privacy-preserving sensors.
**Metric (exact formula):** Classification accuracy = (# test clips with predicted action_id == ground truth) / 405. One prediction per clip (integer 0–39).
**What the metric rewards/punishes:** Pure top-1 accuracy. No calibration, no rank, no partial credit. Every clip weighs 1/405 ≈ 0.247%. Confusable class pairs (e.g. Sweep vs Mop floor, Read documents vs Turn pages, Sit down vs Stand up) are where accuracy is won or lost.

## Constraints
- Compute (local): RTX 4060 Laptop 8 GB VRAM, 16 CPU cores, 15 GB RAM, ~117 GB free disk. torch 2.12.0+cu130 working.
- Model size ≤ 100 MB total. Architectures limited to CNN / RNN / Transformer.
  **ORGANIZER RULING RELAYED BY ATHARV (2026-07-28):** no pretrained weights,
  ensembles legal if total ≤100 MB, and test-time transductive processing
  legal. Preserve the exact reply: the repository currently contains only the
  draft question and a paraphrased research entry.
- **The current stack has a verified sub-100-MB serialized package and a
  streamed loader.** World25 occupies 85,217,859 bytes on disk. Streaming
  reduced persistent fp32 model parameters from 337,973,856 bytes to at most
  10,134,688 bytes while reproducing all 16,200 float64 probabilities exactly.
  Preserve the exact organizer reply because serialized-size versus transient
  dequantization accounting is still not stated publicly.
- No test labels in training; no manual labeling of test samples.
- Reproducibility matters: Top-15 advance to Selection Stage where organizers REPRODUCE the solution. Keep everything scripted, seeded, and documented from day one.
- Submissions per day: assume 5 (verify on Kaggle).

## Timeline
- Kaggle deadline: **2026-09-15**; code upload: **2026-09-22** (EXP-000b rules audit).
- Current research date: 2026-07-30.

## Win condition (updated 2026-07-30 after SUB-012)
- Goal: **Private LB Top 15** (→ Selection Stage → UbiComp finals). Kaggle rank only gates entry; finals decide cash.
- **Public split = 201 clips (derived — all scores are k/201); private = 204.**
  One public clip = 0.4975%. Public/private sampling variation near current
  accuracy is roughly 5 percentage points for their difference, not the
  previously asserted 2--3 points.
- Landscape: Atharv reports current competitors near **0.89**. The local
  repository has no authenticated leaderboard snapshot, method, or private
  transfer evidence for those entries. Treat 0.89 as a real target observation,
  not as proof that the current skeleton/IMU family is one tuning step away.
  **Rank-15 score remains unknown.**
- **Verified public best:** `sub_astgcn_world25_int8_trans05.csv` =
  **0.55721 = 112/201**. It applies tie-safe ordered decoding to the exact
  probabilities from the legal 85.218 MB package. See `LEADERBOARD.md`.
- **Atharv's standing directive:** improve beyond 0.83. A score strictly above
  0.83 requires at least **167/201**, so the present gap is **+55 correct public
  clips**.
- **Sequence result:** ordered transitions transferred, moving the exact
  package output from 109 to 112 public clips. Repeated-recording consensus
  then scored 111/201 despite stronger local OOF, so its marginal is rejected.
  No further sequence/template tuning is the active path.
- The next model-side reset targets the actual remaining error mass: high
  resolution IR + absolute depth + Thermal, with explicit masks and
  candidate-conditioned object pooling. Existing visual preprocessing reduced
  frames to 120×160 or loose ROIs and discarded appearance in motion maps;
  Thermal was absent despite 2,788/2,933 canonical train and 395/405 test
  coverage. A prior 2,891 count included 103 Thermal-only directories outside
  the canonical metadata universe.
- Operational success requires all three simultaneously: competitive private
  accuracy, an inference package ≤100 MB, and reproducible training/inference.
  The serialized-size gate and one exact-package leaderboard measurement are
  passed: world25 is 85.218 MB on disk, its raw output scored 0.54228, and the
  deterministic transition decode scored 0.55721. Live-weight
  accounting, accuracy above 0.83, and complete clean-room reproduction remain
  open.

## What loses
- Overfitting to the 18 training subjects (cross-subject gap is THE challenge — test users 10, 11, 25, 26 are unseen).
- Overfitting public LB (tiny test set).
- Breaking the 100 MB / no-pretrained rule → disqualification at reproduction stage.
- Un-reproducible pipeline → eliminated at Selection Stage even with top score.

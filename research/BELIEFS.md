# Belief Ledger

## B-001 — Cross-subject generalization (not capacity) is the primary bottleneck
- **Confidence:** 90% (↑ from 80%)
- **Importance:** High
- **Evidence for:** EXP-000b — dataset paper: random-split avg 76.5% / visual ~90% vs cross-subject LOSO ≈ 56.4%. A ~30-point cliff.
- **Evidence against:** —
- **Remaining uncertainty:** which invariance lever (augmentation, normalization, adversarial, mixup) closes the most gap on THIS data.
- **Experiments that would settle it:** Q-10..Q-12, Q-32, Q-34.
- **Last updated:** EXP-000b

## B-002 — Skeleton is the best single modality per unit compute
- **Confidence:** 45% (↓ from 65%)
- **Importance:** High
- **Evidence for:** cheapest to train (177 MB raw, tiny input dim); H36M 3D joints are body-normalizable.
- **Evidence against:** EXP-000b — paper: Skeleton 79.1% (with a PRETRAINED MotionBERT, likely banned) vs Depth 90.5/IR 90.2/Thermal 92.6 in-domain; skeletons already pelvis-centered+floor-aligned per frame (global motion destroyed — hurts Walk/Jog/Lie classes); scores useless; multi-person noise.
- **Remaining uncertainty:** cross-subject (not random-split) per-modality ranking is unknown — visual may drop more than skeleton under LOSO.
- **Experiments that would settle it:** Q-02 vs Q-04 under identical subject-CV.
- **Last updated:** EXP-000b

## B-003 — Multi-modal fusion beats best single modality by ≥5%
- **Confidence:** 75% (↑ from 70%)
- **Evidence for:** EXP-000b — test availability is benign for the big-4 (Depth/IR/Skeleton/IMU in ≥404/405 clips); paper published no fusion baseline → uncharted upside; modality error profiles clearly complementary (visual=objects/scene, skeleton=pose, IMU=limb dynamics).
- **Evidence against:** single model.pth ≤100 MB packaging caps how many streams we can ship.
- **Remaining uncertainty:** fusion architecture (late vs mid); whether Thermal (misfit coverage, 25 fps unsynced) is worth a stream.
- **Experiments that would settle it:** Q-20, Q-21, Q-31.
- **Last updated:** EXP-000b

## B-004 — MACRO (balanced) subject-CV predicts LB; micro does not (REWRITTEN by DA-001)
- **Confidence:** 75%
- **Importance:** High — load-bearing metric choice
- **Evidence for:** DA-001 diagnostics — macro-OOF 46.7% vs LB 45.8% (near-exact); micro-OOF 54.3% mispredicted by 9 pts; LB sits at 0.1th percentile of 4-user-subset MICRO distribution (subject variance can't explain the offset, prior mismatch can).
- **Evidence against:** single LB pair so far; needs confirmation on next submissions.
- **Remaining uncertainty:** exact test balance (40×10 + 5?); whether macro tracks LB deltas, not just level.
- **Experiments that would settle it:** prior-adjusted submission pair (files ready); every future submission logs macro-OOF vs LB.
- **Last updated:** DA-001

## B-012 — ~~Test set is class-balanced~~ DEAD (SUB-003/004)
- **Killed:** prior-adjusted submissions dropped −6.5/−6.0 pts (0.393/0.398 vs 0.458 raw) → test prior ≈ TRAIN prior. Metric policy back to micro-OOF. The DA-001 macro↔LB match was coincidental overlap of two effects (subject shift ≈ macro-micro gap in magnitude). Kept here (not just the table) as a warning: a single corroborating coincidence is not confirmation — the submission PAIR was what falsified it.

## B-013 — Current streams structurally cannot reach 0.85; stream quality IS the campaign (NEW)
- **Confidence:** 85%
- **Importance:** Highest — governs compute allocation
- **Evidence for:** DA-001 oracle probe: pick-best-of-3-streams = 63.4% OOF (top-5 any-stream 87.8%). Even perfect fusion of current streams ≤63%.
- **Evidence against:** —
- **Remaining uncertainty:** how far each stream can be pushed (IR untested; ROI untested; capacity 0.2% used; pretrained legality unknown; transduction multiplier unknown).
- **Experiments that would settle it:** EXP-009/010/011 (IR/ROI), pretrained probe, capacity sweep, transduction probes.
- **Last updated:** DA-001

## B-005 — 100 MB budget is not binding — SETTLED by organizer ruling
- **Confidence:** 95%
- **RULING (2026-07-28):** ensembles legal if TOTAL ≤100 MB; strict no-pretrained (none at all); test-time transduction legal. We use ~5 MB → ~20× headroom for multi-stream × multi-seed soups. Efficiency (10% of final) still rewards staying lean.
- **Last updated:** RULING

## B-016 — Rules-clean pipeline is a Selection-Stage asset (NEW)
- **Confidence:** 70%
- **Importance:** High — changes what score ADVANCES
- **Evidence for:** RULING confirms strict no-pretrained; DA-002 put ~85% on the 0.73-0.77 cluster using pretrained visual recipes → they fail reproduction; effective private-LB bar for top-15 advancement is likely well below the public cluster.
- **Evidence against:** cluster teams may also be from-scratch (would mean they out-engineered us — steelman); enforcement rigor unknown.
- **Remaining uncertainty:** rank-15 public score (Atharv harvest pending); how organizers audit training provenance.
- **Last updated:** RULING

## B-006 — Visual modalities are the accuracy backbone (REVISED: latent, recipe-gated)
- **Confidence:** 50% (revised meaning: visual has large headroom but ONLY after recipe fixes — EXP-006: depth random-split just 32.4% vs paper 90% pretrained ⇒ recipe-bottlenecked, not modality-weak; DG gap on depth is only ~9 pts at current quality)
- **Key lever:** person-ROI cropping (person small/off-center at 120×160 full frame), longer training, capacity, more frames/resolution (Q-70/Q-71)
- **Importance:** High — decides where most compute goes
- **Evidence for:** EXP-000b — paper: Depth 90.5/IR 90.2/Thermal 92.6 vs IMU 45.5/mmWave 46.6 (random split); test has Depth+IR in all 405 clips; same apartment/stations in test → environment transfers.
- **Evidence against:** those numbers are pretrained-ResNet-50 + random split; from-scratch small CNN under cross-subject may land far lower; IR contains faces/identity → could overfit subjects.
- **Remaining uncertainty:** from-scratch cross-subject visual accuracy unknown; Depth-inverted-scalar vs raw JET-RGB input choice.
- **Experiments that would settle it:** Q-04, Q-05, Q-52.
- **Last updated:** EXP-000b

## B-007 — Test metadata structure is exploitable for the Kaggle LB (REVISED by EXP-000c)
- **Confidence:** 80% (for the distinctness lever) / legality 85% (metadata ≠ labels; runs inside inference code)
- **Importance:** High for Kaggle gate, ZERO for on-site test (30% of final) — keep detachable
- **Evidence for:** EXP-000c — 405 test clips = 144 recordings (radar-ts key); train ground truth: recording groups 100% user-pure AND 100% all-classes-distinct (741/741) → per-group Hungarian assignment covers 93% of test clips; 2-way cohort split provable (radar-usable E n=199 vs empty L n=206); 4-way ≈ 99/100/105/101.
- **Evidence against:** adjacency prior measured DEAD (0/405 same-class neighbors); station prior effectively dead (recovery ~50% → 0.09 bits); distinctness lift depends on the base model's within-group top-1 collision rate (unmeasured).
- **Remaining uncertainty:** collision rate of real model predictions within groups; whether test groups obey exactly the train distinctness protocol; E1/E2 sub-split boundary confidence low.
- **Experiments that would settle it:** Q-68 (simulate Hungarian on train CV predictions — measurable without LB!), then an LB submission pair with/without.
- **Last updated:** EXP-000c

## B-008 — Radar is not worth a model stream (softened)
- **Confidence:** 70% (↓ from 80)
- **Importance:** Medium (frees compute)
- **Evidence for:** EXP-000b — empty for 100% of users 16-24, 51% of test; 6.6 pts/frame; Doppler clipped ±0.97 m/s; paper mmWave 46.6% even with full coverage.
- **Evidence against:** EXP-000c — radar is usable for ALL 198 early-cohort test clips and its train coverage (users 1-9) includes both batch A+B → an E-cohort-only radar stream could contribute on exactly half of test; its filename timestamp is the recording-group KEY regardless.
- **Remaining uncertainty:** marginal value on the E-half after visual+skel+IMU fusion.
- **Experiments that would settle it:** Q-36 (still deprioritized until fusion exists).
- **Last updated:** EXP-000c

## B-011 — Recording groups: same user + station, pairwise-distinct classes (NEW)
- **Confidence:** 95% (train side is exhaustive ground truth: 741/741)
- **Importance:** High — the main Kaggle-only lever
- **Evidence for:** EXP-000c purity check; group key (radar filename ts) exists for 404/405 test clips; camera-anchor clustering is the backup key.
- **Evidence against:** test protocol could differ (unlikely — same collection pipeline); the one no-radar clip (SM_test_0054) needs camera-anchor fallback.
- **Remaining uncertainty:** measurable lift (simulate on train CV predictions in Q-68).
- **Experiments that would settle it:** Q-68.
- **Last updated:** EXP-000c

## B-009 — IMU is weak alone but cheap ensemble diversity (NEW)
- **Confidence:** 65%
- **Evidence for:** EXP-000b — only ~10 Hz, median 22 samples/trial, paper baseline 45.5%.
- **Evidence against:** paper's IMU baseline may be poor (no augmentation/orientation-invariance); 5-device placement is consistent → room above 45.5%.
- **Remaining uncertainty:** ceiling with proper aug + quaternion/orientation features at 10 Hz.
- **Experiments that would settle it:** Q-03, Q-12.
- **Last updated:** EXP-000b

## B-010 — Class imbalance + ragged coverage needs explicit handling (NEW)
- **Confidence:** 70%
- **Evidence for:** EXP-000b — 12 vs 319 samples/class (27×); classes 25/26 present for only 4-6 users; duration varies 7× by class and test is trimmed shorter.
- **Remaining uncertainty:** whether test is balanced (405 ≈ 40×10?) — check via first submissions / probing-free analysis.
- **Experiments that would settle it:** Q-13b (balanced sampling ablation), submission evidence.
- **Last updated:** EXP-000b

---

## Dead beliefs (killed by evidence — keep, so we don't resurrect them)

| ID | Belief | Killed by | Why |
|----|--------|-----------|-----|
| — | "Test set is anonymized" (implicit) | EXP-000b | Timestamps intact in 4 modalities + IMU internal timestamps + Thermal frame counters |

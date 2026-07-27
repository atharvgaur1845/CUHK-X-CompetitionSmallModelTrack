# Evidence Log

**Counters:** experiments since last devil's-advocate pass: 10 / 10 → DA PASS DUE · since last reset: 10 / 25

---

## SUB-001/002 — First LB results (public)
**Date:** 2026-07-28 · **Tier:** foundation (calibration)

**Results:** sub_fuse3_nohung.csv → **0.45771** · sub_fuse3_hung.csv → **0.44776** (public LB; leader 0.836, target >0.85 per Atharv).
**Finding 1 — CV→LB offset ≈ −9 pts:** OOF 54.3% → LB 45.8%. Subject-CV overestimates. Candidate causes: (a) 4 test users idiosyncratic (2 possibly different hardware/session era); (b) test clips trimmed ~20% shorter + 10 s hard cap (train has longer clips; our temporal sampling sees different length distribution); (c) public LB is a subset — noise ±2-3%; (d) same-user val leakage in our CV? No — folds are user-grouped; but always-train users 5/21 make CV slightly easier than a true 4-unseen-user test.
**Finding 2 — Hungarian NEGATIVE on LB (−1.0%)** vs +1.1% on CV fusion. Hypotheses: (H1) at 46% base accuracy test probs are worse-calibrated than CV probs → assignment noise dominates (consistent with depth-alone −0.9% at 27%); (H2) test recording groups do NOT obey all-distinct (test clips are trimmed segments — organizers may have sliced multiple clips from one activity); (H3) public-subset noise (~2-4 clips). Action: do NOT apply Hungarian until base model ≥60% AND H2 is tested (check: do same-group test clips get near-identical embeddings/predictions more often than distinct-class train groups would?).
**Beliefs updated:** B-004 ↓ 40% (CV predicts LB direction but with −9pt offset; must track deltas not absolutes and fix CV harshness); B-007 ↓ 50% (Hungarian flipped sign on LB); B-011 mechanism still 95% on train, but its TEST transfer now uncertain (H2).
**Next:** devil's-advocate pass NOW; then improvement wave: depth ROI recipe (Q-70/71), skeleton DG block (Q-11b/Q-10/Q-34), CV redesign to include harder folds.

---

---

## EXP-006 — DG-gap diagnostics: subject-split vs random-split per modality
**Date:** 2026-07-27 · **Queue ID:** follow-up of EXP-002 failure analysis · **Tier:** exploit (diagnostic)

**Hypothesis:** If depth's 27% is a cross-subject problem, random-split accuracy will be high (≥60%); if it's a recipe problem, random-split will be low too.
**Setup:** (a) depth no-aug subject fold0; (b) depth no-aug random 25% split; (c) skel no-aug random split. 40/40/60 ep. NOTE: random split leaks same-recording/session clips across train/val → random numbers are OPTIMISTIC upper bounds.
**Results:**
| modality | subject-CV | random-split | DG gap |
|---|---|---|---|
| depth (no-aug) | 23.7% (fold0) | **32.4%** | ~9 pts |
| skel (no-aug) | 53.0% (4-fold) | **73.2%** | ~20 pts |
**Conclusion:** Diagnosis is bimodal and actionable: (1) **Depth is recipe-bottlenecked** — it can't even fit in-domain (32% vs paper's 90% with pretrained ResNet-50 at full res). Root-cause candidates: person small/off-center at 120×160 full-frame; too few epochs; too small CNN; temporal pooling. DG work on depth is premature. (2) **Skeleton is DG-bottlenecked** — solid recipe (73% in-domain), loses 20 pts to unseen subjects → normalization/mixup/adversarial/aug-ablation is where skeleton gains live. Depth aug wasn't the problem (no-aug fold0 23.7 ≤ with-aug).
**Confidence in conclusion:** 85%.
**Unexpected observation:** Even with same-session leakage inflating it, depth random-split is only 32% — the recipe deficit is worse than any DG effect.
**Beliefs updated:** B-006 revised (see ledger): visual weakness = recipe not modality; B-002 ↑ 65%: skeleton confirmed backbone for now; B-001 refined: the cross-subject cliff is modality-dependent (20 pts skel vs 9 pts depth at current depth quality).
**Next ideas:** Q-70 person-ROI cropping via depth foreground (top priority for visual); Q-71 depth recipe scale-up (epochs/width/frames/res); skeleton DG block (Q-10, Q-11b/c, Q-34, Q-32).

---

## SUB-001/002 — First submission pair prepared (NOT yet submitted)
**Date:** 2026-07-27 · files: submissions/sub_fuse3_hung.csv, submissions/sub_fuse3_nohung.csv
Fusion skel_noaug:0.6 + imu_aug:0.2 + depth_aug:0.2, 4-fold checkpoint prob-averaging, temporal-center eval. Hungarian version changed 55/405 predictions across 115 multi-groups (de-duplicated Walk 60→52, Stir 40→25). OOF reference: 54.30% argmax / 55.44% Hungarian. Purpose of the pair: (1) calibrate CV↔LB (B-004), (2) measure the Hungarian lever on real LB (B-007/B-011). ATHARV: submit BOTH on the same day; record public LB scores in this log.

---

---

## EXP-002 — Depth-scalar CNN baseline (TSN-style, with aug)
**Date:** 2026-07-27 · **Queue ID:** Q-04 · **Tier:** exploit

**Hypothesis:** 8 uniform frames of JET-inverted scalar depth (120×160) + small 2D-CNN + temporal pool ≥50% subject-CV from scratch.
**Setup:** train.py --modality depth --aug, FrameCNN 1.19M params, 40 ep, bs 48.
**Expected gain:** ≥50%
**Actual gain:** **26.71% ± 2.21% — hypothesis strongly REFUTED (−23 pts)**
**Conclusion:** From-scratch small visual CNN massively underperforms the paper's 90% (pretrained ResNet-50, random split). Which factor dominates (data scarcity vs cross-subject appearance overfit vs training recipe vs aug harm) is UNKNOWN → EXP-006 diagnostics launched (subject vs random split, no-aug).
**Confidence in conclusion:** 90% that depth@27% is real under this exact recipe; 30% that it's depth's ceiling.

### Failure analysis
**3 reasons:** (1) ~2.3k training clips is tiny for visual texture learning from scratch; (2) CNN may key on subject appearance/body shape (nearest depth blob) — cross-subject collapse; (3) 40 epochs + OneCycle may underfit; no train-acc logging to tell (harness gap — fixed in EXP-006 interpretation via rand-split).
**3 alternatives:** (1) bundled visual aug harmful (crop+erase together); (2) depth scalar input representation poor (invalid=0 conflates with near; no normalization per scene); (3) temporal pooling too lossy (mean+max over 8 frames).
**3 follow-ups (launched as EXP-006):** (a) depth no-aug subject fold0; (b) depth no-aug RANDOM split — the DG-gap probe; (c) skel no-aug random split for the same gap on skeleton.
**Beliefs updated:** B-006 ↓ 70%→40% (visual backbone thesis in doubt pending EXP-006); B-002 ↑ (skeleton back as the backbone candidate).
**Next ideas:** if rand-split depth is HIGH (≥60%): invest in visual DG (mixup, subject-adversarial, stronger aug, IR/depth ensembling). If LOW (≤40%): visual needs capacity/epochs/recipe work first.

---

## EXP-005 — 3-stream late fusion (OOF) + Hungarian
**Date:** 2026-07-27 · **Queue ID:** Q-20 (OOF version) · **Tier:** exploit

**Hypothesis:** Late fusion of skel(53.0)+imu(26.8)+depth(26.7) ≥ +5% over best single.
**Setup:** fuse_oof.py weight grid 0.1 steps on OOF probs (weights tuned on OOF — mild optimism).
**Actual gain:** best weights 0.6/0.2/0.2 → **54.30%** argmax (+1.26 over skeleton), **55.44%** with Hungarian (+2.4 total).
**Conclusion:** Fusion helps but below +5% hypothesis — the weak streams (27%) contribute little; B-003 partially supported. Hungarian on fused probs +1.14% (vs +0.63 on skeleton alone, and NEGATIVE −0.93 on weak depth alone → apply Hungarian only to strong/calibrated probs).
**Confidence:** 80%.
**Unexpected observation:** Hungarian can HURT on badly-calibrated probs — assignment noise displaces correct predictions (depth alone: 26.7→25.8).
**Beliefs updated:** B-003 ↓ 75%→65% (fusion gain scales with stream quality, not stream count); B-007 note: Hungarian gain requires calibrated fused probs.
**Next ideas:** improve streams before widening fusion; re-tune fusion after every stream upgrade.

---

---

## EXP-003 — IMU 1D-CNN baseline (with aug)
**Date:** 2026-07-27 · **Queue ID:** Q-03 · **Tier:** exploit

**Hypothesis:** 5-device 10 Hz IMU tensor (85ch × 64) + SeqTCN ≥35% subject-CV.
**Setup:** train.py --modality imu --aug; 0.65M params, 60 ep.
**Expected gain:** ≥35%
**Actual gain:** **26.75% ± 3.82%** — below hypothesis.
**Conclusion:** IMU is weak cross-subject on this data: ~2 s clips @10 Hz ≈ 20 samples/device is very little; paper's 45.5% was random-split. High fold σ (3.8%) suggests user-dependent sensor signal. Keep only as fusion diversity; do not invest in IMU architecture search yet.
**Confidence in conclusion:** 75% (aug suite untested for harm here, like EXP-001b; orientation-invariant features unexplored — Q-12 could still add a few %).

### Failure analysis
**3 reasons:** (1) 10 Hz × 2 s gives ~20 timesteps — most gesture dynamics are invisible; (2) my aug suite may hurt (same bundling flaw as EXP-001b); (3) angle/mag channels are device-orientation-specific → subject/mounting overfit.
**3 alternatives:** (1) resampling to fixed 64 steps stretches variable durations — destroys absolute rhythm cues; (2) per-channel normalization constants (500°/s etc.) may be poorly scaled; (3) 4% device dropouts unhandled beyond zero-fill.
**3 follow-ups:** (1) no-aug IMU run; (2) acc-magnitude + gyro-magnitude orientation-invariant channels; (3) absolute-time positional channel instead of stretch-resampling.
**Beliefs updated:** B-009 confirmed (65%→80%): weak alone, cheap diversity.
**Next ideas:** defer IMU tuning until after fusion baseline exists.

---

---

## EXP-004 — Hungarian distinctness assignment simulated on skeleton OOF
**Date:** 2026-07-27 · **Queue ID:** Q-68 · **Tier:** exploit

**Hypothesis:** Per-recording-group distinct-label assignment (train-proven constraint, B-011) adds +2-5% over argmax.
**Setup:** code/hungarian_sim.py; OOF probs from skel_noaug 4-fold checkpoints (n=2700 val clips); groups = radar filename ts; scipy linear_sum_assignment on −log p.
**Expected gain:** +2-5%
**Actual gain:** **+0.63%** (53.04% → 53.67%); 98% of val clips sit in multi-clip groups; 246 groups had argmax collisions; on multi-group clips 53.38% → 54.03%.
**Conclusion:** Constraint is genuinely valid (never hurts; every changed prediction was in a collided group) but the lift is bounded by P(true label is the runner-up) — at 53% base accuracy, most collision fixes still miss. Keep as a free, safe post-process; expect the absolute lift to change (direction uncertain: fewer collisions vs better fixes) as base accuracy rises. Re-measure at each milestone.
**Confidence in conclusion:** 85%
**Unexpected observation:** Train val groups reach size 10 (test max 8) — simulation slightly pessimistic vs test's smaller groups? Unclear; group-size-stratified analysis later if it matters.
**Beliefs updated:** B-007 confidence stays 80% but expected magnitude ↓ (+0.5-1.5% not +2-5%); B-011 mechanism confirmed on real predictions.
**Next ideas:** soft-constraint variant (Sinkhorn / duplicate penalty instead of hard assignment); re-run on fusion OOF.

---

---

## EXP-000c — Round-2 audits (Q-60 timeline, Q-61 cohorts, Q-62 fixes, Q-63 depth scale, Q-64 station) + recording-group purity check
**Date:** 2026-07-27 · **Queue IDs:** Q-60,Q-61,Q-62,Q-63,Q-64 · **Tier:** foundation
**Setup:** Workflow wf_2c584dd3-05f (5 agents) + my direct train-radar-timestamp purity check. Artifacts: timeline_merged.csv, test_adjacency_prior.csv, test_cohorts.csv, date_map.csv, depth_jet_scale_probe.csv, station_class_prior.csv.

**Q-60 timeline:** Test clips are CLEAN held-out recordings: 0 timestamp overlaps with train, 0 shared camera recordings (anchor = t0 − frame_idx/10 clustering), min gap to any train trial 284 s, block-level alternation only. **Adjacency prior DEAD: same_class(nearest before, after) = 0/405.** Rules-consistent with users 10/11 (recorded between batch-B slots, May31-Jun2) and 25/26 (Jun12-17). BONUS: 405 test clips = **144 continuous recordings** (29 singletons, 376 clips in 115 multi-clip groups, sizes 2-8; radar filename timestamp = authoritative key, present even on empty radar files).
**Recording-group purity (my check on ALL 2914 train radar files):** 741 multi-trial train groups → **user-pure 741/741, class-pure 0/741, all-classes-DISTINCT 741/741, station-pure 741/741, trial-name-pure 741/741**. So trial A-B-C = one continuous recording pass in which the user performs a SEQUENCE OF DIFFERENT activities; C=1..3 are repeated passes. Implication for test: within each multi-clip recording group, labels are (a) same user, (b) same station, (c) almost surely pairwise DISTINCT → constrained-assignment (Hungarian) post-processing lever covering 93% of test clips. The "same-recording ⇒ same-class pooling" idea is DEAD (checked before building — saved us from a systematically harmful smoother).
**Q-61 cohorts:** 2-way split PROVABLE: E (radar-usable, May31-Jun2, n=199) vs L (radar-empty, Jun12-17, n=206). 4-way E1 99 / E2 100 / L1 105 / L2 101 ≈ 4×101 (moderate confidence; 21 slots and the 143-144 radar recording groups are the provably-pure units; E1/E2 boundary heuristic; S16 has NEW→OLD→NEW leg-MAC blocks—could be two people turn-taking). SM_test_0054 = fully sensor-less outlier (no radar, no IMU rows).
**Q-62 fixes:** True test durations: median 1.90 s, max 10.00 s (hard cap; train max 26.6 s) — test trimmed ~20% shorter. Date map: batch A u1-5 (May7-8), B u6-9 (May30-Jun2), C u16-20 (Jun9-11), D u21-24 (Jun11-13). Radar in TEST is usable for ALL early-cohort clips (198) and empty for ALL late ones → radar could serve the E-half of test (trained on users 1-9). IMU near-complete for all users.
**Q-63 depth scale:** JET mapping is FIXED/ABSOLUTE (inversion lossless, median NN-dist 0.0; background patches constant ±0.5 idx across frames/users/days; per-frame rescale ruled out). Depth index = absolute depth proxy comparable everywhere; guard: physical camera repositioning between dates; mask black invalid (9-21%/frame, corners always dead).
**Q-64 station prior:** DEAD END: I(class;station)=0.711 bits but station-recovery from IR/depth backgrounds tops out ~50-53% (within-station corr 0.186 vs between 0.175) → effective 0.092 bits. Station-only classifier = 12.3%. Revisit only if another station channel appears.

**Conclusion:** Metadata game reshaped: adjacency and station priors dead; the live levers are (1) recording-group distinctness assignment (93% coverage), (2) user-pure grouping → per-user/cohort adaptation, (3) 2-way cohort split for BN adaptation. Test is clean cross-subject — the real model carries the campaign.
**Confidence:** 90% (train-side purity facts are exhaustive, not sampled).
**Beliefs updated:** B-007 revised (see ledger), B-008 softened (radar usable for E-half of test), NEW B-011 (recording groups).
**Next ideas:** Q-68 Hungarian distinctness assignment; Q-69 verify group-key recoverability end-to-end in inference code; per-cohort adaptation stays Q-54.

---

## EXP-001b — Skeleton TCN + augmentation suite
**Date:** 2026-07-27 · **Queue ID:** Q-11 (partial) · **Tier:** exploit

**Hypothesis:** Yaw-rotation ±30° + scale ±15% + jitter + joint-dropout adds +3-8% over EXP-001a.
**Setup:** identical to EXP-001a but --aug (also temporal jitter sampling).
**Expected gain:** +3-8% → ~56-61%
**Actual gain:** **51.49% ± 2.15% — NEGATIVE (−1.6% vs no-aug 53.04%)**
**Conclusion:** Hypothesis REFUTED at these aug strengths. The full suite hurts.
**Confidence in conclusion:** 70% (single seed; σ across folds 2.2% — the deficit is within ~1σ, but direction is clear enough to redesign).

### Failure analysis (mandatory)
**3 reasons why it failed:**
1. Yaw rotation may be harmful: H36M-lifted poses are camera-relative and activities are station-anchored (facing the sink/desk); rotating breaks a genuine, test-transferable orientation prior.
2. Scale ±15% destroys absolute body-size/height in meters, which the floor-aligned z uses to distinguish e.g. Lie_down/Sit vs Stand classes.
3. Joint dropout (2 joints, 20% of clips) may be too destructive with only 17 joints when wrist/head joints carry most class signal.
**3 alternative explanations:**
1. Experiment invalid as a test of augmentation per se — 5 augs bundled; one bad component can mask 4 good ones.
2. Temporal jitter sampling (train-time) changed the effective frame sampling vs eval — train/eval distribution mismatch.
3. Seed noise: ±2.2% fold σ; deficit could be ~luck (needs seed repeat).
**3 follow-up experiments:**
1. Q-11b: one-aug-at-a-time ablation (rotation-only, scale-only, jitter-only, dropout-only, temporal-jitter-only) — 5 cheap runs.
2. Q-11c: milder magnitudes (yaw ±10°, scale ±7%, no joint dropout).
3. Q-10: seq-level normalization (torso-scale) instead of scale-aug — normalization may beat augmentation for size invariance.
**Beliefs updated:** B-001 nuance: not all invariance helps; augmentation must respect station-anchored orientation priors.
**Next ideas:** queued Q-11b/c.

---

---

## EXP-001a — Skeleton TCN baseline, no augmentation
**Date:** 2026-07-27 · **Queue ID:** Q-02 · **Tier:** exploit

**Hypothesis:** Normalized 17×3 H36M skeleton sequences (person-0, T=32, coords+velocity) + small TCN ≥45% subject-CV.
**Setup:** code/train.py --modality skel --tag skel_noaug; SeqTCN 0.66M params, GroupNorm, AdamW 1e-3 OneCycle, 60 ep, label smoothing 0.1, batch 64, seed 0; 4-fold subject CV (cv_folds.json).
**Expected gain:** ≥45% (baseline floor)
**Actual gain:** **53.04% ± 1.32%** (folds in results.csv)
**Conclusion:** Hypothesis confirmed with margin. A 0.66M no-aug skeleton model nearly matches the dataset paper's ~56.4% LOSO (which used contrastive learning). Skeleton stream is real; fold σ is low (1.3%) → CV is stable.
**Confidence in conclusion:** 85%
**Unexpected observation:** —
**Beliefs updated:** B-002 ↑ (skeleton competitive after all); B-004 ↑ (fold σ low → protocol usable).
**Next ideas:** person-selection policy ablation; torso-scale normalization (Q-10).

---

---

## EXP-000 — Environment + data reconnaissance (not a model experiment)
**Date:** 2026-07-27 · **Queue ID:** — · **Tier:** foundation

**Setup:** Local: RTX 4060 8 GB, 16 cores, 15 GB RAM, torch 2.12.0+cu130 OK. Train 43 GB extracted; test zip extracted.

**Findings:**
- Layout: `HAR/data/<modality>/<id>_<name>/user<N>/<s>-<r>-<t>/files`. Modalities: Depth_Color, IMU, IR, Radar, Skeleton, Thermal — all 40 classes each.
- Trial folders look like `1-1-1`, `1-1-2`, `1-1-3` (scene?-?-repeat). Class 0 Depth has 11 users, NOT all 18 → per-class user coverage is partial; must profile the full grid.
- Depth_Color: color-mapped PNG frames (~30 fps, timestamped filenames). IR: PNG frames. Thermal: JPG frames (`frame_000226.jpg`, no timestamps in name).
- IMU: two CSVs per trial — `up(LA+RA+C).csv` (left arm, right arm, chest) and `down(LL+RL).csv` (legs); Chinese headers; per row: device name + acc(3) gyro(3) angle(3) mag(3) quaternion(4) temp battery. Multiple devices interleaved by row → must pivot by device. ~137 rows / ~4.5 s trial → ~10 Hz per device per file. Timestamps present.
- Radar: single CSV per trial, header `timestamp,frame,DetObj#,x,y,z,v,snr,noise` — sparse point cloud per frame.
- Skeleton: `predictions/*.json` per RGB frame (pose-estimator output; RGB itself withheld). Schema TBD.
- Test: 405 clips `SM_test_XXXX/`; sample_submission predicts 34 for all.

**Conclusion:** Data is rich but heterogeneous; profiling grid (modality × class × user × trial counts, frame counts, missing patterns, test availability) is the required next step.
**Beliefs updated:** none yet (pre-experiment).
**Next ideas:** profiling workflow; skeleton JSON schema inspection; test-set modality census.

---

## EXP-000b — Deep profiling workflow (8 agents: censuses, 4 modality deep-dives, literature, critic)
**Date:** 2026-07-27 · **Queue ID:** — · **Tier:** foundation
**Setup:** Workflow wf_411e5e03-190; full reports in workflow transcript; artifacts in research/artifacts/ (train_index.csv, test_index.csv, skeleton_quality_sample200.csv, skeleton_frames_per_trial.csv, imu_duration_by_class.csv, imu_train/test_device_scan.psv, visual_trial_profile.csv, radar_file_sizes.csv, radar_profile_sample300.csv).

**Train census:** 18 users (1-9, 16-24), 3036 unique samples (class,user,trial), ~2900 trials/modality. NO user has all 40 classes (user5 worst: 17; user18 best: 35). Classes 25/26 (Watch_TV/Play_games) missing for 14/12 of 18 users. Class imbalance 12 (cls 25) to 319 (cls 36 Walk) samples. Trial name A-B-C: A=camera station 1-7, C=repetition 1-3. Missing-modality: 2748/3036 have all 6; 103 Thermal-only samples; Thermal is the misfit (coverage differs, ~25 fps unsynced, no timestamps).
**Test census:** 405 clips; Depth/IMU/IR/Skeleton present in all 405; Radar missing 1 + EMPTY in 206 (51%); Thermal missing 10. Timestamps NOT anonymized (Depth/IR/Skeleton/Radar filenames + IMU internal): dates 05-31(62) 06-01(57) 06-02(80) 06-12(23) 06-13(53) 06-16(29) 06-17(101); >30-min-gap clustering → 16 sessions. 275 clips fall on train recording days (interleaved with train users' sessions); 130 clips (06-16/17) on days with zero train coverage. Test clips slightly shorter than train (frames med 20 vs 24-26).
**Skeleton:** H36M 17-joint **3D** (meters), 10 Hz, per-frame JSON list of persons; keypoint_scores always exactly 1.0 (useless); pelvis root-centered x,y and floor-aligned min-z=0 PER FRAME (global translation destroyed by preprocessing). Multi-person 5.6% of frames overall but 84.6% in Comb_hair, 53.5% Brush_teeth (mirror reflections at bathroom station → itself class-discriminative). Frames/trial: min 1 / med 24 / max 236. Test identical format.
**IMU:** 5 devices (chest WTC, arms WTLA/WTRA, legs WTLL/WTRL) at **~10 Hz only**, 21 Chinese-header cols (acc g, gyro °/s, angle °, mag uT, quat, temp, battery). Only 7 physical sensors in whole dataset → MACs shared across users, NO per-user leak; but legs have OLD pair (users 1-5) vs NEW pair epochs, and 50 TEST clips use the OLD leg pair (cohort fingerprint). 4% trials missing ≥1 device; 40 train trials fully empty; SM_test_0054 IMU fully empty; SM_test_0001 down-csv has English 23-col header (parse both). Trial duration med 2.22 s train / 1.80 s test. Duration is class-informative (7× spread: Stand_up 0.87 s vs Sweep 6.22 s) but test is systematically shorter → trimmed.
**Visual:** Depth_Color 640×480 PNG = JET-colormapped depth, exactly invertible to 8-bit scalar (low=near); ~10% black = invalid. IR 640×480 gray PNG = near-IR photo (full texture, faces visible). Thermal 320×240 JPG ironbow-ish colormap, NOT cleanly invertible, ~25 fps, unsynced, no timestamps but CONTINUOUS session frame counter (second session-grouping key for test!). Depth/IR/Skeleton are frame-exact aligned at 10 fps (same timestamps+indices). ONE apartment, ~7 fixed camera stations; station↔activity correlated (bathroom→wash face); same rooms in test; different users share rooms → environment transfers, station prior is exploitable.
**Radar:** sparse (med 6.6 pts/frame, cap ~20), Doppler clipped ±0.97 m/s, med 2.5 s recorded. **Header-only EMPTY for 100% of users 16-24** (51.6% of train files); test 51% empty → radar-empty is a test cohort fingerprint; as a signal, DEFER (weak aux at best).
**Literature/rules:** Kaggle deadline **Sep 15, 2026**; code upload Sep 22; top-15 verification = live Zoom inference on fresh samples + offline reproduction within ≤10% of Kaggle score. Final score: Kaggle private 20%, on-site private test (8 NEW subjects) 30%, reproducibility 10%, report 20%, presentation 10%, efficiency 10%. Submission package = code + SINGLE checkpoints/model.pth ≤100 MB (implicitly caps ensembles). ImageNet-init of small CNNs AMBIGUOUS (organizers' own paper used pretrained ResNet-50) → email cuhkx.competition@gmail.com for ruling. Dataset paper (arXiv 2512.07136): random-split baselines Depth 90.5 / IR 90.2 / Thermal 92.6 (ResNet-50) / Skeleton 79.1 (MotionBERT) / mmWave 46.6 / IMU 45.5; **cross-subject LOSO ≈ 56.4%**. Public LB (135 teams): #1 0.8358, #2 0.7662, cluster 0.73-0.77. No public baseline code.
**Critic (contradictions to resolve):** (1) test-user attribution vs "users 10/11/25/26" rules statement — timestamp overlap means interleaved recording, NOT train-user identity; needs timeline merge audit. (2) Test-duration median 10.3 s figure is likely a bug (all other evidence says ~2 s) — recompute. (3) Train census counted files not contents → its Radar/IMU coverage overstates usable data. (4) Underweighted signals: session→class adjacency leakage for 275 clips; cohort fingerprint = radar-empty × leg-MAC epoch × thermal-missing × session; station prior P(class|station) + scene classifier; multi-person rate as feature; Thermal frame-counter session linking.

**Conclusion:** Strategy reshaped: (1) visual modalities are the accuracy backbone, not skeleton; cross-subject regularization of a small depth/IR CNN is the core battle; (2) skeleton is still the cheapest strong stream but its ceiling is lower than assumed (79% with pretrained MotionBERT on random split); (3) IMU is weak-but-cheap ensemble diversity; radar defer; (4) session/cohort structure of test is a legitimate, reproducible post-processing lever for the Kaggle LB (worth several %), but the on-site test (30% of final) has none of it — core model quality still dominates; (5) 0.75 public ≈ top-5; 0.76+ realistic target from fusion + DG; leader 0.836 suggests either strong DG or session exploitation.
**Confidence in conclusion:** 80% (pending critic follow-ups).
**Beliefs updated:** B-002 ↓45%, B-003 ↑75%, B-005 confirmed shape (single model.pth), NEW B-006..B-010 (see BELIEFS.md).
**Next ideas:** Q-60 timeline merge audit, Q-61 cohort clustering, Q-62 duration recompute, Q-63 depth-scale probe, Q-64 station prior, Q-65 organizer email draft — queued as follow-up profiling round.

---

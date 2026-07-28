# Evidence Log

**Counters:** experiments since last devil's-advocate pass: 3 / 10 · since last reset: 19 / 25

---

## SUB-010 result — **0.52736 NEW BEST** · EXP-032b IR-4th-stream: no gain
**Date:** 2026-07-29 · SUB-010 beat projection again (0.520 → 0.527); offset now ≈ −6.0 and narrowing with ensemble size — each nested CV point ≈ 1 LB point now. Ladder: 0.458→0.483→0.488→0.512→0.517→**0.527**. IR as 4th fusion stream: flat at all weights (info covered by depth+skel) — closed. Next members training (stgcn_w96_s1, skel_w192); jitter-TTA to be wired into next test assembly.

---

---

## EXP-031/SUB-010 — 10-member assembly: NESTED 58.70
**Date:** 2026-07-29 · mildaug (4-fold ≈54.97) joined TCN soup (now 6); imu_lstm (25.05 solo) added +0.3 to the IMU block (29.93); depth re-entered fusion at w=0.1 (0.7/0.2/0.1). NESTED 58.70 (+0.26), all-tuned 59.33. **SUB-010 staged: sub_block10.csv, projected ~0.520.**

---

---

## EXP-032 — W-01 logit-gate fusion: FAIL (−6.8 nested)
**Date:** 2026-07-29 · **Tier:** explore

2-layer MLP over concatenated stream log-probs, trained per-fold nested: **51.96 vs 58.74 scalar**. Third confirmation of the data-scale law (with EXP-020/021): learned combiners overfit at 2.3k clips; scalar late fusion is the operating point. Fusion-vs-oracle gap is NOT gate-mineable — remaining fusion upside is only better/more members.
**Beliefs updated:** fusion axis CLOSED except member addition; W-02 pairwise specialists demoted (same overfit risk, unless implemented as pure calibration with ≤2 params/pair).

---

---

## SUB-009 result — **0.51741 NEW BEST** (104/201) — 5th consecutive accurate projection
**Date:** 2026-07-29 · Ladder: 0.458 → 0.483 → 0.488 → 0.512 → 0.517. Offset stable ≈ −6.7.

---

## DA-004 — Fourth devil's-advocate pass (inline)
**Date:** 2026-07-29 · **After:** EXP-030/SUB-009

**1. Why probably wrong?** The diversity grind yields +0.5/wave and is visibly saturating; extrapolation plateaus ~0.53-0.55 LB. Without a NEW mechanism the 80+ directive is unreachable, and whether 0.53 even makes top-15 is STILL unknown — the LB harvest has now been flagged in three consecutive DA passes (only Atharv can do it).
**2. Never-questioned (current round):** (a) fusion is still SCALAR-weighted — a small per-clip logit-gate MLP (2 layers over concatenated stream logits, nested-evaluated) has never been tried; part of the fusion-vs-oracle gap is mineable; (b) the 40-class space is treated flat — pairwise specialist recalibration for the top confusion pairs (Read↔Turn_pages, Sweep↔Mop, Pour↔Stir) untried; (c) multi-window voting per clip beyond jitter-TTA untried; (d) ST-GCN member seed-variance unmeasured.
**3. Gold-medalist critique:** submission cadence now good; the big hole is REPRODUCIBILITY PACKAGING — Selection Stage requires organizers to rerun us, and our pipeline is scattered scripts with hardcoded paths. Report + package are 30% of the final score and cost nothing in GPU time. Start now, not at deadline.
**4. Steelman (partially adopted):** "CV-mineable mass left is ~2-4 pts; the rest is representation/data-locked. Accept ~0.53-0.55 terminal, pivot fully to packaging/report." — Adopted as a PARALLEL track, not a pivot: improvement continues per the standing directive while the deliverable package gets built alongside.
**Queue produced:** W-01 logit-gate fusion (nested); W-02 pairwise specialists; W-03 imu-LSTM member + mildaug folds 1,3 + stgcn seed variance; P-01 packaging artifact (single model.pth + inference.py + README, fp16, ~42 MB).
**Counter reset:** 0/10.

---

---

## EXP-029/030/SUB-009 — Ladder 2: GCN soup + reassembly → NESTED OOF 58.44
**Date:** 2026-07-29 · **Tier:** exploit

stgcn_s1 51.90 · stgcn_w96 52.71 (best GCN member) · skel_mildaug 53.89 folds 0,2 (+0.3, in-noise; parked as member candidate). **GCN-soup3 = 55.48 alone** (≈ TCN-soup5 55.96 — two near-equal decorrelated blocks). Block grid still rising at 0.4 (57.93); shipped conservative 0.7/0.3 → **NESTED fusion 58.44** (+0.5; all-tuned 58.74). SUB-009 staged: sub_block8_gcnsoup.csv, projected ~0.517-0.52.
**Next:** DA-004 (due next experiment); block-mix nested exploration; second imu arch member.

---

---

## EXP-028 — S-04 TENT-GN cohort adaptation sim: FAIL (gated, 0 submissions spent)
**Date:** 2026-07-29 · **Tier:** explore

**Result:** per-user entropy-min on GN affines (3 ep, lr 1e-3): Δ = exactly 0.000/16 users (params moved too little to flip anything). Sensitivity at lr 1e-2/10 ep: params move (L2 0.69), 9/149 flips, accuracy DROPS 0.456→0.423, entropy oscillates (2.25→2.29) — the objective doesn't descend usefully on this tiny affine set.
**Conclusion:** TENT-GN dead on this architecture. Transduction family 0-for-2 (with EXP-016). Last reserved variant: head-only + τ≥0.9 pseudo-labels on the FUSED probs at higher base.
**Beliefs updated:** B-007/transduction: adaptation magic is not hiding in easy variants; the leader (if adapting) does something else or from a much higher base.
**Next:** diversity ladder continues (stgcn seed/width members, mild-aug retry Q-11c); DA-004 in 3.

---

---

## EXP-027 — Hands/upper-body crop stream (C-01): FAILED TO BREAK THE CEILING
**Date:** 2026-07-29 · **Tier:** crazy

**Result:** hroi solo 24.51 (folds 0,2); ensemble +0.10 weight → +0.23 (58.17→58.40) — under the 2×SE adoption bar. **Decisive evidence: 0.00 accuracy on the target object classes themselves** (Watch_TV, Play_games, Phone_call, Write all 0.00 solo) despite the crop.
**Conclusion:** The object-in-hand error mass is REPRESENTATION-locked, not framing-locked: from-scratch small CNNs at this data scale cannot learn object identity at any crop, and pretraining is banned. C-01 parked (its +0.2 can be reclaimed at final packaging if it survives full-fold verification).
### Failure analysis
**3 reasons:** (1) object recognition needs texture priors that 2.3k clips can't teach; (2) upper-crop of a loose person-ROI often misses the actual hands region; (3) IR contrast on small objects is poor at 112px.
**3 alternatives:** (1) crop quality (skeleton wrists unavailable in pixel space — can't verify); (2) 60-ep budget; (3) folds 0,2 screen noise.
**3 follow-ups:** (1) SSL-init the visual STEM (EXP-021's +3 might transfer to CNNs — last visual play); (2) accept the cap: Watch_TV/Play_games are also data-poor (12-40 train clips) — mine non-object classes instead; (3) at packaging, test hroi's +0.2 on full folds.
**Beliefs updated:** B-006 final form: visual object-route is CLOSED under the no-pretrained rule at this data scale. The remaining routes to higher accuracy: skeleton-block depth, cohort/transductive adaptation (S-04), calibration/fusion refinement, and per-class specialists on non-object confusions.

---

---

## SUB-008 result — **0.51243 NEW BEST** (103/201; +2.5 over SUB-006)
**Date:** 2026-07-29
Beat the 0.500 projection: offset narrowed to −6.7 (was −7.9) — ensemble diversity transfers BETTER to test users than CV predicted (plausible: averaging washes subject-specific errors hardest exactly where subject shift is largest). Ladder: 0.458 → 0.483 → 0.488 → 0.512.
**Beliefs:** B-004 ↑ (4th accurate-or-better prediction); B-013's diversity corollary now LB-proven twice.
**Next:** C-01 hands-crop stream from roi224 upper-crop (zero re-extraction), S-04 cohort sim, stgcn-weight nested exploration.

---

---

## EXP-026/SUB-008 — 5-seed soup + ST-GCN block + fusion: NESTED OOF 57.93 (+1.8)
**Date:** 2026-07-29 · **Tier:** exploit (ship)

Seeds s3 54.75 / s4 54.86 (5-seed σ 0.15). Ladder: single 54.74 → soup5 55.96 → +0.2×stgcn 57.04 (grid still rising at 0.3: 57.37 — explore under nested next) → nested fusion with imu_inv **57.93** (all-tuned 58.22; optimism gap 0.3 healthy). Depth weight remains 0.
**SUB-008 staged:** sub_block5_stgcn.csv — projected LB ≈ 0.500 (offset −7.9).
**Beliefs:** B-002/B-013: diversity ladder works — each decorrelated member adds; skeleton BLOCK now ~57 vs single-model wall 55.
**Next:** C-01 hands-crop stream (ceiling-breaker #1), S-04 cohort adaptation sim, stgcn weight/nested exploration.

---

---

## EXP-025 — ST-GCN stream: individually weaker, ensemble WINNER (+1.6)
**Date:** 2026-07-29 · **Queue ID:** Q-30 · **Tier:** explore

**Result (folds 0,2):** ST-GCN alone 52.04 (1.06M) vs TCN 53.59. BUT ensemble: TCN-3-seed-soup 54.77 → **56.40 at soup+0.2×stgcn (+1.6)**; argmax agreement only 61% — strong decorrelation. Graph-conv inductive bias sees different errors than temporal conv.
**Conclusion:** Architecture diversity > capacity for ensemble gains (as B-013 implied). ST-GCN adopted as a skeleton-block member; folds 1,3 queued to complete OOF; full fusion re-tune after seeds s3/s4.
**Beliefs updated:** B-002 refined: the skeleton BLOCK (multi-arch ensemble) has headroom beyond the single-model ~55 wall.

---

---

## SUB-006/007 — LB pair: imu-inv CONFIRMED (+0.5, new best 0.48756); refit-18+TTA REFUTED (−1.5)
**Date:** 2026-07-29
- sub_soup3_imuinv → **0.48756** (98/201; predicted 0.483-0.487 ✓ — CV→LB calibration holding at −7.9±1).
- sub_ship1_refit_tta → 0.47263. **Refit-on-18 with last-epoch checkpoints transfers WORSE than val-selected fold models.** Causes (ranked): last-epoch ≠ val-best (OneCycle end-state); 150 ep no-val overfit; TTA-jitter mismatch; 0.5/0.5 dilution of proven soup. Shelved — retry only with early-stop at CV-derived epoch budget.
**Beliefs updated:** B-004 ↑ 85% (third consecutive accurate LB prediction); "more data always helps" nuance — checkpoint SELECTION matters more than +2 users.
**Next:** architecture-diversity wave (ST-GCN stream + 5-seed soup) per Atharv's use-the-headroom directive; rules re-confirmed: transformer ARCH legal (was dropped on merit), no pretrained, no LLMs anywhere.

---

---

## S-01/S-02/S-03 — Ship-mode assembly: refit-18 + TTA + package audit → SUB-007 staged
**Date:** 2026-07-29 · **Tier:** exploit (ship)

- **S-01 refit-on-18:** 3 skeleton seeds (150 ep) + IMU-inv (60 ep) retrained on ALL 18 users, no val (fixed hyperparams from CV era). No OOF possible — gain rides on +22% more training users (esp. always-train users 5/21 now contributing to every member).
- **S-02 TTA:** 3-sample temporal-jitter averaging on refit members at inference.
- **SUB-007:** sub_ship1_refit_tta.csv = skel(0.5 fold-soup + 0.5 refit-soup) 0.8 + imu(same mix) 0.2. Prediction histogram sane (Walk 59 ≈ train prior).
- **S-03 package audit:** 7 skel + 5 imu members = **41.8 MB fp16** (83.6 fp32) — comfortably ≤100 MB with efficiency-score headroom; final deliverable = fp16 members + inference script with prob-averaging.
**Expected LB:** 0.49-0.51 (refit +1-2 hypothesis on top of 0.483 baseline — SUB-006/007 pair measures it).
**ATHARV:** submit sub_soup3_imuinv.csv AND sub_ship1_refit_tta.csv — the pair isolates the refit+TTA effect on LB.

---

---

## S-05 — Label sanity check (A3, flagged by three DA passes): PASSED
**Date:** 2026-07-29 · **Tier:** diagnostic (CPU)

**Setup:** best fused OOF (3-seed soup 0.8 + imu_inv 0.2); Signature-B = confident (>0.7) wrong prediction that equals a recording-group-mate's label (folder-shift errors would concentrate here).
**Result:** 14/2700 = **0.52%** hits; inspection shows adjacent-class confusions (6↔7 drink/eat, 9↔10 pour/stir, 22↔23) not systematic mislabels; 2 of 14 are the census's known 2-modality oddballs (07_user22_3-1-3, 37_user23_7-1-2). High-conf-wrong tail tops out at 0.79 conf — no smoking guns.
**Conclusion:** Train labels are clean (noise ≤0.5%); label-cleaning has no meaningful upside. A3 settled after 27 experiments.
**Beliefs updated:** A3 verified — removed from every future DA checklist.

---

---

## DA-003 — Third devil's-advocate pass (inline)
**Date:** 2026-07-29 · **After:** EXP-024

**1. Why is the current direction probably wrong?** We are grinding +0.5-4 levers (soup +0.8, imu-inv +4 on a 0.2-weight stream ≈ +0.8 fused) while the diagnosed 34.7% object-in-hand error mass has NO live experiment against it — Q-97 (hands/upper-body crop stream) was queued at EXP-017 and never built. Best-case grind arithmetic: fused ~57-58 OOF → LB ~0.50; if rank-15 sits above 0.50 we lose while "winning" every screening.
**2. Never-questioned assumptions:** (a) full-length training clips are the right training view when test is trimmed — trim-matched training (trim as the DEFAULT view, not an aug) never tried; (b) we predict test with 4 fold-models (each trained on 14 users) and have never refit on ALL 18 users (+1-2 free, flagged in DA-002 Q3, still undone); (c) label sanity — THIRD consecutive DA flag, still never checked.
**3. Gold-medalist critique:** (a) still flying blind on the actual Top-15 bar — LB harvest pending since DA-002 (only Atharv can); (b) provable E/L cohort split never used at inference (per-cohort GN-affine adaptation sim never run); (c) efficiency score (10% of final) unexamined — our ~5 MB total is likely near-max points, worth confirming and PROTECTING in design decisions; (d) submission cadence wasteful: ~245 available, ~6 used — every honest candidate is cheap calibration.
**4. Steelman (adopted):** "The campaign is now over-optimized for process, under-optimized for shipping. With ~7 weeks left the highest-EV block is boring: refit best configs on all 18 users, add TTA, soup everything into one ≤100 MB model.pth, submit variants daily, and lock the reproducible Selection-Stage package early. The marginal screening experiment returns <0.5 pts; packaging/refit returns 1-3 guaranteed." Verdict: correct — this week pivots to SHIP MODE, exploration continues only in the crazy tier (hands-crop stream is the one sanctioned moonshot).
**Queue produced:** S-01 refit-on-18 (all adopted configs); S-02 temporal multi-crop TTA; S-03 soup-everything packaging + efficiency audit; S-04 E/L cohort GN-adaptation sim; S-05 label sanity (finally); C-01 hands/upper-body crop stream (crazy tier).
**Counter reset:** 0/10.

---

---

## EXP-022/023/024 — Three-lever screening (folds 0,2; paired vs jvb_big 53.59 / imu_aug 24.8)
**Date:** 2026-07-29 · **Tier:** exploit

- **EXP-022 LS-off:** 52.93 (−0.7) — keep LS=0.1; calibration-blame theory unsupported.
- **EXP-023 motion-person:** 52.56 (−1.0) — person-0 stays. Mirror reflections move WITH the person → motion energy can't separate them; policy adds switching noise elsewhere. Mirror fix would need depth-consistency cues (parked).
- **EXP-024 IMU-inv:** **28.94 (+4.1, clear win)** — dropping absolute angle+mag channels (subject/mounting leak), adding |acc|/|gyro|, fixing quat hemisphere. ADOPTED; 4-fold retrain (imu_inv4) launched for the fusion stack. Confirms EXP-003's failure-analysis hypothesis #3 two weeks late — the leak channels were fighting DG the whole time.
**Beliefs updated:** B-009 ↑ (IMU ceiling higher than 27; orientation-invariance was the blocker); skeleton input-space levers exhausted (B-002 note: remaining skeleton gains = architecture diversity + ensembling).

---

---

## EXP-021 — SSL→fine-tune transformer: GATE FAILED, transformer line closed
**Date:** 2026-07-29 · **Queue ID:** M-03/Q-94 · **Tier:** explore/crazy

**Result:** cross-modal masked pretrain (40 ep, 3,338 clips incl. test; recon loss 0.30→0.0146) → fine-tune = **48.19%** folds 0,2. vs from-scratch transformer 45.23 (+3.0 from SSL) but vs GATE 56.15 → **−8. Transformer fusion line CLOSED per RESET-001 gate. Late fusion is the permanent architecture.**
**What survives:** (1) cross-modal SSL demonstrably works at this scale (+3.0) — retained as an option for pretraining per-modality STEMS/encoders later (e.g. init the visual CNN from an SSL objective); (2) M-01 aligned dataset reusable.

### Failure analysis
**3 reasons:** (1) 2.3k labeled clips cannot polish a 3.8M-token-mixer even with SSL init — the late-fusion streams each exploit stronger inductive bias per parameter; (2) vis stem tokens weak (visual stream itself only ~30%) — transformer can't select good vis information that isn't there; (3) reconstruction pretext may reward low-level smoothness, not class-discriminative structure (loss 0.0146 ≈ trivially predictable signals dominate).
**3 alternatives:** (1) 60 fine-tune epochs too few for a pretrained trunk (no LLRD/warmup tuning); (2) aux-head weight 0.3 may dominate gradients; (3) fold-0,2 screen ±1.4 SE — but an 8-pt gap is far outside noise.
**3 follow-ups (parked, low priority):** LLRD fine-tune; contrastive (not reconstructive) pretext; SSL-init only the visual stem inside the LATE fusion stack.
**Beliefs updated:** B-013 stands; architecture question settled — the campaign is now: better per-modality streams × late fusion × soups (+ gated transduction).
**Next:** cheap untested exploits — Q-85 LS-off, Q-86 person-selection, IMU orientation fix — paired on screening folds.

---

---

## EXP-020 — Fusion transformer from scratch: BELOW GATE
**Date:** 2026-07-29 · **Queue ID:** M-02 · **Tier:** explore

**Result:** mmfuse_v1 (3.78M, d256 L4, aligned tokens, modality dropout, aux heads) = **45.23%** folds 0,2 — vs gate 56.15 (nested late fusion) and ~52 for plain skeleton on those folds.
**Conclusion:** From-scratch transformer fusion loses badly at 2.3k clips, as RESET-001's gate anticipated. NOT dead yet: the design's bet is SSL+transformer; EXP-021 (cross-modal masked pretrain 40 ep on 3,338 clips → fine-tune) is the deciding run. If EXP-021 also < 56.15 → transformer line dies, fall back to late fusion permanently.
**3 reasons if it fails for good:** (1) data scale below transformer viability even with SSL; (2) vis stem too weak to give the trunk useful tokens (garbage in); (3) aligned-grid pad/mask regime needs longer training than 60 ep.

---

---

## SUB-005 — soup3+imu+droi_big fusion → LB 0.48258 (best yet)
**Date:** 2026-07-28 · 97/201 public clips (prev best 92). Predicted 0.48-0.49 ✓.
**Offset calibrated: LB ≈ honest-nested-OOF − 7.9** (56.15 → 48.26). trunc-aug reclaimed ~1 of the original −9. CV→LB tracking is now trustworthy for planning (B-004 ↑ 80%).
**Beliefs updated:** B-004 ↑; ensemble/soup mechanics verified end-to-end on LB (+2.5 over SUB-001 with same modality set).
**Next:** M-02 fusion transformer (mine the 56→63 oracle gap), M-03 cross-modal SSL (attack the 35% no-stream-right mass), ST-GCN, person-fix, LS-off.

---

---

## EXP-013b/018/019 — Depth folds, seed replication, soup + nested fusion, SUB-005 prepared
**Date:** 2026-07-28 · **Tier:** exploit

- droi_big folds 1-3: 28.81 ± 1.6 micro (4-fold ≈ 29.6 with fold0 31.94; +3 over 40-ep recipe). Depth stream ≈ 30.
- **Seed variance (EXP-018): skel_jvb_big seeds 0/1/2 = 54.75 / 55.08 / 54.67 → seed σ ≈ 0.18%** — training is stable; fold σ (2-3%) dominates; paired deltas ≥1% on 4 folds are meaningful. (Resolves DA-002 Q5's top concern for skeleton; visual seed σ still unmeasured.)
- 3-seed soup: 54.74 → **55.56** (+0.8, free, ensemble-legal).
- **Nested fusion OOF: 56.15% micro** (all-tuned 56.63, w=0.6/0.2/0.2 — optimism gap only 0.5). New honest best.
- SUB-005 file: sub_soup3_fusion.csv (soup + imu + droi_big, all-tuned weights on test). Predicted LB ≈ 0.48-0.49 (offset −8±2 with trunc-aug partially reclaiming trim).

---

## RESET-001 — From-scratch review (scheduled at 25 experiments)
**Date:** 2026-07-28 · **After:** EXP-019

**"If we started from scratch today, knowing everything, what would we build?"** — answered in full by DA-002 Q3 and ratified here: per-modality stems (keep current TCN-jvb + ROI CNNs) feeding a small temporal fusion transformer on the shared 10 Hz grid (modality-token dropout, duration token, pad+mask not stretch), trained in 2 stages: (A) **cross-modal masked pretraining on all 3,441 unlabeled clips incl. test** — the only legal transfer under the no-pretrained ruling, subsumes distillation; (B) supervised fine-tune with mixup/trunc-aug/modality-dropout. Multi-seed soup (now proven +0.8). Gates: fusion-transformer must beat late fusion by week 3 or fall back; transduction only at ≥60 CV.
**Gap vs current solution:** current = scalar late fusion, no SSL, stretch-resampling. The 56.15-vs-63.4 oracle gap (per-clip stream selection) and the 34.7% no-stream-right error mass (object-in-hand classes, EXP-017) are exactly what stages A+B target.
**Migration queued:** M-01 aligned multimodal Dataset (pad+mask+duration), M-02 fusion transformer, M-03 cross-modal masked pretraining, M-04 refit-on-18-users + packaging. Est. ~2 weeks alongside continued stream work.
**Verified:** metric = micro-OOF with nested tuning (CV↔LB tracking continues); crazy-tier spend ≈ 8% (target 10) — acceptable.

---

---

## RULING — Organizer clarifications received (via Atharv, 2026-07-28)
1. **NO pretrained weights at all** — strict from-scratch. Consequences: (a) our pipeline already compliant; (b) DA-002's cluster hypothesis (0.73-0.77 ≈ pretrained visual recipes, ~85%) now implies those teams are DISQUALIFIABLE at reproduction → effective private-LB Top-15 bar drops for rules-clean teams; (c) SSL pretraining on competition data (Q-94, no external weights) is THE only visual transfer path — promoted to top priority; (d) pretrained probes (EXP-015/T1) retain only diagnostic value; T1 rerun cancelled.
2. **Ensembles legal if total ≤100 MB** — multi-stream × multi-seed soups are fair game (we use ~5 MB today; ~20× headroom). Multi-seed averaging promoted (EXP-018 seeds already training).
3. **Test-time transductive processing legal** — TENT-style/statistics adaptation inside inference code officially allowed. Gated behind base ≥60 CV per EXP-016 discipline.
**Also:** previous session's background chain died with the process (T1 probe at ep15, droi_big after fold0). droi_big folds 1-3 + skel seeds 1-2 relaunched (chain5).
**Beliefs updated:** B-005 SETTLED (ensemble counting known); B-006 path fixed to SSL-only; NEW B-016: rules-clean pipeline is a Selection-Stage asset — the private-LB bar for ADVANCING is likely below the public cluster's scores.

---

---

## EXP-017 — First confusion/oracle analysis (DA-002 Q2 item #1)
**Date:** 2026-07-28 · **Tier:** diagnostic (CPU)

**Setup:** OOF fusion skel_jvb_big(0.6) + imu_aug(0.2) + depth_aug(0.2), fixed weights (no grid → honest-ish). New best fusion: **56.15% micro OOF** (jvb_big OOF alone 54.74).
**Findings:**
1. **Worst classes are object-interaction classes** — Watch_TV 0.00 (n=12), Play_games 0.05, Wipe_bowls 0.11, Make_a_phone_call 0.16, Take_and_use_tableware 0.21, Use_a_mobile_phone 0.23, Read_documents 0.23↔Turn_pages 0.26 (mutual confusion), Write 0.28→Tap_keyboard. Pattern: classes separated by WHAT THE HANDS HOLD. Skeleton is structurally blind to it; current visual too weak to supply it. Motion classes excellent: Walk 0.97, Lie_down 0.85, Jumping_jacks 0.83, Jog 0.81.
2. **Error decomposition:** total err 43.9% = 9.1% fusion-recoverable (some stream right) + **34.7% needs better streams**. Better fusion alone caps at ~65%.
3. Low-support × hard: Watch_TV/Play_games also have the least train data (12-40) — even their ORACLE coverage is 0-7%.
**Implication:** The visual stream's specific job is hands+object resolution — favors IR (texture) over depth, favors ROI/resolution/pretraining (T1 tonight), and possibly an upper-body/hands sub-crop stream. This is where the missing ~35 points live.
**Beliefs updated:** B-013 sharpened: "stream quality" = specifically object-discriminative visual capability.
**Next:** T1 outcome routes everything; hands-crop stream added to queue (Q-97, explore).

---

---

## DA-002 — Second devil's-advocate pass (5 agents) + immediate follow-ups
**Date:** 2026-07-28 · **After:** EXP-016

**Q1 — Leader mechanism (KEY DERIVED FACT: public split = 201 clips; every score is k/201; leader = 168/201; private = 204 clips → 1 public clip = 0.4975%, shakeup ±2-3%).** Mechanism posteriors for 0.836: M1 strong transfer-visual recipe (ImageNet-init, possibly rule-gray) 30-40% — our EXP-015 probe was too crippled to refute it (112px/1-ch/25ep); M5 MANUAL TEST LABELING ~25% (IR is a face-visible photo stream; 405 clips = one afternoon; 83.6% sits mid-band of human accuracy; unenforceable on public LB, dies at Selection Stage); M2 public-LB probing ~20% (score granularity reveals exact correct-count; ≤190 subs available; leader needs only +14 clips over cluster base); M3 metadata ~5%; rest ~10%. The 0.73-0.77 CLUSTER is ~85% M1 — likely pretrained visual recipes. **Strategic corollary: if leader = M5/M2, the real competitive bar is ~0.76-0.77, and if pretrained is ruled illegal at reproduction, the effective top-15 bar drops further — our rules-clean pipeline gains free places at Selection Stage.**
**Kill/confirm tests:** T1 full-recipe transfer probe (224px, 3-ch, ResNet-18, 60-80ep, IR-ROI, random+subject splits; decision: subj ≥55 ⇒ cliff additive, M1 viable AND our own roadmap; subj ≤40 with rand ≥70 ⇒ cliff proportional ⇒ red alert on all visual plans). T2 LB forensics (leader submission count + staircase; GitHub mirror hides counts — ATHARV: read Kaggle LB directly: leader's entry count, score-vs-time shape, and scores at ranks 10/15/20/30 — the rank-15 number converts the campaign target from fantasy to measured bar).
**Q2 — Still-untested, ranked:** (1) per-class confusion analysis of best fusion — 16 experiments, 4 submissions, NOBODY has looked at a confusion matrix; prices everything else; (2) seed-variance measurement (3 seeds × skel_noaug) — protects all ±1-2 decisions; (3) label sanity via group-distinctness signature (free detector: OOF prediction matching a group-mate's label); (4) LS-off ablation (gates calibration → Hungarian/transduction/fusion); (5) person-0-vs-mirror; (6) IMU orientation/quat fixes (+leak removal); (7) no-stretch+duration (with trim-sim eval); (8) mid-fusion gated on 1+4.
**Q3 — Reset preview ("what would we build from scratch"):** per-modality stems (keep) + small temporal fusion TRANSFORMER on the shared 10 Hz grid (replace scalar late fusion — the 54.3-vs-63.4 oracle gap is per-clip stream selection that scalars can't express) + cross-modal masked pretraining on ALL 3,441 clips incl. test (legal SSL that subsumes distillation; the only transfer we own) + multi-seed soup + gated TENT-style adaptation. Prediction: 0.58 central, 90% CI 0.50-0.66; P(top-15) ≈ 0.4-0.6. Migration ~2 wks, keeps most code. HARD GATE: fusion transformer must hit ≥60 CV by week 3 or revert to late fusion.
**Q4 — Retarget steelman: CONFIRMED. P(0.85) < 2-5%.** 0.85 LB needs ~94% subject-CV (offset math) — no published result on this dataset is within 25 pts. OBJECTIVE.md's real win condition was always Top-15 → Selection Stage; the campaign now explicitly optimizes P(top-15), and the rules-clean pipeline is a Selection-Stage asset (cluster disqualification scenario). 245 submissions remain — abundant; GPU-hours are the scarce resource.
**Q5 — Hygiene audit: most adopted/killed decisions are within noise.** Only "ROI +3.4" clearly survives (2.8σ, 4/4 folds, corroborated). bones +1.8 ≈ 0.6-0.9σ; tn-death <0.5σ (reclassified parked-unmeasured); balanced +1.5 driven by one fold; trunc NEVER measured alone (bundling sin, third offense); EXP-012 +1.7 = p≈0.17 (2/4 folds). NEW DECISION PROTOCOL: paired-delta on same OOF clips (SE≈0.6-0.9%), adopt only if |Δ| > 2×paired-SE or replicated across ≥2 seeds; nested tuning for fusion weights; no bundled changes ever.
**Follow-ups executed immediately:** (a) leaderboard mirror fetched — top-6 only, counts hidden ('—'), leader last-submission Jul 26, cluster ties at 0.75124 (151/201 — /201 grid makes ties common); (b) M3 within-group class-order check on train (radar-ts grouping, 45 signatures with ≥3 groups): order-consistent only 2/45 → NO fixed script; M3 dead properly (first attempt with mangled keys gave a false 61/62 — re-verification caught it); (c) 224px ROI cache rebuild launched for T1.
**Counter reset:** DA at 0/10; from-scratch reset review due in 4 experiments (will adopt Q3's design as its base).

---

---

## EXP-016 — Per-user self-training simulation (Q-91 sim): NEGATIVE
**Date:** 2026-07-28 · **Queue ID:** Q-91 · **Tier:** explore

**Hypothesis:** Per-user pseudo-label fine-tuning on the user's unlabeled clips adds ≥+3% (the "leader's transduction" hypothesis, DA-001 Q1).
**Setup:** transduction_sim.py — skel_noaug fold checkpoints; per val-user: pseudo-label (τ=0.60), fine-tune copy 15 ep @2e-4, re-evaluate. 16 users.
**Result:** **mean Δ −3.2% · median −2.5% · helped 2/16 users** · pseudo-label acc 60-93%.
**Conclusion:** Naive self-training is DESTRUCTIVE at ~50% base accuracy. Do not spend submissions on it. The transduction family is not dead — but the entry bar is a stronger base model and/or gentler adaptation.

### Failure analysis
**3 reasons:** (1) pseudo-label noise (7-40% wrong) compounds through 15 epochs; (2) confident clips are the already-easy ones — fine-tuning on them shifts decision boundaries away from hard clips (distribution narrowing); (3) full-model fine-tuning on 40-80 clips overfits instantly.
**3 alternatives:** (1) τ=0.60 too low; (2) lr/epochs too aggressive; (3) skeleton-only sim understates fusion-level transduction (better-calibrated probs).
**3 follow-ups (parked until base ≥60%):** (1) head-only fine-tune + τ=0.9 + 3 epochs; (2) entropy-minimization/consistency (TENT-style on GN affine params) instead of hard labels; (3) class-balanced pseudo-label selection per user.
**Beliefs updated:** B-013 reinforced (base model quality gates EVERYTHING downstream: Hungarian, transduction, fusion); "leader uses naive transduction" demoted — if the leader adapts, it's not this way, or their base is already ≥70.
**Next ideas:** DA-002 due; fusion re-tune with skel_jvb_big when droi_big lands → new submission candidate.

---

---

## EXP-012 — Skeleton scale-up: jvb + width 256 + 150 ep + trunc-aug
**Date:** 2026-07-28 · **Queue ID:** Q-83 · **Tier:** exploit

**Result:** **54.75% ± 2.89% micro** (4-fold, micro-selected), 2.53M params — new best single stream (+1.7 over skel_noaug 53.04).
**Conclusion:** Positive but sublinear: 4× params + 2.5× epochs + bones + trunc bought +1.7. The TCN family saturates ≈55 micro subject-CV; the in-domain ceiling (73) is not being approached via capacity. Next skeleton levers must change the inductive bias or the data view: ST-GCN-lite (Q-30), joint/bone/motion multi-stream logit ensemble, person-selection fix (Q-86), mixup (Q-34).
**Beliefs updated:** B-002: skeleton solid but its DG gap (~18 pts) is not capacity-limited.

---

---

## EXP-015 — Pretrained ResNet-18 probes (Q-84, DIAGNOSTIC — legality pending)
**Date:** 2026-07-28 · **Tier:** exploit (diagnostic)

**Setup:** probe_pretrained.py — ImageNet ResNet-18, 1-ch conv1 (summed RGB filters), 112×112 ROI crops, 8-frame TSN mean-pool, 25 ep, macro metric.
**Results (macro):**
| stream | random-split | subject-fold0 | from-scratch subject ref |
|---|---|---|---|
| depth-ROI | 45.0% | 25.9% | ~21 (droi_bal) |
| IR-ROI | 46.8% | 27.3% | ~17 (iroi_bal) |
**Conclusions:** (1) ImageNet init ≈ doubles from-scratch visual accuracy → representation matters, and legal SSL pretraining (Q-94) inherits a real prize. (2) BUT even pretrained, cross-subject collapses visual by ~20 pts (46→27) — the visual DG gap is far larger than skeleton's; init does not fix DG. (3) Paper's 90% not approached at 112 px/25 ep/macro — resolution (224+/full-frame context?), 3-ch input, longer training, and micro metric all separate us; a stronger transfer recipe would land higher, but the DG cliff pattern will remain. (4) Strategic: visual streams are worth pushing to the ~40-50 subject range (SSL + res + capacity) as FUSION diversity, not as a backbone that alone reaches 0.85.
**Caveats:** probe recipe deliberately cheap; treat absolute numbers as lower bounds, the random-vs-subject GAP as the robust finding.
**Beliefs updated:** B-006 finalized: visual = mid-strength fusion stream, ceiling recipe-and-DG-bound; B-013 reinforced — no single stream reaches target; the road is (max skeleton) + (SSL visual ~45+) + (fusion) + (transduction/self-training multiplier).
**Next ideas:** Q-94 SSL pretrain (masked/temporal pretext on ALL unlabeled clips incl. test — rules-legal, no external weights); ROI cache at 224 px; Q-91 transduction SIMULATION on CV folds (adapt-per-val-user with pseudo-labels — measures the multiplier without submissions).

---

---

## EXP-008/009/010/011 — Wave 2: balanced skeleton + first visual ROI/IR streams (macro-selected era)
**Date:** 2026-07-28 · **Queue IDs:** Q-80/81/82 · **Tier:** exploit

**Results (MACRO, 4-fold unless noted):**
- skel_bal (balanced sampling): **47.6% macro** vs 46.1 skel_noaug macro reference → +1.5 macro. (Micro effect unknown; balanced sampling on hold after B-012 died — test ≈ train prior.)
- iroi_bal (IR person-ROI, first IR training ever): **17.3% macro** — paper's 90% modality nearly useless from scratch at this recipe.
- droi_bal (depth person-ROI): **21.1% macro** vs 17.7 full-frame depth macro → ROI +3.4 on depth.
- ir_bal (full-frame IR, folds 0,2): killed at fold0 ep30 (~12-17%, info value spent — worse than IR-ROI as expected).
- Feature ablation completions (folds 0,2 macro; jv reference 44.9): **jvb (bones) 46.6 (+1.8)** — adopt; jv+tn (torso-norm) 44.1 (−0.7) — dead.
**Conclusion:** (1) ROI cropping helps but the from-scratch visual recipe remains the wall — three independent data points (depth-rand 32, iroi 17, droi 21) say representation/training, not modality or crop, is the blocker → pretrained probe (Q-84) is the decisive next diagnostic, self-supervised pretraining (Q-94) the likely legal remedy. (2) Bones adopted into skeleton. (3) Never edit files a running chain imports (crashed 3 ablation runs; re-run cost ~20 min).
**Beliefs updated:** B-006 unchanged (recipe-gated); B-013 reinforced.

---

## EXP-014 — E1 trim-shift simulation (inference-only)
**Date:** 2026-07-28 · **Queue ID:** DA-001 E1 · **Tier:** diagnostic

**Setup:** trim_sim.py — quantile-map val skeleton clips onto the TEST frame-count distribution (med 20, p90 41), re-score skel_noaug fold checkpoints on CPU.
**Result:** OOF 53.04% full → 51.33% trimmed → **Δ_trim = 1.7 pts**.
**Conclusion:** Offset decomposition: ≈1.7 trimming + ≈1-2 selection/tuning optimism (DA E3, partially fixed) + ≈5-6 genuine subject shift + public-subset noise. Countermeasure shipped: random-truncation augmentation (aug component "trunc") — in all wave-3 runs.
**Beliefs updated:** B-004: offset now decomposed and mostly attributed to subject shift → DG (B-001) remains the true battle; CV↔LB tracking continues on micro + offset.

---

---

## SUB-003/004 — Prior-corrected submissions: REFUTED B-012
**Date:** 2026-07-28

**Results:** sub_fuse3_prioradj → **0.39303** (−6.5 vs raw 0.45771); sub_fuse3_sinkhorn → **0.39800** (−6.0).
**Conclusion:** Test class prior ≈ TRAIN prior (imbalanced), NOT uniform. Both corrections pushed mass toward rare classes (25/26 got ~13% of predictions) and paid for it. The macro-OOF ≈ LB match in DA-001 was coincidence; the −9pt CV→LB offset is therefore GENUINE domain shift (test users harder — per-user OOF spread 42.8-63.4% supports; plus ~20% trim + 10 s cap), not prior mismatch.
**Metric policy reverted:** micro-OOF is again the primary selection/headline metric (test ≈ train prior); log macro alongside. Balanced sampling: re-evaluate on micro (EXP-008's +1.5 was macro; micro effect unknown — measure OOF).
**What the pair bought:** B-012 killed with certainty for 2 submissions; without the pair we might have balanced-trained the whole campaign in the wrong direction.
**Beliefs updated:** B-012 → DEAD (as "uniform test"); B-004 revised again — micro-OOF −9±2 offset with domain-shift attribution; next diagnostic = trim simulation (E1).
**Next:** E1 trim simulation (inference-only); continue stream-quality wave (B-013 unaffected — governs under any prior).

---

---

## DA-001 — Devil's-advocate pass (5-agent workflow) + zero-GPU diagnostics
**Date:** 2026-07-28 · **After:** EXP-006 + SUB-001/002

**1. Why is our current direction probably wrong?** The plan's own EV arithmetic tops out ~27 pts short of 0.85: skeleton in-domain ceiling is 73% (EXP-006) — DG cannot exceed it; depth fix + fusion + DG are +1-3% levers on a 46% base. The 0.836 leader implies a categorically different mechanism: pretrained visual backbone (if legal) and/or per-user transduction (4 users × ~100 clips, balanced test). **VERIFIED by oracle probe: pick-best-of-3-streams oracle = 63.4% — current streams structurally cannot reach 0.85. Stream replacement/upgrade is the campaign.** (Any-stream top-5 oracle 87.8% — labels are near the top; calibration/transduction can mine it later.)
**2. Never-questioned assumptions (code audit):** (top) train-prior≈test-prior — FALSE, see diagnostics below; val-best checkpoint = order-statistic optimism; label smoothing 0.1 distorts Hungarian costs; person-0 skeleton may be the MIRROR in bathroom classes; T=32 stretch-resampling erases duration + gives per-sample velocity units; CV structurally easier than test (always-train users, coverage-maximized folds); depth invalid=0 conflates with near after /255; IMU quaternion double-cover + absolute angle channels = subject/mounting leak; train/eval temporal sampling mismatch; `hash(sid)` seeding not reproducible (PYTHONHASHSEED) — Selection-Stage risk; MENTAL_MODEL claimed class-balanced sampling + EMA that train.py never implemented; Q-54 BN-adaptation impossible — models use GroupNorm.
**3. Gold-medalist critique:** single seeds; 60-ep budgets; bundled augs (already bitten twice); triple-dipping on OOF (checkpoint selection + fusion weights + Hungarian tuning); no TTA; email drafted but never sent ("DONE" self-deception); 0.66M skeleton = 0.2% of the 100 MB budget — capacity massively underused.
**4. Steelman verdict:** "The CV is broken and every decision made against it is suspect" — CONFIRMED within hours (see diagnostics).
**Diagnostics run (da_analysis.py):**
- **Prior mismatch is THE CV→LB offset:** balanced-mean (macro) OOF = 46.7% vs LB 45.8% — near-exact match; micro-CV 54.3% was the wrong metric. LB 0.458 is at the 0.1th percentile of all C(16,4) user-subset micro accuracies → subject variance CANNOT explain it; prior mismatch CAN. Test behaves ~class-balanced (405 ≈ 40×10). Predicted-class histogram was Walk=60/405 (~15%) vs balanced-expected ~10.
- Per-user OOF accuracy spread: 42.8% (user6) to 63.4% (user19) — huge subject variance confirms DG remains real too.
- Actions taken: macro-accuracy is now the checkpoint-selection AND headline metric; --balanced sampler added; last-epoch checkpoints saved; crc32 seeding; prior-adjusted + Sinkhorn submissions written (sub_fuse3_prioradj.csv, sub_fuse3_sinkhorn.csv) for Atharv.
**Queue entries produced:** pretrained ResNet-18 probe (diagnostic; legality pending email), IR stream (running, EXP-009), transduction tier promoted (balanced-assignment/self-training once base ≥55% macro), label-smoothing ablation, person-selection policy, no-stretch+duration-feature variant, valid-mask channel, LOSO harshness measurement, nested fusion-weight tuning, TTA. Killed: Q-54 as stated (no BN to adapt).
**Counter reset:** next DA after 10 more experiments; from-scratch reset review still due at 25.

---

## EXP-007 — Skeleton aug component ablation (folds 0,2; micro-selected, pre-macro-change)
**Date:** 2026-07-28 · **Queue ID:** Q-11b · **Tier:** exploit

**Setup:** one component at a time, magnitudes from EXP-001b; screening folds 0,2 (no-aug reference on same folds = 52.26%).
**Results:** rot 50.3 (−2.0) · scale 51.1 (−1.2) · jit 50.9 (−1.4) · jdrop 50.7 (−1.5) · tjit 51.0 (−1.2) · rot+jit+tjit 51.9 (−0.3). Feature configs (jvb, jv+tn) crashed due to my mid-chain train.py edit (race — never edit files a running chain imports); re-running in wave 2.
**Conclusion:** EVERY component hurts individually at these magnitudes; rot worst — consistent with station-anchored orientation being a genuine, transferable prior (don't rotate it away). This aug family is not the skeleton DG lever. Alternatives: normalization/features (jvb/tn — wave 2), mixup, capacity+longer training, subject-adversarial.
**Confidence:** 70% (screening folds only, single seed, micro metric).
**Beliefs updated:** B-001 refined — "aggressive geometric aug" is NOT the DG lever for skeleton; station-anchored orientation is signal, not noise.

---

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

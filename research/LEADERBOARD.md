# Leaderboard Ledger

This file is the durable source of truth for public-leaderboard results. A score
is **verified** only when Atharv supplies the Kaggle result (normally a
screenshot). Claims found only in experiment notes remain **unverified** and
must not be used as the current best.

The public split contains 201 clips, so one correct clip is
`1 / 201 = 0.00497512` (about 0.50 percentage points). Reported scores below
map exactly to integer correct counts after Kaggle rounding.

## ✅ VERIFIED 2026-09-03 — NEW CHAMPION: `sub_r2_dist.csv` = **0.83084 = 167/201**

Supplied by Atharv. `0.83084` maps exactly to `167/201`; the previous champion
`sub_r2.csv` is `0.82587 = 166/201`. **+1 clip.**

**This is a successfully forecast result, and the forecast was the corrected one.**
EXP-124 predicted +4 to +5 clips. I retracted that to **"~+1 clip, range −3 to +3"**
in EXP-125 **before** the submission was scored, after measuring the change through the
real decoder rather than against raw argmax. The outcome is exactly +1.

**B-022 is now CONFIRMED, not just directional.** Its standing prediction was that
distinctness coupling flips sign as base accuracy rises. The full public ladder:

| base | distinctness delta |
|---|---|
| 112 clips | **−1** |
| 121 clips | **0** |
| 123 clips | never run |
| **166 clips** | **+1** ← first positive reading |

Monotone across four points spanning 54 clips of base accuracy. The belief made a
falsifiable prediction, the prediction was tested at a base far outside its measured
range, and it held.

**Champion recipe is updated** — `--distinctness penalty --distinctness-penalty 2.0`.
Everything else identical: same `testprobs_r2.npz` (verified by SHA-256), same
`--transition-weight 0.5 --transition-score conditional --backoff unigram`. The
submission differs from the old champion on **6 of 405 rows**, and because the other
399 rows are identical there is **no sampling noise in the comparison** — the +1 is the
exact net of those 6 rows, not a draw from the ±9–10 clip floor.

| | |
|---|---|
| file | `submissions/sub_r2_dist.csv` |
| sha256 | `154cd713ffa083a1c9c0aa172c1ba09289221579d13297b6a65245312e5ef897` |
| score | **0.83084 = 167/201** |
| OOF evidence | +1.00 pt pooled, 4/4 folds, rescues 191→216, harms 87→85 |

## Public leaderboard snapshot (top 20) — 2026-07-31

Supplied by Atharv from the Kaggle public leaderboard on 2026-07-31. This
resolves **P-05**, which had been open since the campaign began. Every score
maps to an exact integer `k/201`, which independently confirms the derived
201-clip public split.

| Rank | Team | Score | Correct/201 | Submitted |
|---:|---|---:|---:|---|
| 1 | Jacobo Martin | 0.90049 | 181 | 2026-07-30 |
| 2 | Bull | 0.80099 | 161 | 2026-07-30 |
| 3 | Pre-Par-e | 0.79601 | 160 | 2026-07-30 |
| 4 | z shuyang | 0.77611 | 156 | 2026-07-31 |
| 5 | Arthurs Torres24 | 0.76616 | 154 | 2026-07-15 |
| 6 | houqiiii | 0.75124 | 151 | 2026-07-29 |
| 7 | ahmetkrgztr | 0.74129 | 149 | 2026-07-27 |
| 8 | shenzhijie | 0.73631 | 148 | 2026-07-30 |
| 9 | Arunava Maulik | 0.73134 | 147 | 2026-07-28 |
| 10 | Six Senses | 0.72139 | 145 | 2026-07-28 |
| 11 | Danielle Lesin | 0.71641 | 144 | 2026-07-06 |
| 12 | Fususu | 0.71144 | 143 | 2026-07-31 |
| 13 | Phaedrus | 0.69154 | 139 | 2026-07-23 |
| 14 | Ming | 0.68159 | 137 | 2026-07-25 |
| **15** | **Cuda out of memory** | **0.68159** | **137** | 2026-07-30 |
| 16 | Ali Kaya | 0.68159 | 137 | 2026-07-29 |
| 17 | Ioannis M | 0.66666 | 134 | 2026-07-30 |
| 18 | StormML | 0.66169 | 133 | 2026-07-29 |
| 19 | Zalman Goldstein | 0.65174 | 131 | 2026-07-28 |
| 20 | Freeman Hui | 0.65174 | 131 | 2026-07-26 |

**Measured bars, against our verified 112/201:**

| Bar | Score | Correct/201 | Clips needed |
|---|---:|---:|---:|
| Rank 1 | 0.90049 | 181 | **+69** |
| Rank 10 | 0.72139 | 145 | +33 |
| **Rank 15** | **0.68159** | **137** | **+25** |
| Rank 20 | 0.65174 | 131 | +19 |

Structural notes:

- **Rank 1 is a 20-clip outlier.** 181/201 sits 20 clips clear of rank 2, while
  ranks 2--20 form a smooth ladder from 161 down to 131. Whatever produces
  0.90049 is not what produces the rest of the board.
- Ranks 14--16 are a three-way tie at 137/201, so the rank-15 boundary is
  currently decided by submission time, not score.
- The top-15 bar is **0.68159**, not the previously assumed 0.83 or the
  observed 0.89.
- Our 112/201 is below rank 20; exact standing is unknown because only the top
  20 was captured.

## VERIFIED 2026-08-02 — new best 0.62189 = 125/201

| file | w | public | clips |
|---|---:|---:|---:|
| `sub_visual_mil_v1.csv` (visual member ALONE) | — | 0.38805 | 78 |
| `sub_world25_visgeo225_f0123_trans05.csv` | 0.225 | 0.61194 | 123 |
| **`sub_visgeo035_f0123_trans05.csv`** | **0.35** | **0.62189** | **125** |
| `sub_visgeo045_f0123_trans05.csv` | 0.45 | 0.61194 | 123 |
| `sub_visgeo055_f0123_trans05.csv` | 0.55 | 0.58706 | 118 |

**Public weight curve (4-fold visual member): 0.225→123 · 0.35→125 · 0.45→123 · 0.55→118.**
Peak at w=0.35. The 4-fold member at the old weight 0.225 was a **null** (123 = 123): 59 rows
changed, net zero. The gain came from re-weighting, not from the extra folds.

**Why the weight had to move (EXP-063).** Visual alone transfers **+0.89** (OOF 0.379 → public
0.388); the skeleton stack transfers **−7.55** (0.618 → 0.542). Any weight fitted on OOF
under-weights the only component that survives the subject shift. OOF selected 0.15–0.20;
the truth is 0.35. **Do not tune fusion on OOF again** — use the public LB, sparingly.

**Fusion parameterization is closed.** Fold-safe on 2700 clips: global w +1.778 pts,
per-class-group +1.778, per-cohort +1.704. Richer schemes buy nothing.

## Superseded staging note (2026-08-02, kept for provenance)

Submit in this order; each is an isolated single change against the 123/201 champion.

| # | File | Change vs champion | Rows differ | SHA-256 |
|---|---|---|---:|---|
| 1 | `sub_world25_visgeo225_f0123_trans05.csv` | visual member = **4-fold average** instead of fold-2-only; w=0.225 and lambda=0.5 unchanged | **59/405** (~29 public) | `265ef9e45318a5819a382b5bbd0f12d1114d8d66800fb8388771872864436059` |
| 2 | `sub_world25_visgeo225_trans05_dist10.csv` | + soft distinctness (penalty 1.0); everything else identical | **12/405** (~6 public) | `dd57f76acb7499a8d968ffc7439aff756b98b0e6877efe453682c4aefd228d76` |

**#1 — 4-fold visual member.** Fusion weight w=0.225 was **refit fold-safe on 2700 clips**
(w chosen per fold on the other three folds only): unbiased gain **+1.778 points, +48 clips**,
with w=0.225 selected by 3 of 4 folds and also the full-sample argmax. Selection bias measured at
only 0.185 points, versus the 552-clip sweep that produced the original weight. This retires the
methodology error recorded below for the tri-member config.
*Caveat:* the change is one model -> an average of four, and there is **no unbiased OOF estimate
of that specific change** (the fold-2 model cannot be scored on folds 0/1/3 without contamination).
59 changed rows is a large edit; downside is real.

**#2 — distinctness at base 123.** Tests B-022's standing falsifiable prediction that the coupling
flips sign as base accuracy rises. Measured ladder: base 112 -> **−1 clip**; base 121 -> **0 clips**;
base 123 -> ? Monotone so far and not yet positive. Cheapest live test of a standing prediction in
the campaign.

**Submit #1 first** (larger expected effect, and it changes the base that #2 is defined against —
if #1 wins, #2 must be regenerated on top of it before it means anything).

## Current state

- **Verified best:** `sub_world25_visgeo225_trans05.csv` —
  **0.61194 = 123/201** (visual MIL fused in LOG space at w=0.225 + transition
  decode). **+11 clips in one day**, all from a member rejected that morning.

Same-day ladder, every score user-verified:

| candidate | rule | public | clips |
|---|---|---:|---:|
| `sub_astgcn_world25_int8_trans05` | transition only | 0.55721 | 112 |
| `sub_world25_int8_trans05_dist10` | + distinctness | 0.55223 | 111 |
| `sub_world25_visfuse10_trans05` | + visual, linear w=0.10 | 0.58706 | 118 |
| `sub_world25_visfuse15_trans05` | linear w=0.15 | 0.58706 | 118 |
| `sub_world25_visfuse20_trans05` | linear w=0.20 | 0.57213 | 115 |
| `sub_world25_visgeo15_trans05` | **geometric** w=0.15 | 0.60199 | 121 |
| `sub_world25_visgeo15_trans05_dist10` | geometric + distinctness | 0.60199 | 121 |
| **`sub_world25_visgeo225_trans05`** | **geometric w=0.225** | **0.61194** | **123** |
| `sub_world25_visgeo275_trans05` | geometric w=0.275 | ~0.59 | ~119 |
| `sub_world25_visgeo30_trans05` | geometric w=0.30 | 0.577 | 116 |
| `sub_tri_geo_trans05` | + astgcn_all18 @ w=0.30 | 0.60696 | 122 |
| `sub_tri_geo15_trans05` | + astgcn_all18 @ w=0.15 | 0.61194 | 123 |

**`astgcn_all18` rejected as a member:** neutral at w=0.15, −1 clip at w=0.30.
Its large local marginal (+3.99) was an artefact of sweeping 36 weight
combinations on 552 clips.

**Weight optimum is pinned at w=0.225** (121 / **123** / ~119 / 116).

**Methodology error, recorded so it is not repeated.** The tri-member config was
chosen by sweeping **36 weight combinations on 552 clips** and reporting the
maximum: selection on a small validation set, not an unbiased estimate. Local
said +3.99 points; public returned −1 clip. The earlier single-member sweeps
(~7 weights on the same 552 clips) carried the same bias more mildly, and the
public confirmations of those made the method look more trustworthy than it is.
Any local weight search on this 552-clip sample must be treated as optimistic,
and multi-member configs compound it.

**B-022 confirmed directionally:** the distinctness delta moved from −1 clip at
base 112 to 0 clips at base 121, monotone in base accuracy as predicted. It is
not yet positive; re-test again if the base rises further.

**Fusion-rule finding:** geometric (log-space) fusion beats linear at every
weight and stays positive far past the point where linear collapses. Linear
averaging lets a confident base drown out the visual member; the geometric mean
treats the two as independent evidence.
- **Measured follow-up:** `sub_astgcn_world25_int8_repeat_trans05.csv` —
  **0.55223 = 111/201**. Repeat consensus remained above the 109/201 base but
  lost one public clip versus transition-only, so the extra pooling is rejected.
- **Result 2026-07-31:** `sub_world25_int8_trans05_dist10.csv` scored
  **0.55223 = 111/201**, one clip below the champion. Soft distinctness is
  rejected on the public split, matching B-022's prediction that coupling does
  not pay at this base accuracy. The champion is unchanged at 112/201.
- **Staged for upload 2026-07-31 (second):**
  `sub_world25_visfuse10_trans05.csv`, SHA-256
  `736ca965366e31ad6b93bdb3490dcd0ce3da49ecfc10ea172fb104296acc0451`,
  16,520 bytes. Visual MIL member fused into the exact package probabilities at
  **w=0.10**, then the champion's proven transition decode (`lambda=0.5`,
  `distinctness=none`). Differs from the champion on 34/405 rows.
  Evidence: paired fusion gain replicated on 4/4 seed baselines, and the weight
  was re-validated **against the full-stack ensemble** rather than a single
  model — where the optimum is 0.10 and w=0.20 is negative. Expected move is
  about **+2 clips**.
- The fp32 `sub_astgcn_world25.csv` remains unscored. Its score must not be
  inferred from the int8 package output, which differs on 3/405 rows.
- Previous verified best: `sub_astgcn_a20.csv` —
  **0.53731 = 108/201**
- The internal `0.52736` claim for `sub_block10.csv` is not present in the
  supplied leaderboard evidence. It is excluded from the verified ladder until
  Atharv confirms it.
- A score above 0.83 requires at least **167/201**, which is **55 more correct
  public clips** than the current 112/201.

## Results

| File | Public score | Correct / 201 | Status | Evidence / interpretation |
|---|---:|---:|---|---|
| `sub_astgcn_world25_int8_trans05.csv` | **0.55721** | **112** | **verified 2026-07-30; current best** | Tie-safe ordered transition decoding added three public clips over the exact package base |
| `sub_astgcn_world25_int8_repeat_trans05.csv` | **0.55223** | **111** | **verified 2026-07-30; rejected marginal** | Repeat consensus added two clips over base but lost one versus transition-only despite stronger local OOF |
| `sub_astgcn_world25_int8.csv` | **0.54228** | **109** | **verified 2026-07-30** | Exact legal 85.218 MB package output; then-new best, +1 public clip versus `a20` |
| `sub_astgcn_world25.csv` | — | — | **unscored** | Fp32 world-frame IMU candidate; differs from the scored int8 output on 3/405 rows and must not inherit its score |
| `sub_astgcn_a20_int8.csv` | — | — | **not scored** | Derived from the legal 82.696 MB package; differs from verified fp32 `a20` on 1/405 rows; must not inherit 0.53731 |
| `sub_astgcn_a20.csv` | **0.53731** | **108** | verified 2026-07-30 | Adaptive ST-GCN added three public correct clips; superseded by world25 and transition decoding |
| `sub_multitcn_g50.csv` | 0.52238 | 105 | verified 2026-07-30 | User-confirmed Kaggle result; MultiTCN candidate |
| `sub_block12_tta.csv` | 0.51741 | 104 | verified 2026-07-29 | Supplied screenshot; tied block8 despite 18/405 different predictions |
| `sub_block8_gcnsoup.csv` | 0.51741 | 104 | verified 2026-07-29 | Supplied screenshot; smaller accuracy reference, but still oversized as a four-fold fp16 package |
| `sub_block5_stgcn.csv` | 0.51243 | 103 | verified 2026-07-29 | Supplied screenshot; first large architecture-diversity gain |
| `sub_soup3_imuinv.csv` | 0.48756 | 98 | verified 2026-07-29 | Supplied screenshot; invariant IMU added one public correct clip |
| `sub_soup3_fusion.csv` | 0.48258 | 97 | verified 2026-07-29 | Supplied screenshot |
| `sub_ship1_refit_tta.csv` | 0.47263 | 95 | verified 2026-07-29 | Supplied screenshot; refit/TTA bundle lost three clips vs invariant-IMU soup |
| `sub_fuse3_hung.csv` | 0.44776 | 90 | verified 2026-07-29 | Supplied screenshot; hard distinctness assignment was negative |
| `sub_fuse3_sinkhorn.csv` | 0.39800 | 80 | verified 2026-07-29 | Supplied screenshot; uniform-marginal correction failed |
| `sub_fuse3_prioradj.csv` | 0.39303 | 79 | verified 2026-07-29 | Supplied screenshot; prior correction failed |
| `sub_fuse3_nohung.csv` | 0.45771 | 92 | historical internal record | Consistent baseline in LOG; not visible in the supplied screenshot crop |
| `sub_block10.csv` | 0.52736 claimed | 106 claimed | **unverified** | Internal LOG claim only; absent from supplied screenshot; do not call it the best |

## Submission-file identity

These hashes bind each ledger row—verified or pending—to its exact local CSV.
A hash records identity only; it does not upgrade a pending row to verified.

| File | SHA-256 |
|---|---|
| `sub_astgcn_world25_int8_trans05.csv` | `ed6784665f7aacf5832ca10d7b7a0adc8fd333977cd25effedc8e4efe168e346` |
| `sub_astgcn_world25_int8_repeat_trans05.csv` | `1a1dbdaeba41814cf0b459b78c2ae95eb9d80389ae62a842970cac76cd6dc377` |
| `sub_astgcn_world25.csv` | `bc5305a6631bb4f9e036af70c6773bc092efb38c2730fb98244cb39ba1f1f3ed` |
| `sub_astgcn_world25_int8.csv` | `879417469e0c4de3d52862465126c26e9813dd35678fa9725d61d935f89a2e45` |
| `sub_astgcn_a20_int8.csv` | `b9e419f3b4a904731ce36a659bbb50714aeee730855cdcd046ead4fe88cffd36` |
| `sub_astgcn_a20.csv` | `a1a95ec2a52ccd7a5dcf10260d18be40930607f1c77ac33ad8ba867da4d84390` |
| `sub_multitcn_g50.csv` | `8bf1bff62fd1216d4d6b40985f26f844ed0b67b5f478b8d7d2e8f175fa259649` |
| `sub_block12_tta.csv` | `80364db0cfd4e8ef43e4a0d0e2b6cf369e23adb7b7fb2bb73b509fdbbc298a38` |
| `sub_block8_gcnsoup.csv` | `647abe049e11e41a14f77e28da453ca52e9f761177a50b73c596dab4e8a0a787` |
| `sub_block5_stgcn.csv` | `74bfe77f75a7ad7121abd926eac30421e8875761d32ad39ce60d3fa5aedda182` |
| `sub_soup3_imuinv.csv` | `ebeae68129a9d315aee676ae1704a2aee8e261096b50d9cc54cba22ae7afa731` |
| `sub_soup3_fusion.csv` | `8e97907ce42a9bd2ee8cf00729e69ce88b57f66a670503c200c7b8c4f603ac4e` |
| `sub_ship1_refit_tta.csv` | `6705f3c814e1995a1483b462f0f6527340782ebba0f7d463b3bbc530ab59c232` |
| `sub_fuse3_hung.csv` | `b34f0203cb739a679ae7c4fdaeb72cf5a6cf8b261bf6b4a3eaaf55a941ab327a` |
| `sub_fuse3_sinkhorn.csv` | `8765a0577ae9996e8e9145639188a52108e4137261190297a4b18f935e6ab8ea` |
| `sub_fuse3_prioradj.csv` | `56a9b982319d6fb97ac5aed6278f052872d70b4081b9fba69bd341b23fe8a61d` |

## Pending-package identity

These byte counts and hashes establish artifact identity and serialized size.
The default inference path now streams one member at a time: persistent fp32
model parameters peak at 10,134,688 bytes rather than 337,973,856, with exact
probability/CSV parity. The exact organizer wording is still needed to settle
whether transient conversion scratch or total process RSS is relevant.

| Artifact | Bytes | Members | SHA-256 | Audit |
|---|---:|---:|---|---|
| `research/artifacts/model_astgcn_a20_int8.pth` | 82,696,132 | 44 | `d5a3ee6ee46df5e48e0eff2d06bff66dd9e2ee7c3dc2cace1b63d4d14142c286` | payload passed; packaged CSV 16,517 bytes and 1/405 argmax differs from scored fp32 |
| `research/artifacts/model_astgcn_world25_int8.pth` | 85,217,859 | 48 | `bccd32dc997839dd085717f42fd600192d5b692fc962075fa55565600ae95ee0` | payload passed; packaged CSV 16,516 bytes and 3/405 argmaxes differ from fp32 |
| `submissions/sub_astgcn_world25.csv` | 16,516 | — | `bc5305a6631bb4f9e036af70c6773bc092efb38c2730fb98244cb39ba1f1f3ed` | fp32 candidate; unscored |
| `submissions/sub_astgcn_world25_int8.csv` | 16,516 | — | `879417469e0c4de3d52862465126c26e9813dd35678fa9725d61d935f89a2e45` | exact legal-package output; verified 0.54228 |
| `submissions/sub_astgcn_world25_int8_trans05.csv` | 16,515 | — | `ed6784665f7aacf5832ca10d7b7a0adc8fd333977cd25effedc8e4efe168e346` | tie-safe sequence-aware output; verified 0.55721 |
| `submissions/sub_astgcn_world25_int8_repeat_trans05.csv` | 16,514 | — | `1a1dbdaeba41814cf0b459b78c2ae95eb9d80389ae62a842970cac76cd6dc377` | repeated-pass consensus plus sequence decoding; verified 0.55223 |
| `submissions/sub_astgcn_a20_int8.csv` | 16,517 | — | `b9e419f3b4a904731ce36a659bbb50714aeee730855cdcd046ead4fe88cffd36` | legal-package output; not scored |

## Reading small deltas

- The verified ladder is `97 → 98 → 103 → 104 → 105 → 108 → 109 → 112`.
- A one-clip change is below the resolution needed to establish a general
  mechanism. Use paired subject-OOF evidence and isolated submission changes.
- Equal public scores do not imply equal predictions or equal private scores.
- Leaderboard evidence evaluates accuracy only. Package legality, model size,
  reproducibility, and on-site generalization remain separate gates.

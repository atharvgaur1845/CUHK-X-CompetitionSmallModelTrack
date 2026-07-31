# Leaderboard Ledger

This file is the durable source of truth for public-leaderboard results. A score
is **verified** only when Atharv supplies the Kaggle result (normally a
screenshot). Claims found only in experiment notes remain **unverified** and
must not be used as the current best.

The public split contains 201 clips, so one correct clip is
`1 / 201 = 0.00497512` (about 0.50 percentage points). Reported scores below
map exactly to integer correct counts after Kaggle rounding.

## Current state

- **Verified best:** `sub_astgcn_world25_int8_trans05.csv` —
  **0.55721 = 112/201**
- **Measured follow-up:** `sub_astgcn_world25_int8_repeat_trans05.csv` —
  **0.55223 = 111/201**. Repeat consensus remained above the 109/201 base but
  lost one public clip versus transition-only, so the extra pooling is rejected.
- **Next upload:** none yet. The next candidate must come from the all-user
  high-resolution visual/object reset, not another sequence postprocessor.
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

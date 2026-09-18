# Final testing phase, 2026-09-18 18:17-18:27 UTC (laptop RTX 4060)

Both Selected submissions run from the pushed image `ghcr.io/atharvgaur1845/cuhkx-small-agx:v1` (image id `c5b0d56e147d`,
the same id `docker load` gave from the uploaded `image.tar.gz` on 17 Sept), with the organisers' exact command
(`--gpus all --network none --shm-size 16g`, data mounted `:ro`), compared by `tools/independent_compare.py`
against the CSVs inside the uploaded zip (unzipped fresh from `dist/small_AGX_verification_v1.zip`, sha256 re-verified).

| submission | wall | result |
|---|---|---|
| 56213683 (second) | 293 s | byte-identical to `final_submission_2.csv` (sha256 7c6434ac...) |
| 56212349 (primary) | 304 s | 403/405, the two disclosed decoder near-ties (SM_test_0206 8 vs 7, SM_test_0382 21 vs 22) |

Logs: `run_*.log`, `compare_*.txt`, `final_test.log`; script `final_test.sh`. Static rule-by-rule check: `docs/RULES_COMPLIANCE.md`.

# G14 — clean-room attack on the actual upload archives (2026-09-17, laptop RTX 4060)

Archives (`dist/`, hashes in `UPLOAD_SHA256SUMS`):
    20a00f926b049a602a1fd3d5f788301301333b14c1deb8d4cd816d015bfa2039  small_AGX_verification_v1.zip
    e399522793d82a122b1312b0cf143ad6e5c08af4cf51f829d258dc499934da5b  image.tar.gz

- zip: 96,688,141 bytes, 208 files; image.tar.gz: 3,590,141,281 bytes; image id `c5b0d56e147d`,
  registry `ghcr.io/atharvgaur1845/cuhkx-small-agx:v1` digest `sha256:c5b0d56e147d2dfa8181c655986d9431552c99be173919335d6b6b76df353c85`.
- git tag `verification-v1`.

Procedure (`tools/g14_clean_room.sh`): empty directory, unzip the zip, `sha256sum -c artifacts/SHA256SUMS`,
`docker rmi` of any local image, `docker load` from the archive, a fresh copy of Kaggle Testing mounted
`:ro`, the organisers' exact `docker run --rm --gpus all --network none --shm-size 16g …` for both
submission ids, comparison by `tools/independent_compare.py` (shares no package code), `diff -rq` of the data.

| attack | condition | 56212349 vs final_submission_1 | 56213683 vs final_submission_2 | data dir | wall |
|---|---|---|---|---|---|
| 1 | working repository present | 403/405 (SM_test_0206 8 vs 7, SM_test_0382 21 vs 22 — the disclosed residual) | byte-identical | untouched | 281 s / 272 s |
| 2 | working repository **moved aside** for the whole run | 403/405, same two rows | byte-identical | untouched | ~280 s / 276 s |

G13 (host, no Docker): unzip in an empty directory, laptop environment, `56213683` → byte-identical (275 s).

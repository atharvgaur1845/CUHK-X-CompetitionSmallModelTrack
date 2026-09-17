# G6 / G7 — organisers' docker command on the laptop (RTX 4060, image 3959e5df5a94)

    docker run --rm --gpus all --network none --shm-size 16g \
        -v <Testing>:/data:ro -v <out>:/output cuhkx-small-agx:v1 bash inference.sh /data /output/submission.csv

- G6 (Kaggle Testing, primary config 56212349): exit 0, 4 min 36 s wall, `/output` holds only the CSV.
  Output sha256 7c6434ac… = byte-identical to `sub_final_ship7.csv`; vs `sub_pkgship6.csv` 403/405
  (SM_test_0206 8 vs 7, SM_test_0382 21 vs 22 — the documented residual). Compared with
  `tools/independent_compare.py` (shares no package code).
- G7 (synthetic hostile set `/tmp/synth`, read-only): exit 0, 49 s; 405 shipped windows rejected by
  fingerprint, 24 clips detected live; 24 rows in the (shuffled) sample_submission order, 13 classes.
- Date: 2026-09-17 (UTC 00:00), docker with nvidia-container-toolkit 1.20.0.

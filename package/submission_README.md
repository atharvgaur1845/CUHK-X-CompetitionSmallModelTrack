# Selected Kaggle Final Submission records

`submission_info.yaml` lists both Selected submissions with IDs, timestamps, scores, hashes and
the model/configuration mapping. `final_submission_1.csv` (ID 56212349, primary) and
`final_submission_2.csv` (ID 56213683) are the exact files uploaded to Kaggle, copied with
`cp`, never opened by a writer; their SHA-256 values are in `submission_info.yaml`.

Reproduction: `bash inference.sh <data_dir> <out.csv>` reproduces #2 byte-for-byte and #1 in
403/405 rows (README §1 and §7 explain the two rows).

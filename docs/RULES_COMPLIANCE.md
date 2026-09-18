# Rules compliance check of the uploaded package (2026-09-18, final testing phase)

Checked against every line of the three organiser documents in `templates/` (Reproduction Package Guidelines v2,
Kaggle Submission Verification Guide v4, Verification Process) using the **uploaded archives themselves**
(`dist/small_AGX_verification_v1.zip` sha256 `20a00f92...`, `dist/image.tar.gz` sha256 `e3995227...`, both re-hashed;
the zip unpacked fresh; the image is the one whose `docker load` id, `c5b0d56e147d`, is recorded in
`docs/verification_logs/final_clean/`, and whose registry digest is `sha256:c5b0d56e...353c85`).

## Reproduction Package Guidelines v2

| # | rule (paraphrased) | status | evidence |
|---|---|---|---|
| 1 | inference code that produces the final submission + training code | PASS | `src/code/` holds the inference path (`unpack_stage2.py`, `fuse_general.py`, `ordered_transition_decoder.py`, ...) and the trainers (`train.py`, `train_video_crop.py`, `train_video_thermal.py`, `train_imu_world.py`, ...); `src/CODE_MAP.md`, `docs/PROVENANCE.md` |
| 1 | every custom weight loaded at inference, each ensemble member; Small track size rule | PASS | one file `artifacts/stage2_ship7.pth`, 96,369,921 bytes (< 100 MB), 28 blobs with hashes in its `manifest.json`; `sha256sum -c artifacts/SHA256SUMS` OK; the run prints `96.37 MB on disk, cap 100 MB -> PASS` |
| 1 | public base model need not be included if repo + revision given; downloaded during setup only | PASS (not needed) | MViTv2-S Kinetics init is training-time only (manifest gives file + revision); YOLO weights are inside the package in fp16; nothing is fetched at inference |
| 1 | one entry point `bash inference.sh <test_data_dir> <output_csv>`; published layout; Kaggle CSV format | PASS | contract in `inference.sh`; layout discovery `src/code/locate_test_data.py` (Testing/, Testing/data, clip dir); output `path,prediction`, 405 rows in sample-submission order |
| 1 | no hard-coded file lists, sample counts or machine paths | PASS | grep of the inference path: "405" occurs only in a docstring and a `--help` string; no `/home`, `/scratch`; the 405 shipped crop windows are fingerprint-gated and any other clip is detected live (README 8) |
| 1 | optional third argument = Submission ID if the two Selected submissions differ | PASS | `bash inference.sh <data> <out> 56213683`; `configs/56212349.json`, `configs/56213683.json` |
| 1 | README.md, manifest.yaml, submission/submission_info.yaml from the template | PASS | README sections 0 to 9 match the template headings one for one; `submission_info.yaml` keys identical to the template; `manifest.yaml` has every template key (plus `team.members`, `solution.code_map`, `solution.training_provenance`); no `REPLACE_ME` |
| 1 | honour declaration signed by every member, PDF copy if possible | PASS | `declarations/CUHK-X_Honor_Declaration.docx` + `.pdf`: Small Model Track boxes ticked, both submission ids and UTC times, name/email/signature/date 2026-09-17 |
| 2 | Docker preferred: Dockerfile + `docker save` archive or pinned public tag; image self-contained; `WORKDIR` = package root | PASS | `environment/Dockerfile` (base `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`, pins, build-time self-checks with an empty `TORCH_HOME`), `WORKDIR /package`, `CMD ["bash","inference.sh","/data","/output/submission.csv"]`; `image.tar.gz` uploaded and `ghcr.io/atharvgaur1845/cuhkx-small-agx:v1` public (`docker manifest inspect` resolves without login) |
| 2 | the organisers' exact `docker run --rm --gpus all --network none --shm-size 16g -v data:/data:ro -v out:/output <image> bash inference.sh /data /output/submission.csv` works | PASS | run today for both ids: 293 s and 304 s, outputs written to `/output/submission.csv`, nothing written under `/data` (mounted read-only) |
| 2 | Option B also present | PASS | `environment/requirements.txt` (16 pins incl. torch 2.7.1 / torchvision 0.22.1 / ultralytics 8.4.118) + `pip_freeze_image.txt`; Python/CUDA/torch versions in README 4 |
| 2 | say early if > 1 GPU, > 24 GB, specific generation, not Linux | PASS | README section 0 answers all four (no / no / no / Linux yes) |
| 3 | which submissions reproduced: id, file name, private score, SHA-256; primary marked | PASS | README 1 and `submission_info.yaml`; primary 56212349 (0.87745), second 56213683 (0.87254); CSV hashes match the Kaggle-downloaded files in `templates/` byte for byte |
| 3 | hardware: GPU model/count, peak GPU memory, RAM, disk | PASS | README 3: RTX 4060 8 GB and A100, 1 GPU, peak 5.4 GB, RSS 1.9 GB, 1 GB work + 10.6 GB image |
| 3 | software: OS, driver, CUDA, Python, PyTorch or image tag | PASS | README 3-4, manifest `environment` |
| 3 | commands: setup, exact inference command, where output goes | PASS | README 4-5 |
| 3 | run time of one full inference | PASS | 4 min 53 s laptop, 2 min 39 s A100 (README 3); today's runs 293 s / 304 s agree |
| 3 | internet at run time | PASS | none; empty `TORCH_HOME`, `YOLO_OFFLINE=1`, models built `pretrained=False`; verified with `--network none` |
| 3 | randomness: seeds, exact reproducibility, variation | PASS | README 7 and manifest `reproducibility`: no RNG at inference, byte-identical repeats, cross-GPU table (5 generations, 0/405 changes) |
| 3 | files: checkpoints, sizes, anything unusual (ensembles, caches rebuilt) | PASS | README 2 and 6 (per-member bytes, one-file package, caches rebuilt every run, fingerprint-gated windows, the 48 -> 20 prune and the 403/405 residual disclosed) |
| 4 | closed-source APIs | n/a | none (manifest `api.required: false`) |
| 5 | template layout; no dataset in the package | PASS | layout matches section 5 of the guidelines; 208 files, 95 MB; no images, caches, `__pycache__`, `.npz`, credentials (grep clean) |
| 6 | self-check on a fresh container with the network off, compared with the submitted CSV | PASS | done 17 Sept twice from the actual archives (`final_clean/`, repository moved aside on the second) and again today (`final_clean_0918/`) |
| 7 | one archive, parts <= 9 GB, checksums pasted into the form; up to 10 files, 10 GB each | PASS | zip 96.7 MB + image.tar.gz 3.59 GB, sha256 in `dist/UPLOAD_SHA256SUMS` and in the form |
| 7 | disclose both Selected submissions; primary = higher private score | PASS | both disclosed; primary 56212349 |
| deadline | 18 Sept 2026 23:59 UTC | met | uploaded 17 Sept ~01:00 UTC |

## Kaggle Submission Verification Guide v4

| step | rule | status | evidence |
|---|---|---|---|
| 1 | record every Selected submission (up to two); primary = higher private score, entered as `primary_verification_submission_id` | PASS | `submission_info.yaml`: 56212349 primary, 56213683 second |
| 2 | Submission ID from `kaggle competitions submissions` (`ref`), matched by file name + time + public score, not by file name alone | PASS | ids, file names, UTC times and public scores recorded (`docs/verification_logs/kaggle_download/`) |
| 3 | the exact CSV Kaggle evaluated, not resaved; SHA-256 in `submission_info.yaml` | PASS | `final_submission_1/2.csv` are byte-identical (`cmp`) to the Kaggle downloads `templates/sub_pkgship6.csv` and `sub_final_ship7.csv`; hashes f5ead899... and 7c6434ac... recorded |
| 4 | team name, track, primary id, both submission blocks, model/config mapping | PASS | all fields filled; mapping names the checkpoint hash and the config per submission |
| 4 | the two-argument command reproduces the primary; third argument if configurations differ | PASS with disclosure | primary reproduces 403/405 (two decoder near-ties from the post-submission 48 -> 20 prune, documented in README 1/7, `submission_info.yaml` notes, `docs/verification_logs/config_search_*`); the second is byte-identical; third argument supported |

## Verification Process

| rule | status | evidence |
|---|---|---|
| inference reproducibility on the original test data and an additional held-out dataset; frozen models | PASS | frozen since 13 Sept (tag `stage2-final`, package tag `verification-v1`); new data goes through live detectors and fresh caches (synthetic hostile suite, G5/G7) |
| no manually annotated test labels, no hard-coded or memorised answers, no hidden labels, no prohibited data or services, no change of the frozen solution | PASS | declaration signed; the known test-label leak was closed and never used (`research/RULES_VERIFIED.md` L-1); no external data; the package computes every prediction from raw frames (today's run rebuilt all caches from the mounted data) |
| technical failure alone is not misconduct; clarifications allowed without changing the frozen solution | noted | README 8 lists the fallbacks; every fallback reproduces the frozen behaviour |

## Two cosmetic inconsistencies inside the frozen archives (no rule broken)

1. `manifest.yaml` names the signed PDF `declarations/CUHK-X_Honor_Declaration_signed.pdf`; the file in the package is
   `declarations/CUHK-X_Honor_Declaration.pdf` (same folder, the only PDF there). README 9 says "signed PDF included".
2. Affiliation reads "Independent" in the frozen README/manifest; the report, poster and slides say BITS Pilani, K K Birla Goa
   Campus (`docs/next_action.md` records the change).

Both can be stated in the submission form's notes field before the form closes (18 Sept 23:59 UTC) without touching the
archives; re-uploading a corrected zip would change the hash already entered in the form and is not worth the risk.

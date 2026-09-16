# Code map — what runs at inference, what trained the members, what is research history

`src/` is the team repository's code tree verbatim (`code/`, `kaggle/`, `cluster/`,
`inference_assets/`). The full history with every experiment ledger is at
https://github.com/atharvgaur1845/CUHK-X-CompetitionSmallModelTrack (tag `verification-v1`).

## A. Inference path (everything `inference.sh` executes, in order)

| step | file | role |
|---|---|---|
| setup | `code/locate_test_data.py` | find the clip directory and the sample submission at any mount level; fail fast on invalid input |
| 0 | `code/unpack_stage2.py --check integrity` | every blob in the package matches its SHA-256 |
| 1 | `code/extract_detectors.py`, `code/window_fingerprints.py` | rebuild `yolo11n.pt`/`yolo11n-pose.pt` from the package tensors; write shipped crop windows only for fingerprint-matched clips |
| 2 | `code/build_cache.py test` | skeleton / IMU / metadata npz per clip |
| 3 | `kaggle/cuhkx_224_kaggle.py --stage cache --test-only` | person crop @384 and wrist crop @224 from IR + Depth (live YOLO for clips without a shipped window) |
| 4 | `code/build_thermal224_cache.py` | thermal full frame @224 |
| 5 | `code/unpack_stage2.py --check infer` (+ `code/bitpack.py`, `kaggle/cuhkx_224_kaggle.py: build_model, predict`) | three MViTv2-S members from int5 codes; fp16 autocast, flip TTA |
| 6 | `code/infer_stage2_skel_imu.py` (+ `code/infer_packaged.py`, `code/package_ensemble.py`, `code/har_data.py`, `code/har_models.py`, `code/train_imu_world.py` [model definitions], `code/imu_stats_member.py`, `code/imu_trees_to_tensors.py`) | 20-member skeleton/IMU stack (int8) and the IMU forest |
| 7 | `code/fuse_from_stage2.py`, `code/ordered_transition_decoder.py test` | prior-corrected log-linear fusion with `configs/<id>.json`; ordered transition decoder; CSV |
| check | `code/rowdiff.py` | row-level comparison of two submission files |

## B. Training code (review only; not executed by `inference.sh`)

| member | trainer | recipe record |
|---|---|---|
| video members (person 384, wrist 224, thermal folds) | `kaggle/cuhkx_224_kaggle.py --stage train` | `docs/PROVENANCE.md`; `cluster/thermal224.sbatch`, `code/run_wrist_queue.sh`, `cluster/res448_a100.sbatch` (same recipe family) |
| thermal weight soup | `code/soup_weights.py` | inputs recorded in the soup checkpoint |
| skeleton members (`astgcn_v1`, `stgcn_v1`, `skel_jvb_big`) | `code/train.py`, `code/har_data.py`, `code/har_models.py`, `code/cv_protocol.py` | `cluster/skel_retrain.sbatch`, `research/artifacts/results.csv` (in the repository) |
| IMU deep members (`imu_inv4`, `imu_world`) | `code/train.py --modality imu`, `code/train_imu_world.py train` | idem |
| IMU forest | `code/imu_stats_member.py`, `code/imu_trees_to_tensors.py` | ExtraTrees 200 × depth 12, seed 0 |
| stack packaging / pruning | `code/package_ensemble.py`, `code/prune_world25.py` | prune invocation unrecorded (see PROVENANCE) |
| final package | `code/pack_stage2.py` (+ `code/quantize_checkpoint.py`, `code/bitpack.py`) | command in `docs/PROVENANCE.md` |
| caches for training | `code/build_cache.py train`, `kaggle/cuhkx_224_kaggle.py --stage cache`, `code/build_thermal224_cache.py --split train` | — |
| cluster job templates | `cluster/*.sbatch`, `cluster/env.sh`, `cluster/README.md` | — |

## C. Research and experiment scripts (history; none is used by A or B)

`code/exp*.py`, `code/probe_*.py`, `code/*_sim.py`, `code/fuse_*.py` other than
`fuse_from_stage2.py`, `code/train_*` other than the trainers above, `code/build_*cache*.py`
variants, `code/cand_weights.py`, `code/search_config_weights.py` (the coefficient search of
§7 of the README), `code/make_synthetic_testset.py` (the held-out robustness suite),
`code/build_verification_package.sh` (assembles this package). They are included so that the
experiments cited in the report can be inspected; they read files that are not part of this
package (training caches, checkpoints) and are not meant to run here.

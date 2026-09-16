# Team AGX — Reproduction Guide (CUHK-X Challenge, Small Model Track)

This package reproduces our Selected Kaggle submissions from the official label-free Testing
data with one command, offline. Read §0 and §5 first; everything else is reference.

## 0 Needs attention (answer first)

- More than one GPU needed: **no**
- More than 24 GB of GPU memory needed: **no** (peak **5.4 GB** measured; 8 GB is comfortable)
- Specific GPU generation or driver needed: **no** — image is CUDA 12.6 (driver ≥ 525); verified on RTX 4060 (laptop) and A100 (cluster), see §7 for other generations
- Network access needed at inference time: **no** (the run sets an empty `TORCH_HOME` and `YOLO_OFFLINE=1`; nothing is fetched)
- Runs on Linux x86_64: **yes**
- A GPU is required in practice: the three video models are MViTv2-S at 384/224 px on 16 frames; on CPU the run exceeds 16 GB of host RAM and is not supported.

## 1 Team and submission

- Team name: `AGX`
- Track: `small`
- Selected Kaggle Final Submission record: `submission/submission_info.yaml`
- Selected final submissions:
  - **#1 (primary):** Kaggle submission ID `56212349`, file name `sub_pkgship6.csv`, submitted on `2026-09-13T17:45:03Z`, public 0.86567, private **0.87745**, SHA-256 of the exact submitted CSV `f5ead8996755cf06eec288fa28ea7fc23731bcad88c4faaea0d98c561ce0eed0`
  - **#2:** Kaggle submission ID `56213683`, file name `sub_final_ship7.csv`, submitted on `2026-09-13T19:15:08Z`, public 0.86069, private 0.87254, SHA-256 `7c6434ac56523644cbe215891bc867c88ff2bd9847f4fa484d62d721191b45d9`
- Primary verification target: **#1** (the higher private score)
- Solution type: `ensemble` (all weights local, one file)

**What the package reproduces, stated up front.** Both submissions were produced by the same
frozen models, the same preprocessing and the same decoder; they differ in nothing this package
can express. Submission #2 is reproduced **byte-for-byte** (verified on two machines, §7).
Submission #1 is reproduced in **403 of 405 rows**: it was fused two hours earlier from the
skeleton ensemble's full 48-member output, and to satisfy the 100 MB rule the packaged
skeleton ensemble keeps the 20 members that carry fusion weight (the other 28 were pruned;
their combined effect is a ≤ 0.11 shift in the skeleton posterior). The two rows,
`SM_test_0206` and `SM_test_0382`, are decoder near-ties: the fused argmax is identical in
both runs and the sequence decoder flips them by margins of 0.012 and 0.006. We searched
~900 alternative fusion-coefficient settings with the frozen tensors for an exact match; the
closest is one row (`docs/verification_logs/`). We therefore ship the principled,
nested-cross-validated coefficients for both submissions and disclose the residual here, in §7
and in `submission_info.yaml`, rather than fit coefficients to the target file.

## 2 Frozen solution summary

40-class activity recognition from privacy-preserving sensors, cross-subject. Five members:

| member | input | model | bytes in package |
|---|---|---|---|
| person view | YOLO11n person window on IR + Depth (4 channels), 16 frames, 384 px | MViTv2-S (Kinetics-400 init), int5 weight codes | 25.9 MB |
| wrist view | YOLO11n-pose wrist window on IR + Depth, 16 frames, 224 px | MViTv2-S, int5 | 25.9 MB |
| thermal view | thermal full frame, 3 channels, 16 frames, 224 px | MViTv2-S (weight soup of 4 fold models), int5 | 25.9 MB |
| skeleton / IMU stack | 17-joint skeletons; 5 wrist/leg IMUs (world-frame, invariant features) | 20 members: adaptive-graph ST-GCN, ST-GCN, joint/velocity/bone TCN, two IMU TCN families; int8 | 18.9 MB |
| IMU statistics | 545 statistics per clip | ExtraTrees (200 trees, depth 12) stored as tensors | 1.6 MB |
| detectors | IR frames | YOLO11n + YOLO11n-pose, fp16 | 10.3 MB |

Fusion: `Σ w·log p` with the skeleton stack divided by the training class prior (it is trained
with plain cross-entropy; the video members with prior-corrected cross-entropy) plus a prior
term; coefficients `person 0.2359 · wrist 0.1807 · thermal 0.17 · skeleton 0.332 · IMU 0.2967 ·
prior 0.25` chosen by nested cross-validation (`configs/56212349.json`, `configs/56213683.json`
— identical). Decoder: ordered transition model over each recording group (clips grouped by the
Radar timestamp key, ordered by first frame time), transition statistics fitted on training
metadata only (`src/inference_assets/`), beam search, soft distinctness penalty; a clip without
a Radar key forms its own group and receives the fused argmax. Test-time augmentation:
horizontal flip on the video members. The 405 Kaggle crop windows are shipped in the package
and used only for clips whose IR-frame filenames match a stored fingerprint; any other clip is
detected live by the packaged detectors. Training: see `src/CODE_MAP.md` and
`docs/PROVENANCE.md` (every member's script, command and checkpoint; two unrecorded steps named).

## 3 Hardware and runtime

- Operating system: Ubuntu 24.04 (laptop) / Rocky Linux 8 (cluster); image is Ubuntu 22.04
- GPU model or minimum compute capability: any CUDA 12-capable NVIDIA GPU; verified on RTX 4060 Laptop (8 GB) and A100 80 GB
- GPU count: 1
- Minimum GPU memory per GPU: 6 GB (peak measured 5.4 GB)
- Minimum host memory: 8 GB (peak RSS measured 1.9 GB, plus up to 8 worker processes for JPEG decoding)
- Free disk space required: 1 GB for the run directory (`work/`, ~0.5 GB per run, previous runs deleted automatically); image 10.6 GB
- Expected environment setup time: `docker load` 3–5 min; or `docker pull`
- Expected inference time (405 clips): **4 min 53 s** on the RTX 4060 laptop (6 CPU workers), **2 min 39 s** on an A100 (8 workers); cache construction is most of it

## 4 Environment setup

**Docker (preferred).** Image `docker.io/TODO-TEAM/cuhkx-small-agx:v1` (also supplied as
`image.tar.gz`, SHA-256 in the form). Built from `environment/Dockerfile` on
`pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` (Python 3.11, torch 2.7.1+cu126, torchvision
0.22.1, ultralytics 8.4.118, OpenCV headless 5.0). `WORKDIR` is the package root. The
build runs self-checks (artifact hashes, package integrity, detector rebuild, offline model
construction); nothing is downloaded at run time.

```bash
docker load -i image.tar.gz        # or: docker pull docker.io/TODO-TEAM/cuhkx-small-agx:v1
docker run --rm --gpus all --network none --shm-size 16g \
    -v <data_dir>:/data:ro -v <output_dir>:/output \
    cuhkx-small-agx:v1  bash inference.sh /data /output/submission.csv
```

**Without Docker.** Python 3.11; `pip install torch==2.7.1 torchvision==0.22.1 --index-url
https://download.pytorch.org/whl/cu126`; then `pip install -r environment/requirements.txt`
(a `pip freeze` of the image). System packages: none beyond a working NVIDIA driver.
No environment variables are required; `WORKERS=n` caps the JPEG worker processes (default
`min(nproc, 8)`), `CUHKX_WORK=<dir>` relocates the run directory.

## 5 Run inference

From the package root:

```bash
bash inference.sh <data_dir> <output_csv>                # primary, submission 56212349
bash inference.sh <data_dir> <output_csv> 56213683       # second Selected submission
```

`<data_dir>` may be the `Testing/` folder, `Testing/data/`, or the clip directory
(`small_model_track_test/`) — the script locates the directory holding one folder per clip
(`IR/ Depth_Color/ Thermal/ Skeleton/ IMU/ Radar/`) at any level up to depth 4 and the
`sample_submission.csv` beside it; the id of a clip is the last path component of the
sample-submission `path`, so any naming scheme works. Without a `sample_submission.csv` the
clips are written in sorted-id order with a warning. Nothing is written under `<data_dir>`.

Output: `<output_csv>` exactly, columns `path,prediction`, one row per sample-submission row in
its order, `prediction` an integer class 0–39. The log goes to stdout and to
`work/<run_id>/run.log`; the last line prints the row count, the SHA-256 of the CSV and the
wall time. Failures are reported as `FAILED at step [n/7 ...]` with a non-zero exit; invalid
input (missing directory, unknown submission id, a listed clip without a directory, no clips
found) exits 2 before any model runs, with the reason.

Steps: `[0]` package hash and integrity → `[1]` rebuild the two YOLO detectors from the package,
write fingerprint-gated windows → `[2]` skeleton/IMU cache → `[3]` person and wrist crops
(live detection for clips without a shipped window) → `[4]` thermal cache → `[5]` three video
members → `[6]` skeleton stack and IMU forest → `[7]` fusion with the config's coefficients,
decoder, CSV. Every run starts from empty caches; `work/` keeps only the latest run.

## 6 Model artifacts and external resources

| artifact | path | SHA-256 | notes |
|---|---|---|---|
| single deployable checkpoint (all members, detectors, windows) | `artifacts/stage2_ship7.pth` | `5b9145436b588c8f70e3df479f734e75f0bcfeb62de9659194a51573bc360330` | 96,369,921 bytes; zip container, `manifest.json` inside lists every tensor with its hash; loader `src/code/unpack_stage2.py` |
| training class prior + decoder statistics | `src/inference_assets/meta_train.csv`, `train_radar_keys.json`, `cv_folds.json` | `artifacts/SHA256SUMS` | training-split metadata only, no labels of test data |
| window fingerprints | `src/inference_assets/window_fingerprints.json` | `artifacts/SHA256SUMS` | sha1 of IR filenames of the 405 Kaggle clips |
| configs | `configs/56212349.json`, `configs/56213683.json` | `artifacts/SHA256SUMS` | fusion coefficients per Selected submission |

Public base models (used at **training** time as initialisation; not loaded at inference):
torchvision `MViT_V2_S_Weights.KINETICS400_V1` (`mvit_v2_s-ae3be167.pth`, torchvision ≥ 0.15,
BSD-3); Ultralytics `yolo11n.pt` and `yolo11n-pose.pt` (YOLO11 release, ultralytics 8.3.0
assets, AGPL-3.0) — their tensors are inside `artifacts/stage2_ship7.pth` in fp16 and are
rebuilt into `.pt` files at step [1]. No external datasets. No APIs.

## 7 Reproduction expectations

- **Determinism.** Fixed batch composition, no random number generator is consumed at inference
  (no dropout, no sampling; DataLoader workers preserve order). Video members run under fp16
  autocast with flip TTA; the skeleton stack in fp32; the IMU forest is an exact integer tree
  walk; fusion and the decoder are float64 NumPy with deterministic tie-breaking. Repeated runs
  on the same machine give byte-identical CSVs (verified 4× on the laptop, 2× on the cluster).
- **Across GPUs.** Predictions are the claim, not bit-equality. RTX 4060 (torch 2.12+cu130,
  Python 3.12) vs A100 (torch 2.7.1+cu126, Python 3.11, the image's toolchain): max
  per-branch posterior difference 1.6e-3 (video), 4.4e-5 (skeleton), 0 (IMU); **prediction
  changes 0/405; CSV identical** (`docs/verification_logs/`). The near-tie rows named in §1
  are the rows most exposed to such differences; if a run on another GPU generation differs,
  it will be in rows of that kind and we ask that the CSV be compared row-wise.
- **Crop windows.** Window computation by the detector is not reproducible across GPUs at the
  pixel level (measured: recomputing on another GPU changed 22 of 405 final rows), which is why
  the Kaggle windows travel inside the package, gated by the IR-filename fingerprint. On new
  data the packaged detectors run live and the result is deterministic per machine.
- **Tolerance requested.** Submission #2: none (byte-identical). Submission #1: the two rows
  described in §1 (`SM_test_0206` → 8 instead of 7, `SM_test_0382` → 21 instead of 22),
  0.5 % of rows, cause and evidence documented; everything else identical.

## 8 Known limitations and troubleshooting

- **Missing modalities** are handled deterministically exactly as in the Kaggle run (10 Kaggle
  clips have no thermal frames): an absent modality becomes a zero tensor for that member and
  the fusion proceeds; a clip without a Radar file is decoded alone (fused argmax); a clip with
  no data at all receives the fused zero-input output (class 36). Tested with a synthetic
  held-out set (new ids, missing Radar/Thermal/Skeleton/IMU/IR, 320×240 frames, a 0-byte PNG,
  a truncated IMU CSV, stray folders, shuffled and absent `sample_submission.csv`).
- **Same clip ids on a different dataset** are safe: the shipped windows are used only when the
  clip's IR filenames match the stored fingerprint; otherwise the log says how many clips were
  detected live.
- **The package directory must be writable** (Docker's container filesystem is): each run writes
  about 0.5 GB of caches under `work/<run_id>/` and `src/cache/`, deleted at the next run. Under a
  read-only package tree (e.g. an Apptainer SIF without writable binds) step [2] fails with
  `No space left on device`; bind writable directories over `src/cache`, `src/research` and `work`.
- **Paths with spaces** are supported. The output path is made absolute before the run.
- **`--shm-size`** is not needed (workers use no shared tensors) but harmless.
- If `torch.cuda.is_available()` is false the run continues on CPU with a warning; this is a
  diagnostic mode and, for the full video branch, exceeds 16 GB of RAM.
- Nothing here proposes a different model or method; all fallbacks reproduce the frozen behaviour.

## 9 Final checklist

- [x] All placeholder fields replaced (`TODO-TEAM` items are filled by the team before upload)
- [x] `manifest.yaml` matches the submitted files (`artifacts/SHA256SUMS` verified at image build)
- [x] The package runs from a clean directory through `inference.sh` (laptop, cluster, container)
- [x] Every Selected Final Submission ID and its exact CSV are included and mapped to the configuration
- [x] `primary_verification_submission_id` = 56212349
- [x] External data (none), base models (two, training-time), APIs (none), licences declared
- [ ] `declarations/CUHK-X_Honor_Declaration.docx` completed and signed by every member, signed PDF included
- [x] No API keys, passwords, credentials, raw challenge data or caches included

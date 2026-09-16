# Training provenance of every packaged member

Every entry names the script, the command (or where it is recorded) and the checkpoint the
package tensors came from. Two steps were not recorded at the time and are marked as such;
neither affects inference reproducibility (the package tensors are what they are and are
hashed), and both are disclosed so a reviewer is not left to discover them.

Data: the official Training split only (2,933 clips, 18 subjects). No external data. No test
labels. Cross-validation folds are subject-grouped (`inference_assets/cv_folds.json`; the video
trainer's 4-fold split is hard-coded in `kaggle/cuhkx_224_kaggle.py`).

## Video members — `kaggle/cuhkx_224_kaggle.py --stage train`

Every video checkpoint embeds its full argument namespace under `["args"]`; the recipes below
are read from there. Base weights: torchvision `MViT_V2_S_Weights.KINETICS400_V1`; the stem is
widened to 4 channels (IR + Depth) for the person and wrist views.

| member | checkpoint | recipe |
|---|---|---|
| person 384 (`k224_mvit384_all`) | trained on all 18 subjects, 2026-09-13, cluster A100 job 341219 | `--stage train --all-train --tag k224_mvit384_all --image-size 384 --batch-size 4 --accum 4 --workers 3 --seed 1 --cache-dir <crop_384> --data-root <data>`; 20 epochs, lr 1e-4, warm-up 2, EMA 0.99, label smoothing 0.1 |
| wrist 224 (`k224_mvitwrist_all`) | all 18 subjects, 2026-08-23, laptop | `--stage train --all-train --crop wrist --tag k224_mvitwrist_all` with defaults (batch 4, accum 4, 20 epochs, lr 1e-4); trainer was unseeded at that date (`code/run_wrist_queue.sh`) |
| thermal soup (`k224_mvit_th_soup`) | weight average of `k224_mvit_th_f0..f3` | folds: `cluster/thermal224.sbatch` (`--fold F --tag k224_mvit_th_fF --cache-dir <thermal_224> --batch-size 16 --accum 1 --workers 6 --seed 1`); soup: `code/soup_weights.py --inputs k224_mvit_th_f0 k224_mvit_th_f1 k224_mvit_th_f2 k224_mvit_th_f3 --out k224_mvit_th_soup` (inputs recorded inside the soup checkpoint) |

Crops for training were produced by the same `--stage cache` code path as at inference
(`yolo11n.pt` person windows; `yolo11n-pose.pt` wrist windows with margin 1.8, min side 96).

## Skeleton / IMU stack (`world25`, 20 packaged members) — `code/train.py`, `code/train_imu_world.py`

| tag (× 4 folds) | family | recipe (from `research/artifacts/results.csv`, `run_EXP-040_*.json`, checkpoint metadata) |
|---|---|---|
| `astgcn_v1` | adaptive-graph ST-GCN on 17-joint skeletons | `code/train.py --modality skel --arch adaptive --width 64 --feat jv --aug-spec trunc --t-skel 32 --stretch --person first --batch-size 64 --lr 1e-3 --label-smoothing 0.1 --seed 0` (EXP-040) |
| `stgcn_v1` (merged with `stgcn_s1`, `stgcn_w96` weights) | ST-GCN | `code/train.py --modality skelg --tag stgcn_v1 --epochs 100 --seed 0` (EXP-025) |
| `skel_jvb_big` (merged with seeds s1–s4 weights) | joint/velocity/bone TCN, width 256 | `code/train.py --modality skel --feat jvb --width 256 --epochs 150 --seed 0` (EXP-012) |
| `imu_inv4` | invariant-feature IMU TCN | `code/train.py --modality imu --imu-feat inv --epochs 60 --seed 0` (EXP-024b) |
| `imu_world` | world-frame IMU TCN (gravity-removed world acceleration, angular velocity, magnitudes, presence mask) | `code/train_imu_world.py train --folds 0,1,2,3 --amp` (seed 0, 60 epochs, width 128, depth 4, dropout 0.3; EXP-046) |

The 48-member development stack was packaged by `code/package_ensemble.py build --preset
astgcn_world25` (int8). **Unrecorded step 1:** the pruning of that stack to the 20 members
carrying fusion weight (`code/prune_world25.py --drop … --merge …`, September 2026) was run
without its invocation being logged; the packaged tensors are the pruned artifact's tensors
verbatim, and `pack_stage2.py` declares the member list explicitly in the package manifest
(`skeleton_spec`). This is the step behind the 2-row difference on the primary submission
(README §1).

## IMU statistics forest — `code/imu_stats_member.py`, `code/imu_trees_to_tensors.py`

545 statistics per clip from the five IMUs; `ExtraTreesClassifier(n_estimators=200,
max_depth=12, random_state=0)` on all training clips; Laplace smoothing 1e-3 at inference.
**Unrecorded step 2:** the conversion of the fitted forest to the int8-leaf tensor file
(`pack_int8` in `imu_trees_to_tensors.py`) was run from an interactive session; the function
is in the file, the resulting tensors are in the package and hashed.

## Detectors

Ultralytics YOLO11n and YOLO11n-pose release checkpoints (assets v8.3.0), never fine-tuned,
stored in the package as fp16 tensors (exactly representable; the fp16→fp32 round trip is
verified at packing time) and rebuilt into `.pt` files at inference.

## Package

`code/pack_stage2.py --video person384=k224_mvit384_all --video wrist=k224_mvitwrist_all
--video thermal=k224_mvit_th_soup --bits 5 --pack-bits --detector person=yolo11n.pt
--detector wrist=yolo11n-pose.pt --windows crop_384=cache/crop_384/windows.json
--windows crop_wrist224=cache/crop_wrist224/windows.json --imu-trees imu_trees_t200_d12_int8.npz
--weight person384=0.2359 --weight wrist=0.1807 --weight thermal=0.17 --weight skel=0.332
--weight imu=0.2967 --weight prior=0.25 --out stage2_ship7.pth` (2026-09-13; reconstructed
from the manifest, which records every input). Fusion coefficients: nested cross-validation
on the 2,700 pooled out-of-fold clips, centroid of the top-25 settings (repository log
EXP-154/158).

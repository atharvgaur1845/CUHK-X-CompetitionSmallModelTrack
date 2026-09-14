# Stage-2 reproduction — CUHK-X Small Model Track

Everything needed to regenerate our Kaggle submission from raw test frames is this
repository plus one file, `stage2_ship7.pth` (96,369,921 bytes, SHA-256
`5b9145436b588c8f70e3df479f734e75f0bcfeb62de9659194a51573bc360330`). No training data,
no cached features, no downloads at run time.

## Run

```bash
pip install -r requirements-inference.txt
./inference.sh stage2_ship7.pth <test_dir> out.csv [work_dir]
```

`<test_dir>` is the competition directory holding the `SM_test_XXXX/` clip folders
(`Small-Model-Track/Testing/data/small_model_track_test`). `sample_submission.csv` is
looked for at `<test_dir>/../../` (the competition layout), then `<test_dir>/../`, then
`<test_dir>/`; if none is present the clips are emitted in sorted id order. `work_dir`
(default `./inference_work`) receives the rebuilt caches and can be deleted afterwards.

One GPU with >= 16 GB and ~2 GB of scratch. Wall time on the 405 Kaggle clips: **2 min
39 s** on an A100 with 6 CPU workers (`WORKERS=n`); the shipped crop windows mean YOLO
only runs live for clips it has not seen.

## What the run does

| step | script | what it rebuilds from the package |
|---|---|---|
| 1 | `code/extract_detectors.py` | `yolo11n.pt` and `yolo11n-pose.pt` (fp16, tensor-identical to the Ultralytics releases) and the crop-window cache for the Kaggle test clips |
| 2 | `code/build_cache.py test` | per-clip skeleton / IMU / metadata `.npz` from `Skeleton/`, `IMU/`, `Radar/` |
| 3 | `kaggle/cuhkx_224_kaggle.py --stage cache --test-only` | person crop (IR + Depth, 4 ch, 384 px) and wrist crop (224 px) from the raw frames with the **packaged** detectors |
| 4 | `code/build_thermal224_cache.py` | thermal full frame, 3 ch, 224 px |
| 5 | `code/unpack_stage2.py --check infer` | the three MViTv2-S video members (int5 weight codes -> fp32), test-set posteriors |
| 6 | `code/infer_stage2_skel_imu.py` | the 20 skeleton members (int8) and the IMU ExtraTrees forest (stored as tensors), test-set posteriors |
| 7 | `code/fuse_from_stage2.py` + `code/ordered_transition_decoder.py` | log-linear fusion with the weights declared in the manifest, then the ordered transition decoder |

The fusion formula is written into the package manifest and printed by step 7:
`0.2967·log(imu) + 0.2359·log(person384) + 0.25·log(prior) + 0.332·log(skel/prior) +
0.17·log(thermal) + 0.1807·log(wrist)`. The decoder's transition statistics come from
`inference_assets/` (training metadata only, 0.45 MB); it never sees a test label.

## What is in the 96.37 MB

| branch | bytes | contents |
|---|---|---|
| video | 64.79 MB | 3 × MViTv2-S (Kinetics-400 init, fine-tuned): person crop @384 (4 ch), wrist crop @224 (4 ch), thermal @224 (3 ch); int5 per-output-channel weight codes, bit-packed |
| skeleton stack (`world25`) | 18.93 MB | 20 members in 5 families — skeleton: adaptive-graph ST-GCN, ST-GCN soup, joint/velocity/bone TCN soup; IMU (deep): invariant-feature TCN, world-frame TCN — 4 folds each, int8 |
| detectors | 10.30 MB | YOLO11n person + YOLO11n-pose wrist, fp16 — inside the package as required (topic 738333) |
| IMU | 1.58 MB | ExtraTrees on 545 statistical features, stored as tensors |
| windows | 0.01 MB | crop windows for the 405 Kaggle test clips (see *Determinism*) |

Loader: `code/unpack_stage2.py`. Manifest: `manifest.json` inside the archive
(format `cuhkx.stage2-package`, version 2).

## Determinism and the crop windows

Every model forward pass is deterministic. The one operation we measured to drift across
machines is the **YOLO crop window**: the caches the models were trained against were
cropped on a Kaggle GPU, and recomputing the windows on a different GPU reproduced them
for only 184/405 person and 14/405 wrist clips (sub-pixel box differences), which moved
22 of the 405 final rows. The package therefore ships the windows it computed for the 405
Kaggle clips; `inference.sh` uses them when the clip id is present and runs the packaged
detectors live for any clip that is not (the on-site data). Both detectors are in the
package, so nothing outside the 100 MB is consulted.

## Verified reproductions

| machine | environment | result vs `submissions/sub_final_ship7.csv` |
|---|---|---|
| laptop, RTX | Python 3.12.3, torch 2.12.0+cu130, torchvision 0.27.0 | rowdiff **0** (the file was produced here from raw frames, with every cache moved aside) |
| cluster, A100 80GB, clean directory (code + package only) | Python 3.11.15, torch 2.7.1+cu126, torchvision 0.22.1 | rowdiff **0** (job 345715, `cluster/cleanroom.sbatch`, EXP-159); 2 min 39 s wall |

## The two scored rows, stated up front

`submissions/sub_final_ship7.csv` is exactly this package's output and scored
**0.86069 (173/201)**. Our best public score, `sub_pkgship6.csv` = **0.86567 (174/201)**,
was fused from a 48-member skeleton artifact; the packaged skeleton keeps the 20 members
that fit in the byte budget, and the two files differ in 2 of 405 rows. We submit the
package's own output as the row of record so that what is scored is what is reproduced.

## Code map for a reviewer

- `inference.sh` — the whole pipeline, one command, commented per step
- `code/pack_stage2.py` / `code/unpack_stage2.py` — package writer and loader, including the
  int5 bit-packing (`code/bitpack.py`) and the detector fp16 round-trip check
- `kaggle/cuhkx_224_kaggle.py` — video trainer / cropper / inference (MViTv2-S)
- `code/build_thermal224_cache.py`, `code/build_cache.py` — thermal and skeleton/IMU caches
- `code/ordered_transition_decoder.py` — the decoder; `--transition-weight 0.5
  --transition-score conditional --backoff unigram --distinctness penalty
  --distinctness-penalty 2.0` are the shipped settings
- `research/LOG.md` — every experiment, newest first, with the verdict in the title

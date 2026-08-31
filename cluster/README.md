# Cluster (`ssh sharanga`) — bring-up and job templates

Hardware seen 2026-09-01 (`sinfo`). `/home` is Lustre with **116 TB free**.

| partition | GPUs/node | GRES string | time limit |
|---|---|---|---|
| `gpu_h200_8` | 8× H200 NVL | `gpu:nvidia_h200_nvl:8` | 1 day |
| `gpu_h100_4` | 4× H100 80GB (×2 nodes) | `gpu:nvidia_h100_80gb_hbm3:4` | 3 days |
| `gpu_a100_8` | 8× A100 80GB | `gpu:nvidia_a100-sxm4-80gb:8` | 5 days |
| `gpu_rtx_pro_6000_6_c` | 6× RTX PRO 6000 | `gpu:nvidia_rtx_pro_6000:6` | 2 days |
| `gpu_v100_2` | 2× V100 32GB (×2 nodes) | `gpu:tesla_v100-pcie-32gb:2` | 3 days |

## Why arrays and not DDP

The repo contains **zero** `torch.distributed` code, and the skeleton/IMU members are
0.6–2.5M params — the whole 48-member `world25` stack retrains in **under one GPU-hour**.
The right parallelism is `sbatch --array` over (tag × fold), one GPU per task. Introducing
DDP under a 14-day deadline would be a large change for no gain. The video trainers are
the only ones that would benefit, and an 80 GB card makes large-batch single-GPU simpler.

## Order

```bash
bash cluster/sync.sh                 # 50 GB raw data + repo (run from the laptop)
ssh sharanga 'bash ~/cuhkx/cluster/env.sh'
sbatch cluster/parity.sbatch         # ⚠ GATE — must pass before anything else
sbatch cluster/skel_retrain.sbatch   # T1.5, 12 tasks, minutes each
```

## The parity gate is not optional

`cluster/parity.sbatch` retrains `k224_mvit_f2` and requires **micro = 0.71472**, then
rebuilds `testprobs_r2.npz` and requires **max abs diff 0.0**. A silent environment
difference (cuDNN algo, torchvision version, JPEG decoder) would invalidate every
cluster-vs-laptop comparison downstream, and we would not notice for days.

## Gotchas carried over from the laptop

- **`--workers 0`** for the video trainers. `--workers 2` silently OOM-killed a 384 px run
  with no traceback. Cluster nodes have 1 TB RAM so this can be raised, but raise it
  *deliberately* and watch RSS.
- **Keep `persistent_workers=False` in `code/train.py`.** `epoch_seed` mutates every epoch
  and worker copies must observe it, else every clip gets one frozen augmented view.
- **`research/artifacts/results.csv` has no file locking.** A 12–48 task array will
  interleave rows. Prefer the per-run `run_{exp}_{tag}_{folds}.json` manifests.
- **`CUHKX_ROOT`** now overrides the repo root (was hardcoded in 11 files).
- **`CUHKX_FOLD_FILE`** selects the fold protocol. The default `cv_folds.json` is a
  **16-user** protocol (`always_train: [user5, user21]`); use `cv_folds_all18.json`.

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

## The parity gate is not optional — but check 1 as originally written was unmeetable

`cluster/parity.sbatch` rebuilds `testprobs_r2.npz` and requires **max abs diff 0.0**.
That half is sound: it is pure inference over fixed weights, so it is genuinely
deterministic, and a silent environment difference (cuDNN algo, torchvision version,
JPEG decoder) would invalidate every cluster-vs-laptop comparison downstream.

**Check 1 — "retrain `k224_mvit_f2` and require micro = 0.71472" — was wrong, and I
wrote it.** `kaggle/cuhkx_224_kaggle.py` sets no seed anywhere, so 0.71472 is one draw
of an unseeded process and *the laptop cannot reproduce it either*. Kaggle returned
0.68252 and the gate could not say whether that was a broken environment or an ordinary
draw. See **EXP-120**.

The trainer now takes `--seed`. The gate is therefore:

1. **Deterministic half (hard):** `testprobs_r2.npz` to **max abs diff 0.0**.
2. **Stochastic half (distributional):** three `--seed`-replicated runs of
   `k224_mvit_f2`; the laptop's 0.71472 must fall inside the measured spread.
   Cheap side-effect: this finally measures the **visual branch's seed sigma**, which
   `LOG.md:2069` records as never measured. The 2.80 every visual gate quotes is a
   *fold* sigma inherited from the skeleton branch, whose own seed sigma is 0.18.

Do not re-impose an equality gate on any unseeded trainer.

## Gotchas carried over from the laptop

- **`--workers 0`** for the video trainers. `--workers 2` silently OOM-killed a 384 px run
  with no traceback. Cluster nodes have 1 TB RAM so this can be raised, but raise it
  *deliberately* and watch RSS.
- **Keep `persistent_workers=False` in `code/train.py`.** `epoch_seed` mutates every epoch
  and worker copies must observe it, else every clip gets one frozen augmented view.
  This does **not** generalise to `kaggle/cuhkx_224_kaggle.py`, which draws augmentation
  from the numpy global RNG: measured under fork, torch seeds numpy per worker, so its
  workers do *not* share a stream and `persistent_workers=True` is safe there (EXP-120).
- **`research/artifacts/results.csv` has no file locking.** A 12–48 task array will
  interleave rows. Prefer the per-run `run_{exp}_{tag}_{folds}.json` manifests.
- **`CUHKX_ROOT`** now overrides the repo root (was hardcoded in 11 files).
- **`CUHKX_FOLD_FILE`** selects the fold protocol. The default `cv_folds.json` is a
  **16-user** protocol (`always_train: [user5, user21]`); use `cv_folds_all18.json`.

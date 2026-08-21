# Running the 224px experiment on Kaggle

## Setup

1. New Notebook on **cuhk-x-competition-small-model-track** (the data is already attached).
2. Settings → **Accelerator: GPU T4 x2** (or P100) · **Internet: ON**
   (Internet is needed once, for the torchvision Kinetics weights.)
3. Upload `cuhkx_224_kaggle.py` via *File → Upload* (or paste it into a cell).

## Run

```python
!python cuhkx_224_kaggle.py --stage cache                        # ~25-35 min, once
!python cuhkx_224_kaggle.py --stage train --tag k224_mvit_f2     # ~2-3 h
!python cuhkx_224_kaggle.py --stage infer --tag k224_mvit_f2     # ~3 min
```

The cache lands in `/kaggle/temp` (not persisted). Training checkpoints and outputs
land in `/kaggle/working` (persisted). If a session times out mid-training, rerun the
identical `--stage train` command — it resumes from the last completed epoch.

## What to bring home

Download from `/kaggle/working` and drop into `research/artifacts/`:

| file | use |
|---|---|
| `oof_k224_mvit_f2.npz` | the screen — compare micro against **0.67638** |
| `testprobs_k224_mvit_f2.npz` | the member — add to the bag |

Then locally:

```bash
# add it as a 13th video member (ADDITION, never a replacement)
python3 code/build_adabn_candidates.py     # edit VID to append the new tag
python3 code/ordered_transition_decoder.py test \
    --probs research/artifacts/testprobs_n6.npz \
    --transition-weight 0.5 --start-weight 0.0 \
    --transition-score conditional --backoff unigram \
    --output submissions/sub_n6.csv
python3 code/rowdiff.py submissions/sub_n1.csv submissions/sub_n6.csv
```

## Reading the result

`vid_ig65m_f2` — the best single member we own — is **micro 0.67638, object 289/479**.

- **≥ 0.70** — resolution is the lever. Train folds 0/1/3 too and this becomes the
  new core of the ensemble.
- **0.68–0.70** — comparable to IG-65M but decorrelated; still a strong bag member,
  since the pairing gain has been what pays all campaign.
- **< 0.67** — resolution is not the lever, and B-030 is falsified. Say so and stop;
  do not tune it into looking better.

Object accuracy is the number to watch, not micro: the hypothesis is specifically
that a 3.1x downsample was destroying hand-object detail. If micro rises but object
does not, the mechanism claim is wrong even if the score went up.

## Knobs

| flag | default | note |
|---|---|---|
| `--arch` | `mvit_v2_s` | `swin3d_t`, `swin3d_s`, `s3d` also wired |
| `--batch-size` | 6 | measured peak: mvit bs4 = 5.2 GB, bs6 = 7.6 GB |
| `--epochs` | 20 | 30 matches the local recipe if quota allows |
| `--all-train` | off | for the final shippable model only; its fold-2 number is train-on-test |

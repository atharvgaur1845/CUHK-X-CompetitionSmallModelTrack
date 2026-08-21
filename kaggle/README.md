# The 224px experiment — runs locally

## The correction that made this simple

The competition page hosts only `sample_submission.csv` and `test.csv`. The ~50 GB of
IR/Depth frames is **not** on Kaggle, so there is nothing to train against there
without uploading a cache yourself.

That stopped mattering once the cache was stored as JPEG instead of raw uint8:

| 224px cache | size |
|---|---|
| raw uint8 (what made 224 look impossible) | 10.7 GB |
| **JPEG q90** | **1.3 GB** (measured 8.1×) |

Decode costs 10.9 ms/clip — about **6 s per epoch** across 4 workers against ~230 s of
GPU time. **Raw storage was the constraint, not 224px.** So this runs on the laptop.

Measured on the RTX 4060 (8 GB): `mvit_v2_s` at batch 4 peaks at **5.18 GB**, runs
**446 ms/step**, 4.2 min/epoch, **~1.4 h for 20 epochs**.

## Run

```bash
python3 kaggle/cuhkx_224_kaggle.py --stage cache                       # ~35 min CPU, once
python3 kaggle/cuhkx_224_kaggle.py --stage train --tag k224_mvit_f2    # ~1.4 h
python3 kaggle/cuhkx_224_kaggle.py --stage infer --tag k224_mvit_f2    # ~3 min
```

Resumable — rerun the identical `--stage train` command and it continues from the last
completed epoch. Smoke-test the plumbing first with `--limit 20` (delete the cache dir
afterwards; a limited cache is not trainable).

Then fuse locally, adding it as a 13th video member (an **addition**, never a swap):

```bash
python3 code/rowdiff.py submissions/sub_n1.csv submissions/sub_n6.csv
```

## Optional: Kaggle, for folds in parallel

Only worth it to run folds 0/1/3 while the local card does fold 2. Build the cache
locally, upload the ~1.3 GB `crop_224/` directory as a private Dataset, attach it, then
paste the file into a cell — it detects the kernel and runs every stage — or:

```python
run(stage="train", data_root="/kaggle/input/<your-cache>", batch_size=6, accum=3)
```

Kaggle's 16 GB card takes batch 6–8; the 8 GB laptop card does not.

## Reading the result

`vid_ig65m_f2` — the best single member we own — is **micro 0.67638, object 289/479**.

- **≥ 0.70** — resolution is the lever. Train folds 0/1/3 and this becomes the new core.
- **0.68–0.70** — comparable but decorrelated; still a strong bag member, since the
  pairing gain is what has paid all campaign.
- **< 0.67** — resolution is not the lever. **B-030 is falsified.** Say so and stop; do
  not tune it into looking better.

**Watch object accuracy, not micro.** The hypothesis is specifically that a 3.1×
downsample was destroying hand-object detail. If micro rises but object does not, the
mechanism claim is wrong even if the score went up.

## Knobs

| flag | default | note |
|---|---|---|
| `--arch` | `mvit_v2_s` | `swin3d_t`, `swin3d_s`, `s3d` also wired |
| `--image-size` | 224 | 128 reproduces the existing cache, for an A/B at equal storage |
| `--batch-size` | 4 | 5.18 GB; raise to 6–8 on a 16 GB card |
| `--epochs` | 20 | 30 matches the local recipe |
| `--jpeg-quality` | 90 | 8.1× smaller than raw; lower only if disk is tight |
| `--all-train` | off | final shippable model only; its fold-2 number is train-on-test |

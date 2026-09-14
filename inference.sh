#!/usr/bin/env bash
# inference.sh -- CUHK-X Small Model Track, Stage-2 / on-site reproduction.
#
#   ./inference.sh <package.pth> <test_dir> <out.csv> [work_dir]
#
#   package.pth   the single <=100 MB checkpoint (research/artifacts/stage2_ship7.pth)
#   test_dir      directory holding SM_test_XXXX/ clip folders (IR, Depth_Color, Thermal,
#                 Skeleton, IMU, Radar subfolders); sample_submission.csv is looked for at
#                 <test_dir>/../../ (the competition layout, Testing/sample_submission.csv),
#                 then <test_dir>/../, then <test_dir>/. The layout the code expects is
#                 materialised under the repo with symlinks if it is not already there.
#   out.csv       submission to write
#   work_dir      scratch for caches (default: ./inference_work)
#
# Runs EVERY branch from the package on raw frames: the two YOLO detectors are extracted
# from the package and used for the person and wrist crops; the three MViT views, the 20
# skeleton members and the IMU tree forest are rebuilt from the package's tensors; the
# fusion weights come from the manifest; the decoder fits its transition model from the
# shipped training metadata (inference_assets/), so no training data is required.
#
# Reproduction gate: on the Kaggle test directory this must reproduce
# submissions/sub_final_ship7.csv at rowdiff 0 (python3 code/rowdiff.py) -- laptop RTX
# and cluster A100/H100 both verified (EXP-159).
set -euo pipefail
PKG=$(readlink -f "$1"); TEST=$(readlink -f "$2"); OUT="$3"; WORK=$(readlink -f "${4:-inference_work}")
REPO=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
mkdir -p "$WORK/cache" "$WORK/artifacts"
export CUHKX_ROOT="$REPO"
echo "== package  $PKG"; echo "== test dir $TEST"; echo "== work     $WORK"

# 0. shipped training metadata the decoder and IMU pipeline expect under cache/
mkdir -p "$REPO/cache" "$REPO/research/artifacts"
for f in meta_train.csv; do [ -f "$REPO/cache/$f" ] || cp "$REPO/inference_assets/$f" "$REPO/cache/$f"; done
for f in train_radar_keys.json cv_folds.json; do [ -f "$REPO/research/artifacts/$f" ] || cp "$REPO/inference_assets/$f" "$REPO/research/artifacts/$f"; done

# 0b. the competition layout inside the repo. Three scripts (infer_stage2_skel_imu.py,
#     ordered_transition_decoder.py, imu_stats_member.py) resolve the test clips and the
#     submission order through $REPO/Small-Model-Track/Testing/, not through <test_dir>.
#     Found by the clean-room run (EXP-159): on a machine where the data lives elsewhere
#     the run died at step 6. Materialise that layout with symlinks when it is absent;
#     when it is present (this laptop, the cluster home tree) nothing is touched.
LAYOUT="$REPO/Small-Model-Track/Testing"
if [ ! -e "$LAYOUT/data/small_model_track_test" ]; then
  mkdir -p "$LAYOUT/data" && ln -s "$TEST" "$LAYOUT/data/small_model_track_test"
  echo "== linked $LAYOUT/data/small_model_track_test -> $TEST"
fi
if [ ! -f "$LAYOUT/sample_submission.csv" ]; then
  found=""
  for c in "$TEST/../../sample_submission.csv" "$TEST/../sample_submission.csv" "$TEST/sample_submission.csv"; do
    [ -f "$c" ] && { cp "$c" "$LAYOUT/sample_submission.csv"; found="$c"; break; }
  done
  if [ -n "$found" ]; then echo "== sample_submission.csv taken from $found"
  else  # no ordering file beside the data: emit the clips in sorted id order
    { echo "path,prediction"; ls "$TEST" | grep -E '^SM_test_[0-9]+$' | sort | sed 's#^#small_model_track_test/#; s#$#/,0#'; } > "$LAYOUT/sample_submission.csv"
    echo "== WARNING: no sample_submission.csv found beside $TEST; wrote one in sorted clip order"
  fi
fi

# 1. detectors OUT OF THE PACKAGE (organiser requirement: weights inside the 100 MB file)
echo "== [1/7] extracting detectors from the package"
$PY "$REPO/code/extract_detectors.py" --package "$PKG" --out "$WORK"

# 2. npz cache for skeleton / IMU / decoder metadata, test split only
echo "== [2/7] skeleton+IMU npz cache and meta_test.csv"
( cd "$REPO" && CUHKX_TEST_ROOT="$TEST" $PY code/build_cache.py test )

# 3. video caches from raw frames with the PACKAGED detectors (cwd holds yolo11n*.pt)
echo "== [3/7] person crop @384 and wrist crop @224 (YOLO from the package)"
( cd "$WORK" && $PY "$REPO/kaggle/cuhkx_224_kaggle.py" --stage cache --test-only --crop person --image-size 384 --data-root "$TEST/../.." --cache-workers "${WORKERS:-4}" )
( cd "$WORK" && $PY "$REPO/kaggle/cuhkx_224_kaggle.py" --stage cache --test-only --crop wrist  --image-size 224 --data-root "$TEST/../.." --cache-workers "${WORKERS:-4}" )
echo "== [4/7] thermal full-frame @224"
( cd "$REPO" && $PY code/build_thermal224_cache.py --split test --workers "${WORKERS:-4}" --ref-cache "$WORK/cache/crop_384" --out "$WORK/cache/thermal_224" --test-root "$TEST" )

# 5. video branch from the package
echo "== [5/7] video members from the package"
( cd "$REPO" && $PY code/unpack_stage2.py --package "$PKG" --check infer --cache-root "$WORK/cache" --out-dir "$WORK/artifacts" )

# 6. skeleton + IMU branches from the package
echo "== [6/7] skeleton and IMU members from the package"
( cd "$REPO" && $PY code/infer_stage2_skel_imu.py --package "$PKG" --out-dir "$WORK/artifacts" )

# 7. fuse with the manifest's weights, decode, write
echo "== [7/7] fuse + decode"
( cd "$REPO" && $PY code/fuse_from_stage2.py --package "$PKG" --artifacts "$WORK/artifacts" --out "$WORK/artifacts/testprobs_fused.npz" )
( cd "$REPO" && $PY code/ordered_transition_decoder.py test --probs "$WORK/artifacts/testprobs_fused.npz" \
    --transition-weight 0.5 --transition-score conditional --backoff unigram \
    --distinctness penalty --distinctness-penalty 2.0 --output "$OUT" | tail -2 )
echo "== wrote $OUT"

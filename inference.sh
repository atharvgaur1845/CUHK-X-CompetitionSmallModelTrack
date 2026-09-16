#!/usr/bin/env bash
# inference.sh -- CUHK-X Small Model Track, team AGX. Single entry point for verification.
#
#   bash inference.sh <data_dir> <output_csv> [submission_id]
#
#   data_dir       the official label-free Testing data in the published layout, mounted
#                  wherever the Organizing Committee chooses (read-only is fine). Any of these
#                  work: the Testing/ folder, Testing/data/, or the clip directory itself
#                  (small_model_track_test/ holding one folder per clip with IR/ Depth_Color/
#                  Thermal/ Skeleton/ IMU/ Radar/). sample_submission.csv is looked for beside
#                  the data; without it the clips are emitted in sorted-id order (warning).
#   output_csv     written exactly here, Kaggle submission format (path,prediction)
#   submission_id  optional. 56212349 (default, primary target) or 56213683; selects
#                  configs/<id>.json -- the two Selected submissions share every model tensor
#                  and differ in nothing the package can express except the fusion
#                  coefficients listed there (see README section 2 and 7).
#
# Every run starts from empty caches (work/<run_id>/ and src/cache/test are wiped), needs no
# network (TORCH_HOME is an empty run-local directory; YOLO_OFFLINE=1), never writes under
# <data_dir>, and fails loudly on malformed input before any model runs.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash inference.sh <data_dir> <output_csv> [submission_id]" >&2; exit 2
fi
PKG_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ -d "$PKG_ROOT/src/code" ]; then REPO="$PKG_ROOT/src"; else REPO="$PKG_ROOT"; fi   # package layout vs. git repo
PY=${PYTHON:-python3}

DATA_DIR="$1"
[ -d "$DATA_DIR" ] || { echo "ERROR: input data directory does not exist: $DATA_DIR" >&2; exit 2; }
DATA_DIR="$(cd -- "$DATA_DIR" && pwd)"
mkdir -p "$(dirname -- "$2")"
OUT="$(cd -- "$(dirname -- "$2")" && pwd)/$(basename -- "$2")"     # absolute before any cd
SUB_ID="${3:-56212349}"; [ "$SUB_ID" = "primary" ] && SUB_ID=56212349
CFG="$PKG_ROOT/configs/$SUB_ID.json"
if [ ! -f "$CFG" ]; then
  echo "ERROR: unknown submission id '$SUB_ID'. Known: $(ls "$PKG_ROOT/configs" | sed 's/\.json$//' | tr '\n' ' ')" >&2; exit 2
fi
read -r PKG_REL PKG_SHA < <($PY -c "import json,sys; c=json.load(open(sys.argv[1])); print(c['package'], c['package_sha256'])" "$CFG")
PKG="$PKG_ROOT/$PKG_REL"
[ -f "$PKG" ] || { echo "ERROR: package file missing: $PKG" >&2; exit 3; }
GOT_SHA=$(sha256sum "$PKG" | cut -d' ' -f1)
[ "$GOT_SHA" = "$PKG_SHA" ] || { echo "ERROR: package sha256 mismatch: $GOT_SHA != $PKG_SHA ($PKG)" >&2; exit 3; }

# ---- run namespace: one directory per run, siblings removed, nothing shared between datasets
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$SUB_ID-$$"
WORK_BASE="${CUHKX_WORK:-$PKG_ROOT/work}"
WORK="$WORK_BASE/$RUN_ID"
mkdir -p "$WORK_BASE"
find "$WORK_BASE" -mindepth 1 -maxdepth 1 -type d ! -name "$RUN_ID" -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$WORK/cache" "$WORK/artifacts" "$WORK/torch_home" "$WORK/yolo_config"
exec > >(tee -a "$WORK/run.log") 2>&1
STEP="setup"
trap 'echo "FAILED at step [$STEP]; log: $WORK/run.log" >&2' ERR
T0=$(date +%s)
echo "== inference.sh  package=$PKG_REL sha256=$GOT_SHA  submission=$SUB_ID  config=$(basename "$CFG")"
echo "== data_dir=$DATA_DIR  output=$OUT  work=$WORK"

# ---- environment: offline by construction
export CUHKX_ROOT="$REPO"
export TORCH_HOME="$WORK/torch_home"          # empty: a download attempt fails instead of using a cache
export YOLO_CONFIG_DIR="$WORK/yolo_config" YOLO_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=$(nproc 2>/dev/null || echo 4); WORKERS=${WORKERS:-$(( NPROC < 8 ? NPROC : 8 ))}
$PY - <<'PYEOF' || echo "WARNING: no CUDA device visible to torch; the run continues on CPU (hours, diagnostic only)"
import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)
PYEOF

# ---- fresh derived state under the code tree (paths the members resolve from their own location)
STEP="state-wipe"
rm -rf "$REPO/cache/test" "$REPO/cache/meta_test.csv"
mkdir -p "$REPO/cache" "$REPO/research/artifacts"
cp -f "$REPO/inference_assets/meta_train.csv" "$REPO/cache/meta_train.csv"
cp -f "$REPO/inference_assets/train_radar_keys.json" "$REPO/inference_assets/cv_folds.json" "$REPO/research/artifacts/"
L="$REPO/Small-Model-Track/Testing/data/small_model_track_test"; [ -L "$L" ] && rm -f "$L"   # leftover of an older layout

# ---- locate the clips and the submission order inside <data_dir>
STEP="locate-data"
eval "$($PY "$REPO/code/locate_test_data.py" "$DATA_DIR" --work "$WORK")"
export CUHKX_TEST_ROOT="$TEST_ROOT"
echo "== clips: $N_CLIPS under $TEST_ROOT ; order from $SAMPLE_SUB"

STEP="0/7 package integrity"
echo "== [0/7] package integrity"
( cd "$REPO" && $PY code/unpack_stage2.py --package "$PKG" --check integrity | tail -2 )

STEP="1/7 detectors"
echo "== [1/7] detectors and fingerprint-gated crop windows from the package"
$PY "$REPO/code/extract_detectors.py" --package "$PKG" --out "$WORK" --test-root "$TEST_ROOT"

STEP="2/7 skeleton+IMU cache"
echo "== [2/7] skeleton + IMU npz cache"
( cd "$REPO" && CUHKX_TEST_ROOT="$TEST_ROOT" $PY code/build_cache.py test )

STEP="3/7 video crops"
echo "== [3/7] person crop @384 and wrist crop @224 (packaged YOLO for clips without a shipped window)"
( cd "$WORK" && $PY "$REPO/kaggle/cuhkx_224_kaggle.py" --stage cache --test-only --crop person --image-size 384 --data-root "$TEST_ROOT" --cache-workers "$WORKERS" )
( cd "$WORK" && $PY "$REPO/kaggle/cuhkx_224_kaggle.py" --stage cache --test-only --crop wrist  --image-size 224 --data-root "$TEST_ROOT" --cache-workers "$WORKERS" )

STEP="4/7 thermal"
echo "== [4/7] thermal full frame @224"
( cd "$REPO" && $PY code/build_thermal224_cache.py --split test --workers "$WORKERS" --ref-cache "$WORK/cache/crop_384" --out "$WORK/cache/thermal_224" --test-root "$TEST_ROOT" )

STEP="5/7 video members"
echo "== [5/7] video members from the package"
( cd "$REPO" && $PY code/unpack_stage2.py --package "$PKG" --check infer --cache-root "$WORK/cache" --out-dir "$WORK/artifacts" )

STEP="6/7 skeleton+IMU members"
echo "== [6/7] skeleton and IMU members from the package"
( cd "$REPO" && $PY code/infer_stage2_skel_imu.py --package "$PKG" --out-dir "$WORK/artifacts" )

STEP="7/7 fuse+decode"
echo "== [7/7] fusion (coefficients from $(basename "$CFG")) and ordered transition decoder"
( cd "$REPO" && $PY code/fuse_from_stage2.py --package "$PKG" --artifacts "$WORK/artifacts" --weights "$CFG" --out "$WORK/artifacts/testprobs_fused.npz" )
( cd "$REPO" && $PY code/ordered_transition_decoder.py test --probs "$WORK/artifacts/testprobs_fused.npz" \
    --transition-weight 0.5 --transition-score conditional --backoff unigram \
    --distinctness penalty --distinctness-penalty 2.0 \
    --sample-submission "$SAMPLE_SUB" --output "$WORK/artifacts/submission_$SUB_ID.csv" | tail -2 )
cp -f "$WORK/artifacts/submission_$SUB_ID.csv" "$OUT"
ROWS=$(( $(wc -l < "$OUT") - 1 ))
echo "== wrote $OUT  rows=$ROWS  sha256=$(sha256sum "$OUT" | cut -d' ' -f1)  wall=$(( $(date +%s) - T0 ))s"
trap - ERR

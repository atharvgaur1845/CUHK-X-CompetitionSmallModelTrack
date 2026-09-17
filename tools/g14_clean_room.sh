#!/usr/bin/env bash
# G14 clean-room attack: from the ACTUAL archives to be uploaded, in an empty directory,
# network off, data read-only, compared with tools/independent_compare.py (shares no package code).
# usage: bash tools/g14_clean_room.sh <zip> <image.tar.gz> <Testing dir> <clean dir> <log dir>
set -euo pipefail
ZIP=$(readlink -f "$1"); TAR=$(readlink -f "$2"); TESTING=$(readlink -f "$3"); CLEAN=$4; LOGS=$(readlink -f "$5")
CMP=$(readlink -f "$(dirname "$0")/independent_compare.py")
mkdir -p "$LOGS"; rm -rf "$CLEAN"; mkdir -p "$CLEAN"; cd "$CLEAN"
echo "== $(date -u +%FT%TZ) clean dir $CLEAN"
sha256sum "$ZIP" "$TAR" | tee "$LOGS/archives.sha256"
unzip -q "$ZIP"; PKG=$(ls -d */ | head -1); PKG=${PKG%/}
echo "== unzipped $PKG ($(find "$PKG" -type f | wc -l) files)"
( cd "$PKG" && sha256sum -c artifacts/SHA256SUMS ) | tail -2
docker rmi -f cuhkx-small-agx:v1 ghcr.io/atharvgaur1845/cuhkx-small-agx:v1 >/dev/null 2>&1 || true
docker load -i "$TAR" | tee "$LOGS/docker_load.txt"
IMG=$(docker images cuhkx-small-agx:v1 --format '{{.ID}}'); echo "== image id $IMG" | tee "$LOGS/image_id.txt"
cp -r "$TESTING" data_copy            # fresh copy of the data, mounted read-only
for SUB in 56212349 56213683; do
  rm -rf out_$SUB; mkdir out_$SUB
  echo "== run $SUB"
  /usr/bin/time -f "wall=%es maxrss=%MkB" docker run --rm --gpus all --network none --shm-size 16g \
      -v "$CLEAN/data_copy":/data:ro -v "$CLEAN/out_$SUB":/output cuhkx-small-agx:v1 \
      bash inference.sh /data /output/submission.csv $SUB > "$LOGS/run_$SUB.log" 2>&1 || { echo "RUN FAILED $SUB"; tail -20 "$LOGS/run_$SUB.log"; exit 1; }
  tail -1 "$LOGS/run_$SUB.log"; ls out_$SUB
  cp out_$SUB/submission.csv "$LOGS/submission_$SUB.csv"
done
echo "== independent comparison"
python3 "$CMP" out_56212349/submission.csv "$PKG/submission/final_submission_1.csv" | tee "$LOGS/compare_primary.txt"
python3 "$CMP" out_56213683/submission.csv "$PKG/submission/final_submission_2.csv" | tee "$LOGS/compare_second.txt"
echo "== data dir untouched:"; diff -rq "$TESTING" data_copy && echo "identical"
echo "== $(date -u +%FT%TZ) done"

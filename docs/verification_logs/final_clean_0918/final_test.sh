#!/usr/bin/env bash
set -euo pipefail
cd "/home/atharv/Desktop/projects/KAggle /CUHK-X-CompetitionSmallModelTrack"
L=docs/verification_logs/final_clean_0918; O=/tmp/claude-1000/-home-atharv-Desktop-projects-KAggle--CUHK-X-CompetitionSmallModelTrack/318d8faf-a087-4162-8351-a69a9ef091b1/scratchpad/final_out
rm -rf "$O"; mkdir -p "$O"
echo "== $(date -u +%FT%TZ) image $(docker images ghcr.io/atharvgaur1845/cuhkx-small-agx:v1 --format '{{.ID}}') archive-loaded id per dist log: $(cat docs/verification_logs/final_clean/image_id.txt)" | tee $L/README_run.txt
for SUB in 56213683 56212349; do
  mkdir -p "$O/$SUB"
  /usr/bin/time -f "wall=%es" docker run --rm --gpus all --network none --shm-size 16g \
     -v "$PWD/Small-Model-Track/Testing":/data:ro -v "$O/$SUB":/output ghcr.io/atharvgaur1845/cuhkx-small-agx:v1 \
     bash inference.sh /data /output/submission.csv $SUB > $L/run_$SUB.log 2>&1 || { echo "RUN FAILED $SUB"; tail -20 $L/run_$SUB.log; exit 1; }
  tail -2 $L/run_$SUB.log
done
PKG=/tmp/claude-1000/-home-atharv-Desktop-projects-KAggle--CUHK-X-CompetitionSmallModelTrack/318d8faf-a087-4162-8351-a69a9ef091b1/scratchpad/pkg/small_AGX_verification_v1
python3 tools/independent_compare.py "$O/56213683/submission.csv" "$PKG/submission/final_submission_2.csv" | tee $L/compare_second.txt
python3 tools/independent_compare.py "$O/56212349/submission.csv" "$PKG/submission/final_submission_1.csv" | tee $L/compare_primary.txt
ls -la "$O/56213683" "$O/56212349"
echo "== $(date -u +%FT%TZ) done"

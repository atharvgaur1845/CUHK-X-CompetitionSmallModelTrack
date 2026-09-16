#!/usr/bin/env bash
# Assemble the Selection Stage verification package from this repository (template v5 layout).
#   bash code/build_verification_package.sh [dist/small_AGX_verification_v1]
# src/ is the repo's code tree verbatim (code/, kaggle/, cluster/, inference_assets/), so every
# script's ROOT = parents[1] resolves to src/. No dataset, caches, checkpoints or bytecode.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PKG="${1:-$REPO/dist/small_AGX_verification_v1}"
rm -rf "$PKG"; mkdir -p "$PKG"/{src,configs,artifacts,environment,submission,declarations,docs}
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '*.npz' --exclude '*.npy' --exclude '*.pt' --exclude '*.pth' \
      "$REPO/code" "$REPO/kaggle" "$REPO/cluster" "$REPO/inference_assets" "$PKG/src/"
cp "$REPO/inference.sh" "$PKG/inference.sh"; chmod +x "$PKG/inference.sh"
cp "$REPO"/configs/*.json "$PKG/configs/"
cp -L "$REPO/research/artifacts/stage2_ship7.pth" "$PKG/artifacts/stage2_ship7.pth"
cp "$REPO/environment/Dockerfile" "$REPO/environment/README.md" "$PKG/environment/"
[ -f "$REPO/environment/requirements.txt" ] && cp "$REPO/environment/requirements.txt" "$PKG/environment/"
cp "$REPO/environment/.dockerignore" "$PKG/.dockerignore"
cp "$REPO/submissions/sub_pkgship6.csv"   "$PKG/submission/final_submission_1.csv"
cp "$REPO/submissions/sub_final_ship7.csv" "$PKG/submission/final_submission_2.csv"
for f in README.md manifest.yaml PACKAGE_STRUCTURE.md; do [ -f "$REPO/package/$f" ] && cp "$REPO/package/$f" "$PKG/$f"; done
[ -f "$REPO/package/submission_info.yaml" ] && cp "$REPO/package/submission_info.yaml" "$PKG/submission/submission_info.yaml"
[ -f "$REPO/package/submission_README.md" ] && cp "$REPO/package/submission_README.md" "$PKG/submission/README.md"
[ -f "$REPO/package/CODE_MAP.md" ] && cp "$REPO/package/CODE_MAP.md" "$PKG/src/CODE_MAP.md"
[ -d "$REPO/package/declarations" ] && cp -r "$REPO/package/declarations/." "$PKG/declarations/"
[ -d "$REPO/package/docs" ] && cp -r "$REPO/package/docs/." "$PKG/docs/"
# hashes of everything a run depends on
( cd "$PKG" && sha256sum artifacts/stage2_ship7.pth configs/*.json inference.sh src/inference_assets/* > artifacts/SHA256SUMS )
# no secrets, no dataset
if grep -rIiE "api_key|apikey|secret_key|password|BEGIN (RSA |OPENSSH )?PRIVATE" "$PKG/src" "$PKG/configs" "$PKG/inference.sh" | grep -v "^Binary" | grep -vi "no api\|never put\|credential_environment" ; then
  echo "ERROR: possible credential in the package" >&2; exit 1; fi
[ -e "$PKG/src/Small-Model-Track" ] && { echo "ERROR: dataset in package" >&2; exit 1; }
echo "package at $PKG"; du -sh "$PKG"; ls "$PKG"

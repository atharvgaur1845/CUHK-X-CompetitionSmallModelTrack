#!/usr/bin/env bash
# Push the repo + raw data to the cluster. Run from the laptop.
# Caches are DERIVED (15 GB) and are deliberately not synced — rebuild them on the
# 96-core nodes, which is faster than moving them over the wire.
set -euo pipefail
REMOTE="${REMOTE:-sharanga}"
DEST="${DEST:-cuhkx}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

ssh "$REMOTE" "mkdir -p ~/$DEST"

# 1. code, research ledgers, submissions — small, always fresh
rsync -az --info=progress2 \
  --exclude '.git' --exclude 'cache' --exclude 'checkpoints' \
  --exclude 'Small-Model-Track' --exclude '__pycache__' --exclude '*.pyc' \
  "$LOCAL/" "$REMOTE:~/$DEST/"

# 2. raw data — 50 GB, the long pole. Resumable: rerun if it drops.
rsync -az --info=progress2 --partial \
  "$LOCAL/Small-Model-Track" "$REMOTE:~/$DEST/"

# 3. the two real package artifacts + third_party pretrained weights
rsync -az --info=progress2 "$LOCAL/third_party" "$REMOTE:~/$DEST/" 2>/dev/null || true
echo "synced. Next: ssh $REMOTE 'bash ~/$DEST/cluster/env.sh'"

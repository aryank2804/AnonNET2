#!/usr/bin/env bash
# Build $VEIL_ROOT/clips_756 (symlinks into clips_81f) from the list of evaluated clips.
set -euo pipefail
: "${VEIL_ROOT:?source config.sh first}"
LIST="${1:-$(dirname "$0")/../manifests/clips_756.txt}"
mkdir -p "$VEIL_ROOT/clips_756"
while read -r id; do
  [ -n "$id" ] && ln -sf "$VEIL_ROOT/clips_81f/$id.mp4" "$VEIL_ROOT/clips_756/$id.mp4"
done < "$LIST"
echo "linked $(ls "$VEIL_ROOT/clips_756" | wc -l) clips"

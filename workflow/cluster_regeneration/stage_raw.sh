#!/bin/bash
# stage_raw.sh <lane> <chunk> <chunk shots file> <lane filedb>     (before_chunk hook)
# Hard-links each shot's raw dump + manifest from $REGEN_ROOT/raw into the lane's FileDB (same
# filesystem, no extra space), staged in .tmp and published only when both files are there. A
# shot without them is left alone: the lane runner leaves it out and keeps the chunk open.
set -uo pipefail
FILEDB=${4:?lane filedb}
: "${REGEN_ROOT:?REGEN_ROOT is not set}"
mkdir -p "$FILEDB/raw" || exit 1
while read -r s; do
  [ -z "$s" ] && continue
  src=$REGEN_ROOT/raw/$s dst=$FILEDB/raw/$s
  dump=vest_${s}_daq_raw.json.gz man=vest_${s}_daq_manifest.json
  [ -f "$dst/$dump" ] && [ -f "$dst/$man" ] && continue
  { [ -f "$src/$dump" ] && [ -f "$src/$man" ]; } || continue
  rm -rf "${dst:?}" "${dst:?}.tmp"
  cp -al "$src" "$dst.tmp" && mv "$dst.tmp" "$dst" || { rm -rf "${dst:?}.tmp"; echo "cannot link raw for $s" >&2; }
done < "${3:?chunk shots file}"
exit 0

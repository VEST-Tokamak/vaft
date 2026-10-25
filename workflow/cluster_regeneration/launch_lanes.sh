#!/bin/bash
# launch_lanes.sh <run dir> [lanes] [cores per lane]
# Split every shot in ~/regen2147/raw round-robin over <lanes> lanes (each lane sees every era and
# class in proportion) and submit one g1 allocation per lane. Re-running it on an existing run dir
# keeps the lane shot lists and only resubmits lanes that are not queued/running (resume).
# Prints and records every job id it submits (<run dir>/jobs.txt); cancel ONLY those ids.
set -euo pipefail
RUN=$(readlink -f "${1:?run dir, e.g. ~/regen2147/full}"); LANES=${2:-4}; CORES=${3:-32}
ROOT=${REGEN_ROOT:-$HOME/regen2147}
MEM=${MEM:-128G}; TIME=${TIME:-2-00:00:00}; PART=${PART:-g1}
mkdir -p "$RUN"
if [ ! -f "$RUN/shots_all.txt" ]; then
  ls "$ROOT/raw" | grep -E '^[0-9]+$' | sort -n > "$RUN/shots_all.txt"
  for i in $(seq 0 $((LANES - 1))); do
    mkdir -p "$RUN/lane$i"
    awk -v n="$LANES" -v i="$i" '(NR - 1) % n == i' "$RUN/shots_all.txt" > "$RUN/lane$i/shots.txt"
  done
  cp "$ROOT/PIN_SHA" "$RUN/PIN_SHA"
fi
[ "$(cat "$RUN/PIN_SHA")" = "$(cat "$ROOT/PIN_SHA")" ] \
  || { echo "run was started at $(cat "$RUN/PIN_SHA"), $ROOT is pinned at $(cat "$ROOT/PIN_SHA")" >&2; exit 2; }
for lane in "$RUN"/lane*/; do
  lane=${lane%/}; name=r2147-$(basename "$RUN")-$(basename "$lane")
  if [ -f "$lane/job" ] && [ -n "$(squeue -h -j "$(cat "$lane/job")" 2>/dev/null)" ]; then
    echo "$name: job $(cat "$lane/job") still queued/running"; continue
  fi
  job=$(sbatch --parsable -A "${SLURM_ACCOUNT_NAME:?set SLURM_ACCOUNT_NAME, e.g. tdst}" -p "$PART" -c "$CORES" --mem="$MEM" -t "$TIME" -J "$name" \
        -o "$lane/slurm-%j.out" --wrap "bash $ROOT/run_lane.sh $lane $lane/shots.txt $CORES")
  echo "$job" > "$lane/job"; echo "$(date '+%F %T') $name $job" >> "$RUN/jobs.txt"
  echo "$name: submitted $job"
done

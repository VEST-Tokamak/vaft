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
  # The shot list is frozen here, so the raw copy must be complete and verified first:
  # copy_raw.sh leaves <scratch>/missing_or_different.lst; pass it as RAW_VERIFIED.
  [ -f "${RAW_VERIFIED:-}" ] && [ ! -s "${RAW_VERIFIED:-}" ] \
    || { echo "set RAW_VERIFIED=<copy_raw scratch>/missing_or_different.lst (must exist and be empty)" >&2; exit 2; }
  if [ -n "${SHOTS_ALL:-}" ]; then
    # A subset run (e.g. the shots another run has not started): every listed shot must have raw.
    grep -E '^[0-9]+$' "$SHOTS_ALL" | sort -n | uniq > "$RUN/shots_all.txt.tmp"
    while read -r s; do [ -d "$ROOT/raw/$s" ] || { echo "SHOTS_ALL shot $s has no $ROOT/raw/$s" >&2; exit 2; }; done < "$RUN/shots_all.txt.tmp"
    mv "$RUN/shots_all.txt.tmp" "$RUN/shots_all.txt"
  else
    ls "$ROOT/raw" | grep -E '^[0-9]+$' | sort -n > "$RUN/shots_all.txt.tmp" && mv "$RUN/shots_all.txt.tmp" "$RUN/shots_all.txt"
  fi
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
  if [ -f "$lane/job" ]; then
    # Only an active state counts as running. A job that left slurmctld (MinJobAge) makes squeue
    # exit 1 with "Invalid job id"; that is "not running". Any other squeue failure must not read
    # as "not running" (run_lane.sh's flock would stop a second runner, but do not rely on it).
    id=$(cat "$lane/job"); rc=0
    q=$(squeue -h -j "$id" -t PD,R,CF,CG,S -o %T 2>&1) || rc=$?
    if [ "$rc" -ne 0 ] && ! grep -q "Invalid job id" <<< "$q"; then
      echo "$name: squeue failed ($q); not resubmitting" >&2; continue
    fi
    [ "$rc" -eq 0 ] && [ -n "$q" ] && { echo "$name: job $id is $q"; continue; }
  fi
  [ -f "$lane/STOP" ] && { echo "$name: STOP file present; remove it to resume"; continue; }
  # DEPENDENCY=afterany:<id>[:<id>...] holds the lanes until those jobs end. It is passed as -d:
  # Slurm 22.05's sbatch reads no SBATCH_DEPENDENCY variable, so an exported one is silently ignored.
  job=$(sbatch --parsable ${DEPENDENCY:+-d "$DEPENDENCY"} -A "${SLURM_ACCOUNT_NAME:?set SLURM_ACCOUNT_NAME, e.g. tdst}" -p "$PART" -c "$CORES" --mem="$MEM" -t "$TIME" -J "$name" \
        -o "$lane/slurm-%j.out" --wrap "bash $ROOT/run_lane.sh $lane $lane/shots.txt $CORES") \
    || { echo "$name: sbatch failed" >&2; continue; }
  echo "$job" > "$lane/job"; echo "$(date '+%F %T') $name $job${DEPENDENCY:+ $DEPENDENCY}" >> "$RUN/jobs.txt"
  echo "$name: submitted $job"
done

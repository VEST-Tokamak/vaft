#!/bin/bash
# launch_lanes.sh <run dir> [lanes] [cores per lane]
# Plans a run once -- every shot in $REGEN_ROOT/raw (or SHOTS_FROM) dealt round-robin over <lanes>
# lanes, each split into CHUNK-shot chunks, pinned at $REGEN_ROOT/PIN_SHA -- then submits one
# Slurm job per lane that has pending chunks and no queued/running job, through vaft.code's
# SlurmBackend: the execution policy (VAFT_EXECUTION_POLICY, VAFT_EXECUTION_APPROVAL) is checked
# before every sbatch and every submission lands in its audit log. Account, partition and policy
# come from the site file (site.example.env). Job ids are appended to <run dir>/jobs.txt; cancel
# ONLY those. DEPENDENCY=<sbatch -d value> holds the jobs until others end; EXCLUDE_RUNS="<run
# dir> ..." refuses a plan that shares shots with those runs.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=site.sh
. "$HERE/site.sh"
RUN=$(readlink -f "${1:?run dir}"); LANES=${2:-4}; CORES=${3:-16}
: "${REGEN_ROOT:?set REGEN_ROOT in $VAFT_REGEN_ENV (see site.example.env)}"
MEM_MB=${MEM_MB:-65536}; TIME_S=${TIME_S:-172800}; CHUNK=${CHUNK:-500}
PY=$(readlink -f "$REGEN_ROOT/env")/bin/python
mkdir -p "$RUN"
if ! ls "$RUN"/lane*/chunks >/dev/null 2>&1; then
  # The shot list is frozen here, so the raw copy must be complete and verified first:
  # copy_raw.sh leaves <scratch>/missing_or_different.lst; pass it as RAW_VERIFIED.
  [ -f "${RAW_VERIFIED:-}" ] && [ ! -s "${RAW_VERIFIED:-}" ] \
    || { echo "set RAW_VERIFIED=<copy_raw scratch>/missing_or_different.lst (must exist and be empty)" >&2; exit 2; }
  ls "${SHOTS_FROM:-$REGEN_ROOT/raw}" | grep -E '^[0-9]+$' | sort -n > "$RUN/shots_all.txt"
  # shellcheck disable=SC2086
  "$PY" -m vaft.database.campaign plan "$RUN/shots_all.txt" "$RUN" --lanes "$LANES" --chunk "$CHUNK" \
    --pin "$(cat "$REGEN_ROOT/PIN_SHA")" ${EXCLUDE_RUNS:+--exclude $EXCLUDE_RUNS}
fi
slurm_args=${VAFT_SLURM_ARGS:-}
[ -n "${DEPENDENCY:-}" ] && slurm_args="$slurm_args -d $DEPENDENCY"
launch=("$PY" -m vaft.database.campaign launch "$RUN" --backend slurm --cores "$CORES" --memory-mb "$MEM_MB"
        --time-s "$TIME_S" --lane-command "bash $HERE/run_lane.sh")
[ -n "${VAFT_SLURM_CLUSTER:-}" ] && launch+=(--cluster "$VAFT_SLURM_CLUSTER")
[ -n "${VAFT_EXECUTION_APPROVAL:-}" ] && launch+=(--approval "$VAFT_EXECUTION_APPROVAL")
[ -n "$slurm_args" ] && launch+=(--slurm-args "$slurm_args")
out=$("${launch[@]}")
echo "$out"
echo "$(date '+%F %T') $out" >> "$RUN/jobs.txt"

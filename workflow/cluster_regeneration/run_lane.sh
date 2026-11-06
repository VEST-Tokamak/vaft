#!/bin/bash
# run_lane.sh <lane dir>       (the Slurm job of one lane; also runs by hand on any host)
# The site's half of a lane: the pinned environment, the srvns namespace that shows the lane's
# FileDB as /srv/vest.filedb, the packaged base-era Green table staged inside that FileDB, and
# the before/after-chunk hooks. Everything else -- chunks, resume, Snakemake, the chunk verdict,
# CPU/memory measurement -- is `python -m vaft.database.campaign run-lane`, which runs the same
# way without a scheduler.
#
# CORES (default $SLURM_CPUS_PER_TASK), MEM_MB (default $SLURM_MEM_PER_NODE): what the lane books.
# MAX_CHUNKS=N runs at most N pending chunks; ALLOW_MISSING_RAW=1 closes chunks with absent raw.
# SOLVER_RSS_KILL_GB (64): one watched solver above it, or all of them above MEM_MB - 8 GB, is
# killed (the stock RDCON grows to 60-210 GB on msing >= 25 cases before aborting anyway).
set -euo pipefail
LANE=$(readlink -f "${1:?lane dir}")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=site.sh
. "$HERE/site.sh"
regen_cluster_env
WF=$REGEN_SRC/workflow/automatic_pipeline_1_routine_data_processing
CONFIG=${REGEN_CONFIG:-$REGEN_ROOT/regen-config.yaml}
CORES=${CORES:-${SLURM_CPUS_PER_TASK:-8}}
MEM_MB=${MEM_MB:-${SLURM_MEM_PER_NODE:-65536}}
FILEDB=$LANE/filedb
mkdir -p "$FILEDB"
python -c "import vaft,sys; sys.exit(0 if vaft.__file__.startswith('$REGEN_SRC/') else 3)" \
  || { echo "vaft.__file__ outside $REGEN_SRC" >&2; exit 3; }

# The packaged base-era Green table, staged inside the FileDB when the run config points
# efit.table_dir there, so base-era k-files name a /srv/vest.filedb path rather than this host's
# worktree (byte-identical to vaft/data/efit at the pin; checked).
BASE_TABLE=$(python - "$CONFIG" <<'PY'
import sys, yaml
d = (yaml.safe_load(open(sys.argv[1])) or {}).get("efit", {}).get("table_dir", "")
print(d[len("/srv/vest.filedb/"):].rstrip("/") if d.startswith("/srv/vest.filedb/") else "")
PY
)
if [ -n "$BASE_TABLE" ]; then
  if [ ! -d "$FILEDB/$BASE_TABLE" ]; then
    mkdir -p "$(dirname "$FILEDB/$BASE_TABLE")"
    rm -rf "${FILEDB:?}/${BASE_TABLE:?}.tmp"
    cp -a "$REGEN_SRC/vaft/data/efit" "$FILEDB/$BASE_TABLE.tmp" && mv "$FILEDB/$BASE_TABLE.tmp" "$FILEDB/$BASE_TABLE"
  fi
  for t in "$REGEN_SRC"/vaft/data/efit/*; do
    [ -f "$t" ] || continue
    cmp -s "$t" "$FILEDB/$BASE_TABLE/$(basename "$t")" \
      || { echo "$BASE_TABLE/$(basename "$t") differs from the pin's vaft/data/efit" >&2; exit 2; }
  done
fi

KILL_MB=$(( ${SOLVER_RSS_KILL_GB:-64} * 1024 )); [ "$KILL_MB" -gt $(( MEM_MB / 2 )) ] && KILL_MB=$(( MEM_MB / 2 ))
TOTAL_MB=$(( MEM_MB - 8192 ))
args=(--cores "$CORES" --workflow-dir "$WF" --config "$CONFIG" --filedb "$FILEDB" --base-dir /srv/vest.filedb
      --pin "$REGEN_SHA" --mem-mb "$TOTAL_MB" --guard-kill-mb "$KILL_MB" --guard-total-mb "$TOTAL_MB"
      --prefix "$REGEN_ROOT/srvns $FILEDB nice -n 5"
      --before-chunk "$HERE/stage_raw.sh" --after-chunk "$HERE/prune_scratch.sh")
[ -n "${MAX_CHUNKS:-}" ] && args+=(--max-chunks "$MAX_CHUNKS")
[ "${ALLOW_MISSING_RAW:-0}" = 1 ] && args+=(--allow-missing-inputs)
export TMPDIR=${TMPDIR:-/tmp}/vaft-lane-${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
trap 'rm -rf "${TMPDIR:?}"' EXIT
python -m vaft.database.campaign run-lane "$LANE" "${args[@]}"

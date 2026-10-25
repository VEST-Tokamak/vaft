#!/bin/bash
# run_lane.sh <lane dir> <shots file> [cores]
# One lane = one FileDB root (<lane dir>/filedb, seen as /srv/vest.filedb through srvns) processed
# in sequential chunks of CHUNK shots, each one `snakemake --cores N --scheduler greedy` over the
# UNCHANGED pipeline-1 Snakefile at the pinned SHA. Lanes never share a FileDB root, so the
# run-scoped preflight/IMPA-selection checkpoints of two lanes cannot race.
#
# Resumable: a chunk is listed in <lane dir>/done_chunks.txt only when Snakemake finished it (exit 0,
# or exit 1 with nothing but per-shot rule errors under --keep-going); anything else stops the lane
# with exit 1 for a look. Rerunning continues; Snakemake skips finished products. A STOP file in
# <lane dir> ends the lane after the current chunk. One run_lane per lane at a time (flock).
set -uo pipefail
LANE=$(readlink -f "${1:?lane dir}"); SHOTS_FILE=$(readlink -f "${2:?shots file}"); CORES=${3:-${SLURM_CPUS_PER_TASK:-8}}
. "${REGEN_ROOT:-$HOME/regen2147}/env.sh"          # sets REGEN_ROOT/REGEN_SRC/REGEN_ENV, isolates HOME + HSDS
WF=$REGEN_SRC/workflow/automatic_pipeline_1_routine_data_processing
CONFIG=${REGEN_CONFIG:-$REGEN_ROOT/regen-config.yaml}
FILEDB=$LANE/filedb; WORK=$LANE/work; META=$FILEDB/pipeline/preflight/metadata
mkdir -p "$FILEDB/raw" "$WORK" "$LANE/chunks" "$LANE/logs"
log() { echo "$(date '+%F %T') $*" | tee -a "$LANE/lane.log"; }

exec 9>"$LANE/.lock"
flock -n 9 || { log "another run_lane holds $LANE/.lock; not starting (job ${SLURM_JOB_ID:-none})"; exit 0; }
export TMPDIR=${TMPDIR:-/tmp}/regen2147-${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# The lane belongs to one pin and one chunk size for its whole life.
[ -f "$LANE/PIN_SHA" ] || echo "$REGEN_SHA" > "$LANE/PIN_SHA"
[ "$(cat "$LANE/PIN_SHA")" = "$REGEN_SHA" ] \
  || { log "lane is pinned at $(cat "$LANE/PIN_SHA"), env.sh resolves $REGEN_SHA: refusing"; exit 2; }
CHUNK=${CHUNK:-$(cat "$LANE/chunk_size" 2>/dev/null || echo 500)}
[ -f "$LANE/chunk_size" ] || echo "$CHUNK" > "$LANE/chunk_size"
[ "$(cat "$LANE/chunk_size")" = "$CHUNK" ] || { log "lane was split with CHUNK=$(cat "$LANE/chunk_size"), not $CHUNK"; exit 2; }
log "lane start config=$CONFIG sha=$REGEN_SHA cores=$CORES chunk=$CHUNK host=$(hostname) job=${SLURM_JOB_ID:-none}"
python -c "import vaft,sys; sys.exit(0 if vaft.__file__.startswith('$REGEN_SRC/') else 3)" \
  || { log "vaft.__file__ outside $REGEN_SRC"; exit 3; }

# Split once: chunk names must keep meaning the same shots across resumes.
ls "$LANE"/chunks/shots.[0-9][0-9][0-9] >/dev/null 2>&1 || split -l "$CHUNK" -d -a 3 "$SHOTS_FILE" "$LANE/chunks/shots."

# The packaged base-era Green table, staged inside the FileDB when the run config points
# efit.table_dir there, so base-era k-files name a /srv/vest.filedb path rather than this host's
# worktree (byte-identical to vaft/data/efit at the pin; checked below).
BASE_TABLE=$(python - "$CONFIG" <<'PY'
import sys, yaml
d = (yaml.safe_load(open(sys.argv[1])) or {}).get("efit", {}).get("table_dir", "")
print(d[len("/srv/vest.filedb/"):].rstrip("/") if d.startswith("/srv/vest.filedb/") else "")
PY
)
if [ -n "$BASE_TABLE" ]; then
  if [ ! -d "$FILEDB/$BASE_TABLE" ]; then
    mkdir -p "$(dirname "$FILEDB/$BASE_TABLE")"
    cp -a "$REGEN_SRC/vaft/data/efit" "$FILEDB/$BASE_TABLE.tmp" && mv "$FILEDB/$BASE_TABLE.tmp" "$FILEDB/$BASE_TABLE"
  fi
  for t in "$REGEN_SRC"/vaft/data/efit/*; do
    [ -f "$t" ] || continue
    cmp -s "$t" "$FILEDB/$BASE_TABLE/$(basename "$t")" || { log "$BASE_TABLE/$(basename "$t") differs from the pin's vaft/data/efit"; exit 2; }
  done
fi

touch "$LANE/done_chunks.txt"
for f in "$LANE"/chunks/shots.[0-9][0-9][0-9]; do
  c=$(basename "$f")
  grep -qx "$c" "$LANE/done_chunks.txt" && continue
  [ -f "$LANE/STOP" ] && { log "STOP before $c"; exit 0; }
  # Raw: hard links into this lane's root (same filesystem, no extra space). A shot without
  # both its dump and its manifest is left out: the raw rule would otherwise try VEST SQL.
  : > "$LANE/chunks/$c.run"
  while read -r s; do
    [ -z "$s" ] && continue
    src=$REGEN_ROOT/raw/$s
    if [ -f "$src/vest_${s}_daq_raw.json.gz" ] && [ -f "$src/vest_${s}_daq_manifest.json" ]; then
      [ -e "$FILEDB/raw/$s" ] || cp -al "$src" "$FILEDB/raw/$s"
      echo "$s" >> "$LANE/chunks/$c.run"
    else
      echo "$c $s" >> "$LANE/missing_raw.txt"; log "$c: no raw dump+manifest for $s, left out"
    fi
  done < "$f"
  if [ ! -s "$LANE/chunks/$c.run" ]; then log "$c: no shot with raw, nothing to run"; echo "$c" >> "$LANE/done_chunks.txt"; continue; fi
  cfg=$LANE/chunks/$c.yaml
  { cat "$CONFIG"; echo "cores: $CORES"; echo "shots:"; sed 's/^/- /' "$LANE/chunks/$c.run"; } > "$cfg"
  # The checkpoints' outputs are run-scoped and newer than every (hard-linked) raw dump, so under
  # --rerun-triggers mtime the previous chunk's verdict would be reused for this chunk's shots.
  rm -f "$META/eligible_shots.json" "$META/excluded_shots.json" "$META/impa_shots.json"
  t0=$(date +%s); log "$c start ($(wc -l < "$LANE/chunks/$c.run") shots)"
  # Memory: the Snakefile declares none, so cap it here. GPEC cells (DCON/RDCON `match` peaks
  # 12-25 GB) are booked at GPEC_MEM_MB, CHEASE at 4 GB; running jobs stay under the allocation - 8 GB.
  ALLOC_MB=${SLURM_MEM_PER_NODE:-65536}
  # Peak-RSS sampler for the solver binaries (no Slurm RSS accounting); may include other lanes' on the node.
  ( while :; do ps -u "$(id -u)" -o rss=,args= | grep -E "$REGEN_ROOT|$GPECHOME|$CHEASEHOME|$EFITHOME" \
      | grep -vE "grep|ps -u" | awk -v t="$(date +%s)" '{print t, $1, $2, $3}' >> "$LANE/logs/$c.rss"; sleep 20; done ) &
  SAMPLER=$!
  attempt=$LANE/logs/$c.attempt-$(date +%Y%m%d%H%M%S).log
  SM=(python -m snakemake --snakefile "$WF/Snakefile" --configfile "$cfg" --directory "$WORK")
  # A time-limit kill leaves the lock behind; we hold the lane lock, so no live Snakemake owns it.
  "$REGEN_ROOT/srvns" "$FILEDB" "${SM[@]}" --unlock > "$attempt" 2>&1
  "$REGEN_ROOT/srvns" "$FILEDB" nice -n 5 "${SM[@]}" --cores "$CORES" --scheduler greedy \
      --keep-going --rerun-incomplete --rerun-triggers mtime --printshellcmds --nocolor \
      --resources mem_mb=$(( ALLOC_MB - 8192 )) \
      --set-resources "run_gpec_module:mem_mb=${GPEC_MEM_MB:-24000}" "run_chease:mem_mb=4000" \
      >> "$attempt" 2>&1
  rc=$?
  kill "$SAMPLER" 2>/dev/null
  cp "$attempt" "$LANE/logs/$c.snakemake.log"      # the latest attempt is what harvest.py reads
  log "$c exit=$rc wall=$(( $(date +%s) - t0 ))s log=$(basename "$attempt")"
  for j in eligible_shots excluded_shots impa_shots; do
    cp -p "$META/$j.json" "$LANE/chunks/$c.$j.json" 2>/dev/null || true
  done
  if [ "$rc" -ne 0 ]; then
    # Under --keep-going a failed shot exits 1 with "Error in rule"; that is a per-shot outcome.
    # A lock, a workflow/config error, a missing checkpoint verdict or a killed Snakemake is not.
    if ! grep -q "^Error in rule" "$attempt" \
       || grep -qE "LockException|Directory cannot be locked|WorkflowError|SyntaxError|MissingInputException|^Traceback" "$attempt" \
       || [ ! -f "$LANE/chunks/$c.eligible_shots.json" ]; then
      log "$c: Snakemake failed as a whole (rc=$rc); lane stopped, see $attempt"; exit 1
    fi
  fi
  echo "$c" >> "$LANE/done_chunks.txt"
done
log "lane done"

#!/bin/bash
# run_lane.sh <lane dir> <shots file> [cores]
# One lane = one FileDB root (<lane dir>/filedb, seen as /srv/vest.filedb through srvns) processed
# in sequential chunks of CHUNK shots, each one `snakemake --cores N --scheduler greedy` over the
# UNCHANGED pipeline-1 Snakefile at the pinned SHA. Resumable: finished chunks are listed in
# <lane dir>/done_chunks.txt; rerunning continues (Snakemake skips finished products). A STOP file in
# <lane dir> ends the lane after the current chunk. Lanes never share a FileDB root, so the
# run-scoped preflight/IMPA-selection checkpoints of two lanes cannot race.
set -uo pipefail
LANE=$(readlink -f "${1:?lane dir}"); SHOTS_FILE=$(readlink -f "${2:?shots file}"); CORES=${3:-${SLURM_CPUS_PER_TASK:-8}}
CHUNK=${CHUNK:-500}
. "${REGEN_ROOT:-$HOME/regen2147}/env.sh"          # sets REGEN_ROOT/REGEN_SRC/REGEN_ENV, isolates HOME + HSDS
WF=$REGEN_SRC/workflow/automatic_pipeline_1_routine_data_processing
FILEDB=$LANE/filedb; WORK=$LANE/work
mkdir -p "$FILEDB/raw" "$WORK" "$LANE/chunks" "$LANE/logs"
export TMPDIR=${TMPDIR:-/tmp}/regen2147-${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
log() { echo "$(date '+%F %T') $*" | tee -a "$LANE/lane.log"; }
log "lane start config=${REGEN_CONFIG:-$REGEN_ROOT/regen-config.yaml} sha=$REGEN_SHA cores=$CORES chunk=$CHUNK host=$(hostname) job=${SLURM_JOB_ID:-none}"
python -c "import vaft,sys; sys.exit(0 if vaft.__file__.startswith('$REGEN_SRC/') else 3)" \
  || { log "vaft.__file__ outside $REGEN_SRC"; exit 3; }
# Raw: hard links into this lane's root (same filesystem, no extra space). The copy in
# $REGEN_ROOT/raw is the read-only-by-convention source.
while read -r s; do
  [ -z "$s" ] && continue
  [ -e "$FILEDB/raw/$s" ] || cp -al "$REGEN_ROOT/raw/$s" "$FILEDB/raw/$s" 2>/dev/null \
    || log "no raw dump for $s in $REGEN_ROOT/raw"
done < "$SHOTS_FILE"
split -l "$CHUNK" -d -a 3 "$SHOTS_FILE" "$LANE/chunks/shots."
touch "$LANE/done_chunks.txt"
for f in "$LANE"/chunks/shots.[0-9][0-9][0-9]; do
  c=$(basename "$f")
  grep -qx "$c" "$LANE/done_chunks.txt" && continue
  [ -f "$LANE/STOP" ] && { log "STOP before $c"; exit 0; }
  cfg=$LANE/chunks/$c.yaml
  { cat "${REGEN_CONFIG:-$REGEN_ROOT/regen-config.yaml}"; echo "cores: $CORES"; echo "shots:"; sed 's/^/- /' "$f"; } > "$cfg"
  t0=$(date +%s); log "$c start ($(wc -l < "$f") shots)"
  # Memory: the Snakefile declares none, so cap it here (DCON/RDCON match peaks at 12-25 GB).
  # Every job gets Snakemake's default mem_mb; GPEC cells are booked at GPEC_MEM_MB; the sum of
  # running jobs stays under the allocation minus 8 GB.
  ALLOC_MB=${SLURM_MEM_PER_NODE:-65536}
  # Peak-RSS sampler for the solver binaries (no Slurm accounting of RSS on tdst).
  ( while :; do ps -u "$(id -u)" -o rss=,args= | grep -E "$REGEN_ROOT|$GPECHOME|$CHEASEHOME|$EFITHOME" \
      | grep -vE "grep|ps -u" | awk -v t="$(date +%s)" '{print t, $1, $2, $3}' >> "$LANE/logs/$c.rss"; sleep 20; done ) &
  SAMPLER=$!
  # Snakemake runs inside the namespace; its jobs inherit it.
  "$REGEN_ROOT/srvns" "$FILEDB" nice -n 5 python -m snakemake --snakefile "$WF/Snakefile" \
      --configfile "$cfg" --directory "$WORK" --cores "$CORES" --scheduler greedy \
      --keep-going --rerun-incomplete --rerun-triggers mtime --printshellcmds --nocolor \
      --resources mem_mb=$(( ALLOC_MB - 8192 )) \
      --set-resources "run_gpec_module:mem_mb=${GPEC_MEM_MB:-16000}" "run_chease:mem_mb=4000" \
      >> "$LANE/logs/$c.snakemake.log" 2>&1
  rc=$?
  kill "$SAMPLER" 2>/dev/null
  log "$c exit=$rc wall=$(( $(date +%s) - t0 ))s"
  # The preflight / IMPA-selection checkpoints are run-scoped: the next chunk overwrites them.
  # Keep each chunk's verdict next to its shot list (merge.sh rebuilds the run-wide ones).
  for j in eligible_shots excluded_shots impa_shots; do
    cp -p "$FILEDB/pipeline/preflight/metadata/$j.json" "$LANE/chunks/$c.$j.json" 2>/dev/null || true
  done
  # Snakemake exits non-zero when any shot's job failed under --keep-going; that is a
  # per-shot outcome (harvested from the products), not a lane failure.
  echo "$c" >> "$LANE/done_chunks.txt"
done
log "lane done"

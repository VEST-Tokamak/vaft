#!/bin/bash
# run_lane.sh <lane dir> <shots file> [cores]
# One lane = one FileDB root (<lane dir>/filedb, seen as /srv/vest.filedb through srvns) processed
# in sequential chunks of CHUNK shots, each one `snakemake --cores N --scheduler greedy` over the
# UNCHANGED pipeline-1 Snakefile at the pinned SHA. Lanes never share a FileDB root, so the
# run-scoped preflight/IMPA-selection checkpoints of two lanes cannot race.
#
# Resumable: a chunk is listed in <lane dir>/done_chunks.txt only when Snakemake finished it (exit 0,
# or exit 1 with nothing but per-shot rule errors under --keep-going) and every shot of it had its
# raw dump; anything else stops the lane with exit 1 for a look. Rerunning continues; Snakemake
# skips finished products. A STOP file in <lane dir> ends the lane after the current chunk.
# One run_lane per lane at a time (flock).
set -uo pipefail
LANE=$(readlink -f "${1:?lane dir}"); SHOTS_FILE=$(readlink -f "${2:?shots file}"); CORES=${3:-${SLURM_CPUS_PER_TASK:-8}}
. "${REGEN_ROOT:-$HOME/regen2147}/env.sh"          # sets REGEN_ROOT/REGEN_SRC/REGEN_ENV, isolates HOME + HSDS
WF=$REGEN_SRC/workflow/automatic_pipeline_1_routine_data_processing
CONFIG=${REGEN_CONFIG:-$REGEN_ROOT/regen-config.yaml}
FILEDB=$LANE/filedb; WORK=$LANE/work; META=$FILEDB/pipeline/preflight/metadata
mkdir -p "$FILEDB/raw" "$WORK" "$LANE/chunks" "$LANE/logs"
log() { echo "$(date '+%F %T') $*" | tee -a "$LANE/lane.log"; }

exec 9>"$LANE/.lock"
flock -w 60 9 || { log "another run_lane holds $LANE/.lock; not starting (job ${SLURM_JOB_ID:-none})"; exit 0; }
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

# Split once (into a temporary dir, published only when complete): chunk names must keep meaning
# the same shots across resumes.
if ! ls "$LANE"/chunks/shots.[0-9][0-9][0-9] >/dev/null 2>&1; then
  rm -rf "$LANE/chunks.tmp"; mkdir -p "$LANE/chunks.tmp"
  split -l "$CHUNK" -d -a 3 "$SHOTS_FILE" "$LANE/chunks.tmp/shots." || { log "split failed"; exit 2; }
  mv "$LANE"/chunks.tmp/shots.* "$LANE/chunks/" && rmdir "$LANE/chunks.tmp"
fi

# The packaged base-era Green table, staged inside the FileDB when the run config points
# efit.table_dir there, so base-era k-files name a /srv/vest.filedb path rather than this host's
# worktree (byte-identical to vaft/data/efit at the pin; checked below).
BASE_TABLE=$(python - "$CONFIG" <<'PY'
import sys, yaml
d = (yaml.safe_load(open(sys.argv[1])) or {}).get("efit", {}).get("table_dir", "")
print(d[len("/srv/vest.filedb/"):].rstrip("/") if d.startswith("/srv/vest.filedb/") else "")
PY
) || { log "cannot read efit.table_dir from $CONFIG"; exit 2; }
if [ -n "$BASE_TABLE" ]; then
  if [ ! -d "$FILEDB/$BASE_TABLE" ]; then
    mkdir -p "$(dirname "$FILEDB/$BASE_TABLE")"
    rm -rf "$FILEDB/$BASE_TABLE.tmp"
    cp -a "$REGEN_SRC/vaft/data/efit" "$FILEDB/$BASE_TABLE.tmp" && mv "$FILEDB/$BASE_TABLE.tmp" "$FILEDB/$BASE_TABLE"
  fi
  for t in "$REGEN_SRC"/vaft/data/efit/*; do
    [ -f "$t" ] || continue
    cmp -s "$t" "$FILEDB/$BASE_TABLE/$(basename "$t")" || { log "$BASE_TABLE/$(basename "$t") differs from the pin's vaft/data/efit"; exit 2; }
  done
fi

ALLOC_MB=${SLURM_MEM_PER_NODE:-65536}
# Runaway guard limits: one solver above KILL_MB, or the lane's solvers together above the
# allocation minus 8 GB (the largest is stopped first).
KILL_MB=$(( ${SOLVER_RSS_KILL_GB:-64} * 1024 )); [ "$KILL_MB" -gt $(( ALLOC_MB / 2 )) ] && KILL_MB=$(( ALLOC_MB / 2 ))
TOTAL_MB=$(( ALLOC_MB - 8192 ))

touch "$LANE/done_chunks.txt"
for f in "$LANE"/chunks/shots.[0-9][0-9][0-9]; do
  c=$(basename "$f")
  grep -qx "$c" "$LANE/done_chunks.txt" && continue
  [ -f "$LANE/STOP" ] && { log "STOP before $c"; exit 0; }
  # Raw: hard links into this lane's root (same filesystem, no extra space), staged in .tmp and
  # published only when both files are there. A shot without its dump and manifest is left out
  # (the raw rule would otherwise try VEST SQL) and keeps the chunk from being marked done.
  : > "$LANE/chunks/$c.run"; missing=0
  while read -r s; do
    [ -z "$s" ] && continue
    src=$REGEN_ROOT/raw/$s dst=$FILEDB/raw/$s
    dump=vest_${s}_daq_raw.json.gz man=vest_${s}_daq_manifest.json
    if [ ! -f "$dst/$dump" ] || [ ! -f "$dst/$man" ]; then
      if [ -f "$src/$dump" ] && [ -f "$src/$man" ]; then
        rm -rf "$dst" "$dst.tmp"
        cp -al "$src" "$dst.tmp" && mv "$dst.tmp" "$dst" || { rm -rf "$dst.tmp"; log "$c: cannot link raw for $s"; }
      fi
    fi
    if [ -f "$dst/$dump" ] && [ -f "$dst/$man" ]; then
      echo "$s" >> "$LANE/chunks/$c.run"
    else
      missing=$((missing + 1))
      grep -qx "$c $s" "$LANE/missing_raw.txt" 2>/dev/null || echo "$c $s" >> "$LANE/missing_raw.txt"
      log "$c: no raw dump+manifest for $s, left out"
    fi
  done < "$f"
  if [ ! -s "$LANE/chunks/$c.run" ]; then
    log "$c: no shot with raw, nothing to run"
    [ "${ALLOW_MISSING_RAW:-0}" = 1 ] && echo "$c" >> "$LANE/done_chunks.txt"
    continue
  fi
  cfg=$LANE/chunks/$c.yaml
  { cat "$CONFIG"; echo "cores: $CORES"; echo "shots:"; sed 's/^/- /' "$LANE/chunks/$c.run"; } > "$cfg"
  # The checkpoints' outputs are run-scoped and newer than every (hard-linked) raw dump, so under
  # --rerun-triggers mtime the previous chunk's verdict would be reused for this chunk's shots.
  rm -f "$META"/{eligible,excluded,impa}_shots.json "$LANE/chunks/$c".{eligible,excluded,impa}_shots.json
  t0=$(date +%s); log "$c start ($(wc -l < "$LANE/chunks/$c.run") shots)"
  attempt=$LANE/logs/$c.attempt-$(date +%Y%m%d%H%M%S).log
  SM=(python -m snakemake --snakefile "$WF/Snakefile" --configfile "$cfg" --directory "$WORK")
  # A time-limit kill leaves the lock behind; we hold the lane lock, so no live Snakemake owns it.
  "$REGEN_ROOT/srvns" "$FILEDB" "${SM[@]}" --unlock > "$attempt" 2>&1
  # Memory: the Snakefile declares none, so cap it here. GPEC cells (DCON/RDCON `match` peaks
  # 12-25 GB) are booked at GPEC_MEM_MB, CHEASE at 4 GB; running jobs stay under the allocation - 8 GB.
  "$REGEN_ROOT/srvns" "$FILEDB" nice -n 5 "${SM[@]}" --cores "$CORES" --scheduler greedy \
      --keep-going --rerun-incomplete --rerun-triggers mtime --printshellcmds --nocolor \
      --resources mem_mb="$TOTAL_MB" \
      --set-resources "run_gpec_module:mem_mb=${GPEC_MEM_MB:-24000}" "run_chease:mem_mb=4000" \
      >> "$attempt" 2>&1 9>&- &
  SM_PID=$!
  # RSS sampler + runaway guard over this lane's own process tree (descendants of SM_PID; tdst has
  # no Slurm RSS accounting). The stock RDCON on msing >= 25 cases grows to 60-210 GB before it
  # aborts in gal.f anyway, which otherwise takes the allocation (and its Snakemake) with it.
  # Before a kill the process is re-read from /proc, so a reused PID is never hit.
  ( descends() { local p=$1; while [ "$p" -gt 1 ]; do [ "$p" = "$SM_PID" ] && return 0
      p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null) || return 1; [ -n "$p" ] || return 1; done; return 1; }
    rss_mb() { awk '/^VmRSS:/ {print int($2 / 1024)}' "/proc/$1/status" 2>/dev/null; }
    while kill -0 "$SM_PID" 2>/dev/null; do
      total=0 big_pid= big_mb=0 big_comm=
      while read -r pid rss comm; do
        descends "$pid" || continue
        case "$comm" in dcon|rdcon|match|rmatch|stride|gpec|chease|efit.bin|efund.bin|ld-linux*) ;; *) continue ;; esac
        mb=$(( rss / 1024 )); total=$(( total + mb ))
        echo "$(date +%s) $pid $rss $comm $(readlink "/proc/$pid/cwd" 2>/dev/null)" >> "$LANE/logs/$c.rss"
        [ "$mb" -gt "$big_mb" ] && { big_pid=$pid big_mb=$mb big_comm=$comm; }
      done < <(ps -u "$(id -u)" -o pid=,rss=,comm=)
      if [ -n "$big_pid" ] && { [ "$big_mb" -gt "$KILL_MB" ] || [ "$total" -gt "$TOTAL_MB" ]; }; then
        now=$(rss_mb "$big_pid")
        if [ "$(cat "/proc/$big_pid/comm" 2>/dev/null)" = "$big_comm" ] && [ -n "$now" ] && [ "$now" -ge $(( big_mb / 2 )) ]; then
          kill -KILL "$big_pid" 2>/dev/null && log "$c: killed $big_comm pid $big_pid at ${now} MB RSS (lane solvers ${total} MB; limits ${KILL_MB}/${TOTAL_MB} MB) ($(readlink "/proc/$big_pid/cwd" 2>/dev/null))"
        fi
      fi
      sleep 5
    done ) 9>&- &
  SAMPLER=$!
  wait "$SM_PID"
  rc=$?
  kill "$SAMPLER" 2>/dev/null
  cp "$attempt" "$LANE/logs/$c.snakemake.log"      # the latest attempt is what harvest.py reads
  log "$c exit=$rc wall=$(( $(date +%s) - t0 ))s log=$(basename "$attempt")"
  for j in eligible_shots excluded_shots impa_shots; do
    [ -f "$META/$j.json" ] && cp -p "$META/$j.json" "$LANE/chunks/$c.$j.json"
  done
  if [ "$rc" -ne 0 ]; then
    # Under --keep-going a failed shot exits 1 with "Error in rule" and the run ends with
    # "Exiting because a job execution failed" + "WorkflowError: At least one job did not complete
    # successfully."; that is a per-shot outcome. A lock, any other WorkflowError, a failed
    # run-scoped rule, a missing checkpoint verdict or a killed Snakemake (which just stops) is not.
    odd_workflow_error=$(grep -A1 "^WorkflowError" "$attempt" | grep -vE "^WorkflowError:?$|^At least one job did not complete successfully|^--$")
    if ! grep -q "^Error in rule" "$attempt" \
       || ! grep -q "Exiting because a job execution failed" "$attempt" \
       || grep -qE "LockException|Directory cannot be locked|SyntaxError|MissingInputException|^Traceback" "$attempt" \
       || grep -qE "^Error in rule (generate_static_ods|generate_efit_table|preflight_raw_dumps|select_impa_shots|plot_static):" "$attempt" \
       || [ -n "$odd_workflow_error" ] \
       || [ ! -f "$LANE/chunks/$c.eligible_shots.json" ]; then
      log "$c: Snakemake failed as a whole (rc=$rc); lane stopped, see $attempt"; exit 1
    fi
  fi
  # Solver scratch: DCON's euler.bin and the vacuum vmat.bin are ~98 % of the GPEC tree (~5 GB per
  # MHD shot) and only feed later solves of the same chunk, all finished now. They are not
  # Snakemake outputs and no stage product reads them, but a later rerun of an ideal-GPEC cell
  # alone needs its DCON cell rerun first (euler.bin). Size, mtime and path stay in pruned.tsv
  # (PRUNE_HASH=1 adds sha256: ~10 s per MHD shot of serial reading, so off by default).
  if [ "${PRUNE_SOLVER_SCRATCH:-1}" = 1 ] && [ -d "$FILEDB/gpec" ]; then
    while read -r s; do
      find "$FILEDB"/gpec/magnetic/chease/*/"$s"/n=*/work -type f \( -name euler.bin -o -name vmat.bin \) \
        -printf "%s\t%TY-%Tm-%Td %TH:%TM\t%p\n" 2>/dev/null | while IFS=$'\t' read -r size mtime path; do
          sha=-; [ "${PRUNE_HASH:-0}" = 1 ] && sha=$(sha256sum < "$path" | cut -d' ' -f1)
          printf "%s\t%s\t%s\t%s\t%s\n" "$c" "$size" "$mtime" "$sha" "${path#"$FILEDB"/}" >> "$LANE/pruned.tsv"
          rm -f "$path"
        done
    done < "$LANE/chunks/$c.run"
  fi
  if [ "$missing" -gt 0 ] && [ "${ALLOW_MISSING_RAW:-0}" != 1 ]; then
    log "$c: $missing shot(s) without raw (missing_raw.txt); chunk left open for a rerun (ALLOW_MISSING_RAW=1 to close)"
    continue
  fi
  echo "$c" >> "$LANE/done_chunks.txt"
done
log "lane done"

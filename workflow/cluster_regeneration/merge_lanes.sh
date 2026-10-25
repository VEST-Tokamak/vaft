#!/bin/bash
# merge.sh <merged filedb dir> <lane dir> [<lane dir> ...]
# Union of lane FileDB roots into one root shaped like /srv/vest.filedb, by HARD LINKS (no copy,
# same filesystem). Per-shot directories are disjoint between lanes; a shot present in two lanes
# is refused. Run-scoped products:
#   omas/static/<era>, pipeline/efit_tables/<era>   one copy (first lane); the others must match
#                                                     in their .json.gz / .ddd content (checked)
#   pipeline/preflight/metadata/*.json              rebuilt over ALL merged shots by the same
#                                                     scripts the checkpoints run
# Run on a compute node (sbatch -c 2): the preflight re-read touches every raw dump.
set -euo pipefail
OUT=$(readlink -f "$1"); shift
. "${REGEN_ROOT:-$HOME/regen2147}/env.sh"
WF=$REGEN_SRC/workflow/automatic_pipeline_1_routine_data_processing
mkdir -p "$OUT"
declare -A OWNER
for lane in "$@"; do
  L=$(readlink -f "$lane")/filedb
  # Every per-shot directory, wherever the FileDB grammar puts it (raw/<s>, omas/<stage>/<s>,
  # omas/<stage>/magnetic[/chease/<product>]/<s>, efit|chease/magnetic/<s>,
  # gpec/magnetic/chease/<product>/<s>): a directory whose name is the shot number.
  while IFS= read -r key; do
    key=${key#./}
    if [ -e "$OUT/$key" ]; then
      [ "${OWNER[$key]:-}" = "$L" ] && continue
      echo "shot dir $key exists in two lanes (${OWNER[$key]:-earlier merge} and $L)" >&2; exit 2
    fi
    mkdir -p "$OUT/$(dirname "$key")"
    cp -al "$L/$key" "$OUT/$key"; OWNER[$key]=$L
  done < <(cd "$L" && find . -path ./pipeline -prune -o -type d -regextype posix-extended -regex '.*/[0-9]+' -print -prune)
  for d in omas/static pipeline/efit_tables; do
    [ -d "$L/$d" ] || continue
    mkdir -p "$OUT/$d"
    for e in $(ls "$L/$d"); do
      if [ ! -e "$OUT/$d/$e" ]; then cp -al "$L/$d/$e" "$OUT/$d/$e"; continue; fi
      [ -d "$L/$d/$e" ] || continue   # a build log beside the table: keep the first lane's
      for f in $(cd "$L/$d/$e" && find . -type f \( -name '*.ddd' -o -name '*.json.gz' \)); do
        cmp -s "$L/$d/$e/$f" "$OUT/$d/$e/$f" || { echo "run-scoped $d/$e/$f differs between lanes" >&2; exit 2; }
      done
    done
  done
done
# Run-wide preflight + IMPA selection over every merged shot, exactly as the checkpoints run
# them (required fields / min samples from the pinned workflow config.yaml), seen as /srv.
FIELDS=$(python - "$WF/config.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))["raw"]["preflight"]
print(",".join(map(str, c["required_fields"])), c["min_samples"])
PY
)
read -r REQ MINS <<< "$FIELDS"
ls "$OUT/raw" | sort -n | sed 's#.*#/srv/vest.filedb/raw/&/vest_&_daq_raw.json.gz#' > "$OUT.rawlist"
META=/srv/vest.filedb/pipeline/preflight/metadata
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576
"$REGEN_ROOT/srvns" "$OUT" bash -c "mkdir -p $META && \
  python $WF/validate_raw_dumps.py --required-fields $REQ --min-samples $MINS \
    --eligible-output $META/eligible_shots.json --excluded-output $META/excluded_shots.json \$(cat $OUT.rawlist) && \
  python $WF/select_impa_shots.py --eligible $META/eligible_shots.json --output $META/impa_shots.json \$(cat $OUT.rawlist)"
echo "merged $(ls "$OUT/raw" | wc -l) shots into $OUT"

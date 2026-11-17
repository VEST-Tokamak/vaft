#!/bin/bash
# push_products.sh <merged filedb dir on the cluster> <staging dir on the server>   (run on the workstation)
# Resumable relay copy cluster -> server through this workstation (the two hosts hold no key for each
# other), one shot-tree chunk at a time, with a sha256 manifest verified on arrival.
# The staging dir must be on the same filesystem as /srv/vest.filedb (so the final step is a
# rename), e.g. /srv/vest.filedb.regen-staging. It is created by the operator, not by this script.
set -uo pipefail
SRC=${1:?merged filedb dir on the cluster}; DST=${2:?staging dir on the server}
# Hosts stay out of the repository: ssh destinations come from the environment or from
# ${VAFT_REGEN_ENV:-~/.config/vaft/regen.env}, e.g.
#   VAFT_REGEN_CLUSTER_SSH="-p <port> <user>@<cluster login host>"   VAFT_REGEN_SERVER_SSH="-p <port> <user>@<server>"
# (see site.example.env)
ENV_FILE=${VAFT_REGEN_ENV:-$HOME/.config/vaft/regen.env}
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
SSH_OPTS="-o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30"
SERVER="ssh $SSH_OPTS ${VAFT_REGEN_SERVER_SSH:?set VAFT_REGEN_SERVER_SSH}"
CLUSTER="ssh $SSH_OPTS ${VAFT_REGEN_CLUSTER_SSH:?set VAFT_REGEN_CLUSTER_SSH}"
RROOT=${VAFT_REGEN_REMOTE_ROOT:?set VAFT_REGEN_REMOTE_ROOT}   # the run root on the cluster, relative to its home
WORK=${WORK:-$HOME/.cache/vaft-regen-push/$(printf "%s %s" "$SRC" "$DST" | shasum | cut -c1-12)}; mkdir -p "$WORK"
echo "$SRC -> $DST" > "$WORK/target"
$SERVER "test -d '$DST'" || { echo "create $DST on the server first" >&2; exit 2; }
# Unit = one top-level product directory per shot (raw dumps are NOT pushed: the server has them).
# The unit list is written once: chunk names (and their .ok markers) must keep meaning the same units.
[ -s "$WORK/units.txt" ] || { $CLUSTER "cd '$SRC' && { find . -path ./pipeline -prune -o -path ./raw -prune -o -type d -regextype posix-extended \
        -regex '.*/[0-9]+' -print -prune | sed 's#^./##'; echo omas/static; echo pipeline/efit_tables; \
        echo pipeline/preflight; ls -d raw/[0-9]*/plot 2>/dev/null; } | sort" > "$WORK/units.txt.tmp" \
  && mv "$WORK/units.txt.tmp" "$WORK/units.txt" || exit 3; }
ls "$WORK"/unit.[0-9][0-9][0-9][0-9] >/dev/null 2>&1 || split -l "${CHUNK:-2000}" -d -a 4 "$WORK/units.txt" "$WORK/unit." || exit 3
for f in "$WORK"/unit.[0-9][0-9][0-9][0-9]; do
  c=$(basename "$f"); [ -f "$WORK/$c.ok" ] && continue
  list=$(tr '\n' ' ' < "$f")
  $CLUSTER "cd '$SRC' && find $list -type f -print0 | sort -z | xargs -0r sha256sum" > "$WORK/$c.sha256" || { echo "$c: sha failed"; continue; }
  $CLUSTER "cd '$SRC' && tar -cf - $list" | $SERVER "tar -C '$DST' -xf -"
  rc=("${PIPESTATUS[@]}")
  if [ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ] && $SERVER "cd '$DST' && sha256sum --quiet -c -" < "$WORK/$c.sha256"; then
    touch "$WORK/$c.ok"; echo "$(date '+%F %T') $c ok"
  else
    echo "$(date '+%F %T') $c FAILED (tar rc ${rc[*]})"
  fi
done

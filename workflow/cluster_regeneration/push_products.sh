#!/bin/bash
# push_to_vest.sh <merged filedb dir on tdst> <staging dir on vestserver>   (run on the workstation)
# Resumable relay copy tdst -> vestserver through this workstation (the two hosts hold no key for each
# other), one shot-tree chunk at a time, with a sha256 manifest verified on arrival.
# The staging dir must be on the same filesystem as /srv/vest.filedb (so the final step is a
# rename), e.g. /srv/vest.filedb.regen2147-staging. It is created by the operator, not by this script.
set -uo pipefail
SRC=${1:?merged filedb dir on tdst}; DST=${2:?staging dir on vestserver}
# Hosts stay out of the repository: ssh destinations come from the environment or from
# ${VAFT_REGEN_ENV:-~/.config/vaft/regen.env}, e.g.
#   VAFT_REGEN_TDST_SSH="-p <port> <user>@<cluster head>"   VAFT_REGEN_VEST_SSH="-p <port> <user>@<server>"
ENV_FILE=${VAFT_REGEN_ENV:-$HOME/.config/vaft/regen.env}
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
SSH_OPTS="-o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30"
VEST="ssh $SSH_OPTS ${VAFT_REGEN_VEST_SSH:?set VAFT_REGEN_VEST_SSH}"
TDST="ssh $SSH_OPTS ${VAFT_REGEN_TDST_SSH:?set VAFT_REGEN_TDST_SSH}"
RROOT=${VAFT_REGEN_REMOTE_ROOT:-regen2147}   # the run root on the cluster, relative to its home
WORK=${WORK:-$HOME/regen2147-push}; mkdir -p "$WORK"
$VEST "test -d '$DST'" || { echo "create $DST on vestserver first" >&2; exit 2; }
# Unit = one top-level product directory per shot (raw dumps are NOT pushed: vestserver has them).
$TDST "cd '$SRC' && { find . -path ./pipeline -prune -o -path ./raw -prune -o -type d -regextype posix-extended \
        -regex '.*/[0-9]+' -print -prune | sed 's#^./##'; echo omas/static; echo pipeline/efit_tables; \
        echo pipeline/preflight; ls -d raw/[0-9]*/plot 2>/dev/null; } | sort" > "$WORK/units.txt" || exit 3
split -l "${CHUNK:-2000}" -d -a 4 "$WORK/units.txt" "$WORK/unit."
for f in "$WORK"/unit.[0-9][0-9][0-9][0-9]; do
  c=$(basename "$f"); [ -f "$WORK/$c.ok" ] && continue
  list=$(tr '\n' ' ' < "$f")
  $TDST "cd '$SRC' && find $list -type f -print0 | sort -z | xargs -0 sha256sum" > "$WORK/$c.sha256" || { echo "$c: sha failed"; continue; }
  $TDST "cd '$SRC' && tar -cf - $list" | $VEST "tar -C '$DST' -xf -" && \
  $VEST "cd '$DST' && sha256sum --quiet -c -" < "$WORK/$c.sha256" && touch "$WORK/$c.ok" \
    && echo "$(date '+%F %T') $c ok" || echo "$(date '+%F %T') $c FAILED"
done

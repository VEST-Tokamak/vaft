#!/bin/bash
# copy_raw.sh <local scratch dir>          (run on a workstation that reaches both hosts)
# Resumable relay copy vestserver:/srv/vest.filedb/raw (only read) -> cluster:~/$RROOT/raw through
# this workstation (the two hosts hold no key for each other). Chunks of CHUNK shot dirs; done
# markers on the cluster, so a re-run continues. Measured 2026-10-25: 143 GB at ~37 MB/s.
set -uo pipefail
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
CHUNK=${CHUNK:-500}
$VEST 'ls /srv/vest.filedb/raw | sort -n' > "$1/shotdirs.txt" || exit 3
split -l "$CHUNK" -d -a 3 "$1/shotdirs.txt" "$1/chunk."
T0=$(date +%s)
for f in "$1"/chunk.*; do
  c=$(basename "$f")
  if $TDST "test -f ~/$RROOT/transfer/done/$c"; then continue; fi
  t=$(date +%s)
  $VEST "cd /srv/vest.filedb/raw && tar -cf - $(tr '\n' ' ' < "$f")" \
    | $TDST "mkdir -p ~/$RROOT/transfer/done && tar -C ~/$RROOT/raw -xf - && touch ~/$RROOT/transfer/done/$c"
  rc=("${PIPESTATUS[@]}")
  echo "$(date '+%F %T') $c rc=${rc[*]} $(( $(date +%s) - t ))s"
done
echo "ALL DONE in $(( $(date +%s) - T0 ))s"

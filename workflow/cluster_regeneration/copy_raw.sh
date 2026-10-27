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
# The shot listing is frozen at the first run in this scratch dir: chunk names (and their done
# markers, prefixed RUN_TAG) must keep meaning the same shots. Shots added later: a new scratch
# dir with NEW_SINCE=<last copied shot> and a new RUN_TAG.
if [ ! -s "$1/shotdirs.txt" ]; then
  $VEST "ls /srv/vest.filedb/raw" | grep -E '^[0-9]+$' | sort -n \
    | awk -v n="${NEW_SINCE:-0}" '$1 > n' > "$1/shotdirs.txt.tmp" || exit 3
  split -l "$CHUNK" -d -a 3 "$1/shotdirs.txt.tmp" "$1/chunk." || exit 3
  mv "$1/shotdirs.txt.tmp" "$1/shotdirs.txt"
fi
T0=$(date +%s)
for f in "$1"/chunk.[0-9][0-9][0-9]; do
  c=${RUN_TAG:-}$(basename "$f")
  if $TDST "test -f ~/$RROOT/transfer/done/$c"; then continue; fi
  t=$(date +%s)
  $VEST "cd /srv/vest.filedb/raw && tar -cf - $(tr '\n' ' ' < "$f")" \
    | $TDST "mkdir -p ~/$RROOT/transfer/done && tar -C ~/$RROOT/raw -xf -"
  rc=("${PIPESTATUS[@]}")
  # Done only when both ends succeeded: a sender-side "Cannot stat" still yields a valid archive.
  if [ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ]; then $TDST "touch ~/$RROOT/transfer/done/$c"; fi
  echo "$(date '+%F %T') $c rc=${rc[*]} $(( $(date +%s) - t ))s"
done
echo "ALL DONE in $(( $(date +%s) - T0 ))s"
# Verify: name + size of every dump and manifest on both sides, sha256 of a ~1 % random sample.
$VEST "cd /srv/vest.filedb/raw && find . -name 'vest_*_daq_*' -printf '%P %s\n'" | sort > "$1/vest.lst"
$TDST "cd ~/$RROOT/raw && find . -name 'vest_*_daq_*' -printf '%P %s\n'" | sort > "$1/tdst.lst"
comm -23 "$1/vest.lst" "$1/tdst.lst" > "$1/missing_or_different.lst"
echo "server files $(wc -l < "$1/vest.lst"), cluster files $(wc -l < "$1/tdst.lst"), missing or size-different $(wc -l < "$1/missing_or_different.lst")"
awk 'BEGIN { srand(2147) } rand() < 0.01 { print $1 }' "$1/vest.lst" > "$1/sample.lst"
$VEST "cd /srv/vest.filedb/raw && xargs -r sha256sum" < "$1/sample.lst" > "$1/sample.sha256"
if $TDST "cd ~/$RROOT/raw && sha256sum --quiet -c -" < "$1/sample.sha256"; then
  echo "sha256 sample OK ($(wc -l < "$1/sample.lst") files)"
else
  echo "sha256 sample MISMATCH"
fi

#!/usr/bin/env bash
# Provision per-shot HSDS folders. hsload does not create them, so a shot's
# first replication fails with "Domain ... not found" until this has run.
# Idempotent: hstouch opens with mode='x' and refuses an existing folder.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
SOURCE="${1:?usage: provision_hsds_shots.sh <source> <first> <last> [owner]}"
FIRST="${2:?}" ; LAST="${3:?}" ; OWNER="${4:-admin}"
made=0 ; existed=0 ; failed=0
for shot in $(ls /srv/vest.filedb/archive | awk -v a="$FIRST" -v b="$LAST" '$1>=a && $1<=b' | sort -n); do
  if hstouch -o "$OWNER" "/$SOURCE/$shot/" >/dev/null 2>&1; then made=$((made+1))
  elif python3 -c "
import h5pyd,logging,sys; logging.disable(logging.WARNING)
try: h5pyd.Folder('/$SOURCE/$shot/', mode='r'); sys.exit(0)
except Exception: sys.exit(1)" 2>/dev/null; then existed=$((existed+1))
  else failed=$((failed+1)); echo "  FAILED: /$SOURCE/$shot/" >&2
  fi
done
echo "source=$SOURCE created=$made already_existed=$existed failed=$failed"

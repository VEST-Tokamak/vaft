# Sourced by the regeneration scripts: the site's settings, from a file outside the repository
# (see site.example.env), then -- on the cluster -- the run's pinned environment.
# shellcheck shell=bash
VAFT_REGEN_ENV=${VAFT_REGEN_ENV:-$HOME/.config/vaft/regen.env}
if [ -f "$VAFT_REGEN_ENV" ]; then
  # shellcheck disable=SC1090
  set -a; . "$VAFT_REGEN_ENV"; set +a
fi
regen_cluster_env() {  # the pinned environment of the run root (env.sh, from env.example.sh)
  : "${REGEN_ROOT:?set REGEN_ROOT in $VAFT_REGEN_ENV (see site.example.env)}"
  # shellcheck disable=SC1091
  . "$REGEN_ROOT/env.sh"
}

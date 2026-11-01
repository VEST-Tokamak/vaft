#!/bin/bash
# Build (or re-pin) the #2147 regeneration environment on tdst.
#   PIN=<develop SHA or ref> bash $REGEN_ROOT/setup_env.sh
# One variable re-pins everything: a private ref refs/regen2147/pin in our own clone,
# a detached worktree src-<sha9>, a uv venv env-<sha9> (py3.14 from ~/work/vaft-ci/py314),
# the real EFIT .ddd tables (no git-lfs here), and ~/regen2147/{src,env,env.sh} -> that pin.
# Never touches ~/git/vaft, ~/work/vaft-ci/repo or any other checkout.
set -euo pipefail
ROOT=${REGEN_ROOT:-$HOME/regen2147}
PIN=${PIN:?set PIN=<develop sha or ref>}
PY=${REGEN_PYTHON:-$HOME/work/vaft-ci/py314/bin/python}   # a Python 3.14 with uv beside it
UV=$(dirname "$PY")/uv
REPO=$ROOT/repo
# Versions matched to vestserver conda vaft-ml (the #2191 qualification env); omas is the
# pyproject pin (==0.95.2), not vaft-ml's 0.94.2.
PINS="numpy==2.5.3 scipy==1.18.1 snakemake==9.27.0 snakemake-executor-plugin-cluster-generic"

if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  git clone -q --no-checkout https://github.com/VEST-Tokamak/vaft.git "$REPO"
fi
git -C "$REPO" fetch -q --no-write-fetch-head origin '+refs/heads/develop:refs/remotes/origin/develop'
SHA=$(git -C "$REPO" rev-parse --verify "${PIN}^{commit}")
git -C "$REPO" merge-base --is-ancestor "$SHA" origin/develop \
  || { echo "PIN $PIN ($SHA) is not on develop" >&2; exit 2; }
git -C "$REPO" update-ref refs/regen2147/pin "$SHA"
S9=${SHA:0:9}
SRC=$ROOT/src-$S9 ENV=$ROOT/env-$S9
if [ ! -d "$SRC" ]; then
  git -C "$REPO" worktree add -q --detach "$SRC" refs/regen2147/pin
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$SHA" ] || { echo "worktree $SRC is not at $SHA" >&2; exit 2; }
# Real Green tables: each must hash to the oid its LFS pointer names.
for f in ec129129.ddd ep129129.ddd rfcoil.ddd rv129129.ddd; do
  oid=$(git -C "$SRC" show "HEAD:vaft/data/efit/$f" | sed -n 's/^oid sha256://p')
  [ "$(sha256sum < "$ROOT/efit-ddd/$f" | cut -d' ' -f1)" = "$oid" ] || { echo "table $f != LFS oid $oid" >&2; exit 2; }
  cp --remove-destination "$ROOT/efit-ddd/$f" "$SRC/vaft/data/efit/$f"
done
if [ ! -x "$ENV/bin/python" ]; then
  "$UV" venv -q -p "$PY" "$ENV"
fi
VIRTUAL_ENV=$ENV "$UV" pip install -q --python "$ENV/bin/python" --no-sources -e "$SRC" $PINS
ln -sfn "src-$S9" "$ROOT/src"
ln -sfn "env-$S9" "$ROOT/env"
echo "$SHA" > "$ROOT/PIN_SHA"
# Record the exact environment next to the pin.
"$ENV/bin/python" -m pip --version >/dev/null 2>&1 || true
VIRTUAL_ENV=$ENV "$UV" pip freeze --python "$ENV/bin/python" > "$ROOT/env-$S9.freeze.txt"
( cd /; PYTHONNOUSERSITE=1 "$ENV/bin/python" -c "
import sys, vaft, numpy, scipy, omas, snakemake
assert vaft.__file__.startswith('$SRC/'), vaft.__file__
print('python', sys.version.split()[0], 'vaft', vaft.__file__, 'numpy', numpy.__version__,
      'scipy', scipy.__version__, 'omas', omas.__version__, 'snakemake', snakemake.__version__)" ) | tee "$ROOT/env-$S9.versions.txt"
echo "pinned $SHA -> $SRC, $ENV"

#!/bin/bash
# Usage: .github/scripts/tdst_gate.sh <PR number> [--comment]
#
# The develop gate, run on the lab's Slurm cluster instead of GitHub Actions.
# It repeats what Package CI's `core-test` and `package` jobs do, on GitHub's
# own merge ref of PR <N> (refs/pull/N/merge, the tree Actions tests), and on a
# pass posts the commit status `merge-gate` on the PR head -- the one context
# the develop ruleset requires. Actions posts the same status when its two jobs
# pass, so whichever finishes first unblocks the merge (CONTRIBUTING.md,
# "Two ways to satisfy the develop gate").
#
# A failure posts nothing to `merge-gate`: the context stays "expected" and the
# PR stays blocked until a run passes. The informational `tdst-ci` status
# carries the verdict either way.
#
# Runs on a workstation with `gh` authenticated for VEST-Tokamak/vaft and ssh
# key access to the cluster. Every step (install, tests, build) runs inside the
# Slurm allocation; the login node only fetches and submits.
#
# Configuration (host and account stay out of the repository). Read from the
# environment, after sourcing ${VAFT_TDST_ENV:-~/.config/vaft/tdst-gate.env}:
#   VAFT_TDST_SSH      ssh destination and options, e.g. "-p <port> <user>@<head>"  (required)
#   VAFT_TDST_SBATCH   sbatch options, e.g. "-A <account> -p g1"                     (required)
#   VAFT_TDST_ROOT     remote work root                     (default ~/work/vaft-ci)
#   VAFT_TDST_PYTHON   remote Python 3.14 with `uv`         (default $ROOT/py314/bin/python)
#
# Exit: 0 passed, 1 failed, 2 not mergeable / not a develop PR, 3 infrastructure.
set -uo pipefail

N=${1:?usage: .github/scripts/tdst_gate.sh <PR number> [--comment]}
COMMENT=${2:-}
ENV_FILE=${VAFT_TDST_ENV:-$HOME/.config/vaft/tdst-gate.env}
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
SSH_DEST=${VAFT_TDST_SSH:?set VAFT_TDST_SSH (or write it to $ENV_FILE)}
SBATCH_ARGS=${VAFT_TDST_SBATCH:?set VAFT_TDST_SBATCH (or write it to $ENV_FILE)}
REPO=VEST-Tokamak/vaft

# -o LogLevel=ERROR: the cluster's post-quantum key-exchange warning would
# otherwise land in every captured stderr.
ssh_() { ssh -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=60 $SSH_DEST "$@"; }
die() { echo "PR #$N: CI infrastructure error: $*" >&2; exit 3; }
status() {  # status <context> <state> <description>
  # VAFT_TDST_NO_STATUS=1: report on stdout only (a dry run of the gate).
  [ "${VAFT_TDST_NO_STATUS:-0}" = 1 ] && return 0
  gh api -X POST "repos/$REPO/statuses/$HEAD" -f context="$1" -f state="$2" \
    -f description="${3:0:139}" >/dev/null || echo "could not post status $1=$2" >&2
}

read -r STATE BASE HEAD MERGEABLE <<< "$(gh pr view "$N" --repo "$REPO" \
  --json state,baseRefName,headRefOid,mergeable \
  --jq '"\(.state) \(.baseRefName) \(.headRefOid) \(.mergeable)"')" || die "cannot read PR #$N"
[ "$STATE" = OPEN ] || { echo "PR #$N is $STATE"; exit 2; }
# main keeps its own gate (test, package, tutorial on Actions); this one is develop's.
[ "$BASE" = develop ] || { echo "PR #$N targets $BASE, not develop"; exit 2; }
[ "$MERGEABLE" != CONFLICTING ] || { echo "PR #$N (${HEAD:0:9}): MERGE CONFLICT with develop"; exit 2; }

# Fetch the merge ref and submit. Prints "<job id> <base sha> <run dir>".
# (A function, not a heredoc inside $(...): macOS's bash 3.2 cannot parse that.)
remote_submit() { cat <<'REMOTE'
set -euo pipefail
N=$1 HEAD=$2 SBATCH_ARGS=$3
ROOT=${VAFT_TDST_ROOT:-$HOME/work/vaft-ci}
PY=${VAFT_TDST_PYTHON:-$ROOT/py314/bin/python}
UV=$(dirname "$PY")/uv
CLONE=$ROOT/repo
mkdir -p "$ROOT/runs"

# One clone owned by this gate, never the shared ~/git/vaft. No git-lfs on the
# cluster, so LFS files are pointers -- exactly what Actions' checkout gives
# core-test and package. flock: several sessions may gate PRs at once.
exec 9>"$ROOT/repo.lock"
flock 9
[ -d "$CLONE/.git" ] || git clone -q --no-checkout https://github.com/VEST-Tokamak/vaft.git "$CLONE" >&2
# GitHub recomputes refs/pull/N/merge asynchronously after a push; wait until
# its second parent is the head we were asked to gate.
for _ in 1 2 3 4 5 6; do
  git -C "$CLONE" fetch -q origin "+refs/pull/$N/merge:refs/remotes/pr/$N/merge" >&2
  [ "$(git -C "$CLONE" rev-parse "pr/$N/merge^2")" = "$HEAD" ] && break
  sleep 20
done
[ "$(git -C "$CLONE" rev-parse "pr/$N/merge^2")" = "$HEAD" ] \
  || { echo "refs/pull/$N/merge does not contain head $HEAD" >&2; exit 1; }
MERGE=$(git -C "$CLONE" rev-parse "pr/$N/merge")
BASE_SHA=$(git -C "$CLONE" rev-parse "pr/$N/merge^1")
RUN=$ROOT/runs/pr$N-${HEAD:0:9}-$(date +%Y%m%d%H%M%S)
mkdir -p "$RUN"
git -C "$CLONE" worktree add -q --detach "$RUN/src" "$MERGE" >&2
# Keep the logs of the 30 newest runs; drop older ones entirely (their src
# worktree is removed at job end, `worktree prune` forgets any leftover).
ls -1dt "$ROOT"/runs/*/ 2>/dev/null | tail -n +31 | xargs -r rm -rf
git -C "$CLONE" worktree prune
flock -u 9

cat > "$RUN/job.sh" <<JOB
#!/bin/bash
set -o pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# No MPLBACKEND: core-test runs without one, and test_help asserts the
# backend is still unresolved after import.
export PYTHONNOUSERSITE=1 TMPDIR=\${TMPDIR:-/tmp}/vaft-ci-\$SLURM_JOB_ID
mkdir -p "\$TMPDIR"
RUN=$RUN SRC=$RUN/src
finish() { echo "\$1" > \$RUN/exit; cd /; rm -rf "\$TMPDIR" \$RUN/venv \$RUN/wheel-venv \$RUN/dist; git -C $CLONE worktree remove --force \$SRC; exit 0; }
step() { local name=\$1; shift; echo "== \$name: \$*" >> \$RUN/steps.log; "\$@" > \$RUN/\$name.log 2>&1; local rc=\$?; echo "   \$name rc=\$rc" >> \$RUN/steps.log; return \$rc; }
cd \$SRC

# Environment, as core-test builds it: the dev extra, the CPU torch wheel, onnx.
step venv $UV venv -q -p $PY \$RUN/venv || finish 3
. \$RUN/venv/bin/activate
# --no-sources: Actions installs with pip, which ignores [tool.uv.sources]
# (an imas_core git tag that has to be built from source); match pip.
step install $UV pip install --no-sources -e ".[dev]" build twine || finish 3
step torch $UV pip install "torch>=2.4,<2.12" --index-url https://download.pytorch.org/whl/cpu || finish 3
step onnx $UV pip install onnx skl2onnx || finish 3
step import python -c "import vaft; assert vaft.__file__.startswith('\$SRC/'), vaft.__file__; print(vaft.__file__)" || finish 3

# core-test
FAIL=0
step ruff python -m ruff check --select F821 --output-format concise vaft/formula/ || FAIL=1
step core python -m pytest -q -m "core and not perf" --durations=25 -p no:cacheprovider --basetemp=\$TMPDIR/pytest || FAIL=1

# package
step build python -m build --outdir \$RUN/dist || FAIL=1
step twine twine check \$RUN/dist/* || FAIL=1
step verify_dist python test/verify_dist.py \$RUN/dist --max-wheel-mib 26 || FAIL=1
smoke() {
  $PY -m venv \$RUN/wheel-venv && \$RUN/wheel-venv/bin/pip install -q --no-deps \$RUN/dist/*.whl \
  && cd \$TMPDIR && \$RUN/wheel-venv/bin/python -c "import vaft; print(vaft.__version__)" \
  && \$RUN/wheel-venv/bin/vaft --help >/dev/null && \$RUN/wheel-venv/bin/vaft plot --help >/dev/null \
  && \$RUN/wheel-venv/bin/vaft gui --help >/dev/null
}
step smoke smoke || FAIL=1
cd \$SRC
finish \$FAIL
JOB
chmod +x "$RUN/job.sh"
JOB=$(sbatch --parsable $SBATCH_ARGS -c 4 --mem=16G -t 01:30:00 \
      -J "vaft-ci-$N" -o "$RUN/slurm.out" "$RUN/job.sh")
echo "${JOB%%;*} ${BASE_SHA:0:9} $RUN"
REMOTE
}

OUT=$(remote_submit | ssh_ bash -s -- "$N" "$HEAD" "\"$SBATCH_ARGS\"") \
  || die "cannot fetch the merge ref or submit the job"
read -r JOB BASE_SHA RUN <<< "$(printf '%s\n' "$OUT" | tail -1)"
[ -n "${JOB:-}" ] || die "sbatch returned no job id"
echo "PR #$N (${HEAD:0:9}) on develop $BASE_SHA: Slurm job $JOB, run $RUN" >&2
status tdst-ci pending "Slurm job $JOB: core-test + package on develop $BASE_SHA"

# squeue drops a finished job within seconds and returns empty with rc 0.
while ssh_ "squeue -h -j $JOB 2>/dev/null" | grep -q .; do sleep 60; done

CODE=$(ssh_ "cat $RUN/exit 2>/dev/null")
CORE=$(ssh_ "tail -1 $RUN/core.log 2>/dev/null" | sed 's/^=* *//; s/ *=*$//')
SLURM_STATE=$(ssh_ "sacct -n -X -j $JOB -o State%20 2>/dev/null" | head -1 | tr -d ' ')
FAILED_STEPS=$(ssh_ "grep -B1 'rc=[1-9]' $RUN/steps.log 2>/dev/null | sed -n 's/^== \([a-z_]*\):.*/\1/p'" | paste -sd, -)
case "$CODE" in
  0) VERDICT=PASSED ;;
  1) VERDICT="FAILED ($FAILED_STEPS)" ;;
  3) VERDICT="ERROR (environment step $FAILED_STEPS)" ;;
  *) VERDICT="ERROR (job $SLURM_STATE, exit '$CODE')" ;;
esac
LINE="PR #$N (${HEAD:0:9}) merged onto develop $BASE_SHA: $VERDICT | core: ${CORE:-n/a} | Slurm job $JOB, logs $RUN"
echo "$LINE"
if [ "$CODE" = 0 ]; then
  status tdst-ci success "core-test + package passed (Slurm job $JOB)"
  status merge-gate success "develop gate passed on tdst (Slurm job $JOB)"
else
  status tdst-ci failure "$VERDICT (Slurm job $JOB)"
fi
if [ "$COMMENT" = "--comment" ]; then
  gh pr comment "$N" --repo "$REPO" --body "**develop gate (tdst cluster):** $LINE" >/dev/null \
    || echo "could not post the PR comment" >&2
fi
case "$CODE" in 0) exit 0 ;; 1) exit 1 ;; *) exit 3 ;; esac

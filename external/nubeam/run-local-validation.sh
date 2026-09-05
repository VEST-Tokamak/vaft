#!/usr/bin/env bash
# Run a NUBEAM reference case entirely on this machine and compare the result
# against the reference Plasma State that ships with the NTCC distribution.
#
# The NTCC archive provides two complete cases -- D3D and TFTR -- each as an
# input Plasma State plus a reference output Plasma State. They need no
# external data and no plasma-state generator, which makes them the right way
# to establish that this macOS build computes correct physics before any of it
# is pointed at experimental data.
#
# Usage:
#   ./run-local-validation.sh [--case d3d|tftr] [--nptcls N] [--keep]
#
# The reference cases run nubeam_comp_exec directly through its documented
# environment-variable interface (README section IV) rather than through
# nbx_share.csh, which additionally requires cstate (sglib tek-vek graphics)
# and nubeam_cleanup_files (absent from this distribution).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# Three distinct roots. SCRIPT_DIR is this directory in the VAFT checkout;
# NUBEAM_ROOT is the external NUBEAM source tree, which VAFT does not vendor;
# PREFIX is the installation macos.sh produced inside it, and is what
# $NUBEAMHOME names.
NUBEAM_ROOT="${NUBEAM_SOURCE_DIR:-}"
CASE='d3d'
NPTCLS=''
SEED=''
KEEP=0
REPEAT=''

usage() {
  cat <<'EOF'
Usage: ./run-local-validation.sh [--case d3d|tftr] [--nptcls N] [--keep]

  --nubeam-root PATH  the NUBEAM source tree holding the reference cases
                    (or set NUBEAM_SOURCE_DIR)
  --case d3d|tftr   reference case to run (default: d3d)
  --nptcls N        override the Monte Carlo particle count (minimum 100).
                    The reference namelist uses 20000 and takes a couple of
                    minutes; --nptcls 100 gives a fast smoke pass, but its
                    statistical noise is far too large for the profile
                    comparison to mean anything.
  --seed N          override the namelist RNG seed. Running the same case
                    under two different seeds measures how much the code
                    disagrees with itself, which is the noise floor any
                    comparison against the reference has to be read against.
  --keep            keep an existing work directory instead of recreating it

Environment overrides:
  PREACTDIR   PREACT reaction-table directory (default: ./local/share/preact)
  ADASDIR     ADAS data directory             (default: ./local/share/adas)
EOF
}

die() { printf 'run-local-validation.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

CASE_EDIT="$SCRIPT_DIR/_case_edit.py"
[[ -f "$CASE_EDIT" ]] || die "missing helper: $CASE_EDIT"
# Text edits go through Python, never sed/awk: `sed -i ''` is correct on BSD
# and broken on GNU, and the two spell a whole-line replacement differently.
# See external/nubeam/_case_edit.py.
case_edit() { python3 "$CASE_EDIT" "$@"; }

while (($#)); do
  case "$1" in
    --nubeam-root) (($# >= 2)) || die '--nubeam-root needs a path'; NUBEAM_ROOT="$2"; shift 2 ;;
    --case) (($# >= 2)) || die '--case needs a value'; CASE="$2"; shift 2 ;;
    --nptcls) (($# >= 2)) || die '--nptcls needs a value'; NPTCLS="$2"; shift 2 ;;
    --seed) (($# >= 2)) || die '--seed needs a value'; SEED="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ "$CASE" == d3d || "$CASE" == tftr ]] || die "unknown case: $CASE (expected d3d or tftr)"
# Step count is per-case and must match the reference run, because the fast ion
# slowing-down distribution is still filling in over these steps: run TFTR for
# D3D's two steps instead of its own four and every beam-driven profile lands
# roughly half-built. The values come from the shipped {d3d,tftr}_test.csh.
case "$CASE" in
  d3d) REPEAT='2x0.010' ;;
  tftr) REPEAT='4x0.010' ;;
esac
if [[ -n "$NPTCLS" ]]; then
  [[ "$NPTCLS" =~ ^[0-9]+$ ]] || die "--nptcls needs an integer"
  ((NPTCLS >= 100)) || die 'nubeam_comp_exec rejects nptcls < 100'
fi
if [[ -n "$SEED" ]]; then
  [[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed needs an integer"
fi

[[ -n "$NUBEAM_ROOT" ]] ||
  die "--nubeam-root is required: the reference cases ship with the NUBEAM source tree, which VAFT does not vendor."
NUBEAM_ROOT="$(cd "$NUBEAM_ROOT" 2>/dev/null && pwd -P)" ||
  die "NUBEAM source tree does not exist"
DATA_DIR="$NUBEAM_ROOT/nubeam_comp_exec"
BUILD_DIR="$NUBEAM_ROOT/build/darwin-arm64"
PREFIX="${NUBEAMHOME:-$NUBEAM_ROOT/local}"

NUBEAM_EXEC="$PREFIX/bin/nubeam_comp_exec"
UPDATE_STATE="$PREFIX/bin/update_state"
[[ -x "$NUBEAM_EXEC" ]] || die "not built: $NUBEAM_EXEC (run ./install.sh first)"
[[ -x "$UPDATE_STATE" ]] || die "not built: $UPDATE_STATE (run ./install.sh first)"

export PREACTDIR="${PREACTDIR:-$PREFIX/share/preact}"
export ADASDIR="${ADASDIR:-$PREFIX/share/adas}"
[[ -d "$PREACTDIR/tables" ]] || die "PREACTDIR is not initialized: $PREACTDIR"
[[ -d "$ADASDIR/data" ]] || die "ADASDIR has no data tree: $ADASDIR"
# Both databases are caches: the table code writes newly computed reaction
# tables back into them on first use.
[[ -w "$PREACTDIR" && -w "$ADASDIR/tables" ]] ||
  die "PREACTDIR and ADASDIR/tables must be writable"

INPUT_STATE="${CASE}_input_state.cdf"
REFERENCE_STATE="${CASE}_output_state.cdf"
OUTPUT_STATE="${CASE}_output_state_local.cdf"
for file in "$INPUT_STATE" "$REFERENCE_STATE" nubeam_init_example.dat \
            nubeam_step_example.dat nubeam_init_files.dat nubeam_step_files.dat; do
  [[ -f "$DATA_DIR/$file" ]] || die "missing reference data: $DATA_DIR/$file"
done

WORK_DIR="$BUILD_DIR/validation/$CASE${SEED:+-seed$SEED}"
if ((KEEP)); then
  [[ -d "$WORK_DIR" ]] || die "--keep given but no work directory exists: $WORK_DIR"
else
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"
fi

note "case=$CASE  workdir=$WORK_DIR"
note "PREACTDIR=$PREACTDIR"
note "ADASDIR=$ADASDIR"

cp "$DATA_DIR/$INPUT_STATE" "$WORK_DIR/"
cp "$DATA_DIR/nubeam_init_example.dat" "$WORK_DIR/"
cp "$DATA_DIR/nubeam_step_example.dat" "$WORK_DIR/"

# nubeam_{init,step}_files.dat name the state and namelist files. The shipped
# copies carry the placeholder "my_old_state.cdf"; nbx_share.csh rewrites it,
# and so do we.
sed "s#my_old_state.cdf#$INPUT_STATE#" "$DATA_DIR/nubeam_init_files.dat" \
  > "$WORK_DIR/nubeam_init_files.dat"
sed "s#my_old_state.cdf#$INPUT_STATE#" "$DATA_DIR/nubeam_step_files.dat" \
  > "$WORK_DIR/nubeam_step_files.dat"

if [[ -n "$SEED" ]]; then
  note "overriding RNG seed with nseed=$SEED"
  case_edit set-seed "$SEED" "$WORK_DIR/nubeam_init_example.dat" ||
    die "could not set nseed in the init namelist"
fi

if [[ -n "$NPTCLS" ]]; then
  note "overriding particle counts with nptcls=nptclf=$NPTCLS"
  case_edit set-particles "$NPTCLS" \
    "$WORK_DIR/nubeam_init_example.dat" "$WORK_DIR/nubeam_step_example.dat"
fi

export NUBEAM_WORKPATH="$WORK_DIR"

# init_hold, not init: it holds the RNG seed at the namelist's nseed rather
# than reseeding from the system clock. Without it every run draws a different
# Monte Carlo sample and the comparison below has no fixed meaning.
note "NUBEAM INIT (init_hold; RNG seed held at namelist nseed)"
(
  export NUBEAM_ACTION=init_hold
  export FRANTIC_INIT=50
  unset FRANTIC_ACTION || true
  cd "$WORK_DIR" && "$NUBEAM_EXEC"
) > "$WORK_DIR/init.log" 2> "$WORK_DIR/init.err" || {
  tail -n 60 "$WORK_DIR/init.log" >&2 || true
  tail -n 60 "$WORK_DIR/init.err" >&2 || true
  die "NUBEAM INIT failed; full output in $WORK_DIR/init.{log,err}"
}
grep -Fq 'nubeam INIT completed:  normal exit.' "$WORK_DIR/init.log" ||
  die "NUBEAM INIT did not report normal completion; see $WORK_DIR/init.log"

note "NUBEAM STEP (NUBEAM_REPEAT_COUNT=$REPEAT)"
(
  export NUBEAM_ACTION=step
  export NUBEAM_REPEAT_COUNT="$REPEAT"
  export NUBEAM_POSTPROC=summary_test
  export FRANTIC_ACTION=execute
  unset FRANTIC_INIT || true
  cd "$WORK_DIR" && "$NUBEAM_EXEC"
) > "$WORK_DIR/step.log" 2> "$WORK_DIR/step.err" || {
  tail -n 60 "$WORK_DIR/step.log" >&2 || true
  tail -n 60 "$WORK_DIR/step.err" >&2 || true
  die "NUBEAM STEP failed; full output in $WORK_DIR/step.{log,err}"
}
grep -Fq 'nubeam STEP completed:  normal exit.' "$WORK_DIR/step.log" ||
  die "NUBEAM STEP did not report normal completion; see $WORK_DIR/step.log"

[[ -s "$WORK_DIR/state_changes.cdf" ]] ||
  die "NUBEAM STEP produced no state_changes.cdf"

note "merging NUBEAM output into a full Plasma State"
"$UPDATE_STATE" -input "$WORK_DIR/$INPUT_STATE" \
                -output "$WORK_DIR/$OUTPUT_STATE" \
                -updates "$WORK_DIR/state_changes.cdf" \
  > "$WORK_DIR/update_state.log" 2>&1 || {
  tail -n 40 "$WORK_DIR/update_state.log" >&2 || true
  die "update_state failed; see $WORK_DIR/update_state.log"
}
[[ -s "$WORK_DIR/$OUTPUT_STATE" ]] || die "update_state produced no $OUTPUT_STATE"

cp "$DATA_DIR/$REFERENCE_STATE" "$WORK_DIR/"

interpolation_warnings="$(grep -Ec 'x arguments for interpolation are out of bounds' "$WORK_DIR/step.log" || true)"
{
  printf 'case=%s\n' "$CASE"
  printf 'init=normal\n'
  printf 'step=normal\n'
  printf 'nptcls=%s\n' "${NPTCLS:-namelist default (20000)}"
  printf 'nseed=%s\n' "${SEED:-namelist default}"
  printf 'interpolation_warnings=%s\n' "$interpolation_warnings"
  printf 'output_state=%s\n' "$WORK_DIR/$OUTPUT_STATE"
  printf 'reference_state=%s\n' "$WORK_DIR/$REFERENCE_STATE"
} > "$WORK_DIR/validation-manifest.txt"

note "NUBEAM completed normally"
printf 'RESULT_DIR=%s\n' "$WORK_DIR"
printf 'INTERPOLATION_WARNINGS=%s\n' "$interpolation_warnings"

if [[ -x "$SCRIPT_DIR/compare-plasma-state.py" ]]; then
  note "comparing against the reference Plasma State"
  "$SCRIPT_DIR/compare-plasma-state.py" \
    "$WORK_DIR/$REFERENCE_STATE" "$WORK_DIR/$OUTPUT_STATE" \
    --changes "$WORK_DIR/state_changes.cdf" \
    | tee "$WORK_DIR/comparison.txt"
else
  note "compare-plasma-state.py not found; skipping the profile comparison"
fi

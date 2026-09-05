#!/usr/bin/env bash
# Generate a Plasma State from a G-EQDSK and run NUBEAM on it, entirely on this
# machine. This is the local equivalent of run-server-validation.sh, which does
# the same work on the NANOLAB server.
#
# Usage:
#   ./run-local-vest.sh --input-dir PATH --gfile PATH [--run-name NAME]
#                       [--repeat 1x0.001] [--nptcls N]
#
# The input set is the same one run-server-validation.sh stages, unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# Three distinct roots. SCRIPT_DIR is this directory in the VAFT checkout;
# NUBEAM_ROOT is the external NUBEAM source tree, which VAFT does not vendor;
# PREFIX is the installation macos.sh produced inside it, and is what
# $NUBEAMHOME names.
NUBEAM_ROOT="${NUBEAM_SOURCE_DIR:-}"
INPUT_DIR=''
GFILE=''
RUN_NAME='vest'
REPEAT='1x0.001'
NPTCLS=''

usage() {
  cat <<'EOF'
Usage:
  ./run-local-vest.sh --input-dir PATH --gfile PATH [--run-name NAME]
                      [--repeat COUNTxSTEP] [--nptcls N]

Creates a Plasma State from the supplied G-EQDSK with plasma_state_test, then
runs NUBEAM INIT and STEP against it. Everything runs locally; no server.

  --nubeam-root PATH  the NUBEAM installation root (or set NUBEAM_SOURCE_DIR
                    or NUBEAMHOME)
  --input-dir PATH  directory holding inputf, profiles, mdescr_*.dat,
                    sconfig_*.dat and the four nubeam_*.dat files
  --gfile PATH      G-EQDSK equilibrium
  --run-name NAME   work directory name under build/darwin-arm64/ (default: vest)
  --repeat CxS      NUBEAM_REPEAT_COUNT (default: 1x0.001)
  --nptcls N        override nptcls/nptclf in the init namelist (minimum 100)
EOF
}

die() { printf 'run-local-vest.sh: %s\n' "$*" >&2; exit 1; }
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
    --input-dir) (($# >= 2)) || die '--input-dir needs a path'; INPUT_DIR="$2"; shift 2 ;;
    --gfile) (($# >= 2)) || die '--gfile needs a path'; GFILE="$2"; shift 2 ;;
    --run-name) (($# >= 2)) || die '--run-name needs a name'; RUN_NAME="$2"; shift 2 ;;
    --repeat) (($# >= 2)) || die '--repeat needs a value'; REPEAT="$2"; shift 2 ;;
    --nptcls) (($# >= 2)) || die '--nptcls needs a value'; NPTCLS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$INPUT_DIR" && -n "$GFILE" ]] || { usage >&2; exit 1; }
[[ -d "$INPUT_DIR" ]] || die "input directory does not exist: $INPUT_DIR"
[[ -f "$GFILE" ]] || die "G-EQDSK file does not exist: $GFILE"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
  die 'run name may contain only letters, digits, dot, underscore and hyphen'
if [[ -n "$NPTCLS" ]]; then
  [[ "$NPTCLS" =~ ^[0-9]+$ ]] || die '--nptcls needs an integer'
  ((NPTCLS >= 100)) || die 'nubeam_comp_exec rejects nptcls < 100'
fi

if [[ -z "${NUBEAMHOME:-}" ]]; then
  [[ -n "$NUBEAM_ROOT" ]] ||
    die "set \$NUBEAMHOME, or pass --nubeam-root pointing at the NUBEAM source tree."
  NUBEAM_ROOT="$(cd "$NUBEAM_ROOT" 2>/dev/null && pwd -P)" ||
    die "NUBEAM source tree does not exist"
fi
PREFIX="${NUBEAMHOME:-$NUBEAM_ROOT/local}"
BUILD_DIR="${NUBEAM_WORK_ROOT:-$PREFIX/run}"

GENERATOR="$PREFIX/bin/plasma_state_test"
NUBEAM_EXEC="$PREFIX/bin/nubeam_comp_exec"
[[ -x "$GENERATOR" ]] || die "not built: $GENERATOR (run ./install.sh first)"
[[ -x "$NUBEAM_EXEC" ]] || die "not built: $NUBEAM_EXEC (run ./install.sh first)"

export PREACTDIR="${PREACTDIR:-$PREFIX/share/preact}"
export ADASDIR="${ADASDIR:-$PREFIX/share/adas}"
[[ -d "$PREACTDIR/tables" ]] || die "PREACTDIR is not initialized: $PREACTDIR"
[[ -d "$ADASDIR/data" ]] || die "ADASDIR has no data tree: $ADASDIR"

required=(inputf profiles nubeam_init.dat nubeam_init_files.dat
          nubeam_step.dat nubeam_step_files.dat)
for file in "${required[@]}"; do
  [[ -f "$INPUT_DIR/$file" ]] || die "missing input: $INPUT_DIR/$file"
done
shopt -s nullglob
descriptors=("$INPUT_DIR"/mdescr_*.dat "$INPUT_DIR"/sconfig_*.dat)
shopt -u nullglob
((${#descriptors[@]} >= 2)) ||
  die "no mdescr_*.dat / sconfig_*.dat machine description in $INPUT_DIR"

WORK_DIR="$BUILD_DIR/$RUN_NAME"
# nubeam_comp_exec holds file paths in fixed-length CHARACTER buffers and
# silently truncates past roughly 140 characters, which shows up as an
# unrelated-looking "file open failure" on the input state. Fail here with the
# real reason instead.
((${#WORK_DIR} < 100)) ||
  die "work directory path is too long for NUBEAM's fixed-length path buffers (${#WORK_DIR} chars): $WORK_DIR"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
note "workdir=$WORK_DIR"

for file in "${required[@]}"; do cp "$INPUT_DIR/$file" "$WORK_DIR/"; done
for file in "${descriptors[@]}"; do cp "$file" "$WORK_DIR/"; done
cp "$GFILE" "$WORK_DIR/equilibrium.gfile"

# plasma_state_test reads the equilibrium filename from line 2 of inputf.
case_edit set-equilibrium "$WORK_DIR/inputf" equilibrium.gfile

if [[ -n "$NPTCLS" ]]; then
  note "overriding particle counts with nptcls=nptclf=$NPTCLS"
  case_edit set-particles "$NPTCLS" "$WORK_DIR/nubeam_init.dat"
fi

note "creating the Plasma State from $(basename "$GFILE")"
( cd "$WORK_DIR" && "$GENERATOR" ) \
  > "$WORK_DIR/plasma_state_test.log" 2>&1 || {
  tail -n 40 "$WORK_DIR/plasma_state_test.log" >&2 || true
  die "plasma_state_test failed; see $WORK_DIR/plasma_state_test.log"
}
# The generator reports its own errors on stdout and still exits 0, so the
# output file is the only reliable success signal.
STATE_FILE="$(case_edit read-field "$WORK_DIR/inputf" state)"
[[ -s "$WORK_DIR/$STATE_FILE" ]] ||
  die "the generator did not create $STATE_FILE; see $WORK_DIR/plasma_state_test.log"
note "generated $STATE_FILE ($(wc -c < "$WORK_DIR/$STATE_FILE" | tr -d ' ') bytes)"

export NUBEAM_WORKPATH="$WORK_DIR"
export NUBEAM_REPEAT_COUNT="$REPEAT"

note "NUBEAM INIT"
(
  export NUBEAM_ACTION=init_hold
  export FRANTIC_INIT=50
  unset FRANTIC_ACTION || true
  cd "$WORK_DIR" && "$NUBEAM_EXEC"
) > "$WORK_DIR/init.log" 2> "$WORK_DIR/init.err" || {
  tail -n 40 "$WORK_DIR/init.log" >&2 || true
  tail -n 20 "$WORK_DIR/init.err" >&2 || true
  die "NUBEAM INIT failed; see $WORK_DIR/init.{log,err}"
}
grep -Fq 'nubeam INIT completed:  normal exit.' "$WORK_DIR/init.log" ||
  die "NUBEAM INIT did not report normal completion; see $WORK_DIR/init.log"

note "NUBEAM STEP (NUBEAM_REPEAT_COUNT=$REPEAT)"
(
  export NUBEAM_ACTION=step
  export NUBEAM_POSTPROC=summary_test
  export FRANTIC_ACTION=execute
  unset FRANTIC_INIT || true
  cd "$WORK_DIR" && "$NUBEAM_EXEC"
) > "$WORK_DIR/step.log" 2> "$WORK_DIR/step.err" || {
  tail -n 40 "$WORK_DIR/step.log" >&2 || true
  tail -n 20 "$WORK_DIR/step.err" >&2 || true
  die "NUBEAM STEP failed; see $WORK_DIR/step.{log,err}"
}
grep -Fq 'nubeam STEP completed:  normal exit.' "$WORK_DIR/step.log" ||
  die "NUBEAM STEP did not report normal completion; see $WORK_DIR/step.log"

interpolation_warnings="$(grep -Ec 'x arguments for interpolation are out of bounds' "$WORK_DIR/step.log" || true)"
{
  printf 'gfile=%s\n' "$GFILE"
  printf 'input_dir=%s\n' "$INPUT_DIR"
  printf 'generator=normal\n'
  printf 'init=normal\n'
  printf 'step=normal\n'
  printf 'repeat=%s\n' "$REPEAT"
  printf 'nptcls=%s\n' "${NPTCLS:-namelist default}"
  printf 'plasma_state=%s\n' "$WORK_DIR/$STATE_FILE"
  printf 'interpolation_warnings=%s\n' "$interpolation_warnings"
} > "$WORK_DIR/vest-manifest.txt"

note "completed normally"
printf 'RESULT_DIR=%s\n' "$WORK_DIR"
printf 'PLASMA_STATE=%s\n' "$WORK_DIR/$STATE_FILE"
printf 'INTERPOLATION_WARNINGS=%s\n' "$interpolation_warnings"

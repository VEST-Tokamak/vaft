#!/usr/bin/env bash
# Build the GACODE suite (NEO first) natively on Apple Silicon macOS.
#
# Usage:
#   bash external/gacode/macos.sh --gacode-root PATH [--codes neo,tglf] [--check]
#
# VAFT does not vendor the GACODE source. This script owns the reproducible
# build recipe and operates on a GACODE tree you already hold, named by
# --gacode-root. Unlike NUBEAM, GACODE builds in place: there is no separate
# installation prefix, so $GACODEHOME points at the checkout itself.
#
# GACODE's own build contract is $GACODE_ROOT plus $GACODE_PLATFORM, which
# selects platform/build/make.inc.$GACODE_PLATFORM. VAFT does not redefine
# either -- it sets both from $GACODEHOME rather than replacing them, so a tree
# built here stays usable from a plain shell with shared/bin/gacode_setup.
#
# macOS/Apple Silicon only. Linux and Windows are not covered here.

set -euo pipefail
IFS=$'\n\t'

GACODE_SOURCE="${GACODE_SOURCE_DIR:-}"
CODES="neo"
RUN_CHECK=0

usage() {
  cat <<'EOF'
Usage: bash external/gacode/macos.sh --gacode-root PATH [--codes neo,tglf] [--check]

  --gacode-root PATH   the GACODE source tree to build (or set GACODE_SOURCE_DIR)
  --codes LIST         comma-separated suite members to build; default "neo"
  --check              after building, run the NEO reg18 regression case

Environment overrides:
  GACODE_SOURCE_DIR=/absolute/path   default for --gacode-root
  GACODE_PLATFORM=NAME               default GFORTRAN_OSX_BREW

The build happens in place. Afterwards, export:

  export GACODEHOME=<the tree you passed>

which is what vaft.code.gacode reads. Nothing is written into the VAFT checkout.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --gacode-root) GACODE_SOURCE="${2:-}" ; shift 2 ;;
    --codes)       CODES="${2:-}"         ; shift 2 ;;
    --check)       RUN_CHECK=1            ; shift ;;
    -h|--help)     usage ; exit 0 ;;
    *) echo "unknown argument: $1" >&2 ; usage >&2 ; exit 2 ;;
  esac
done

if [ -z "$GACODE_SOURCE" ]; then
  echo "error: --gacode-root is required (or set GACODE_SOURCE_DIR)" >&2
  usage >&2
  exit 2
fi

GACODE_SOURCE="$(cd "$GACODE_SOURCE" && pwd -P)"

for marker in Makefile shared/bin/gacode_setup platform/build neo/src; do
  if [ ! -e "$GACODE_SOURCE/$marker" ]; then
    echo "error: $GACODE_SOURCE is missing $marker, so it is not a GACODE tree" >&2
    exit 1
  fi
done

if ! command -v brew >/dev/null 2>&1; then
  echo "error: Homebrew is required. See https://brew.sh" >&2
  exit 1
fi

# gcc supplies gfortran; open-mpi supplies the mpif90 wrapper the makefiles call
# unconditionally, even for the serial build. fftw and netcdf are linked by the
# suite makefiles whether or not NEO itself uses them.
MISSING=()
for formula in gcc open-mpi netcdf netcdf-fortran fftw; do
  brew --prefix "$formula" >/dev/null 2>&1 || MISSING+=("$formula")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "Installing missing dependencies: ${MISSING[*]}"
  brew install "${MISSING[@]}"
fi

export GACODE_ROOT="$GACODE_SOURCE"
export GACODE_PLATFORM="${GACODE_PLATFORM:-GFORTRAN_OSX_BREW}"
export FFTW_INC="$(brew --prefix fftw)/include"
export BREW_LIB="$(brew --prefix)/lib"
export PATH="$GACODE_ROOT/shared/bin:$PATH"

MAKE_INC="$GACODE_ROOT/platform/build/make.inc.$GACODE_PLATFORM"
if [ ! -f "$MAKE_INC" ]; then
  echo "error: no platform file $MAKE_INC" >&2
  echo "Available platforms:" >&2
  ls "$GACODE_ROOT/platform/build" | sed 's/^make\.inc\./  /' >&2
  exit 1
fi

echo "GACODE_ROOT     = $GACODE_ROOT"
echo "GACODE_PLATFORM = $GACODE_PLATFORM"

# Order matters: the per-code makefiles link shared/*/*.a and f2py/*/*.a as
# EXTRA_LIBS, so both must exist before any suite member is built.
echo "==> shared libraries"
make -C "$GACODE_ROOT/shared"
echo "==> f2py libraries (expro, geo)"
make -C "$GACODE_ROOT/f2py"

IFS=',' read -r -a CODE_LIST <<< "$CODES"
for code in "${CODE_LIST[@]}"; do
  if [ ! -d "$GACODE_ROOT/$code" ]; then
    echo "error: no suite member '$code' in $GACODE_ROOT" >&2
    exit 1
  fi
  echo "==> $code"
  make -C "$GACODE_ROOT/$code"
done

echo
echo "Build complete. Export:"
echo
echo "  export GACODEHOME=$GACODE_ROOT"
echo

if [ "$RUN_CHECK" -eq 1 ]; then
  echo "==> NEO reg18 regression"
  # The neo launcher shells out to neo_parse.py, which imports gacodeinput from
  # f2py/pygacode. Without it on PYTHONPATH the parse step fails silently and
  # NEO then aborts on a missing input.neo.gen -- see install/check_gacode.py.
  export PYTHONPATH="$GACODE_ROOT/f2py:$GACODE_ROOT/f2py/pygacode:${PYTHONPATH:-}"
  export PATH="$GACODE_ROOT/neo/bin:$PATH"
  SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/vaft-gacode-reg18.XXXXXX")"
  trap 'rm -rf "$SCRATCH"' EXIT
  cp -R "$GACODE_ROOT/neo/tools/input/reg18" "$SCRATCH/reg18"
  EXPECTED="$(tr -d '[:space:]' < "$SCRATCH/reg18/out.neo.prec")"
  rm -f "$SCRATCH/reg18/out.neo.prec"
  ( cd "$SCRATCH" && neo -e reg18 -n 1 >/dev/null )
  ACTUAL="$(tr -d '[:space:]' < "$SCRATCH/reg18/out.neo.prec")"
  if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "reg18 PASS: $ACTUAL"
  else
    echo "reg18 FAIL: got $ACTUAL, expected $EXPECTED" >&2
    exit 1
  fi
fi

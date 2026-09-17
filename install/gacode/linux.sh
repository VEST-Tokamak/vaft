#!/usr/bin/env bash
# Build the GACODE suite (NEO first) on Linux.
#
# Usage:
#   bash install/gacode/linux.sh --gacode-root PATH [--codes neo,tglf] [--check]
#
# VAFT does not vendor the GACODE source. This script owns the reproducible
# build recipe and operates on a GACODE tree you already hold, named by
# --gacode-root. Unlike NUBEAM, GACODE builds in place: there is no separate
# installation prefix, so $GACODEHOME points at the checkout itself and the
# executable is <code>/bin/<code>, not bin/<code>.
#
# GACODE's own build contract is $GACODE_ROOT plus $GACODE_PLATFORM, which
# selects platform/build/make.inc.$GACODE_PLATFORM at build time and
# platform/exec/exec.$GACODE_PLATFORM at run time. VAFT does not redefine
# either -- it sets both from $GACODEHOME rather than replacing them, so a tree
# built here stays usable from a plain shell with shared/bin/gacode_setup.
#
# Linux only; the macOS recipe is install/gacode/macos.sh.
#
# On the platform tag: upstream ships ~90 of them and every one is named for a
# site or a distribution, so there is no generic "Linux + gfortran" entry to
# pick. TUMBLEWEED is the default here because its settings are the ones a
# stock Linux box satisfies -- mpifort, -fallow-argument-mismatch (which
# gfortran 10+ requires), and system lapack/blas/fftw rather than a hand-built
# OpenBLAS at somebody's home directory, which is what rules out MINT. Override
# with --platform if your site has its own. Both the build file and the exec
# file must exist for the tag, because a tag that only builds fails later,
# inside a shell script, without naming itself.
#
# Two differences from macos.sh, both deliberate:
#
#   * It installs nothing. macos.sh runs `brew install` for what is missing;
#     the apt equivalent needs root, and a compiler is the operator's decision.
#     This script names the package and stops.
#   * TUMBLEWEED compiles with -march=native, so the binaries are tuned to the
#     machine that built them. That is right for a local build and wrong for
#     one you intend to copy to a different CPU; use --platform if you need
#     portable objects.
set -euo pipefail
IFS=$'\n\t'

GACODE_SOURCE="${GACODE_SOURCE_DIR:-}"
# Both, not just neo: install/check_gacode.py:45 requires neo and tglf, and
# vaft/code/gacode resolves both, so a neo-only tree fails its own verification.
# macos.sh still defaults to neo alone, which predates TGLF support (#553).
CODES="neo,tglf"
PLATFORM_TAG="${GACODE_PLATFORM:-TUMBLEWEED}"
RUN_CHECK=0

usage() {
  cat <<'EOF'
Usage: bash install/gacode/linux.sh --gacode-root PATH [options]

  --gacode-root PATH   the GACODE tree to build (or set GACODE_SOURCE_DIR)
  --codes LIST         comma-separated suite members (default: neo,tglf -- both,
                       because install/check_gacode.py requires both)
  --platform TAG       GACODE_PLATFORM (default: TUMBLEWEED, or $GACODE_PLATFORM)
  --check              run NEO's reg18 regression after building
  -h, --help

GACODE builds in place, so $GACODEHOME is the checkout itself. This script
prints the export lines; it edits no shell profile.
EOF
}

die() { printf 'linux.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --gacode-root) (($# >= 2)) || die '--gacode-root needs a path'; GACODE_SOURCE="$2"; shift 2 ;;
    --codes) (($# >= 2)) || die '--codes needs a list'; CODES="$2"; shift 2 ;;
    --platform) (($# >= 2)) || die '--platform needs a tag'; PLATFORM_TAG="$2"; shift 2 ;;
    --check) RUN_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$GACODE_SOURCE" ]] || { usage >&2; die "--gacode-root is required (or set GACODE_SOURCE_DIR)"; }
GACODE_SOURCE="$(cd "$GACODE_SOURCE" 2>/dev/null && pwd -P)" || die "GACODE tree does not exist: $GACODE_SOURCE"
for marker in Makefile shared/bin/gacode_setup platform/build neo/src; do
  [[ -e "$GACODE_SOURCE/$marker" ]] || die "$GACODE_SOURCE is missing $marker, so it is not a GACODE tree"
done

# --- toolchain, named rather than installed -----------------------------------
# gfortran compiles it; mpifort is called by the suite makefiles unconditionally,
# even for a serial build. lapack/blas/fftw and netCDF are linked by the suite
# makefiles whether or not the member you asked for uses them.
require() {
  command -v "$1" >/dev/null || die "$1 is required (e.g. apt install $2)"
}
require gfortran gfortran
require mpifort "openmpi-bin libopenmpi-dev"
require make make
MISSING_LIBS=()
# Captured once and matched from a here-string. `ldconfig -p | grep -q` under
# pipefail reports an installed library as missing: grep exits on its first
# match, ldconfig takes SIGPIPE on a listing larger than the pipe buffer (a
# desktop or a login node, not a minimal container), and the pipeline's 141
# runs the `||` branch.
LDCONFIG_CACHE="$(ldconfig -p 2>/dev/null || true)"
for probe in liblapack.so:liblapack-dev libblas.so:libblas-dev \
             libfftw3.so:libfftw3-dev libnetcdff.so:libnetcdff-dev; do
  library="${probe%%:*}"; package="${probe##*:}"
  # ldconfig knows the real search path, which is multiarch-dependent.
  grep -q "^\s*${library}" <<<"$LDCONFIG_CACHE" || MISSING_LIBS+=("$package")
done
if ((${#MISSING_LIBS[@]})); then
  die "missing libraries. Install them yourself: apt install ${MISSING_LIBS[*]}"
fi

export GACODE_ROOT="$GACODE_SOURCE"
export GACODE_PLATFORM="$PLATFORM_TAG"
# TUMBLEWEED reads both, and an unset FFTW_INC silently becomes `-I` with no
# argument, which swallows the next flag instead of failing.
export FFTW_INC="${FFTW_INC:-/usr/include}"
export NETCDF_FORTRAN_INC="${NETCDF_FORTRAN_INC:-/usr/include}"
export PATH="$GACODE_ROOT/shared/bin:$PATH"

MAKE_INC="$GACODE_ROOT/platform/build/make.inc.$GACODE_PLATFORM"
EXEC_INC="$GACODE_ROOT/platform/exec/exec.$GACODE_PLATFORM"
if [[ ! -f "$MAKE_INC" ]]; then
  printf 'no platform build file %s\nAvailable platforms:\n' "$MAKE_INC" >&2
  ls "$GACODE_ROOT/platform/build" | sed 's/^make\.inc\./  /' >&2
  exit 1
fi
# The launcher execs this at run time. A tag that builds but cannot run fails
# deep inside a shell script without naming the variable, so check it here.
[[ -f "$EXEC_INC" ]] || die "platform $GACODE_PLATFORM has no exec file at $EXEC_INC, so a built code could not be launched. Pick a tag that has both."

note "GACODE_ROOT     = $GACODE_ROOT"
note "GACODE_PLATFORM = $GACODE_PLATFORM"
note "gfortran        = $(gfortran --version | head -1)"

# --- build --------------------------------------------------------------------
# Order matters: the per-code makefiles link shared/*/*.a and f2py/*/*.a as
# EXTRA_LIBS, so both must exist before any suite member is built.
note "shared libraries"
make -C "$GACODE_ROOT/shared"
note "f2py libraries (expro, geo)"
make -C "$GACODE_ROOT/f2py"

IFS=',' read -r -a CODE_LIST <<< "$CODES"
for code in "${CODE_LIST[@]}"; do
  [[ -d "$GACODE_ROOT/$code" ]] || die "no suite member '$code' in $GACODE_ROOT"
  note "$code"
  make -C "$GACODE_ROOT/$code"
  # Assert on src/<code>, the compiled ELF, not on bin/<code>. The latter is the
  # shell launcher VAFT resolves, and it is committed -- it exists in a fresh
  # clone, so checking it would pass on a build that produced nothing.
  [[ -s "$GACODE_ROOT/$code/src/$code" ]] || die "$code did not compile: no binary at $GACODE_ROOT/$code/src/$code"
  [[ -x "$GACODE_ROOT/$code/bin/$code" ]] || die "$code has no launcher at $GACODE_ROOT/$code/bin/$code, which is what VAFT resolves"
done

printf '\nBuild complete. Export:\n'
printf '  export GACODEHOME=%s\n' "$GACODE_ROOT"
printf '  export GACODE_PLATFORM=%s\n' "$GACODE_PLATFORM"
printf '\n'

# --- acceptance ---------------------------------------------------------------
if ((RUN_CHECK)); then
  note "NEO reg18 regression"
  # The neo launcher shells out to neo_parse.py, which imports gacodeinput from
  # f2py/pygacode. Without it on PYTHONPATH the parse step fails silently and
  # NEO then aborts on a missing input.neo.gen -- see install/check_gacode.py.
  export PYTHONPATH="$GACODE_ROOT/f2py:$GACODE_ROOT/f2py/pygacode:${PYTHONPATH:-}"
  export PATH="$GACODE_ROOT/neo/bin:$PATH"
  SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/vaft-gacode-reg18.XXXXXX")"
  trap 'rm -rf "$SCRATCH"' EXIT
  [[ -d "$GACODE_ROOT/neo/tools/input/reg18" ]] || die "reg18 is not in this checkout, so there is nothing to check against"
  cp -R "$GACODE_ROOT/neo/tools/input/reg18" "$SCRATCH/reg18"
  EXPECTED="$(tr -d '[:space:]' < "$SCRATCH/reg18/out.neo.prec")"
  rm -f "$SCRATCH/reg18/out.neo.prec"
  ( cd "$SCRATCH" && neo -e reg18 -n 1 >/dev/null )
  ACTUAL="$(tr -d '[:space:]' < "$SCRATCH/reg18/out.neo.prec")"
  if [[ "$ACTUAL" == "$EXPECTED" ]]; then
    printf 'reg18 PASS: %s\n' "$ACTUAL"
  else
    printf 'reg18 FAIL: got %s, expected %s\n' "$ACTUAL" "$EXPECTED" >&2
    exit 1
  fi
fi

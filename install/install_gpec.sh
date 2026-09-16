#!/usr/bin/env bash
# Build and install the DCON/GPEC suite from a source tree you already hold.
#
# Usage:
#   bash install/install_gpec.sh --source PATH [options]
#   bash install/install_gpec.sh --source PATH --check-only
#   bash install/install_gpec.sh --source PATH --uninstall
#
# GPEC is public (https://github.com/PrincetonUniversity/GPEC), but VAFT neither
# bundles nor fetches it: which revision was built is a statement the operator
# makes, not something a script infers. Obtain the tree yourself and pass its
# path. This script never clones, fetches, pulls, checks out or changes a
# revision of the tree it is given.
#
# What it does: resolve the toolchain explicitly, show upstream's own `make v`
# summary, build the six executables VAFT drives -- dcon, match, rdcon, rmatch,
# stride, gpec -- install them into a prefix that $GPECHOME points at, run
# install/check_gpec.py as the build's acceptance, and write
# vaft-external-install.json there recording the source revision (and, if you
# insisted on building a dirty tree, the diff digest), the make command, the
# compiler, the dependency providers and the executables' checksums.
#
# Linux (system gfortran, distro netCDF and LAPACK) is the verified platform.
# Nothing is installed system-wide.
#
# Four things that differ from install_gpec_windows.ps1, all deliberate:
#
#   * OpenMP is ON here. The Windows build disables it only because LSODE and
#     ZVODE mark a COMMON block threadprivate, which gfortran cannot express in
#     PE object format. ELF has no such limitation, so do not "harmonise" this
#     with the Windows script -- you would be making the Linux build serial for
#     a reason that does not apply to it.
#   * No local HDF5/netCDF build. -BuildDependencies exists on Windows to dodge
#     an AWS S3 SDK shutdown deadlock in the MSYS2 packages. A distro
#     libnetcdff-dev does not carry that SDK.
#   * gfortran 10+ makes argument mismatches fatal, so -fallow-argument-mismatch
#     is mandatory. Without it dcon_interface.mod is never produced and every
#     dependent package cascades into errors that name the wrong thing.
#   * xdraw is skipped. It is an X11 viewer no VAFT workflow uses, and building
#     it would drag libX11-dev into the requirements of a computational install.
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MANIFEST_NAME="vaft-external-install.json"

#: The executables install/check_gpec.py looks for, in its order.
PROGRAMS=(dcon match rdcon rmatch stride gpec)
#: `all` minus `v` (a no-op report) and `xdraw` (X11).
TARGETS=(neededdeps equil lsode zlange zvode orbit vacuum pentrc dcon match
         rdcon rmatch multi sum slayer coil gpec stride)

SOURCE="${GPEC_SOURCE_DIR:-}"
PREFIX=""
LAPACK_HOME=""
NETCDF_F_HOME=""
NETCDF_INC=""
ALLOW_DIRTY=0
JOBS=""
SKIP_TESTS=0
CHECK_ONLY=0
UNINSTALL=0

usage() {
  cat <<'EOF'
Usage: bash install/install_gpec.sh --source PATH [options]

  --source PATH                the GPEC source tree to build (or set GPEC_SOURCE_DIR)
  --prefix PATH                install here (default: <source>/vaft-install)
  --lapack-home PATH           LAPACKHOME for upstream's makefile (default: /usr)
  --netcdf-fortran-home PATH   directory holding libnetcdff (default: derived from
                               an nf-config whose --fc matches the Fortran compiler)
  --netcdf-include PATH        netCDF include directory (default: derived likewise)
  --allow-dirty                build a tree with uncommitted tracked changes; the diff
                               digest and file list are then recorded in the manifest
  --jobs N                     parallel build jobs (default: all cores)
  --skip-tests                 do not run install/check_gpec.py after building
  --check-only                 run install/check_gpec.py and change nothing
  --uninstall                  remove the prefix this script created
  -h, --help

The prefix is what $GPECHOME should point at: VAFT resolves $GPECHOME/bin/dcon and
its five siblings. This script prints the export line; it edits no shell profile.

GPEC builds in place, so its objects and its own bin/ land inside <source> and stay
there. --uninstall removes only the prefix; run `make clean` in <source>/install
yourself if you want the objects gone, because that tree is yours, not this script's.
EOF
}

die() { printf 'install_gpec.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --source) (($# >= 2)) || die '--source needs a path'; SOURCE="$2"; shift 2 ;;
    --prefix) (($# >= 2)) || die '--prefix needs a path'; PREFIX="$2"; shift 2 ;;
    --lapack-home) (($# >= 2)) || die '--lapack-home needs a path'; LAPACK_HOME="$2"; shift 2 ;;
    --netcdf-fortran-home) (($# >= 2)) || die '--netcdf-fortran-home needs a path'; NETCDF_F_HOME="$2"; shift 2 ;;
    --netcdf-include) (($# >= 2)) || die '--netcdf-include needs a path'; NETCDF_INC="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --jobs) (($# >= 2)) || die '--jobs needs a number'; JOBS="$2"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$SOURCE" ]] || die "--source is required: VAFT does not vendor GPEC. Clone it from https://github.com/PrincetonUniversity/GPEC and pass its path, or set GPEC_SOURCE_DIR."
SOURCE="$(cd "$SOURCE" 2>/dev/null && pwd -P)" || die "GPEC source tree does not exist: $SOURCE"
# The same markers install/check_gpec.py validates, so the two agree on what a
# GPEC checkout is.
for marker in install/makefile install/DEFAULTS.inc install/TARGETS.inc dcon gpec; do
  [[ -e "$SOURCE/$marker" ]] || die "not a GPEC source tree (missing $marker): $SOURCE"
done

case "$(uname -s)" in
  Darwin) PLATFORM="darwin-$(uname -m)" ;;
  Linux) PLATFORM="linux-$(uname -m)" ;;
  *) die "unsupported platform: $(uname -s)" ;;
esac
[[ -n "$PREFIX" ]] || PREFIX="$SOURCE/vaft-install"
MANIFEST="$PREFIX/$MANIFEST_NAME"
BUILD_DIR="$SOURCE/install"

# The prefix is removed wholesale by --uninstall, so it must never be inside the
# VAFT checkout. The PowerShell installers enforce this through
# Resolve-InstallPrefix; the POSIX ones did not. Made absolute first, or a
# relative --prefix would slip past the comparison and past the manifest.
[[ "$PREFIX" == /* ]] || PREFIX="$PWD/$PREFIX"
VAFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
case "$PREFIX/" in
  "$VAFT_ROOT"/*) die "the install prefix must be outside the VAFT checkout, because --uninstall removes it: $PREFIX is inside $VAFT_ROOT" ;;
esac

PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "$PYTHON" ]] || die "python3 is required (the VAFT environment provides it)"

if ((CHECK_ONLY)); then
  # With the home variable set, so --check-only reports what the
  # post-install verification reports. Without it the discovery layer
  # asks VAFT to resolve $GPECHOME from the ambient environment and fails
  # on a perfectly good installation.
  exec env GPECHOME="$PREFIX" "$PYTHON" "$SCRIPT_DIR/check_gpec.py" --source "$SOURCE" --prefix "$PREFIX"
fi

if ((UNINSTALL)); then
  [[ -f "$MANIFEST" ]] || die "no $MANIFEST_NAME under $PREFIX; nothing this script installed is there to remove"
  note "removing prefix $PREFIX"
  rm -rf "$PREFIX"
  note "left alone: objects and bin/ inside $SOURCE, which are yours. Run 'make clean' in $BUILD_DIR to remove them."
  exit 0
fi

command -v gfortran >/dev/null || die "gfortran is required (e.g. apt install gfortran)"
command -v gcc >/dev/null || die "gcc is required (e.g. apt install gcc)"
command -v make >/dev/null || die "GNU make is required (e.g. apt install make)"
command -v git >/dev/null || die "git is required"

FC_PATH="$(command -v gfortran)"
FC_VERSION="$("$FC_PATH" --version | head -1)"

# --- netCDF-Fortran, resolved by compiler rather than by PATH order -----------
# nf-config reports the compiler its library was built with. That matters more
# than which one comes first on PATH: a netCDF-Fortran built with ifort ships
# ifort .mod files, and linking them into a gfortran build fails with errors
# that never mention the compiler. Pick the first candidate whose --fc agrees
# with ours, and say so if none does.
if [[ -z "$NETCDF_F_HOME" || -z "$NETCDF_INC" ]]; then
  nf_config=""
  for candidate in "$(command -v nf-config || true)" /usr/bin/nf-config /usr/local/bin/nf-config; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    candidate_fc="$("$candidate" --fc 2>/dev/null || true)"
    if [[ "$(basename "${candidate_fc%% *}")" == gfortran* ]]; then
      nf_config="$candidate"
      break
    fi
    note "skipping $candidate: built with ${candidate_fc:-an unknown compiler}, not gfortran"
  done
  [[ -n "$nf_config" ]] || die "no netCDF-Fortran built with gfortran was found (e.g. apt install libnetcdff-dev). If yours lives elsewhere, pass --netcdf-fortran-home and --netcdf-include."
  note "using netCDF-Fortran from $nf_config"
  # --flibs is `-L<dir> -lnetcdff`; the -L directory is what upstream wants,
  # because on Debian multiarch the library is not under <prefix>/lib.
  if [[ -z "$NETCDF_F_HOME" ]]; then
    NETCDF_F_HOME="$("$nf_config" --flibs 2>/dev/null | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
    [[ -n "$NETCDF_F_HOME" ]] || NETCDF_F_HOME="$("$nf_config" --prefix)/lib"
  fi
  [[ -n "$NETCDF_INC" ]] || NETCDF_INC="$("$nf_config" --includedir 2>/dev/null || echo /usr/include)"
fi
[[ -n "$LAPACK_HOME" ]] || LAPACK_HOME=/usr
[[ -n "$JOBS" ]] || JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# --- revision and cleanliness -------------------------------------------------
REVISION="$(git -C "$SOURCE" rev-parse --short HEAD 2>/dev/null || true)"
[[ -n "$REVISION" ]] || die "the source tree is not under git, so its revision cannot be recorded"
DESCRIBED="$(git -C "$SOURCE" describe --always --dirty 2>/dev/null || echo "$REVISION")"
BRANCH="$(git -C "$SOURCE" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
REMOTE="$(git -C "$SOURCE" remote get-url origin 2>/dev/null || echo "")"
DIRTY_FILES="$(git -C "$SOURCE" status --porcelain --untracked-files=no)"
DIRTY_DIFF_SHA=""
if [[ -n "$DIRTY_FILES" ]]; then
  if ((!ALLOW_DIRTY)); then
    printf 'the source tree has uncommitted tracked changes:\n%s\n' "$DIRTY_FILES" >&2
    die "a build from a dirty tree has no statable provenance. Commit or set the changes aside, or rerun with --allow-dirty to record the diff digest alongside the revision."
  fi
  DIRTY_DIFF_SHA="$(git -C "$SOURCE" diff | "$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')"
  note "building a dirty tree ($DESCRIBED); diff sha256 $DIRTY_DIFF_SHA will be recorded"
fi

# --- the build environment, set rather than inferred --------------------------
# MKLROOT and friends would silently change which math library is linked.
export FC=gfortran CC=gcc
export LAPACKHOME="$LAPACK_HOME"
export NETCDF_FORTRAN_HOME="$NETCDF_F_HOME"
export NETCDFINC="$NETCDF_INC"
export FFLAGS="-fallow-argument-mismatch -O2"
# -rpath, so the executables find the netCDF-Fortran they were linked against
# rather than whichever one LD_LIBRARY_PATH happens to name first. Without it a
# machine carrying a second, differently-compiled netCDF -- an ifort build on
# LD_LIBRARY_PATH is the ordinary case on a cluster -- links correctly and then
# dies at run time with `undefined symbol: __netcdf_MOD_nf90_put_var_*`, which
# names neither the library nor the variable that chose it.
export OMPFLAG=-fopenmp RECURSFLAG=-frecursive
# --disable-new-dtags is what makes this work. Current binutils defaults to
# --enable-new-dtags, which emits DT_RUNPATH, and LD_LIBRARY_PATH takes
# precedence over RUNPATH -- so the rpath is present and ignored. DT_RPATH is
# searched *before* LD_LIBRARY_PATH, which is the ordering needed here: an
# ABI-incompatible netCDF-Fortran substituted at run time is a symbol-lookup
# crash, not a preference worth honouring.
export LDFLAGS="-fopenmp -Wl,--disable-new-dtags -Wl,-rpath,$NETCDF_F_HOME"
unset MKLROOT ACML_HOME NETCDFHOME NETCDF_DIR F90HOME X11_HOME || true

mkdir -p "$PREFIX/logs"
LOG="$PREFIX/logs/gpec-build-$(date +%Y%m%d-%H%M%S).log"

# --- preflight: upstream's own summary of what it resolved --------------------
note "upstream's resolved toolchain (make v):"
MAKE_V="$(cd "$BUILD_DIR" && make v 2>&1)" || die "make v failed; see above"
printf '%s\n' "$MAKE_V" | sed 's/^/    /'
printf '%s\n' "$MAKE_V" >>"$LOG"
# If NEEDED_DEPS is non-empty, upstream is about to build its dependencies from
# submodules inside the operator's checkout -- the one thing this script promises
# not to cause. Stop while the message can still name the two variables that fix it.
NEEDED_DEPS="$(printf '%s\n' "$MAKE_V" | sed -n 's/.*Compiling supporting modules //p' | tr -d '[:space:]')"
if [[ -n "$NEEDED_DEPS" ]]; then
  die "upstream wants to build supporting modules ($NEEDED_DEPS) inside your checkout, which means it did not find a dependency. Set --lapack-home and --netcdf-fortran-home to libraries that exist rather than letting it compile its own."
fi

# --- build --------------------------------------------------------------------
# Upstream's rules judge themselves by a `cp`, so a failed link can leave a
# zero-length binary that make then treats as up to date. Clear the targets
# first, and assert on size rather than on make's exit status afterwards.
for program in "${PROGRAMS[@]}"; do
  rm -f "$SOURCE/bin/$program"
done
# Space-joined rather than "${TARGETS[*]}": IFS is \n\t here, so [*] would join
# with a newline and break both the log line and the manifest below.
TARGETS_LINE="$(printf '%s ' "${TARGETS[@]}")"
PROGRAMS_LINE="$(printf '%s ' "${PROGRAMS[@]}")"
MAKE_COMMAND="make -j$JOBS ${TARGETS_LINE% }"
note "building with $JOBS jobs (log: $LOG)"
(cd "$BUILD_DIR" && make -j"$JOBS" "${TARGETS[@]}") >>"$LOG" 2>&1 || die "build failed; see $LOG"
for program in "${PROGRAMS[@]}"; do
  [[ -s "$SOURCE/bin/$program" ]] || die "$program was not produced at $SOURCE/bin/$program, or is empty (see $LOG)"
done

# --- install ------------------------------------------------------------------
mkdir -p "$PREFIX/bin"
for program in "${PROGRAMS[@]}"; do
  cp -f "$SOURCE/bin/$program" "$PREFIX/bin/$program"
  chmod +x "$PREFIX/bin/$program"
done
note "installed ${#PROGRAMS[@]} executables into $PREFIX/bin"

# --- manifest -----------------------------------------------------------------
"$PYTHON" - "$MANIFEST" <<EOF
import hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
prefix = Path("$PREFIX")
programs = "$PROGRAMS_LINE".split()
record = {
    "code": "gpec",
    "installer": "install/install_gpec.sh",
    "prefix": str(prefix),
    "source": "$SOURCE",
    "source_revision": "$REVISION",
    "source_described": "$DESCRIBED",
    "source_branch": "$BRANCH",
    "source_remote": "$REMOTE",
    "source_dirty": bool("""$DIRTY_FILES""".strip()),
    "source_dirty_files": [l for l in """$DIRTY_FILES""".splitlines() if l.strip()],
    "source_dirty_diff_sha256": "$DIRTY_DIFF_SHA" or None,
    "build_dir": "$BUILD_DIR",
    "build_in_place": True,
    "make_command": "$MAKE_COMMAND",
    "make_targets": "$TARGETS_LINE".split(),
    "openmp": True,
    "fflags": "$FFLAGS",
    "compiler": {"fortran": "$FC_PATH", "version": "$FC_VERSION"},
    "dependency_providers": {
        "lapack_home": "$LAPACK_HOME",
        "netcdf_fortran_home": "$NETCDF_F_HOME",
        "netcdf_include": "$NETCDF_INC",
    },
    "platform": "$PLATFORM",
    "host": platform.node(),
    "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "log": "$LOG",
    "executables": {
        name: {
            "path": str(prefix / "bin" / name),
            "sha256": sha(prefix / "bin" / name),
            "size": (prefix / "bin" / name).stat().st_size,
        }
        for name in programs
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print("wrote", sys.argv[1])
EOF

# --- acceptance ---------------------------------------------------------------
# check_gpec.py runs DCON on upstream's self-contained Solov'ev regression case
# and then GPEC on DCON's output, so it exercises the real handoff rather than
# starting six binaries separately. The executables are installed either way so
# a failure can be examined.
if ((!SKIP_TESTS)); then
  note "verifying with install/check_gpec.py"
  GPECHOME="$PREFIX" "$PYTHON" "$SCRIPT_DIR/check_gpec.py" --source "$SOURCE" --prefix "$PREFIX" || true
fi
printf '\nPoint VAFT at this build:\n  export GPECHOME=%s\n' "$PREFIX"

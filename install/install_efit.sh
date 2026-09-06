#!/usr/bin/env bash
# Build and install the EFIT toolchain (efit + efund) from an EFIT source tree you already hold.
#
# Usage:
#   bash install/install_efit.sh --source PATH --accept-efit-users-agreement [options]
#   bash install/install_efit.sh --source PATH --check-only
#   bash install/install_efit.sh --source PATH --uninstall
#
# EFIT is licensed software. Its source is distributed by the EFIT-AI team under
# a users agreement (LICENSE.rst in the source tree) that forbids redistributing
# the original or any modified source and asks each recipient to register with
# the authors. VAFT therefore neither bundles, mirrors nor fetches EFIT: this
# script builds a tree you have already obtained through the EFIT-AI channel
# (efit-support@fusion.gat.com, GitLab access granted after agreeing to the
# terms). Passing --accept-efit-users-agreement states that you have read and
# agreed to that agreement and hold authorized access; the script never agrees
# on your behalf and never clones, fetches, pulls, checks out or changes a
# revision of the tree it is given.
#
# What it does: configure out of tree with CMake (Release, NetCDF on so EFIT
# writes m-files), build efit and efund, run EFIT's own ctest suite as the
# build's acceptance, install bin/efit and bin/efund into a prefix that
# $EFITHOME points at, and write vaft-external-install.json there recording
# the source revision (and, if you insisted on building a dirty tree, the diff
# digest), the CMake arguments, the compiler, the NetCDF/BLAS providers and the
# executables' checksums. Everything generated stays inside the source tree or
# the prefix, never in the VAFT checkout.
#
# macOS (Apple Silicon, Homebrew gcc/netcdf) and Linux (system gfortran,
# nf-config) are supported; nothing is installed system-wide by this script.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
MANIFEST_NAME="vaft-external-install.json"

SOURCE="${EFIT_SOURCE_DIR:-}"
PREFIX=""
BUILD_DIR=""
ACCEPT_TERMS=0
ALLOW_DIRTY=0
WITH_NETCDF=1
JOBS=""
SKIP_TESTS=0
CHECK_ONLY=0
UNINSTALL=0

usage() {
  cat <<'EOF'
Usage: bash install/install_efit.sh --source PATH --accept-efit-users-agreement [options]

  --source PATH                    the EFIT source tree to build (or set EFIT_SOURCE_DIR)
  --accept-efit-users-agreement    you have read and agreed to EFIT's users agreement
                                   (LICENSE.rst) and hold authorized access to this source
  --prefix PATH                    install here (default: <source>/vaft-install)
  --build-dir PATH                 configure here (default: <source>/build-vaft-<platform>)
  --allow-dirty                    build a tree with uncommitted tracked changes; the diff
                                   digest and file list are then recorded in the manifest
  --without-netcdf                 build without NetCDF (EFIT then writes no m-files)
  --jobs N                         parallel build jobs (default: all cores)
  --skip-tests                     do not run EFIT's ctest suite after building
  --check-only                     run install/check_efit.py and change nothing
  --uninstall                      remove the build directory and prefix this script created
  -h, --help

The prefix is what $EFITHOME should point at: VAFT resolves $EFITHOME/bin/efit and
$EFITHOME/bin/efund. This script prints the export line; it edits no shell profile.
EOF
}

die() { printf 'install_efit.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --source) (($# >= 2)) || die '--source needs a path'; SOURCE="$2"; shift 2 ;;
    --prefix) (($# >= 2)) || die '--prefix needs a path'; PREFIX="$2"; shift 2 ;;
    --build-dir) (($# >= 2)) || die '--build-dir needs a path'; BUILD_DIR="$2"; shift 2 ;;
    --accept-efit-users-agreement) ACCEPT_TERMS=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --without-netcdf) WITH_NETCDF=0; shift ;;
    --jobs) (($# >= 2)) || die '--jobs needs a number'; JOBS="$2"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$SOURCE" ]] || die "--source is required: VAFT does not vendor EFIT. Obtain the source through the EFIT-AI channel (see LICENSE.rst there) and pass its path, or set EFIT_SOURCE_DIR."
SOURCE="$(cd "$SOURCE" 2>/dev/null && pwd -P)" || die "EFIT source tree does not exist: $SOURCE"
for marker in CMakeLists.txt efit/efit.F90 green/efund.f90 LICENSE.rst; do
  [[ -e "$SOURCE/$marker" ]] || die "not an EFIT source tree (missing $marker): $SOURCE"
done

case "$(uname -s)" in
  Darwin) PLATFORM="darwin-$(uname -m)" ;;
  Linux) PLATFORM="linux-$(uname -m)" ;;
  *) die "unsupported platform: $(uname -s)" ;;
esac
[[ -n "$PREFIX" ]] || PREFIX="$SOURCE/vaft-install"
[[ -n "$BUILD_DIR" ]] || BUILD_DIR="$SOURCE/build-vaft-$PLATFORM"
MANIFEST="$PREFIX/$MANIFEST_NAME"

PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "$PYTHON" ]] || die "python3 is required (the VAFT environment provides it)"

if ((CHECK_ONLY)); then
  exec "$PYTHON" "$SCRIPT_DIR/check_efit.py" --source "$SOURCE" --prefix "$PREFIX"
fi

if ((UNINSTALL)); then
  if [[ -f "$MANIFEST" ]]; then
    recorded_build="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("build_dir",""))' "$MANIFEST")"
    if [[ -n "$recorded_build" && -d "$recorded_build" ]]; then
      note "removing build directory $recorded_build"; rm -rf "$recorded_build"
    fi
    note "removing prefix $PREFIX"; rm -rf "$PREFIX"
  else
    die "no $MANIFEST_NAME under $PREFIX; nothing this script installed is there to remove"
  fi
  exit 0
fi

if ((!ACCEPT_TERMS)); then
  cat >&2 <<'EOF'
EFIT is licensed software. Its users agreement (LICENSE.rst in the source tree)
forbids redistribution of the original or any modified source and asks each
recipient to register with the authors (efit-support@fusion.gat.com). VAFT does
not bundle, mirror or fetch EFIT; this script only builds a tree you already
hold. If you have read and agreed to the agreement and obtained authorized
access to this source, rerun with --accept-efit-users-agreement.
EOF
  exit 2
fi

command -v cmake >/dev/null || die "cmake is required (macOS: brew install cmake; Linux: your package manager)"
command -v git >/dev/null || die "git is required"

# --- revision and cleanliness -------------------------------------------------
REVISION="$(git -C "$SOURCE" rev-parse --short HEAD 2>/dev/null || true)"
[[ -n "$REVISION" ]] || die "the source tree is not a git checkout, so its revision cannot be recorded"
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

# --- toolchain ---------------------------------------------------------------
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DTEST_EFUND=ON)
FC_PATH=""
NETCDF_C_DIR=""; NETCDF_F_DIR=""
if [[ "$PLATFORM" == darwin-* ]]; then
  command -v brew >/dev/null || die "Homebrew is required on macOS: https://brew.sh"
  brew list --versions gcc >/dev/null 2>&1 || die "Homebrew gcc (gfortran) is required: brew install gcc"
  FC_PATH="$(brew --prefix gcc)/bin/gfortran"
  [[ -x "$FC_PATH" ]] || die "gfortran not found at $FC_PATH"
  if ((WITH_NETCDF)); then
    brew list --versions netcdf >/dev/null 2>&1 && brew list --versions netcdf-fortran >/dev/null 2>&1 ||
      die "NetCDF is required for m-files: brew install netcdf netcdf-fortran (or pass --without-netcdf)"
    NETCDF_C_DIR="$(brew --prefix netcdf)"; NETCDF_F_DIR="$(brew --prefix netcdf-fortran)"
  fi
else
  FC_PATH="$(command -v gfortran || true)"
  [[ -n "$FC_PATH" ]] || die "gfortran is required (e.g. apt install gfortran)"
  if ((WITH_NETCDF)); then
    command -v nf-config >/dev/null || die "NetCDF-Fortran is required for m-files (e.g. apt install libnetcdff-dev), or pass --without-netcdf"
    NETCDF_C_DIR="$(nc-config --prefix 2>/dev/null || echo /usr)"; NETCDF_F_DIR="$(nf-config --prefix 2>/dev/null || echo /usr)"
  fi
fi
CMAKE_ARGS+=("-DCMAKE_Fortran_COMPILER=$FC_PATH")
if ((WITH_NETCDF)); then
  CMAKE_ARGS+=(-DENABLE_NETCDF=ON "-DNetCDF_C_DIR=$NETCDF_C_DIR" "-DNetCDF_FORTRAN_DIR=$NETCDF_F_DIR")
else
  CMAKE_ARGS+=(-DENABLE_NETCDF=OFF)
fi
FC_VERSION="$("$FC_PATH" --version | head -1)"
[[ -n "$JOBS" ]] || JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# --- configure, build, test ----------------------------------------------------
mkdir -p "$PREFIX/logs"
LOG="$PREFIX/logs/efit-build-$(date +%Y%m%d-%H%M%S).log"
note "configuring $SOURCE -> $BUILD_DIR (log: $LOG)"
cmake -S "$SOURCE" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" >>"$LOG" 2>&1 || die "cmake configure failed; see $LOG"
note "building with $JOBS jobs"
cmake --build "$BUILD_DIR" -j "$JOBS" >>"$LOG" 2>&1 || die "build failed; see $LOG"
[[ -x "$BUILD_DIR/efit/efit" ]] || die "efit was not produced at $BUILD_DIR/efit/efit"
[[ -x "$BUILD_DIR/green/efund" ]] || die "efund was not produced at $BUILD_DIR/green/efund"

CTEST_STATUS="skipped"
if ((!SKIP_TESTS)); then
  note "running EFIT's own ctest suite as the build's acceptance"
  if (cd "$BUILD_DIR" && ctest --output-on-failure >>"$LOG" 2>&1); then
    CTEST_STATUS="passed"
  else
    CTEST_STATUS="failed"
    printf '[WARN] ctest reported failures; see %s. The binaries are installed anyway so the failure can be examined, and the manifest records it.\n' "$LOG"
  fi
fi

# --- install ----------------------------------------------------------------
mkdir -p "$PREFIX/bin"
cp -f "$BUILD_DIR/efit/efit" "$PREFIX/bin/efit"
cp -f "$BUILD_DIR/green/efund" "$PREFIX/bin/efund"
chmod +x "$PREFIX/bin/efit" "$PREFIX/bin/efund"
note "installed bin/efit and bin/efund into $PREFIX"

# --- manifest ---------------------------------------------------------------
BLAS_LIBS="$(grep -m1 '^BLAS_LIBRARIES' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2- || true)"
LAPACK_LIBS="$(grep -m1 '^LAPACK_LIBRARIES' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2- || true)"
"$PYTHON" - "$MANIFEST" <<EOF
import hashlib, json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
prefix = Path("$PREFIX")
record = {
    "code": "efit",
    "installer": "install/install_efit.sh",
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
    "cmake_arguments": ${CMAKE_ARGS[@]+$(printf '%s\n' "${CMAKE_ARGS[@]}" | "$PYTHON" -c 'import json,sys; print(json.dumps(sys.stdin.read().split("\n")[:-1]))')},
    "compiler": {"fortran": "$FC_PATH", "version": "$FC_VERSION"},
    "netcdf": {"enabled": bool($WITH_NETCDF), "c_dir": "$NETCDF_C_DIR" or None, "fortran_dir": "$NETCDF_F_DIR" or None},
    "blas_libraries": "$BLAS_LIBS" or None,
    "lapack_libraries": "$LAPACK_LIBS" or None,
    "platform": "$PLATFORM",
    "host": platform.node(),
    "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "ctest": "$CTEST_STATUS",
    "log": "$LOG",
    "executables": {
        role: {"path": str(prefix / "bin" / role), "sha256": sha(prefix / "bin" / role), "size": (prefix / "bin" / role).stat().st_size}
        for role in ("efit", "efund")
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print("wrote", sys.argv[1])
EOF

note "verifying with install/check_efit.py"
"$PYTHON" "$SCRIPT_DIR/check_efit.py" --source "$SOURCE" --prefix "$PREFIX" || true
printf '\nPoint VAFT at this build:\n  export EFITHOME=%s\n' "$PREFIX"

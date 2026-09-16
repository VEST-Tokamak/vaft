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
  (set VAFT_PYTHON to choose the interpreter the VAFT-side checks run under;
   by default the first one on PATH that can import vaft and f90nml)
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

# The prefix is removed wholesale by --uninstall, so it must never be inside the
# VAFT checkout. The PowerShell installers enforce this through
# Resolve-InstallPrefix; the POSIX ones did not. Made absolute first, or a
# relative --prefix would slip past the comparison and past the manifest.
[[ "$PREFIX" == /* ]] || PREFIX="$PWD/$PREFIX"
VAFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
case "$PREFIX/" in
  "$VAFT_ROOT"/*) die "the install prefix must be outside the VAFT checkout, because --uninstall removes it: $PREFIX is inside $VAFT_ROOT" ;;
esac

PYTHON="${VAFT_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  # The manifest needs only the standard library, but check_efit.py imports
  # VAFT, so the interpreter has to be the one VAFT is installed into. The
  # first python3 on PATH need not be: a conda environment shadows the system
  # one, and either may be the one carrying VAFT and f90nml. Choosing by "can
  # it import what the check will import" beats choosing by PATH order, and a
  # check that fails on a missing module reads as a broken EFIT build when it
  # is a broken interpreter choice.
  for candidate in python3 /usr/bin/python3 python; do
    resolved="$(command -v "$candidate" 2>/dev/null)" || continue
    if "$resolved" -c 'import vaft, f90nml' >/dev/null 2>&1; then PYTHON="$resolved"; break; fi
    [[ -n "$PYTHON" ]] || PYTHON="$resolved"   # remember the first that exists
  done
fi
[[ -n "$PYTHON" ]] || die "python3 is required (the VAFT environment provides it)"
if ! "$PYTHON" -c 'import vaft, f90nml' >/dev/null 2>&1; then
  printf '[WARN] %s cannot import vaft and f90nml, so the VAFT-side checks will report a failure that is about this interpreter and not about EFIT. Set VAFT_PYTHON to the interpreter VAFT is installed into.\n' "$PYTHON" >&2
fi

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

# --- BLAS link order ----------------------------------------------------------
# EFIT's CMakeLists puts its own static archives (libgreen, liblsode,
# libr8slatec) after the shared LAPACK and BLAS on the link line. GNU ld
# resolves left to right and, since binutils 2.22 defaults to
# --no-copy-dt-needed-entries, then refuses liblapack's own reference to
# dscal_ with
#
#   liblapack.so: undefined reference to symbol 'dscal_'
#   libblas.so: error adding symbols: DSO missing from command line
#
# even though libblas is on the line, because it was already passed. This is
# not a NetCDF problem -- the same link fails with NetCDF off -- and it is not
# specific to this machine; it is what a current binutils does with that
# ordering. CMAKE_<LANG>_STANDARD_LIBRARIES is appended last, which is exactly
# where BLAS has to appear.
blas_link_repair() {
  local dir
  [[ "$PLATFORM" == linux-* ]] || return 0   # macOS resolves BLAS via Accelerate
  for dir in /usr/lib /usr/lib64 "/usr/lib/$(uname -m)-linux-gnu" /usr/local/lib; do
    if [[ -e "$dir/libblas.so" || -e "$dir/libblas.a" ]]; then
      CMAKE_ARGS+=("-DCMAKE_Fortran_STANDARD_LIBRARIES=-lblas")
      note "appending -lblas after EFIT's static archives (link-order repair)"
      return 0
    fi
  done
}

# --- NetCDF discovery ---------------------------------------------------------
# EFIT's FindNetCDF searches ONE prefix for both the C and the Fortran halves
# (io/FindNetCDF.cmake), so handing it a split pair fails even when both halves
# exist somewhere. That is easy to do by accident: nf-config and nc-config are
# resolved off PATH independently, and a Fortran-only NetCDF belonging to some
# other package can shadow the system one. GPEC ships exactly that -- a private
# netcdf-fortran whose bin/ goes on PATH ahead of /usr/bin and which contains no
# libnetcdf at all -- so nf-config reports its prefix, nc-config reports /usr,
# and the configure dies on "Can not locate NetCDF C library".
#
# Pick a prefix that provides both halves, and say which one and why.

netcdf_libdirs() {  # every place a distribution might put the libraries
  local prefix="$1"
  printf '%s\n' "$prefix/lib" "$prefix/lib64" "$prefix/lib/$(uname -m)-linux-gnu"
}

netcdf_has_half() {  # netcdf_has_half PREFIX c|fortran
  local prefix="$1" half="$2" stem dir
  [[ "$half" == c ]] && stem=libnetcdf || stem=libnetcdff
  while read -r dir; do
    [[ -e "$dir/$stem.so" || -e "$dir/$stem.a" || -e "$dir/$stem.dylib" ]] && return 0
  done < <(netcdf_libdirs "$prefix")
  return 1
}

netcdf_has_module() {  # gfortran needs the module or the include file
  local prefix="$1"
  [[ -e "$prefix/include/netcdf.mod" || -e "$prefix/include/netcdf.inc" ]]
}

netcdf_libdir_of() {  # where this prefix actually keeps libnetcdf
  local prefix="$1" dir
  while read -r dir; do
    [[ -e "$dir/libnetcdf.so" || -e "$dir/libnetcdf.a" || -e "$dir/libnetcdf.dylib" ]] &&
      { printf '%s\n' "$dir"; return 0; }
  done < <(netcdf_libdirs "$prefix")
  return 1
}

netcdf_prefix_for_efit() {
  local candidate candidate_fc prefix chosen=""
  # First filter: the compiler the Fortran half was built with. nf-config
  # reports its own --fc, and a library built with ifort ships ifort .mod
  # files; linking those into a gfortran build fails with errors that never
  # mention a compiler. On a machine carrying a hand-built netCDF ahead of the
  # distribution one -- ordinary on a cluster -- taking the first nf-config
  # silently configures the wrong one.
  #
  # Second filter: the prefix has to hold all three pieces. A Fortran-only
  # netCDF passes the compiler test and still leaves EFIT with no C library,
  # because FindNetCDF searches one prefix for both halves.
  for candidate in "$(command -v nf-config || true)" /usr/bin/nf-config /usr/local/bin/nf-config; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    candidate_fc="$("$candidate" --fc 2>/dev/null || true)"
    if [[ "$(basename "${candidate_fc%% *}")" != gfortran* ]]; then
      note "skipping $candidate: built with ${candidate_fc:-an unknown compiler}, not gfortran"
      continue
    fi
    prefix="$("$candidate" --prefix 2>/dev/null || true)"
    [[ -n "$prefix" && -d "$prefix" ]] || continue
    if netcdf_has_half "$prefix" c && netcdf_has_half "$prefix" fortran &&
       netcdf_has_module "$prefix"; then
      chosen="$prefix"
      note "using netCDF from $candidate (prefix $prefix)"
      break
    fi
    note "skipping $candidate: $prefix has no NetCDF C library beside the Fortran one"
  done
  if [[ -z "$chosen" ]]; then
    die "no NetCDF provides a gfortran-built Fortran library, its module and a C library under one prefix. EFIT searches one prefix for all three. Install a complete NetCDF (e.g. apt install libnetcdf-dev libnetcdff-dev), or pass --without-netcdf and accept that EFIT will write no m-files."
  fi
  NETCDF_C_DIR="$chosen"; NETCDF_F_DIR="$chosen"
  # EFIT's FindNetCDF only looks in <prefix>/lib and <prefix>/Lib
  # (io/FindNetCDF.cmake: netcdf_lib_suffixes), so on a multiarch distribution
  # -- Debian and Ubuntu put it in lib/<triplet> -- a system NetCDF is never
  # found from the prefix alone. Passing the library directory explicitly takes
  # the NetCDF_LIBRARY_DIR branch instead, which searches it directly. The
  # Fortran half needs its own variable: that branch has no fallback for it.
  # The chosen prefix is what was vetted, so look inside it first; nc-config
  # may well describe a different installation from the one that survived the
  # filters above, and is only useful here when it happens to agree.
  NETCDF_LIB_DIR="$(netcdf_libdir_of "$chosen" || true)"
  if [[ -z "$NETCDF_LIB_DIR" ]]; then
    NETCDF_LIB_DIR="$(nc-config --libdir 2>/dev/null || true)"
    [[ -n "$NETCDF_LIB_DIR" && -e "$NETCDF_LIB_DIR/libnetcdf.so" ]] || NETCDF_LIB_DIR=""
  fi
  [[ -n "$NETCDF_LIB_DIR" ]] ||
    die "found a NetCDF prefix at $chosen but no directory in it holding libnetcdf"
}

# --- toolchain ---------------------------------------------------------------
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DTEST_EFUND=ON)
FC_PATH=""
NETCDF_C_DIR=""; NETCDF_F_DIR=""; NETCDF_LIB_DIR=""
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
    netcdf_prefix_for_efit
  fi
fi
CMAKE_ARGS+=("-DCMAKE_Fortran_COMPILER=$FC_PATH")
if ((WITH_NETCDF)); then
  CMAKE_ARGS+=(-DENABLE_NETCDF=ON "-DNetCDF_C_DIR=$NETCDF_C_DIR" "-DNetCDF_FORTRAN_DIR=$NETCDF_F_DIR")
  if [[ -n "$NETCDF_LIB_DIR" ]]; then
    CMAKE_ARGS+=("-DNetCDF_LIBRARY_DIR=$NETCDF_LIB_DIR"
                 "-DNetCDF_C_LIBRARY_DIR=$NETCDF_LIB_DIR"
                 "-DNetCDF_FORTRAN_LIBRARY_DIR=$NETCDF_LIB_DIR")
  fi
else
  CMAKE_ARGS+=(-DENABLE_NETCDF=OFF)
fi
blas_link_repair
FC_VERSION="$("$FC_PATH" --version | head -1)"
[[ -n "$JOBS" ]] || JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# --- configure, build, test ----------------------------------------------------
mkdir -p "$PREFIX/logs"
LOG="$PREFIX/logs/efit-build-$(date +%Y%m%d-%H%M%S).log"

# Turning NetCDF on or off changes config.h, and CMake does not track config.h
# as a dependency of the Fortran sources. An incremental build therefore leaves
# write_m.F90.o compiled against the *old* setting: it references no NetCDF
# symbol, the linker drops the libraries as unneeded, and the result is a
# binary with HAVE_NETCDF in its config.h and no NetCDF in it. Reconfiguring a
# build tree whose NetCDF state disagrees with what is being asked for is the
# one case that has to start clean.
if [[ -f "$BUILD_DIR/config.h" ]]; then
  had_netcdf=0
  grep -qE '^[[:space:]]*#define[[:space:]]+HAVE_NETCDF' "$BUILD_DIR/config.h" && had_netcdf=1
  if ((had_netcdf != WITH_NETCDF)); then
    note "NetCDF is changing from $had_netcdf to $WITH_NETCDF; removing $BUILD_DIR so the sources are recompiled against it"
    rm -rf "$BUILD_DIR"
  fi
fi

note "configuring $SOURCE -> $BUILD_DIR (log: $LOG)"
cmake -S "$SOURCE" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" >>"$LOG" 2>&1 || die "cmake configure failed; see $LOG"

# EFIT's io/efitIO.cmake calls find_package(NetCDF) without REQUIRED and has no
# else branch, so ENABLE_NETCDF=ON plus a NetCDF it cannot find is not an error:
# it is a silent build that writes no m-file. The generated config.h is the only
# statement of what was actually achieved, so believe that and not the request.
NETCDF_ACHIEVED=0
if [[ -f "$BUILD_DIR/config.h" ]] && grep -qE '^[[:space:]]*#define[[:space:]]+HAVE_NETCDF' "$BUILD_DIR/config.h"; then
  NETCDF_ACHIEVED=1
fi
if ((WITH_NETCDF)) && ((!NETCDF_ACHIEVED)); then
  grep -iE 'netcdf' "$LOG" | tail -5 >&2 || true
  die "NetCDF was requested but the configure did not link it, so this build would write no m-files. The lines above are what CMake said; searched prefix $NETCDF_C_DIR. Fix the NetCDF installation, or pass --without-netcdf to accept a build with no m-files."
fi
((WITH_NETCDF)) && note "NetCDF linked from $NETCDF_C_DIR (m-files will be written)"
note "building with $JOBS jobs"
cmake --build "$BUILD_DIR" -j "$JOBS" >>"$LOG" 2>&1 || die "build failed; see $LOG"
[[ -x "$BUILD_DIR/efit/efit" ]] || die "efit was not produced at $BUILD_DIR/efit/efit"
[[ -x "$BUILD_DIR/green/efund" ]] || die "efund was not produced at $BUILD_DIR/green/efund"

# config.h said what the configure decided; this asks the executable. They can
# disagree -- a stale object file leaves no NetCDF symbol to reference, the
# linker drops the libraries even though they are on the link line, and the
# binary writes no m-file while every configuration artifact claims it will.
# The executable is the only thing that runs, so it is what gets believed.
binary_links_netcdf() {
  local bin="$1"
  if command -v ldd >/dev/null && ldd "$bin" 2>/dev/null | grep -qi netcdf; then return 0; fi
  if command -v otool >/dev/null && otool -L "$bin" 2>/dev/null | grep -qi netcdf; then return 0; fi
  strings -a "$bin" 2>/dev/null | grep -qi 'netcdf'
}
if ((WITH_NETCDF)) && ! binary_links_netcdf "$BUILD_DIR/efit/efit"; then
  die "the configure linked NetCDF but the built efit does not reference it, so it would write no m-files. This is what a stale object file in $BUILD_DIR looks like: remove that directory and build again."
fi

CTEST_STATUS="skipped"
if ((!SKIP_TESTS)); then
  # EFIT generates its test drivers with configure_file, which does not carry
  # an execute bit, so every one of them is "Process not started / permission
  # denied" and the suite reports 0 of 44 passed. That looks like a catastrophic
  # physics failure and is a file mode.
  while IFS= read -r script; do chmod +x "$script"; done \
    < <(find "$BUILD_DIR/test" -name '*.sh' -type f ! -perm -u+x 2>/dev/null)
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
    # "requested" is what the flags asked for; "linked" is what config.h says was
    # achieved. They can differ, and only the second one decides whether this
    # build writes m-files.
    "netcdf": {"enabled": bool($NETCDF_ACHIEVED), "requested": bool($WITH_NETCDF),
               "linked": bool($NETCDF_ACHIEVED),
               "c_dir": "$NETCDF_C_DIR" or None, "fortran_dir": "$NETCDF_F_DIR" or None,
               "library_dir": "$NETCDF_LIB_DIR" or None},
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

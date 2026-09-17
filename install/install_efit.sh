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
# Prefix ownership: what --uninstall may remove, and what an install may claim.
# shellcheck disable=SC1091
. "$SCRIPT_DIR/_external_code_common.sh"
MANIFEST_NAME="$VAFT_EXTERNAL_MANIFEST_NAME"
PREFIX_CREATED=0

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
  --uninstall                      remove the build directory this script configured and what
                                   it installed into the prefix; the prefix itself only if
                                   this script created it and it is then empty
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

# Nothing this script writes may land inside the VAFT checkout. Both sides of
# the comparison are physical paths: a relative --prefix, a symlink or a
# spelling such as /tmp/../<checkout>/x would otherwise compare unequal to the
# checkout it is inside. (The PowerShell installers do the same through
# Resolve-InstallPrefix.)
PREFIX="$(vaft_external_canonical_path "$PREFIX")" || die "cannot resolve the install prefix (a '..' below a directory that does not exist?): $PREFIX"
MANIFEST="$PREFIX/$MANIFEST_NAME"
VAFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
if vaft_external_is_inside "$PREFIX" "$VAFT_ROOT"; then
  die "the install prefix must be outside the VAFT checkout: $PREFIX is inside $VAFT_ROOT"
fi

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

# A build directory is removed recursively, so it has to prove it is one: an
# absolute path holding a CMakeCache.txt that was configured from this source.
efit_build_dir_is_ours() {  # efit_build_dir_is_ours DIR SOURCE
  [[ "$1" == /* && -f "$1/CMakeCache.txt" ]] || return 1
  grep -qxF "CMAKE_HOME_DIRECTORY:INTERNAL=$2" "$1/CMakeCache.txt"
}

if ((UNINSTALL)); then
  recorded_build=""
  if [[ -f "$MANIFEST" ]]; then
    recorded_build="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("build_dir",""))' "$MANIFEST")"
  fi
  # First, because it is what refuses a prefix this script does not own.
  vaft_external_uninstall_prefix "$PREFIX" efit "$PYTHON"
  if [[ -n "$recorded_build" && -d "$recorded_build" ]]; then
    if efit_build_dir_is_ours "$recorded_build" "$SOURCE"; then
      note "removing build directory $recorded_build"; rm -rf "$recorded_build"
    else
      note "left $recorded_build: it is not an absolute path to a CMake build tree configured from $SOURCE"
    fi
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
# Refuses a directory that holds somebody else's files, and records whether the
# prefix is this script's creation, before anything is written into it.
vaft_external_claim_prefix "$PREFIX" efit "$PYTHON"
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

# --- did NetCDF actually get in? ----------------------------------------------
# Asking for it is not the same as getting it. io/efitIO.cmake is
#
#   option(ENABLE_NETCDF "Enable NetCDF" off)
#   if(${ENABLE_NETCDF})
#     find_package (NetCDF)
#     if(${NetCDF_FOUND})
#       set(USE_NETCDF ...)
#     endif()
#   endif()
#
# with no else on the inner if. A find_package that comes up empty is silent:
# the build finishes, ENABLE_NETCDF:BOOL=ON stays in CMakeCache and in this
# script's own manifest, and the first sign of trouble is a reconstruction that
# quietly writes no m-file hours later. The binary is the only honest witness --
# EFIT compiles that message in exactly when write_m was compiled out.
if ((WITH_NETCDF)); then
  if grep -qa 'netcdf needs to be linked to write m-files' "$BUILD_DIR/efit/efit"; then
    printf '[FAIL] NetCDF was requested but the build did not get it, so this efit writes no m-files.\n' >&2
    printf '       find_package(NetCDF) found nothing under:\n' >&2
    printf '         NetCDF_C_DIR       %s\n' "$NETCDF_C_DIR" >&2
    printf '         NetCDF_FORTRAN_DIR %s\n' "$NETCDF_F_DIR" >&2
    printf '       Check that each holds the library and the .mod, or pass --without-netcdf\n' >&2
    printf '       to accept a build with no m-files.\n' >&2
    die "see the configure output in $LOG for what find_package looked at"
  fi
  # Second, independent witness, because the one above is a string that lives in
  # EFIT's source and can be reworded upstream without anyone here noticing.
  # This one reads the linkage instead, and catches the same stale-object case:
  # write_m.F90.o compiled before NetCDF was turned on references no NetCDF
  # symbol, so the linker drops the libraries even though they are on the link
  # line. Measured on a real build before it was fixed.
  if ! { { command -v ldd >/dev/null && ldd "$BUILD_DIR/efit/efit" 2>/dev/null | grep -qi netcdf; } ||
         { command -v otool >/dev/null && otool -L "$BUILD_DIR/efit/efit" 2>/dev/null | grep -qi netcdf; } ||
         strings -a "$BUILD_DIR/efit/efit" 2>/dev/null | grep -qi netcdf; }; then
    die "the configure linked NetCDF but the built efit references none of it, so it would write no m-files. That is what a stale object file in $BUILD_DIR looks like: remove that directory and build again."
  fi
  note "NetCDF is compiled in; this build writes m-files"
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
CMAKE_ARGS_LINES="$(printf '%s\n' "${CMAKE_ARGS[@]}")"
# Values reach Python through the environment and the heredoc is quoted, so no
# shell text is ever parsed as Python source. Interpolating them broke on the
# first quoted path in `git status --porcelain` ("""...name"""" is a
# SyntaxError) -- after bin/ was installed and before the manifest existed.
VAFT_MANIFEST_NETCDF_ACHIEVED="$NETCDF_ACHIEVED" \
VAFT_MANIFEST_WITH_NETCDF="$WITH_NETCDF" \
VAFT_MANIFEST_PREFIX="$PREFIX" \
VAFT_MANIFEST_PREFIX_CREATED="$PREFIX_CREATED" \
VAFT_MANIFEST_SOURCE="$SOURCE" \
VAFT_MANIFEST_REVISION="$REVISION" \
VAFT_MANIFEST_DESCRIBED="$DESCRIBED" \
VAFT_MANIFEST_BRANCH="$BRANCH" \
VAFT_MANIFEST_REMOTE="$REMOTE" \
VAFT_MANIFEST_DIRTY_DIFF_SHA="$DIRTY_DIFF_SHA" \
VAFT_MANIFEST_BUILD_DIR="$BUILD_DIR" \
VAFT_MANIFEST_FC_PATH="$FC_PATH" \
VAFT_MANIFEST_FC_VERSION="$FC_VERSION" \
VAFT_MANIFEST_NETCDF_C_DIR="$NETCDF_C_DIR" \
VAFT_MANIFEST_NETCDF_F_DIR="$NETCDF_F_DIR" \
VAFT_MANIFEST_NETCDF_LIB_DIR="$NETCDF_LIB_DIR" \
VAFT_MANIFEST_BLAS_LIBS="$BLAS_LIBS" \
VAFT_MANIFEST_LAPACK_LIBS="$LAPACK_LIBS" \
VAFT_MANIFEST_PLATFORM="$PLATFORM" \
VAFT_MANIFEST_CTEST_STATUS="$CTEST_STATUS" \
VAFT_MANIFEST_LOG="$LOG" \
VAFT_MANIFEST_DIRTY_FILES="$DIRTY_FILES" \
VAFT_MANIFEST_CMAKE_ARGS_LINES="$CMAKE_ARGS_LINES" \
"$PYTHON" - "$MANIFEST" <<'EOF'
import hashlib, json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
env = {k[len("VAFT_MANIFEST_"):]: v for k, v in os.environ.items() if k.startswith("VAFT_MANIFEST_")}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
prefix = Path(env["PREFIX"])
record = {
    "code": "efit",
    "installer": "install/install_efit.sh",
    "prefix": str(prefix),
    # What --uninstall may remove: these files, this script's build logs and
    # its two records; the directory itself only when prefix_created is true.
    "prefix_created": bool(int(env["PREFIX_CREATED"])),
    "installed_files": ["bin/efit", "bin/efund"],
    "source": env["SOURCE"],
    "source_revision": env["REVISION"],
    "source_described": env["DESCRIBED"],
    "source_branch": env["BRANCH"],
    "source_remote": env["REMOTE"],
    "source_dirty": bool(env["DIRTY_FILES"].strip()),
    "source_dirty_files": [l for l in env["DIRTY_FILES"].splitlines() if l.strip()],
    "source_dirty_diff_sha256": env["DIRTY_DIFF_SHA"] or None,
    "build_dir": env["BUILD_DIR"],
    "cmake_arguments": [a for a in env["CMAKE_ARGS_LINES"].split("\n") if a],
    "compiler": {"fortran": env["FC_PATH"], "version": env["FC_VERSION"]},
    # "requested" is what the flags asked for; "linked" is what config.h says was
    # achieved. They can differ, and only the second one decides whether this
    # build writes m-files.
    "netcdf": {"enabled": bool(int(env["NETCDF_ACHIEVED"])), "requested": bool(int(env["WITH_NETCDF"])),
               "linked": bool(int(env["NETCDF_ACHIEVED"])),
               "c_dir": env["NETCDF_C_DIR"] or None, "fortran_dir": env["NETCDF_F_DIR"] or None,
               "library_dir": env["NETCDF_LIB_DIR"] or None},
    "blas_libraries": env["BLAS_LIBS"] or None,
    "lapack_libraries": env["LAPACK_LIBS"] or None,
    "platform": env["PLATFORM"],
    "host": platform.node(),
    "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "ctest": env["CTEST_STATUS"],
    "log": env["LOG"],
    "executables": {
        role: {"path": str(prefix / "bin" / role), "sha256": sha(prefix / "bin" / role), "size": (prefix / "bin" / role).stat().st_size}
        for role in ("efit", "efund")
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("wrote", sys.argv[1])
EOF

note "verifying with install/check_efit.py"
"$PYTHON" "$SCRIPT_DIR/check_efit.py" --source "$SOURCE" --prefix "$PREFIX" || true
printf '\nPoint VAFT at this build:\n  export EFITHOME=%s\n' "$PREFIX"

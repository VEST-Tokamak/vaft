#!/usr/bin/env bash
# Build the serial NTCC NUBEAM distribution on Linux.
#
# Usage:
#   bash install/nubeam/linux.sh --nubeam-root PATH --accept-ntcc-terms [--resume]
#
# VAFT does not vendor the NUBEAM source: NTCC requires each user to accept its
# licence before downloading it. This script owns the reproducible build recipe
# and operates on a NUBEAM tree you already hold, named by --nubeam-root. The
# resulting installation prefix, <root>/local, is what $NUBEAMHOME should point
# at for vaft.code.nubeam.
#
# The acceptance flag is required before this script downloads the NTCC
# dependency modules (PSPLINE, PREACT, XPLASMA). It signifies that the person
# running it has read and accepted
# https://w3.pppl.gov/NTCC/NUBEAM/downloads.shtml. The script never accepts the
# agreement implicitly. A tree that already carries those modules under
# vendor/ntcc/ is used as-is and nothing is downloaded.
#
# Linux only; macOS is install/nubeam/macos.sh and native Windows is
# install/nubeam/windows.ps1.
#
# ---------------------------------------------------------------------------
# What differs from macos.sh, and why. Most of that script's Make.local exists
# to defeat branches in share/Make.flags that upstream gates on Linux, so on
# Linux the correct action is to *stop* overriding them:
#
#   * MACHINE, OS. Make.flags:171-175 derives both from `sysname`, so setting
#     them here would only risk disagreeing with it.
#   * DEFS/CDEFS = -D__OSX. macOS needs that because portlib guards its Darwin
#     paths on __OSX while Make.flags only derives -D__$(OS). On Linux the
#     glibc paths those guards avoid are the correct ones.
#   * USEFC and FORTLIBS. FORTRAN_VARIANT=GCC arms MKGCC (Make.flags:216-219),
#     and the MKGCC block sets USEFC=Y and fills FORTLIBS itself
#     (Make.flags:303-344). Setting them here would shadow upstream's own
#     values for no reason.
#   * The empty endian.h and bits/byteswap.h compatibility headers. macOS lacks
#     both and trsocket.c includes them without using any symbol from them; on
#     Linux they are the real system headers and must not be shadowed.
#
# And one thing that is genuinely different rather than merely absent:
#
#   * The pspline symbol assertion greps `nm` output. Mach-O prefixes global
#     symbols with an underscore and ELF does not, so the macOS pattern
#     " T _mkbicub_" matches nothing here and the check would pass vacuously on
#     a truncated archive -- which is the exact failure it exists to catch.
# ---------------------------------------------------------------------------
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
NUBEAM_ROOT="${NUBEAM_SOURCE_DIR:-}"
ACCEPT_NTCC_TERMS=0
RESUME=0

usage() {
  cat <<'EOF'
Usage: bash install/nubeam/linux.sh --nubeam-root PATH --accept-ntcc-terms [--resume]

  --nubeam-root PATH   the NUBEAM source tree to build (or set NUBEAM_SOURCE_DIR)
  --accept-ntcc-terms  you have read and accepted the NTCC agreement, which is
                       required before any dependency source is downloaded
  --resume             reuse a recorded build directory and prefix after this
                       installer stopped with an error
  -h, --help

Relative to the NUBEAM tree: the installation prefix is ./local, generated
output is ./build/linux-<arch>, and NTCC sources stay in ./vendor/ntcc. Point
$NUBEAMHOME at ./local afterwards.

To undo an installation, run install/nubeam/uninstall.sh against the same tree.

Install the toolchain yourself first; this script names what is missing rather
than installing it:
  apt install gfortran gcc g++ make curl libnetcdff-dev liblapack-dev libblas-dev
EOF
}

die() { printf 'linux.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

is_child_of_root() {
  case "$1" in
    "$ROOT_DIR"/*) return 0 ;;
    *) return 1 ;;
  esac
}
require_child_path() {
  is_child_of_root "$1" || die "refusing path outside source tree: $1"
}

while (($#)); do
  case "$1" in
    --nubeam-root) (($# >= 2)) || die '--nubeam-root needs a path'; NUBEAM_ROOT="$2"; shift 2 ;;
    --accept-ntcc-terms) ACCEPT_NTCC_TERMS=1; shift ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$NUBEAM_ROOT" ]] ||
  die "--nubeam-root is required: the NUBEAM source tree is not vendored in VAFT. Obtain it from https://w3.pppl.gov/NTCC/NUBEAM/ and pass its path (or set NUBEAM_SOURCE_DIR)."
ROOT_DIR="$(cd "$NUBEAM_ROOT" 2>/dev/null && pwd -P)" ||
  die "NUBEAM source tree does not exist: $NUBEAM_ROOT"
[[ -f "$ROOT_DIR/Makefile" && -d "$ROOT_DIR/nubeam_comp_exec" ]] ||
  die "not a NUBEAM source tree (no Makefile and nubeam_comp_exec/): $ROOT_DIR"

[[ "$(uname -s)" == "Linux" ]] || die "this installer supports Linux only; macOS is install/nubeam/macos.sh"

# Everything this script generates stays inside the NUBEAM tree, never in the
# VAFT checkout.
PREFIX="$ROOT_DIR/local"
BUILD_DIR="$ROOT_DIR/build/linux-$(uname -m)"
NTCC_SOURCE_DIR="$ROOT_DIR/vendor/ntcc"
MANIFEST="$ROOT_DIR/.nubeam-install-manifest"
LOG_FILE="$BUILD_DIR/install.log"

require_child_path "$PREFIX"
require_child_path "$BUILD_DIR"
require_child_path "$NTCC_SOURCE_DIR"

command -v make >/dev/null || die "GNU make is required (apt install make)"
make --version 2>/dev/null | grep -q 'GNU Make' || die "GNU make is required"
command -v gfortran >/dev/null || die "gfortran is required (apt install gfortran)"
command -v gcc >/dev/null || die "gcc is required (apt install gcc)"
command -v g++ >/dev/null || die "g++ is required (apt install g++)"
command -v curl >/dev/null || die "curl is required to fetch NTCC sources (apt install curl)"
command -v ar >/dev/null || die "binutils ar is required (apt install binutils)"

if [[ -e "$MANIFEST" ]]; then
  ((RESUME)) || die "an existing NUBEAM installation manifest was found; remove it with 'bash install/nubeam/uninstall.sh --nubeam-root $ROOT_DIR', or resume the interrupted installer with --resume"
elif [[ -e "$PREFIX" ]]; then
  die "installation prefix already exists without a manifest: $PREFIX"
elif [[ -e "$BUILD_DIR" ]]; then
  die "generated build directory already exists without a manifest: $BUILD_DIR"
fi
if [[ -e "$ROOT_DIR/share/Make.local" ]] && ! grep -qE 'Generated by (VAFT (install|external)/nubeam|.*/install\.sh)' "$ROOT_DIR/share/Make.local"; then
  die "refusing to overwrite an existing user Make.local: $ROOT_DIR/share/Make.local"
fi

FC="$(command -v gfortran)"
CC="$(command -v gcc)"
CXX="$(command -v g++)"
PLATFORM="linux-$(uname -m)"
PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "$PYTHON" ]] || die "python3 is required to write the install record"

# netCDF-Fortran resolved by the compiler it was built with, not by PATH order:
# a library built with ifort ships ifort .mod files, and linking those into a
# gfortran build fails with errors that never name a compiler.
NETCDF_FORTRAN_HOME=""
for candidate in "$(command -v nf-config || true)" /usr/bin/nf-config /usr/local/bin/nf-config; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  candidate_fc="$("$candidate" --fc 2>/dev/null || true)"
  if [[ "$(basename "${candidate_fc%% *}")" == gfortran* ]]; then
    NETCDF_FORTRAN_HOME="$("$candidate" --prefix)"
    NETCDF_FORTRAN_INC="$("$candidate" --includedir 2>/dev/null || echo /usr/include)"
    NETCDF_FORTRAN_LIB="$("$candidate" --flibs 2>/dev/null | tr ' ' '\n' | sed -n 's/^-L//p' | head -1)"
    [[ -n "$NETCDF_FORTRAN_LIB" ]] || NETCDF_FORTRAN_LIB="$NETCDF_FORTRAN_HOME/lib"
    note "using netCDF-Fortran from $candidate"
    break
  fi
  note "skipping $candidate: built with ${candidate_fc:-an unknown compiler}, not gfortran"
done
[[ -n "$NETCDF_FORTRAN_HOME" ]] || die "no netCDF-Fortran built with gfortran was found (apt install libnetcdff-dev)"
NETCDF_C_HOME="$(nc-config --prefix 2>/dev/null || echo /usr)"

# Reference netlib LAPACK/BLAS is what a stock Linux carries; OpenBLAS is used
# instead when it is installed, since NUBEAM's linear algebra is a real cost.
LAPACK_LIB_DIR="$(dirname "$(find /usr/lib -name 'liblapack.so*' -print -quit 2>/dev/null || echo /usr/lib)")"
if ldconfig -p 2>/dev/null | grep -q 'libopenblas\.so'; then
  BLAS_FLAGS="-lopenblas"
  LAPACK_FLAGS="-llapack -lopenblas"
  note "linking OpenBLAS"
else
  BLAS_FLAGS="-lblas"
  LAPACK_FLAGS="-llapack -lblas"
  note "linking reference netlib BLAS/LAPACK"
fi

# The NTCC 2021 sources predate the stricter cross-procedure argument checks
# current GNU Fortran enforces, so they need -std=legacy and its companions.
#
# Those switches cannot be delivered through FFLAGS. Make.local is read before
# Make.flags' MKGCC block, which assigns FFLAGS itself (Make.flags:333), so a
# plain assignment here is discarded -- and an `override` is worse: upstream's
# convention is that FFLAGS ends with `-o`, appended per submodule, and
# `MFFLAGS = $(FFLAGS)` (Make.flags:658) is used as `$(FC90) $(MFFLAGS) $@ $<`.
# An override freezes FFLAGS, the append never happens, and gfortran is handed
# an object path with no -o in front of it.
#
# So the switches travel with the compiler instead. Make.local sets FC, whose
# origin is then "file", so MKGCC's `ifeq "$(origin FC)" "default"` leaves it
# alone (Make.flags:303-306) and every rule picks up the wrapper.
FC_WRAPPER="$BUILD_DIR/bin/vaft-gfortran"
mkdir -p "$BUILD_DIR/bin"
cat > "$FC_WRAPPER" <<WRAPPER
#!/bin/sh
# Generated by VAFT install/nubeam/linux.sh.
exec "$FC" -std=legacy -fallow-argument-mismatch -fallow-invalid-boz "\$@"
WRAPPER
chmod +x "$FC_WRAPPER"

mkdir -p "$BUILD_DIR"
mkdir -p "$PREFIX/bin" "$PREFIX/include" "$PREFIX/lib" "$PREFIX/mod"
exec > >(tee -a "$LOG_FILE") 2>&1

GENERATED_CONFIGS=("$ROOT_DIR/share/Make.local")
write_manifest() {
  {
    printf 'managed_dir\t%s\n' "$PREFIX"
    printf 'managed_dir\t%s\n' "$BUILD_DIR"
    for config in "${GENERATED_CONFIGS[@]}"; do
      printf 'generated_config\t%s\n' "$config"
    done
  } > "$MANIFEST"
}
write_manifest

write_make_local() {
  local target_root="$1"
  local config_dir="$target_root/share"
  [[ -d "$config_dir" ]] || die "NTCC module has no share directory: $target_root"
  if [[ -e "$config_dir/Make.local" ]] && ! grep -qE 'Generated by (VAFT (install|external)/nubeam|.*/install\.sh)' "$config_dir/Make.local"; then
    die "refusing to overwrite an existing user Make.local: $config_dir/Make.local"
  fi
  cat > "$config_dir/Make.local" <<EOF
# Generated by VAFT install/nubeam/linux.sh; removed by install/nubeam/uninstall.sh.
# MACHINE, OS, USEFC and FORTLIBS are deliberately absent: Make.flags derives
# MACHINE/OS from sysname, and FORTRAN_VARIANT=GCC arms its MKGCC branch, which
# sets USEFC=Y and fills FORTLIBS. See the header of install/nubeam/linux.sh.
FORTRAN_VARIANT = GCC
# MKGCC keys its 64-bit FFLAGS and the lib64 search paths off this.
_64 = 1
FC = $FC_WRAPPER
FC90 = $FC_WRAPPER
CC = $CC
CXX = $CXX
# GNU ld makes a single pass and discards an archive whose members resolve
# nothing yet. libnscrunch.a and libr8bloat.a call cspline_, cspeval_, spvec_
# and genxpkg_, but the link line reaches -lpspline long before either of them,
# so those come out undefined. A second PSPLINE occurrence at the end fixes it,
# after every static library that consumes its symbols. -lstdc++ is there for
# the same reason: PREACT contributes C++ objects while the executable is
# linked with gfortran.
#
# override, because Make.flags:370 assigns CLIBS unconditionally after reading
# this file. Unlike FFLAGS that is safe here: nothing on the GCC path appends
# to CLIBS -- the one append in Make.flags is inside an Intel-compiler branch
# this build never takes -- so freezing it costs nothing. Upstream's own value
# is kept and only extended.
override CLIBS = -lc -lstdc++ -L$PREFIX/lib -lpspline
# FFLAGS and DFFLAGS are deliberately left to Make.flags; see the header.
# sglib contains K&R-style C wrappers, whose implicit types and declarations
# modern GCC rejects in its default language mode.
override CFLAGS = -c -O -m64 -std=gnu89 -Wno-implicit-int -Wno-implicit-function-declaration
PREFIX = $PREFIX
NETCDF_DIR = $NETCDF_FORTRAN_HOME
NETCDF_FORTRAN_HOME = $NETCDF_FORTRAN_HOME
NETCDF_C_HOME = $NETCDF_C_HOME
PSPLINE_HOME = $PREFIX
BLAS = -L$LAPACK_LIB_DIR $BLAS_FLAGS
# -lpspline is appended here, which reads oddly until you look at the link
# rules. update_state links with
#   $(FC) $(LDFLAGS) -o $@ $< $(LDLIBS) $(NETCDF) $(LAPACK)
# so $(LAPACK) is the last slot on that line, and it is the only end-of-line
# hook shared with the nubeam_comp_exec link, which ends with $(CLIBS). Both
# reach -lpspline early and then libnscrunch.a, libr8bloat.a and libxplasma2.a
# after it; single-pass ld leaves cspline_, ezspline_init1_ and their siblings
# undefined. A trailing occurrence resolves them in both links at once.
LAPACK = -L$LAPACK_LIB_DIR $LAPACK_FLAGS -L$PREFIX/lib -lpspline
NO_MDSPLUS = 1
NO_EDITLIBS = 1
# NUBEAM documents FFTW 2.1.5, which is not ABI/API compatible with FFTW 3.
# Leave it unset unless a compatible FFTW 2 is supplied.
FFTW = \$(NUBEAM_FFTW_FLAGS)
EOF
}

download_ntcc_module() {
  local module="$1"
  local destination="$NTCC_SOURCE_DIR/$module"
  local archive="$BUILD_DIR/$module.tar.gz"
  local stage="$BUILD_DIR/$module.extract"
  local url="https://w3.pppl.gov/rib/repositories/NTCC/files/$module.tar.gz"

  [[ -d "$destination" ]] && return 0
  ((ACCEPT_NTCC_TERMS)) || die "missing $module source. Read and accept the NTCC agreement, then rerun with --accept-ntcc-terms"

  note "downloading NTCC $module source after explicit agreement acceptance"
  mkdir -p "$NTCC_SOURCE_DIR" "$stage"
  curl --fail --location --show-error --silent "$url" -o "$archive" ||
    die "NTCC did not provide the $module download; obtain it manually from https://w3.pppl.gov/NTCC/ and place its extracted source in $destination"
  if file "$archive" | grep -qi 'HTML'; then
    die "NTCC returned an HTML page instead of $module source. Download it manually and extract it to $destination"
  fi
  tar -xf "$archive" -C "$stage" || die "unrecognized NTCC archive for $module; extract it manually to $destination"

  local candidate
  if [[ -f "$stage/Makefile" && -d "$stage/share" ]]; then
    candidate="$stage"
  else
    candidate="$(find "$stage" -type f -name Makefile -print | sed 's#/Makefile$##' | head -n 1 || true)"
  fi
  [[ -n "$candidate" ]] || die "downloaded $module archive has no recognizable Makefile; extract it manually to $destination"
  mkdir -p "$destination"
  cp -R "$candidate"/. "$destination"/
  rm -rf "$stage" "$archive"
}

find_module_root() {
  local module="$1" module_lower candidate
  module_lower="$(printf '%s' "$module" | tr '[:upper:]' '[:lower:]')"
  candidate="$NTCC_SOURCE_DIR/$module"
  [[ -f "$candidate/Makefile" ]] && { printf '%s\n' "$candidate"; return 0; }
  candidate="$(find "$NTCC_SOURCE_DIR" -maxdepth 3 -type f -name Makefile -print | while read -r makefile; do
    parent="${makefile%/Makefile}"
    name_lower="$(printf '%s' "${parent##*/}" | tr '[:upper:]' '[:lower:]')"
    [[ "$name_lower" == "$module_lower" ]] && { printf '%s\n' "$parent"; break; }
  done)"
  [[ -n "$candidate" ]] || die "could not identify a build root for NTCC module $module under $NTCC_SOURCE_DIR"
  printf '%s\n' "$candidate"
}

# The NTCC pspline archive ships a Makefile whose module list still names
# pspline_calls.o, but the distribution no longer contains pspline_calls.f.
# GNU make then aborts libpspline.a with "no rule to make target", and the
# top-level "for m in ..." loop discards that status, leaving a stub archive.
build_pspline_archive() {
  local module_root="$1"
  local srcdir="$module_root/pspline"
  local objs=() mods=()

  [[ -d "$srcdir" ]] || die "NTCC pspline source directory not found: $srcdir"
  local -a module_specs=(
    "ezspline_mod:ezspline.mod ezspline_obj.mod ezspline_type.mod"
    "pspline_calls:pspline_calls.mod"
    "czspline_pointer_types:czspline_pointer_types.mod"
  )
  local spec stem
  for spec in "${module_specs[@]}"; do
    stem="${spec%%:*}"
    if compgen -G "$srcdir/$stem.[fF]90" >/dev/null || compgen -G "$srcdir/$stem.[fF]" >/dev/null; then
      objs+=("$stem.o")
      # shellcheck disable=SC2206  # deliberate word split: a space-separated list
      mods+=(${spec#*:})
    else
      note "skipping pspline module $stem.o; no source in $srcdir"
    fi
  done

  note "rebuilding complete pspline archive"
  rm -f "$BUILD_DIR/pspline/lib/libpspline.a"
  ( IFS=' '
    make -C "$srcdir" libs "OBJ=$BUILD_DIR/pspline" \
      "Mobjs=${objs[*]}" "Mnams=${mods[*]}" )

  # A stub archive links cleanly against nothing; fail here instead of at the
  # final NUBEAM link with several hundred unresolved spline symbols.
  # ELF has no leading underscore on global symbols -- the macOS spelling of
  # this pattern would match nothing and pass a truncated archive silently.
  local defined symbol
  defined="$(nm -g "$BUILD_DIR/pspline/lib/libpspline.a" 2>/dev/null || true)"
  for symbol in mkbicub_ ezspline_init2_ xlookup_; do
    grep -q " T $symbol\$" <<<"$defined" ||
      die "libpspline.a is missing $symbol; see $LOG_FILE"
  done
}

install_ntcc_artifacts() {
  local module="$1"
  local module_root="$2"
  local module_build="$BUILD_DIR/$module"
  local library

  mkdir -p "$PREFIX/include" "$PREFIX/lib" "$PREFIX/mod"
  shopt -s nullglob
  for library in "$module_build/lib"/*.a; do
    cp "$library" "$PREFIX/lib/"
  done
  # NUBEAM 2021 calls this historical SGLIB archive jclib, while the current
  # archive is named libjc.a. Preserve both link names in the private prefix.
  if [[ -f "$PREFIX/lib/libjc.a" && ! -e "$PREFIX/lib/libjclib.a" ]]; then
    ln -s libjc.a "$PREFIX/lib/libjclib.a"
  fi
  for library in "$module_build/mod"/*; do
    [[ -f "$library" ]] && cp "$library" "$PREFIX/mod/"
  done
  if [[ -d "$module_root/include" ]]; then
    cp -R "$module_root/include"/. "$PREFIX/include/"
  fi
  for library in "$module_root/include/cpp"/*.h; do
    [[ -f "$library" ]] && cp "$library" "$PREFIX/include/"
  done
  shopt -u nullglob
}

build_ntcc_module() {
  local module="$1" module_root
  module_root="$(find_module_root "$module")"
  note "building NTCC dependency $module from $module_root"
  write_make_local "$module_root"
  GENERATED_CONFIGS+=("$module_root/share/Make.local")
  write_manifest
  # Keep values containing spaces (BLAS/LAPACK) in Make.local: GNU Make
  # corrupts them when forwarding command-line definitions to recursive make.
  make -C "$module_root" libs "OBJ=$BUILD_DIR/$module"
  [[ "$module" == pspline ]] && build_pspline_archive "$module_root"
  install_ntcc_artifacts "$module" "$module_root"
}

# install_ntcc_artifacts copies each module's archives into the shared prefix,
# so when two modules ship an archive of the same name the last one installed
# wins outright. Only portlib is duplicated, and the PSPLINE copy carries
# members the NUBEAM tree's copy does not. Back-fill rather than reorder: every
# member the prefix already has keeps the implementation the main tree built.
backfill_shared_archives() {
  local basename_a source_archive target missing scratch
  for basename_a in libportlib.a portlib.a; do
    target="$PREFIX/lib/$basename_a"
    [[ -f "$target" ]] || continue
    for source_archive in "$BUILD_DIR"/*/lib/"$basename_a"; do
      [[ -f "$source_archive" ]] || continue
      [[ "$source_archive" -ef "$target" ]] && continue
      missing="$(comm -23 \
        <(ar t "$source_archive" | sort -u) \
        <(ar t "$target" | sort -u))"
      [[ -n "$missing" ]] || continue
      note "back-filling $(wc -w <<<"$missing" | tr -d ' ') member(s) into $basename_a"
      scratch="$(mktemp -d)"
      ( cd "$scratch"
        IFS=' '
        # shellcheck disable=SC2086  # deliberate word split: a member list
        ar x "$source_archive" $missing && ar r "$target" $missing )
      ranlib "$target"
      rm -rf "$scratch"
    done
  done
}

# The top-level `exec` target drives submodules through
#   for m in $(MEXEC); do (cd $m; $(MAKE) exec); done
# which discards each exit status -- the same construct that let the truncated
# libpspline.a through. Build each program by naming its target directly and
# assert the binary afterwards.
build_aux_executables() {
  local preact_root preact_build generator_build src source_file

  note "building update_state"
  make -C "$ROOT_DIR/update_state" exec "OBJ=$BUILD_DIR/nubeam"
  [[ -x "$BUILD_DIR/nubeam/test/update_state" ]] ||
    die "update_state was not created; see $LOG_FILE"

  preact_root="$(find_module_root preact)/preact"
  preact_build="$BUILD_DIR/preact"
  note "building preact_init"
  make -C "$preact_root" "$preact_build/test/preact_init" \
    "OBJ=$preact_build" "THISLIB=$preact_build/lib/libpreact.a"
  [[ -x "$preact_build/test/preact_init" ]] ||
    die "preact_init was not created; see $LOG_FILE"

  # The Plasma State generator. The NTCC archive ships no main program for it;
  # plasma_state_test.f90 in the vendored 2021 server tree is the complete
  # source. Its own Makefile is unusable (absolute paths, MDSplus, termcap) and
  # a top-level directory here would be swept into the main tree's MEXEC list,
  # so compile and link it directly.
  src="$ROOT_DIR/vendor/server-ntcc-2021/plasma_state_test"
  generator_build="$BUILD_DIR/generator"
  [[ -f "$src/plasma_state_test.f90" ]] ||
    die "Plasma State generator source not found: $src/plasma_state_test.f90"
  note "building plasma_state_test"
  mkdir -p "$generator_build"
  ( cd "$generator_build"
    IFS=' '
    for source_file in ps_momtest.F90 plasma_state_test.f90; do
      # shellcheck disable=SC2086  # deliberate word split: a flag list
      "$FC_WRAPPER" -c -O -m64 -cpp -I"$PREFIX/mod" -I"$PREFIX/include" \
        -I"$NETCDF_FORTRAN_INC" \
        -o "${source_file%.*}.o" "$src/$source_file"
    done
    "$FC_WRAPPER" -o plasma_state_test plasma_state_test.o ps_momtest.o \
      -L"$PREFIX/lib" -lplasma_state -lps_xplasma2 -lplasma_state_kernel \
      -lxplasma2 -lgeqdsk_mds -lmdstransp -lvaxonly -lnscrunch -lfluxav \
      -lr8bloat -lpspline -lezcdf -llsode -llsode_linpack -lsmlib -lcomput \
      -lportlib \
      -L"$NETCDF_FORTRAN_LIB" -lnetcdff -L"$NETCDF_C_HOME/lib" -lnetcdf \
      -L"$LAPACK_LIB_DIR" $LAPACK_FLAGS -lstdc++ )
  [[ -x "$generator_build/plasma_state_test" ]] ||
    die "plasma_state_test was not created; see $LOG_FILE"

  cp "$BUILD_DIR/nubeam/test/update_state" "$PREFIX/bin/update_state"
  cp "$preact_build/test/preact_init" "$PREFIX/bin/preact_init"
  cp "$generator_build/plasma_state_test" "$PREFIX/bin/plasma_state_test"
}

# nubeam_comp_exec requires both PREACTDIR and ADASDIR and calls bad_exit when
# either is unset, so a usable installation has to ship them. Both are runtime
# data directories the table code writes back into: PREACT and ADAS generate
# missing reaction tables on demand and cache them here, so both stay writable.
stage_reaction_databases() {
  local preact_root preact_data adas_dir preact_dir

  preact_root="$(find_module_root preact)"
  preact_data="$preact_root/preact"
  preact_dir="$PREFIX/share/preact"
  adas_dir="$PREFIX/share/adas"

  [[ -f "$preact_data/ORNL6086.DAT" ]] ||
    die "PREACT Aladdin cross-section data not found: $preact_data/ORNL6086.DAT"
  [[ -d "$preact_root/data" ]] ||
    die "ADAS data tree not found: $preact_root/data"

  note "initializing PREACT reaction-table directory: $preact_dir"
  mkdir -p "$preact_dir"
  ( cd "$preact_data" &&
    PREACTDIR="$preact_dir" PATH="$PREFIX/bin:$PATH" \
      ./preactinit DATA "$preact_data" PATH "$PREFIX/bin" )
  [[ -f "$preact_dir/data/ORNL6086.DAT" ]] ||
    die "preactinit did not populate $preact_dir; see $LOG_FILE"

  note "staging ADAS data directory: $adas_dir"
  mkdir -p "$adas_dir/tables"
  if [[ ! -e "$adas_dir/data" ]]; then
    ln -s "$preact_root/data" "$adas_dir/data"
  fi
}

download_ntcc_module pspline
download_ntcc_module preact
download_ntcc_module xplasma

build_ntcc_module pspline
build_ntcc_module preact
build_ntcc_module xplasma

write_make_local "$ROOT_DIR"

note "checking NUBEAM link dependencies"
make -C "$ROOT_DIR" checklibs "OBJ=$BUILD_DIR/nubeam" ||
  die "NUBEAM dependency check failed; see $LOG_FILE"
missing_links="$(make -C "$ROOT_DIR" checklibs "OBJ=$BUILD_DIR/nubeam" | awk '/--- NEED/{print $3}' | sort -u)"
if [[ -n "$missing_links" ]]; then
  printf '%s\n' "NUBEAM still requires these libraries after building PSPLINE, PREACT, and XPLASMA:" "$missing_links" >&2
  die "download the NTCC package(s) that provide the listed libraries into $NTCC_SOURCE_DIR, then rerun this installer"
fi

# The 2021 NUBEAM distribution's nubeam.cpp calls a PSPLINE C API that no
# PSPLINE in this tree provides. There are three generations of that API:
#
#   2021 nubeam.cpp       czspline_init1(&n1, bcs1, &ier)            no handle
#   2018 server tree      F77NAME(czspline_init1_r8)(handle, ...)    handle, typed
#   current PPPL PSPLINE  F77NAME(czspline_init1)(handle, ...)       handle, untyped
#
# The in-tree 2021 file matches neither, and it is a signature mismatch rather
# than a name-mangling one -- forcing identity mangling still leaves four
# parameters against three arguments -- so no preprocessor setting reconciles
# it. The vendored 2021 server tree ships the same file already written against
# the handle-based API, differing only in these five calls.
#
# Bridging the last step is mechanical and numerically exact: upstream removed
# the r4 variants outright, so the surviving untyped entry point *is* the r8
# one. `nm` on the built archive confirms it -- ezspline_init1_ is present and
# no _r4 symbol exists anywhere in it. Stripping the suffix therefore selects
# the same double-precision routine the server tree asked for by name.
#
# Upstream's source is not edited: the adapted copy is staged in the build
# directory, compiled there, and inserted into the archive before `make` runs,
# so make finds the member newer than the .cpp and leaves it alone.
stage_nubeam_cpp_replacement() {
  local original="$ROOT_DIR/nubeam/nubeam.cpp"
  local adapted="$ROOT_DIR/vendor/server-ntcc-2021/nubeam/nubeam.cpp"
  local staged="$BUILD_DIR/src/nubeam.cpp"
  local objdir="$BUILD_DIR/nubeam/obj/nubeam"
  local libdir="$BUILD_DIR/nubeam/lib"
  local header="$PREFIX/include/czspline_capi.h"

  [[ -f "$original" ]] || die "NUBEAM source not found: $original"
  # If upstream ever ships a nubeam.cpp already written against the installed
  # API, this substitution is unnecessary and must not happen silently.
  if ! grep -q 'czspline_init1(&n1' "$original"; then
    note "nubeam.cpp already matches the installed PSPLINE API; no substitution"
    NUBEAM_CPP_SOURCE="$original"
    return 0
  fi
  [[ -f "$adapted" ]] ||
    die "nubeam.cpp calls a PSPLINE API this PSPLINE does not provide, and the adapted copy is not in this tree at $adapted. Supply a nubeam.cpp matching the installed PSPLINE, or a PSPLINE matching this one."
  [[ -f "$header" ]] || die "PSPLINE C header not installed: $header"

  mkdir -p "$BUILD_DIR/src" "$objdir" "$libdir"
  if grep -q 'czspline_init1_r8' "$header"; then
    # The installed PSPLINE still carries the typed entry points, so the
    # server tree's file applies unchanged.
    note "substituting the vendored server-tree nubeam.cpp (typed PSPLINE API)"
    cp -f "$adapted" "$staged"
  else
    note "substituting the vendored server-tree nubeam.cpp, with the r4/r8 suffixes upstream removed"
    sed -E 's/F77NAME\((czspline_[a-z0-9_]+)_r8\)/F77NAME(\1)/g' "$adapted" > "$staged"
    ! grep -q 'czspline_[a-z0-9_]*_r8' "$staged" ||
      die "some typed PSPLINE calls survived the rewrite in $staged"
  fi
  NUBEAM_CPP_SOURCE="$adapted"

  # The same command Make.flags' cxx_proc builds, run from the source directory
  # so that -I./ and -I../include resolve as they do under make.
  ( cd "$ROOT_DIR/nubeam" &&
    "$CXX" -c -O -m64 -I./ -I../include -I../include/cpp \
      -I"$PREFIX/include" -I"$NETCDF_FORTRAN_INC" \
      -D__LINUX -D__NOMDSPLUS \
      -o "$objdir/nubeam.o" "$staged" )
  [[ -s "$objdir/nubeam.o" ]] || die "the substituted nubeam.cpp did not compile; see $LOG_FILE"

  # Compiling proves the header agrees; this proves the library does. Without
  # it a mismatch would surface a thousand lines later as an unresolved symbol
  # in the final link, naming neither this file nor the substitution.
  # Capture both sides once. Under `pipefail`, `nm | grep -q` fails whenever
  # grep exits on its first match and nm takes SIGPIPE -- the same trap
  # build_pspline_archive documents, which is why that one captures too.
  local defined needed symbol
  defined="$(nm -g "$PREFIX/lib/libpspline.a" 2>/dev/null || true)"
  needed="$(nm -u "$objdir/nubeam.o" | awk '/czspline/{print $NF}')"
  for symbol in $needed; do
    grep -q " T $symbol\$" <<<"$defined" ||
      die "the substituted nubeam.cpp needs $symbol, which the installed PSPLINE does not define"
  done

  ar r "$libdir/libnubeam.a" "$objdir/nubeam.o"
  ranlib "$libdir/libnubeam.a"
  # Make compares the archive member against the .cpp, so the member has to be
  # the newer of the two for make to accept it.
  touch "$libdir/libnubeam.a"
}

# The provenance record install/check_nubeam.py looks for. Separate from
# .nubeam-install-manifest, which is the uninstaller's list of what to remove:
# this one answers "what produced these binaries", and without it the checker's
# build-record layer reports SKIP. It also carries which nubeam.cpp the archive
# holds, which is the first question anyone comparing two NUBEAM runs will ask.
#
# The NUBEAM distribution is a tarball rather than a checkout, so there is no
# revision to record; the executable digests are what identifies this build.
write_install_record() {
  local record="$PREFIX/vaft-external-install.json"
  "$PYTHON" - "$record" <<EOF
import hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
prefix = Path("$PREFIX")
source = Path("$NUBEAM_CPP_SOURCE")
record = {
    "code": "nubeam",
    "installer": "install/nubeam/linux.sh",
    "prefix": str(prefix),
    "source": "$ROOT_DIR",
    "source_revision": None,
    "source_dirty": False,
    "build_dir": "$BUILD_DIR",
    "build_in_place": True,
    "nubeam_cpp": {
        "compiled_from": str(source),
        "sha256": sha(source),
        "in_tree_copy_untouched": "$ROOT_DIR/nubeam/nubeam.cpp",
        "substituted": str(source) != "$ROOT_DIR/nubeam/nubeam.cpp",
    },
    "compiler": {"fortran": "$FC", "c": "$CC", "cxx": "$CXX",
                 "wrapper": "$FC_WRAPPER"},
    "dependency_providers": {
        "netcdf_fortran_home": "$NETCDF_FORTRAN_HOME",
        "netcdf_c_home": "$NETCDF_C_HOME",
        "lapack_lib_dir": "$LAPACK_LIB_DIR",
    },
    "platform": "$PLATFORM",
    "host": platform.node(),
    "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "log": "$LOG_FILE",
    "executables": {
        p.name: {"path": str(p), "sha256": sha(p), "size": p.stat().st_size}
        for p in sorted((prefix / "bin").iterdir()) if p.is_file()
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print("wrote", sys.argv[1])
EOF
}

note "building serial NUBEAM"
NUBEAM_CPP_SOURCE=""
stage_nubeam_cpp_replacement
make -C "$ROOT_DIR" libs "OBJ=$BUILD_DIR/nubeam"
make -C "$ROOT_DIR/nubeam_comp_exec" exec "OBJ=$BUILD_DIR/nubeam"
NUBEAM_EXEC="$BUILD_DIR/nubeam/test/nubeam_comp_exec"
[[ -x "$NUBEAM_EXEC" ]] || die "serial executable was not created: $NUBEAM_EXEC"
install_ntcc_artifacts nubeam "$ROOT_DIR"
cp "$NUBEAM_EXEC" "$PREFIX/bin/nubeam_comp_exec"

# Every module has now installed its archives, so this is the first point at
# which the prefix copy of a duplicated archive is final.
backfill_shared_archives
build_aux_executables
stage_reaction_databases

write_manifest
write_install_record

note "NUBEAM installed successfully"
note "binaries: $PREFIX/bin"
note "nubeam.cpp built from: $NUBEAM_CPP_SOURCE"
note "PREACT tables: $PREFIX/share/preact"
note "ADAS data: $PREFIX/share/adas"
note "validate this build against the shipped reference cases with:"
note "  bash $SCRIPT_DIR/run-local-validation.sh --nubeam-root $ROOT_DIR --case d3d"
note "point \$NUBEAMHOME at: $PREFIX"

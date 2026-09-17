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
# Prefix ownership: what --uninstall may remove, and what an install may claim.
# shellcheck disable=SC1091
. "$SCRIPT_DIR/_external_code_common.sh"
MANIFEST_NAME="$VAFT_EXTERNAL_MANIFEST_NAME"
PREFIX_CREATED=0

#: The executables install/check_gpec.py looks for, in its order.
PROGRAMS=(dcon match rdcon rmatch stride gpec)
#: `all` minus `v` (a no-op report) and `xdraw` (X11).
TARGETS=(neededdeps equil lsode zlange zvode orbit vacuum pentrc dcon match
         rdcon rmatch multi sum slayer coil gpec stride)

SOURCE="${GPEC_SOURCE_DIR:-}"
PREFIX=""
LAPACK_HOME=""
NETCDF_F_HOME=""
NETCDF_C_HOME=""
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
  --netcdf-c-home PATH         prefix holding libnetcdf, when netCDF-C is packaged
                               apart from netCDF-Fortran (default: derived from
                               nc-config, and only when they are in fact apart)
  --netcdf-include PATH        netCDF include directory (default: derived likewise)
  --allow-dirty                build a tree with uncommitted tracked changes; the diff
                               digest and file list are then recorded in the manifest
  --jobs N                     parallel build jobs (default: all cores)
  --skip-tests                 do not run install/check_gpec.py after building
  --check-only                 run install/check_gpec.py and change nothing
  --uninstall                  remove what this script installed into the prefix, and the
                               prefix itself if this script created it and it is then empty
  -h, --help

The prefix is what $GPECHOME should point at: VAFT resolves $GPECHOME/bin/dcon and
its five siblings. This script prints the export line; it edits no shell profile.

GPEC builds in place, so its objects and its own bin/ land inside <source> and stay
there. --uninstall touches only the prefix; run `make clean` in <source>/install
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
    --netcdf-c-home) (($# >= 2)) || die '--netcdf-c-home needs a path'; NETCDF_C_HOME="$2"; shift 2 ;;
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
BUILD_DIR="$SOURCE/install"

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
  vaft_external_uninstall_prefix "$PREFIX" gpec "$PYTHON"
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
# --- netCDF-C, but only when it is packaged apart from netCDF-Fortran --------
# upstream's DEFAULTS.inc, given NETCDF_FORTRAN_HOME and nothing else, links
# -lnetcdf out of netCDF-Fortran's own library directory. On a distro that is
# right, because the two ship together. Homebrew puts them in separate kegs, so
# the link line asks for -lnetcdf in a directory that holds only libnetcdff:
#
#     ld: library 'netcdf' not found
#     make: *** [rmatch] Error 1
#
# NETCDF_C_HOME is the branch upstream provides for exactly this, so it is set
# only when the split is real. Setting it unconditionally would break Debian
# multiarch, where nc-config reports /usr but libnetcdf is under
# /usr/lib/<triplet> rather than the /usr/lib that DEFAULTS.inc would derive.
if [[ -z "$NETCDF_C_HOME" ]] && ! compgen -G "$NETCDF_F_HOME/libnetcdf.*" >/dev/null; then
  nc_config="$(command -v nc-config || true)"
  if [[ -n "$nc_config" ]]; then
    candidate="$("$nc_config" --prefix 2>/dev/null || true)"
    if [[ -n "$candidate" ]] && compgen -G "$candidate/lib/libnetcdf.*" >/dev/null; then
      NETCDF_C_HOME="$candidate"
      note "netCDF-C is packaged apart from netCDF-Fortran; taking it from $NETCDF_C_HOME"
    fi
  fi
  [[ -n "$NETCDF_C_HOME" ]] || die "netCDF-Fortran is in $NETCDF_F_HOME but libnetcdf is not, and nc-config did not name a prefix that holds it. Pass --netcdf-c-home."
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
[[ -z "$NETCDF_C_HOME" ]] || export NETCDF_C_HOME="$NETCDF_C_HOME"
export NETCDFINC="$NETCDF_INC"
export FFLAGS="-fallow-argument-mismatch -O2"
export OMPFLAG=-fopenmp RECURSFLAG=-frecursive
# An rpath, so the executables find the netCDF-Fortran they were linked against
# rather than whichever one LD_LIBRARY_PATH names first. Without it a machine
# carrying a second, differently-compiled netCDF -- an ifort build on
# LD_LIBRARY_PATH is the ordinary case on a cluster -- links correctly and then
# dies at run time with `undefined symbol: __netcdf_MOD_nf90_put_var_*`, naming
# neither the library nor the variable that chose it.
#
# --disable-new-dtags is the load-bearing half on ELF: current binutils emits
# DT_RUNPATH, which LD_LIBRARY_PATH overrides, so the rpath would be present
# and ignored. DT_RPATH is searched first.
#
# Note what that costs, because it is wider than the case that motivated it.
# ELF has no per-library rpath, so the entry is a whole directory -- usually the
# multiarch one -- and DT_RPATH then wins over LD_LIBRARY_PATH for *everything*
# resolvable there: LAPACK, BLAS, HDF5, libgfortran. An operator who puts a
# tuned OpenBLAS on LD_LIBRARY_PATH will find these six binaries ignoring it.
# That is the deliberate trade: an ABI-incompatible library substituted at run
# time is a crash, while a deliberate override is a preference.
#
# Linux only. Mach-O has no DT_RUNPATH and does not resolve through
# LD_LIBRARY_PATH, and ld64 rejects --disable-new-dtags outright, so applying
# this on Darwin would fail the first link rather than harden it.
LDFLAGS="-fopenmp"
if [[ "$PLATFORM" == linux-* ]]; then
  LDFLAGS="$LDFLAGS -Wl,--disable-new-dtags -Wl,-rpath,$NETCDF_F_HOME"
fi
export LDFLAGS
unset MKLROOT ACML_HOME NETCDFHOME NETCDF_DIR F90HOME X11_HOME || true

# Refuses a directory that holds somebody else's files, and records whether the
# prefix is this script's creation, before anything is written into it.
vaft_external_claim_prefix "$PREFIX" gpec "$PYTHON"
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
# Values reach Python through the environment and the heredoc is quoted, so no
# shell text is ever parsed as Python source. Interpolating them broke on the
# first quoted path in `git status --porcelain` ("""...name"""" is a
# SyntaxError) -- after bin/ was installed and before the manifest existed.
VAFT_MANIFEST_PREFIX="$PREFIX" \
VAFT_MANIFEST_PREFIX_CREATED="$PREFIX_CREATED" \
VAFT_MANIFEST_PROGRAMS_LINE="$PROGRAMS_LINE" \
VAFT_MANIFEST_SOURCE="$SOURCE" \
VAFT_MANIFEST_REVISION="$REVISION" \
VAFT_MANIFEST_DESCRIBED="$DESCRIBED" \
VAFT_MANIFEST_BRANCH="$BRANCH" \
VAFT_MANIFEST_REMOTE="$REMOTE" \
VAFT_MANIFEST_DIRTY_DIFF_SHA="$DIRTY_DIFF_SHA" \
VAFT_MANIFEST_BUILD_DIR="$BUILD_DIR" \
VAFT_MANIFEST_MAKE_COMMAND="$MAKE_COMMAND" \
VAFT_MANIFEST_TARGETS_LINE="$TARGETS_LINE" \
VAFT_MANIFEST_FFLAGS="$FFLAGS" \
VAFT_MANIFEST_FC_PATH="$FC_PATH" \
VAFT_MANIFEST_FC_VERSION="$FC_VERSION" \
VAFT_MANIFEST_LAPACK_HOME="$LAPACK_HOME" \
VAFT_MANIFEST_NETCDF_F_HOME="$NETCDF_F_HOME" \
VAFT_MANIFEST_NETCDF_C_HOME="$NETCDF_C_HOME" \
VAFT_MANIFEST_NETCDF_INC="$NETCDF_INC" \
VAFT_MANIFEST_PLATFORM="$PLATFORM" \
VAFT_MANIFEST_LOG="$LOG" \
VAFT_MANIFEST_DIRTY_FILES="$DIRTY_FILES" \
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
programs = env["PROGRAMS_LINE"].split()
record = {
    "code": "gpec",
    "installer": "install/install_gpec.sh",
    "prefix": str(prefix),
    # What --uninstall may remove: these files, this script's build logs and
    # its two records; the directory itself only when prefix_created is true.
    "prefix_created": bool(int(env["PREFIX_CREATED"])),
    "installed_files": ["bin/" + name for name in programs],
    "source": env["SOURCE"],
    "source_revision": env["REVISION"],
    "source_described": env["DESCRIBED"],
    "source_branch": env["BRANCH"],
    "source_remote": env["REMOTE"],
    "source_dirty": bool(env["DIRTY_FILES"].strip()),
    "source_dirty_files": [l for l in env["DIRTY_FILES"].splitlines() if l.strip()],
    "source_dirty_diff_sha256": env["DIRTY_DIFF_SHA"] or None,
    "build_dir": env["BUILD_DIR"],
    "build_in_place": True,
    "make_command": env["MAKE_COMMAND"],
    "make_targets": env["TARGETS_LINE"].split(),
    "openmp": True,
    "fflags": env["FFLAGS"],
    "compiler": {"fortran": env["FC_PATH"], "version": env["FC_VERSION"]},
    "dependency_providers": {
        "lapack_home": env["LAPACK_HOME"],
        "netcdf_fortran_home": env["NETCDF_F_HOME"],
        "netcdf_c_home": env["NETCDF_C_HOME"] or None,
        "netcdf_include": env["NETCDF_INC"],
    },
    "platform": env["PLATFORM"],
    "host": platform.node(),
    "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "log": env["LOG"],
    "executables": {
        name: {
            "path": str(prefix / "bin" / name),
            "sha256": sha(prefix / "bin" / name),
            "size": (prefix / "bin" / name).stat().st_size,
        }
        for name in programs
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
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

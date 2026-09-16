#!/usr/bin/env bash
# Build and install DCON/GPEC on macOS from a source tree you already hold.
#
# Usage:
#   bash install/install_gpec_macos.sh --source PATH [options]
#   bash install/install_gpec_macos.sh --source PATH --check-only
#   bash install/install_gpec_macos.sh --source PATH --uninstall
#
# GPEC's build system is already environment-driven: install/DEFAULTS.inc picks
# compiler, math and netCDF libraries out of FC, LAPACKHOME, NETCDFHOME and
# friends, and install/makefile passes them down to every module. So this
# script patches nothing -- it sets the environment GPEC asks for, names the
# targets VAFT needs, and collects the executables. The source tree is left as
# you gave it, apart from the object files and module libraries the build
# itself writes, and one vaft-build.log in install/.
#
# What the environment has to say, and why:
#
#   LAPACKHOME=/usr           the SDK ships liblapack.tbd and libblas.tbd,
#                             which resolve to Accelerate. GPEC then links
#                             -llapack -lblas and needs no separate numerical
#                             library. (Upstream's DEFAULTS.inc documents
#                             pointing this at vecLib.framework instead; both
#                             reach the same implementation. --lapack-home
#                             overrides it.)
#   NETCDF_FORTRAN_HOME       Homebrew splits netCDF into netcdf and
#   NETCDF_C_HOME             netcdf-fortran, and DEFAULTS.inc has a branch for
#                             exactly that split: -lnetcdff from one, -lnetcdf
#                             from the other.
#   -fallow-argument-mismatch GPEC carries old interfaces that current gfortran
#                             rejects outright; upstream's own macOS notes call
#                             for this flag.
#
# Setting both LAPACKHOME and a netCDF home also keeps NEEDED_DEPS empty, so
# the build never falls back to compiling OpenBLAS, HDF5 and netCDF from
# submodules -- which would fetch over the network and take far longer than the
# thing being installed.
#
# Two things this script deliberately does not do:
#
#   xdraw is skipped. It wants X11, VAFT never calls it, and on the older build
#   layout the `all` target's mkbin step fails without it.
#
#   Targets are named one by one rather than through `all`. That covers both
#   build layouts in circulation -- the older install/makefile with its own
#   recipes and `.IGNORE:`, and the current one built on TARGETS.inc -- and it
#   omits the modules VAFT has no caller for.
#
# Because the older layout sets `.IGNORE:`, make exits 0 even when a module
# failed to compile. The exit status is therefore not evidence of anything:
# what this script trusts is that each required executable exists afterwards.
#
# VAFT does not vendor GPEC, and this script never clones, fetches, pulls or
# changes the revision of the tree it is given: provenance is something you
# state, not something a script infers (#226).
#
# Apple Silicon macOS. Linux is a separate recipe, tracked in #855.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MANIFEST_NAME="vaft-external-install.json"

# The six VAFT resolves through vaft.code.gpec, and the six check_gpec.py
# requires. The other modules are built only where one of these links them.
REQUIRED_EXECUTABLES=(dcon match rdcon rmatch stride gpec)

# Ordered so each module's libraries exist before something links them.
BUILD_TARGETS=(equil lsode zlange zvode vacuum pentrc coil slayer dcon match rdcon rmatch gpec stride)

SOURCE="${GPEC_SOURCE_DIR:-}"
PREFIX=""
JOBS=""
LAPACK_HOME_OPT=""
NETCDF_C_OPT=""
NETCDF_F_OPT=""
CHECK_ONLY=0
UNINSTALL=0

usage() {
  cat <<'EOF'
Build and install DCON/GPEC on macOS from a source tree you already hold.

  --source PATH       the GPEC checkout to build (or set GPEC_SOURCE_DIR)
  --prefix PATH       where to install; default ~/.local/vaft/gpec
  --jobs N            parallel build jobs (default: all cores)
  --lapack-home PATH  override LAPACKHOME; default /usr (Accelerate via the SDK)
  --netcdf-c PATH     override the netCDF C home
  --netcdf-fortran PATH  override the netCDF Fortran home
  --check-only        run install/check_gpec.py and change nothing
  --uninstall         remove what this script installed into the prefix
  -h, --help

The prefix is what $GPECHOME should point at: VAFT resolves $GPECHOME/bin/dcon
and its siblings. This script prints the export line; it edits no shell
profile.

Prerequisites: brew install gcc netcdf netcdf-fortran
EOF
}

die() { printf 'install_gpec_macos.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --source) (($# >= 2)) || die '--source needs a path'; SOURCE="$2"; shift 2 ;;
    --prefix) (($# >= 2)) || die '--prefix needs a path'; PREFIX="$2"; shift 2 ;;
    --jobs) (($# >= 2)) || die '--jobs needs a number'; JOBS="$2"; shift 2 ;;
    --lapack-home) (($# >= 2)) || die '--lapack-home needs a path'; LAPACK_HOME_OPT="$2"; shift 2 ;;
    --netcdf-c) (($# >= 2)) || die '--netcdf-c needs a path'; NETCDF_C_OPT="$2"; shift 2 ;;
    --netcdf-fortran) (($# >= 2)) || die '--netcdf-fortran needs a path'; NETCDF_F_OPT="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "this recipe is macOS only; Linux is tracked in issue #855"
PLATFORM="darwin-$(uname -m)"

[[ -n "$SOURCE" ]] || die "--source is required: VAFT does not vendor GPEC. Pass the path of a checkout you already hold, or set GPEC_SOURCE_DIR."
SOURCE="$(cd "$SOURCE" 2>/dev/null && pwd -P)" || die "GPEC source tree does not exist: $SOURCE"
INSTALL_DIR="$SOURCE/install"
for marker in install/makefile install/DEFAULTS.inc dcon gpec; do
  [[ -e "$SOURCE/$marker" ]] || die "not a GPEC source tree (missing $marker): $SOURCE"
done

PREFIX="${PREFIX:-$HOME/.local/vaft/gpec}"
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd -P)"
MANIFEST="$PREFIX/$MANIFEST_NAME"
PYTHON="$(command -v python3 || true)"
[[ -n "$PYTHON" ]] || die "python3 is required to write the install manifest"

if ((CHECK_ONLY)); then
  exec "$PYTHON" "$SCRIPT_DIR/check_gpec.py" --source "$SOURCE" --prefix "$PREFIX"
fi

if ((UNINSTALL)); then
  [[ -f "$MANIFEST" ]] || die "no $MANIFEST_NAME under $PREFIX; nothing this script installed is there to remove"
  # ${PREFIX:?} rather than $PREFIX: an empty variable here would make this
  # `rm -rf /bin`. It cannot be empty by this point, but the guard costs
  # nothing and the failure mode is unrecoverable (SC2115).
  rm -rf "${PREFIX:?}/bin" "${MANIFEST:?}"
  note "removed bin/ and $MANIFEST_NAME from $PREFIX"
  note "the source tree was never patched; run 'make clean' in its install/ to drop the object files"
  exit 0
fi

# --- toolchain ---------------------------------------------------------------
FC_PATH="$(command -v gfortran || true)"
[[ -n "$FC_PATH" ]] || die "gfortran not found: brew install gcc"
FC_VERSION="$("$FC_PATH" -dumpversion 2>/dev/null || echo unknown)"
CC_PATH="$(command -v gcc-15 || command -v gcc-14 || command -v gcc || true)"
[[ -n "$CC_PATH" ]] || die "no C compiler found"
command -v git >/dev/null || die "git is required: GPEC stamps its own version from git at compile time"

LAPACKHOME="${LAPACK_HOME_OPT:-/usr}"
SDKROOT_PATH="$(xcrun --show-sdk-path 2>/dev/null || true)"
[[ -n "$SDKROOT_PATH" ]] || die "no macOS SDK: xcode-select --install"
if [[ -z "$LAPACK_HOME_OPT" ]]; then
  # Only meaningful for the default. A --lapack-home the caller chose is theirs
  # to justify, and may legitimately be a framework directory with no lib/.
  [[ -e "$SDKROOT_PATH/usr/lib/liblapack.tbd" ]] \
    || die "the SDK has no liblapack.tbd, so LAPACKHOME=/usr will not link. Pass --lapack-home explicitly."
fi

BREW_PREFIX="$(command -v brew >/dev/null && brew --prefix 2>/dev/null || echo /opt/homebrew)"
NETCDF_C_HOME="${NETCDF_C_OPT:-$BREW_PREFIX/opt/netcdf}"
NETCDF_FORTRAN_HOME="${NETCDF_F_OPT:-$BREW_PREFIX/opt/netcdf-fortran}"
[[ -d "$NETCDF_C_HOME" ]] || die "no netCDF C at $NETCDF_C_HOME: brew install netcdf (or pass --netcdf-c)"
[[ -d "$NETCDF_FORTRAN_HOME" ]] || die "no netCDF Fortran at $NETCDF_FORTRAN_HOME: brew install netcdf-fortran (or pass --netcdf-fortran)"
[[ -e "$NETCDF_FORTRAN_HOME/include/netcdf.mod" ]] \
  || die "$NETCDF_FORTRAN_HOME has no netcdf.mod; that is the netCDF C package, not the Fortran one"

note "gfortran $FC_VERSION at $FC_PATH"
note "LAPACKHOME $LAPACKHOME (Accelerate via the SDK at $SDKROOT_PATH)"
note "netCDF C $NETCDF_C_HOME"
note "netCDF Fortran $NETCDF_FORTRAN_HOME"

REVISION=""; DESCRIBED=""; BRANCH=""; DIRTY_FILES=""
if git -C "$SOURCE" rev-parse --git-dir >/dev/null 2>&1; then
  # --short, because check_build_record compares this against source_revision(),
  # which asks git for the abbreviated form. A full SHA here would report a
  # mismatch against the very tree it was built from.
  REVISION="$(git -C "$SOURCE" rev-parse --short HEAD 2>/dev/null || true)"
  DESCRIBED="$(git -C "$SOURCE" describe --tags --always --dirty 2>/dev/null || true)"
  BRANCH="$(git -C "$SOURCE" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  # Tracked files only: a build drops object files and libraries all over the
  # tree, and those say nothing about whether the sources were modified.
  DIRTY_FILES="$(git -C "$SOURCE" status --porcelain --untracked-files=no 2>/dev/null || true)"
fi

# --- build --------------------------------------------------------------------
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
LOG="$INSTALL_DIR/vaft-build.log"
FFLAGS_USED="-O3 -fallow-argument-mismatch"

# A target absent from this tree is skipped rather than failing: the two
# layouts in circulation do not define the same set, and the check that matters
# is whether the executables exist at the end. Both the makefile and the .inc
# files it includes are searched -- the older layout carries its recipes in the
# makefile itself, the current one in TARGETS.inc.
MAKEFILES=("$INSTALL_DIR/makefile" "$INSTALL_DIR"/*.inc)
has_target() { grep -qhE "^$1[[:space:]]*:" "${MAKEFILES[@]}" 2>/dev/null; }

TARGETS=()
for target in "${BUILD_TARGETS[@]}"; do
  has_target "$target" && TARGETS+=("$target")
done
((${#TARGETS[@]})) || die "install/ defines none of the expected targets; GPEC's build system has changed shape"
# Older trees build harvest from a submodule and link -lharvest. Build it only
# when the makefile still knows the target *and* the submodule is populated --
# its own recipe would otherwise run `git submodule update`, which fetches.
if has_target harvest; then
  if [[ -e "$SOURCE/harvest/Makefile" ]]; then
    TARGETS=(harvest "${TARGETS[@]}")
  else
    die "this tree links -lharvest but harvest/ holds no Makefile. Populate that submodule yourself, the way GPEC's own build instructions describe; this script does not fetch."
  fi
fi

# Parallelism is a property of the tree, not a preference. The current layout
# declares inter-module dependencies in DEPENDENCIES.inc and builds correctly
# under -j. The older one states each module as a bare `cd ../x; make` with no
# ordering between them, so -j races the libraries against the executables that
# link them: rmatch comes out with _dcfode_ undefined because liblsode.a was
# still being written. `.IGNORE:` then swallows the link error. Serial is
# therefore a correctness requirement there, and overrides --jobs rather than
# letting the caller ask for a build that silently loses modules.
if [[ ! -e "$INSTALL_DIR/DEPENDENCIES.inc" ]] && ((JOBS > 1)); then
  note "this tree declares no inter-module dependencies; building serially instead of with $JOBS jobs"
  JOBS=1
fi

note "building ${#TARGETS[@]} targets with $JOBS jobs; log at $LOG"
(
  cd "$INSTALL_DIR"
  export FC="$FC_PATH" CC="$CC_PATH"
  export LAPACKHOME="$LAPACKHOME"
  export NETCDF_C_HOME="$NETCDF_C_HOME"
  export NETCDF_FORTRAN_HOME="$NETCDF_FORTRAN_HOME"
  export FFLAGS="$FFLAGS_USED"
  make v
  make -j "$JOBS" "${TARGETS[@]}"
) >"$LOG" 2>&1 || true   # `.IGNORE:` makes the status meaningless; the executables decide

# --- collect --------------------------------------------------------------------
# Both layouts leave the executable in its module directory; the newer one also
# copies to <source>/bin, and the older one only via mkbin, which needs xdraw.
# Take whichever exists.
mkdir -p "$PREFIX/bin"
INSTALLED=()
MISSING=()
for exe in "${REQUIRED_EXECUTABLES[@]}"; do
  found=""
  for candidate in "$SOURCE/bin/$exe" "$SOURCE/$exe/$exe"; do
    [[ -x "$candidate" && ! -d "$candidate" ]] && { found="$candidate"; break; }
  done
  if [[ -n "$found" ]]; then
    cp -f "$found" "$PREFIX/bin/$exe"
    chmod +x "$PREFIX/bin/$exe"
    INSTALLED+=("$exe")
  else
    MISSING+=("$exe")
  fi
done

if ((${#MISSING[@]})); then
  printf '[FAIL] the build produced no %s; the log'"'"'s last 30 lines:\n' "${MISSING[*]}" >&2
  tail -30 "$LOG" >&2
  die "see $LOG"
fi
note "installed ${#INSTALLED[@]} executables into $PREFIX/bin"

# --- manifest ---------------------------------------------------------------
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
names = """${INSTALLED[*]}""".split()
record = {
    "code": "gpec",
    "installer": "install/install_gpec_macos.sh",
    "prefix": str(prefix),
    "source": "$SOURCE",
    "source_revision": "$REVISION" or None,
    "source_described": "$DESCRIBED" or None,
    "source_branch": "$BRANCH" or None,
    "source_dirty": bool("""$DIRTY_FILES""".strip()),
    "source_dirty_files": [l for l in """$DIRTY_FILES""".splitlines() if l.strip()],
    "source_patched": False,
    "build_targets": """${TARGETS[*]}""".split(),
    "compiler": {"fortran": "$FC_PATH", "version": "$FC_VERSION", "c": "$CC_PATH"},
    "fflags": "$FFLAGS_USED",
    "lapack_home": "$LAPACKHOME",
    "math_libraries": "-llapack -lblas (Accelerate via the SDK)",
    "netcdf_c_home": "$NETCDF_C_HOME",
    "netcdf_fortran_home": "$NETCDF_FORTRAN_HOME",
    "sdkroot": "$SDKROOT_PATH",
    "xdraw": False,
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
        for name in names
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print("wrote", sys.argv[1])
EOF

note "verifying with install/check_gpec.py"
"$PYTHON" "$SCRIPT_DIR/check_gpec.py" --source "$SOURCE" --prefix "$PREFIX" || true
printf '\nPoint VAFT at this build:\n  export GPECHOME=%s\n' "$PREFIX"

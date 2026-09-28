#!/usr/bin/env bash
# Build and install GENRAY from a source tree you already hold.
#
# Usage:
#   bash install/install_genray.sh --source PATH [options]
#   bash install/install_genray.sh --source PATH --check-only
#   bash install/install_genray.sh --source PATH --uninstall
#
# GENRAY is open source (https://github.com/compxco/genray), but VAFT neither
# bundles nor fetches it: which revision was built is a statement the operator
# makes. Clone it yourself and pass its path. This script never clones,
# fetches, pulls or changes the tree it is given.
#
# What it does: export the committed revision (`git archive HEAD`) into a
# temporary build directory, build `xgenray` there with gfortran and netCDF-Fortran,
# install bin/xgenray into a prefix that $GENRAYHOME points at, run
# install/check_genray.py as the build's acceptance, and write
# vaft-external-install.json recording the revision, the make command, the
# compiler and the executable's checksum. No source file is modified and
# uncommitted changes are not built; the only thing written beside the source is
# the default prefix <source>/vaft-install, an untracked directory (pass
# --prefix to put it elsewhere). The temporary build tree is removed afterwards,
# so the prefix holds exactly what --uninstall removes.
#
# Three things about the build, each of which fails quietly if you get it wrong:
#
#   * PGPLOT. GENRAY draws diagnostic plots through PGPLOT and upstream links
#     -lpgplot -lX11. VAFT reads genray.nc and nothing else, so this build links
#     install/genray/pgplot_stub.f -- empty routines -- instead. The physics and
#     genray.nc are unchanged; the plots are gone.
#   * BSPECIAL. Upstream's makefiles pass -Wl,-noinhibit-exec, which makes GNU
#     ld write an executable even when symbols are unresolved. It is cleared
#     here so a missing routine is a failed build, not a binary that crashes on
#     the first call.
#   * netCDF on Homebrew is two kegs. nf-config names only netcdf-fortran's
#     library directory; libnetcdf lives in the other keg, so both -L paths are
#     passed explicitly.
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
. "$SCRIPT_DIR/_external_code_common.sh"
MANIFEST_NAME="$VAFT_EXTERNAL_MANIFEST_NAME"
PREFIX_CREATED=0

SOURCE="${GENRAY_SOURCE_DIR:-}"
PREFIX=""
JOBS=""
SKIP_TESTS=0
CHECK_ONLY=0
UNINSTALL=0
MAKEFILE="makefile_gfortran.Ubuntu"
STUB="$SCRIPT_DIR/genray/pgplot_stub.f"

usage() {
  cat <<'EOF'
Usage: bash install/install_genray.sh --source PATH [options]

  --source PATH   your GENRAY clone to build (or set GENRAY_SOURCE_DIR)
  --prefix PATH   install here (default: <source>/vaft-install)
  --jobs N        parallel build jobs (default: 1)
  --skip-tests    do not run install/check_genray.py after building
  --check-only    run install/check_genray.py and change nothing
  --uninstall     remove what this script installed into the prefix, and the
                  prefix itself if this script created it and it is then empty
  -h, --help

The prefix is what $GENRAYHOME should point at: VAFT resolves
$GENRAYHOME/bin/xgenray. This script prints the export line; it edits no shell
profile. Requirements: gfortran, GNU make, git, netCDF-Fortran (nf-config) and
netCDF-C (nc-config).
EOF
}

die() { printf 'install_genray.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --source) (($# >= 2)) || die '--source needs a path'; SOURCE="$2"; shift 2 ;;
    --prefix) (($# >= 2)) || die '--prefix needs a path'; PREFIX="$2"; shift 2 ;;
    --jobs) (($# >= 2)) || die '--jobs needs a number'; JOBS="$2"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$SOURCE" ]] || die "--source is required: VAFT does not vendor GENRAY. Clone https://github.com/compxco/genray and pass its path, or set GENRAY_SOURCE_DIR."
SOURCE="$(cd "$SOURCE" 2>/dev/null && pwd -P)" || die "GENRAY source tree does not exist: $SOURCE"
# The same markers install/check_genray.py validates.
for marker in genray.f "$MAKEFILE" 00_Genray_Regression_Tests/ci-tests/test-EC-ITER-Centra-CD/gold-genray.nc; do
  [[ -e "$SOURCE/$marker" ]] || die "not a GENRAY source tree (missing $marker): $SOURCE"
done

case "$(uname -s)" in
  Darwin) PLATFORM="darwin-$(uname -m)" ;;
  Linux) PLATFORM="linux-$(uname -m)" ;;
  *) die "unsupported platform: $(uname -s)" ;;
esac
[[ -n "$PREFIX" ]] || PREFIX="$SOURCE/vaft-install"
PREFIX="$(vaft_external_canonical_path "$PREFIX")" || die "cannot resolve the install prefix: $PREFIX"
MANIFEST="$PREFIX/$MANIFEST_NAME"
VAFT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
if vaft_external_is_inside "$PREFIX" "$VAFT_ROOT"; then
  die "the install prefix must be outside the VAFT checkout: $PREFIX is inside $VAFT_ROOT"
fi

PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "$PYTHON" ]] || die "python3 is required (the VAFT environment provides it)"

if ((CHECK_ONLY)); then
  exec env GENRAYHOME="$PREFIX" "$PYTHON" "$SCRIPT_DIR/check_genray.py" --source "$SOURCE" --prefix "$PREFIX"
fi

if ((UNINSTALL)); then
  vaft_external_uninstall_prefix "$PREFIX" genray "$PYTHON"
  exit 0
fi

command -v gfortran >/dev/null || die "gfortran is required (e.g. brew install gcc, apt install gfortran)"
command -v make >/dev/null || die "GNU make is required"
command -v git >/dev/null || die "git is required"
command -v nf-config >/dev/null || die "netCDF-Fortran is required (nf-config not found; brew install netcdf-fortran, apt install libnetcdff-dev)"
command -v nc-config >/dev/null || die "netCDF-C is required (nc-config not found)"

REVISION="$(git -C "$SOURCE" rev-parse --short HEAD 2>/dev/null || true)"
[[ -n "$REVISION" ]] || die "the source tree is not under git, so its revision cannot be recorded"
DESCRIBED="$(git -C "$SOURCE" describe --always 2>/dev/null || echo "$REVISION")"
BRANCH="$(git -C "$SOURCE" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
REMOTE="$(git -C "$SOURCE" remote get-url origin 2>/dev/null || echo "")"
DIRTY_FILES="$(git -C "$SOURCE" status --porcelain --untracked-files=no)"
[[ -z "$DIRTY_FILES" ]] || note "the source tree has uncommitted changes; they are NOT built (git archive HEAD builds $REVISION)"

NF_INCLUDE="$(nf-config --includedir)"
NF_LIBDIR="$(nf-config --prefix)/lib"
NC_LIBDIR="$(nc-config --libdir)"
FC_PATH="$(command -v gfortran)"
FC_VERSION="$("$FC_PATH" --version | head -1)"
[[ -n "$JOBS" ]] || JOBS=1
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer, got: $JOBS"

vaft_external_claim_prefix "$PREFIX" genray "$PYTHON"
mkdir -p "$PREFIX/logs"
LOG="$PREFIX/logs/genray-build-$(date +%Y%m%d-%H%M%S).log"
# Built outside the prefix, in a directory this script creates and removes, so
# --uninstall never meets files the manifest does not list.
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vaft-genray-build.XXXXXX")" || die "cannot create a build directory"
cleanup_build() { [[ -n "${BUILD_DIR:-}" && -d "$BUILD_DIR" && "$BUILD_DIR" == */vaft-genray-build.* ]] && rm -rf -- "$BUILD_DIR"; }
trap cleanup_build EXIT
note "exporting $SOURCE at $REVISION into $BUILD_DIR"
git -C "$SOURCE" archive HEAD | tar -x -C "$BUILD_DIR"
cp -f "$STUB" "$BUILD_DIR/vaft_pgplot_stub.f"

MAKE_COMMAND="make -j1 -f $MAKEFILE <F90 module objects> INCLUDE=$NF_INCLUDE; make -j$JOBS -f $MAKEFILE INCLUDE=$NF_INCLUDE LOCATION='-L$NF_LIBDIR -L$NC_LIBDIR' LIBRARIES='<build>/vaft_pgplot_stub.o -lnetcdff -lnetcdf' BSPECIAL="
note "building xgenray (log: $LOG)"
(
  cd "$BUILD_DIR"
  unset FFLAGS FCFLAGS LDFLAGS
  gfortran -c -O2 vaft_pgplot_stub.f -o vaft_pgplot_stub.o
  # The makefile does not declare the Fortran-module dependencies among its
  # F90 sources, so under -j config_ext.f90 can compile before the
  # const_and_precisions.mod it uses exists. Build those serially, in the
  # makefile's own order, then the rest in parallel.
  make -j1 -f "$MAKEFILE" kind_spec.o const_and_precisions.o quanc8.o config_ext.o green_func_ext.o \
    INCLUDE="$NF_INCLUDE"
  make -j"$JOBS" -f "$MAKEFILE" \
    INCLUDE="$NF_INCLUDE" \
    LOCATION="-L$NF_LIBDIR -L$NC_LIBDIR" \
    LIBRARIES="$BUILD_DIR/vaft_pgplot_stub.o -lnetcdff -lnetcdf" \
    BSPECIAL=""
) >>"$LOG" 2>&1 || die "build failed; see $LOG"
[[ -s "$BUILD_DIR/xgenray" ]] || die "xgenray was not produced at $BUILD_DIR/xgenray (see $LOG)"

mkdir -p "$PREFIX/bin"
cp -f "$BUILD_DIR/xgenray" "$PREFIX/bin/xgenray"
chmod +x "$PREFIX/bin/xgenray"
note "installed bin/xgenray into $PREFIX"

VAFT_MANIFEST_PREFIX="$PREFIX" \
VAFT_MANIFEST_PREFIX_CREATED="$PREFIX_CREATED" \
VAFT_MANIFEST_SOURCE="$SOURCE" \
VAFT_MANIFEST_REVISION="$REVISION" \
VAFT_MANIFEST_DESCRIBED="$DESCRIBED" \
VAFT_MANIFEST_BRANCH="$BRANCH" \
VAFT_MANIFEST_REMOTE="$REMOTE" \
VAFT_MANIFEST_DIRTY_FILES="$DIRTY_FILES" \
VAFT_MANIFEST_BUILD_DIR="$BUILD_DIR" \
VAFT_MANIFEST_MAKE_COMMAND="$MAKE_COMMAND" \
VAFT_MANIFEST_FC_PATH="$FC_PATH" \
VAFT_MANIFEST_FC_VERSION="$FC_VERSION" \
VAFT_MANIFEST_PLATFORM="$PLATFORM" \
VAFT_MANIFEST_LOG="$LOG" \
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
executable = prefix / "bin" / "xgenray"
record = {
    "code": "genray",
    "installer": "install/install_genray.sh",
    "prefix": str(prefix),
    "prefix_created": bool(int(env["PREFIX_CREATED"])),
    "installed_files": ["bin/xgenray"],
    "source": env["SOURCE"],
    "source_revision": env["REVISION"],
    "source_described": env["DESCRIBED"],
    "source_branch": env["BRANCH"],
    "source_remote": env["REMOTE"],
    # Built from `git archive HEAD`, so uncommitted changes never reach the binary.
    "source_dirty_files_not_built": [l for l in env["DIRTY_FILES"].splitlines() if l.strip()],
    "build_dir": "temporary (removed after install)",
    "build_in_place": False,
    "make_command": env["MAKE_COMMAND"],
    "pgplot": "install/genray/pgplot_stub.f (no-op; plots disabled)",
    "compiler": {"fortran": env["FC_PATH"], "version": env["FC_VERSION"]},
    "platform": env["PLATFORM"],
    "host": platform.node(),
    "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "log": env["LOG"],
    "executables": {
        "xgenray": {"path": str(executable), "sha256": sha(executable), "size": executable.stat().st_size}
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("wrote", sys.argv[1])
EOF

ACCEPTANCE_STATUS=0
if ((!SKIP_TESTS)); then
  note "verifying with install/check_genray.py"
  GENRAYHOME="$PREFIX" "$PYTHON" "$SCRIPT_DIR/check_genray.py" --source "$SOURCE" --prefix "$PREFIX" || ACCEPTANCE_STATUS=$?
else
  note "--skip-tests: install/check_genray.py was not run, so this build is unverified"
fi
printf '\nPoint VAFT at this build:\n  export GENRAYHOME=%s\n' "$PREFIX"
if ((ACCEPTANCE_STATUS)); then
  printf '[FAIL] install/check_genray.py rejected this build (status %s). It is installed so the failure can be examined; rerun the check with --check-only.\n' "$ACCEPTANCE_STATUS" >&2
  exit "$ACCEPTANCE_STATUS"
fi

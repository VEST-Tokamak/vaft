#!/usr/bin/env bash
# Build and install CHEASE on macOS from a source tree you already hold.
#
# Usage:
#   bash install/install_chease_macos.sh --source PATH [options]
#   bash install/install_chease_macos.sh --source PATH --check-only
#   bash install/install_chease_macos.sh --source PATH --uninstall
#
# CHEASE's build system knows about macOS -- CHEASE_MACHINE=darwin -- but the
# darwin block in src-f90/Makefile.define_FLAGS only ever defined flags for
# ifort. There is no gfortran branch, so a plain `make` on an Apple Silicon
# machine runs with empty F90FLAGS and fails somewhere that does not name the
# cause. (Upstream does define gfortran branches -- for linux_nohdf5, aug,
# marconi and ubuntu_22.04 -- just not for darwin.)
#
# The flags are supplied on the make command line rather than patched into that
# file. Command-line variables override a makefile's own assignments, so the
# source tree is left exactly as you gave it: nothing to mark, nothing to
# reverse, no way to overwrite an edit of yours, and the revision recorded in
# the manifest still describes the files that were compiled.
#
# What the flags have to say, and why:
#
#   -ffree-line-length-none   CHEASE has free-form lines past gfortran's
#                             132-column default; without it the build dies on
#                             truncated continuations.
#   -Wl,-syslibroot,$SDK      from `xcrun --show-sdk-path`. Homebrew's gfortran
#                             does not find the system libraries by itself on
#                             recent macOS.
#   -framework Accelerate     LAPACK and BLAS, already on every Mac. CHEASE
#                             needs no separate numerical library here.
#   HDF5 emptied              CHEASE builds without it, and VAFT reads the
#                             EQDSK output rather than the HDF5 one.
#
# VAFT does not vendor CHEASE, and this script never clones, fetches, pulls or
# changes the revision of the tree it is given: provenance is something you
# state, not something a script infers (#226).
#
# Apple Silicon macOS. Linux is a separate recipe and still belongs to #226.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MANIFEST_NAME="vaft-external-install.json"

SOURCE="${CHEASE_SOURCE_DIR:-}"
PREFIX=""
JOBS=""
CHECK_ONLY=0
UNINSTALL=0

usage() {
  cat <<'EOF'
Build and install CHEASE on macOS from a source tree you already hold.

  --source PATH     the CHEASE checkout to build (or set CHEASE_SOURCE_DIR)
  --prefix PATH     where to install; default ~/.local/vaft/chease
  --jobs N          parallel build jobs (default: all cores)
  --check-only      run install/check_chease.py and change nothing
  --uninstall       remove what this script installed into the prefix
  -h, --help

The prefix is what $CHEASEHOME should point at: VAFT resolves
$CHEASEHOME/bin/chease. This script prints the export line; it edits no shell
profile, and it writes nothing into the source tree beyond the object files
the build itself leaves there and one vaft-build.log beside them.
EOF
}

die() { printf 'install_chease_macos.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --source) (($# >= 2)) || die '--source needs a path'; SOURCE="$2"; shift 2 ;;
    --prefix) (($# >= 2)) || die '--prefix needs a path'; PREFIX="$2"; shift 2 ;;
    --jobs) (($# >= 2)) || die '--jobs needs a number'; JOBS="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "this recipe is macOS only; Linux belongs to issue #226"
PLATFORM="darwin-$(uname -m)"

[[ -n "$SOURCE" ]] || die "--source is required: VAFT does not vendor CHEASE. Pass the path of a checkout you already hold, or set CHEASE_SOURCE_DIR."
SOURCE="$(cd "$SOURCE" 2>/dev/null && pwd -P)" || die "CHEASE source tree does not exist: $SOURCE"
SRC90="$SOURCE/src-f90"
for marker in src-f90/Makefile src-f90/Makefile.define_FLAGS src-f90/Makefile.define_MACHINE; do
  [[ -e "$SOURCE/$marker" ]] || die "not a CHEASE source tree (missing $marker): $SOURCE"
done

PREFIX="${PREFIX:-$HOME/.local/vaft/chease}"
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd -P)"
MANIFEST="$PREFIX/$MANIFEST_NAME"
PYTHON="$(command -v python3 || true)"
[[ -n "$PYTHON" ]] || die "python3 is required to write the install manifest"

if ((CHECK_ONLY)); then
  exec "$PYTHON" "$SCRIPT_DIR/check_chease.py" --source "$SOURCE" --prefix "$PREFIX"
fi

if ((UNINSTALL)); then
  [[ -f "$MANIFEST" ]] || die "no $MANIFEST_NAME under $PREFIX; nothing this script installed is there to remove"
  # ${PREFIX:?} rather than $PREFIX: an empty variable here would make this
  # `rm -rf /bin`. It cannot be empty by this point, but the guard costs
  # nothing and the failure mode is unrecoverable (SC2115).
  rm -rf "${PREFIX:?}/bin" "${MANIFEST:?}"
  note "removed bin/ and $MANIFEST_NAME from $PREFIX"
  note "the source tree was never modified, so there is nothing to undo there"
  exit 0
fi

# --- toolchain ---------------------------------------------------------------
FC_PATH="$(command -v gfortran || true)"
[[ -n "$FC_PATH" ]] || die "gfortran not found: brew install gcc"
FC_VERSION="$("$FC_PATH" -dumpversion 2>/dev/null || echo unknown)"
SDKROOT_PATH="$(xcrun --show-sdk-path 2>/dev/null || true)"
[[ -n "$SDKROOT_PATH" ]] || die "no macOS SDK: xcode-select --install"
note "gfortran $FC_VERSION at $FC_PATH"
note "SDK $SDKROOT_PATH"

REVISION=""; DESCRIBED=""; BRANCH=""; DIRTY_FILES=""
if command -v git >/dev/null && git -C "$SOURCE" rev-parse --git-dir >/dev/null 2>&1; then
  # --short, because check_build_record compares this against source_revision(),
  # which asks git for the abbreviated form. A full SHA here would report a
  # mismatch against the very tree it was built from.
  REVISION="$(git -C "$SOURCE" rev-parse --short HEAD 2>/dev/null || true)"
  DESCRIBED="$(git -C "$SOURCE" describe --tags --always --dirty 2>/dev/null || true)"
  BRANCH="$(git -C "$SOURCE" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  # Tracked files only: a build leaves chease.o and friends in src-f90, and an
  # untracked object file says nothing about whether the sources were modified.
  DIRTY_FILES="$(git -C "$SOURCE" status --porcelain --untracked-files=no 2>/dev/null || true)"
fi

# --- build --------------------------------------------------------------------
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
LOG="$SRC90/vaft-build.log"
FREE_FORM='-ffree-line-length-none'
# Left for make to expand: DIR_interpos is defined by CHEASE's own makefiles.
INTERPOS='-I$(DIR_interpos)'
MAKE_VARS=(
  "CHEASE_F90=gfortran"
  "CHEASE_MACHINE=darwin"
  "F90=gfortran"
  "F90FLAGS=-g -O3 $FREE_FORM $INTERPOS"
  "F90FLAGS_O0=-g -O0 $FREE_FORM $INTERPOS"
  "F90FLAGS_O2=-g -O2 $FREE_FORM $INTERPOS"
  "F90FLAGS_parser=-O0 $FREE_FORM"
  "F90FLAGS_parser_nor8=-O0 $FREE_FORM"
  "LDFLAGS=-g -O0 -Wl,-syslibroot,$SDKROOT_PATH"
  "HDF5="
  "INCL_HDF5="
  "LIBS_HDF5="
  "LIBS=-framework Accelerate -lm"
)

note "building with $JOBS jobs; log at $LOG"
(cd "$SRC90" && make -j "$JOBS" chease "${MAKE_VARS[@]}") >"$LOG" 2>&1 || {
  printf '[FAIL] the build failed; its last 20 lines:\n' >&2
  tail -20 "$LOG" >&2
  die "see $LOG"
}
[[ -x "$SRC90/chease" ]] || die "the build reported success but produced no src-f90/chease; see $LOG"

# --- install --------------------------------------------------------------------
mkdir -p "$PREFIX/bin"
cp -f "$SRC90/chease" "$PREFIX/bin/chease"
chmod +x "$PREFIX/bin/chease"
note "installed bin/chease into $PREFIX"

# --- manifest ---------------------------------------------------------------
MAKE_VARS_JSON="$(printf '%s\n' "${MAKE_VARS[@]}" | "$PYTHON" -c 'import json,sys; print(json.dumps([l for l in sys.stdin.read().splitlines() if l]))')"
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
exe = prefix / "bin" / "chease"
record = {
    "code": "chease",
    "installer": "install/install_chease_macos.sh",
    "prefix": str(prefix),
    "source": "$SOURCE",
    "source_revision": "$REVISION" or None,
    "source_described": "$DESCRIBED" or None,
    "source_branch": "$BRANCH" or None,
    "source_dirty": bool("""$DIRTY_FILES""".strip()),
    "source_dirty_files": [l for l in """$DIRTY_FILES""".splitlines() if l.strip()],
    "source_patched": False,
    "make_variables": $MAKE_VARS_JSON,
    "compiler": {"fortran": "$FC_PATH", "version": "$FC_VERSION"},
    "chease_machine": "darwin",
    "sdkroot": "$SDKROOT_PATH",
    "math_libraries": "-framework Accelerate",
    "hdf5": False,
    "platform": "$PLATFORM",
    "host": platform.node(),
    "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "log": "$LOG",
    "executables": {
        "chease": {"path": str(exe), "sha256": sha(exe), "size": exe.stat().st_size}
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print("wrote", sys.argv[1])
EOF

note "verifying with install/check_chease.py"
"$PYTHON" "$SCRIPT_DIR/check_chease.py" --source "$SOURCE" --prefix "$PREFIX" || true
printf '\nPoint VAFT at this build:\n  export CHEASEHOME=%s\n' "$PREFIX"

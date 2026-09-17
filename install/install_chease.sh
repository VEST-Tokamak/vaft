#!/usr/bin/env bash
# Build and install CHEASE from a source tree you already hold.
#
# Usage:
#   bash install/install_chease.sh --source PATH [options]
#   bash install/install_chease.sh --source PATH --check-only
#   bash install/install_chease.sh --source PATH --uninstall
#
# CHEASE is Apache-2.0 and public (https://gitlab.epfl.ch/spc/chease), but VAFT
# still neither bundles nor fetches it: which revision was built is a statement
# the operator makes, not something a script infers. Obtain the tree yourself
# and pass its path. This script never clones, fetches, pulls, checks out or
# changes a revision of the tree it is given.
#
# Clone it with symlink support -- CHEASE commits several sources as symlinks:
#   git -c core.symlinks=true clone https://gitlab.epfl.ch/spc/chease.git
#
# What it does: build the `chease` target in place with gfortran, install
# bin/chease into a prefix that $CHEASEHOME points at, run install/check_chease.py
# as the build's acceptance, and write vaft-external-install.json there recording
# the source revision (and, if you insisted on building a dirty tree, the diff
# digest), the make command, the compiler and the executable's checksum.
#
# Linux (system gfortran) is the verified platform; the macOS arm uses upstream's
# `darwin` machine and is untested. Nothing is installed system-wide.
#
# Three things about the build, each of which fails quietly if you get it wrong:
#
#   * CHEASE_MACHINE is not cosmetic. `Makefile.define_FLAGS` matches
#     linux_nohdf5 in exactly one branch, and that branch is the only one that
#     sets -fdefault-real-8 -fdefault-double-8. Upstream's default machine
#     (`none`, from Makefile.define_MACHINE) compiles cleanly in single
#     precision and produces a numerically different code.
#   * CHEASE_F90 and CHEASE_MACHINE are both exported so the makefile's host
#     detection never runs. Left to itself it shells out to `dnsdomainname` and
#     picks a branch from the hostname, which makes the build machine-dependent.
#   * The goal must be literally `chease`. The makefile keys its XML handling
#     off MAKECMDGOALS being exactly that, and `all` pulls in libxml2.
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MANIFEST_NAME="vaft-external-install.json"

SOURCE="${CHEASE_SOURCE_DIR:-}"
PREFIX=""
MACHINE=""
ALLOW_DIRTY=0
JOBS=""
SKIP_TESTS=0
CHECK_ONLY=0
UNINSTALL=0

usage() {
  cat <<'EOF'
Usage: bash install/install_chease.sh --source PATH [options]

  --source PATH        the CHEASE source tree to build (or set CHEASE_SOURCE_DIR)
  --prefix PATH        install here (default: <source>/vaft-install)
  --machine NAME       upstream CHEASE_MACHINE (default: linux_nohdf5 on Linux,
                       darwin on macOS). Only branches that set the real-8 flags
                       produce a correct code; see the header.
  --allow-dirty        build a tree with uncommitted tracked changes; the diff
                       digest and file list are then recorded in the manifest
  --jobs N             parallel build jobs (default: 1, upstream is not -j safe)
  --skip-tests         do not run install/check_chease.py after building
  --check-only         run install/check_chease.py and change nothing
  --uninstall          remove the prefix this script created
  -h, --help

The prefix is what $CHEASEHOME should point at: VAFT resolves $CHEASEHOME/bin/chease.
This script prints the export line; it edits no shell profile.

CHEASE builds in place, so its object files land in <source>/src-f90 and stay
there. --uninstall removes only the prefix; run `make clean` in src-f90 yourself
if you want the objects gone, because that directory is yours, not this script's.
EOF
}

die() { printf 'install_chease.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --source) (($# >= 2)) || die '--source needs a path'; SOURCE="$2"; shift 2 ;;
    --prefix) (($# >= 2)) || die '--prefix needs a path'; PREFIX="$2"; shift 2 ;;
    --machine) (($# >= 2)) || die '--machine needs a name'; MACHINE="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --jobs) (($# >= 2)) || die '--jobs needs a number'; JOBS="$2"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$SOURCE" ]] || die "--source is required: VAFT does not vendor CHEASE. Clone it from https://gitlab.epfl.ch/spc/chease (with -c core.symlinks=true) and pass its path, or set CHEASE_SOURCE_DIR."
SOURCE="$(cd "$SOURCE" 2>/dev/null && pwd -P)" || die "CHEASE source tree does not exist: $SOURCE"
# The same markers install/check_chease.py validates, so the two agree on what a
# CHEASE checkout is.
# chease_prog.f90 rather than chease_prog_effxml.f90: the latter is generated,
# not committed. src-f90/Makefile deletes it at parse time
# (`$(shell rm -f ... chease_prog_effxml.f90 ...)`) and the build writes it
# again, so a checkout that has never been built does not have it -- and a
# freshly cloned tree is exactly the case this marker has to accept.
for marker in src-f90/Makefile src-f90/Makefile.define_FLAGS src-f90/chease_prog.f90; do
  [[ -e "$SOURCE/$marker" ]] || die "not a CHEASE source tree (missing $marker): $SOURCE"
done

case "$(uname -s)" in
  Darwin) PLATFORM="darwin-$(uname -m)"; DEFAULT_MACHINE="darwin" ;;
  Linux) PLATFORM="linux-$(uname -m)"; DEFAULT_MACHINE="linux_nohdf5" ;;
  *) die "unsupported platform: $(uname -s)" ;;
esac
[[ -n "$MACHINE" ]] || MACHINE="$DEFAULT_MACHINE"
[[ -n "$PREFIX" ]] || PREFIX="$SOURCE/vaft-install"
MANIFEST="$PREFIX/$MANIFEST_NAME"
BUILD_DIR="$SOURCE/src-f90"

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
  # asks VAFT to resolve $CHEASEHOME from the ambient environment and fails
  # on a perfectly good installation.
  exec env CHEASEHOME="$PREFIX" "$PYTHON" "$SCRIPT_DIR/check_chease.py" --source "$SOURCE" --prefix "$PREFIX"
fi

if ((UNINSTALL)); then
  [[ -f "$MANIFEST" ]] || die "no $MANIFEST_NAME under $PREFIX; nothing this script installed is there to remove"
  note "removing prefix $PREFIX"
  rm -rf "$PREFIX"
  note "left alone: object files in $BUILD_DIR, which are yours. Run 'make clean' there to remove them."
  exit 0
fi

command -v gfortran >/dev/null || die "gfortran is required (e.g. apt install gfortran)"
command -v make >/dev/null || die "GNU make is required (e.g. apt install make)"
command -v git >/dev/null || die "git is required"

# --- symlink placeholders -----------------------------------------------------
# CHEASE commits several sources as symlinks. A tree cloned without symlink
# support holds one-line text files naming their target instead, and gfortran
# then reports a syntax error that names nothing useful. Detect it here, where
# the message can say what actually happened.
PLACEHOLDERS=""
while IFS= read -r candidate; do
  [[ -f "$candidate" ]] || continue
  [[ "$(stat -c %s "$candidate" 2>/dev/null || stat -f %z "$candidate" 2>/dev/null || echo 9999)" -le 512 ]] || continue
  target="$(tr -d '\n\r' <"$candidate")"
  [[ -n "$target" && "$target" != *$'\n'* && -e "$SOURCE/src-f90/$target" ]] || continue
  PLACEHOLDERS+="${candidate#"$SOURCE/"}"$'\n'
done < <(find "$SOURCE/src-f90" -maxdepth 1 -type f -name '*.f90' -size -1k)
if [[ -n "$PLACEHOLDERS" ]]; then
  printf 'these files are symlink placeholders, not source:\n%s\n' "$PLACEHOLDERS" >&2
  die "this tree was cloned without symlink support. Obtain it again with: git -c core.symlinks=true clone https://gitlab.epfl.ch/spc/chease.git"
fi

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

FC_PATH="$(command -v gfortran)"
FC_VERSION="$("$FC_PATH" --version | head -1)"
[[ -n "$JOBS" ]] || JOBS=1

# --- build -------------------------------------------------------------------
mkdir -p "$PREFIX/logs"
LOG="$PREFIX/logs/chease-build-$(date +%Y%m%d-%H%M%S).log"
MAKE_COMMAND="CHEASE_F90=gfortran CHEASE_MACHINE=$MACHINE make -j$JOBS chease"
note "building $SOURCE/src-f90 with $MAKE_COMMAND (log: $LOG)"
# IMAS_HOME and a caller's F90/F90FLAGS/LIBS would each redirect the build into
# upstream's IMAS-linked targets or override the precision flags above.
(
  cd "$BUILD_DIR"
  unset IMAS_HOME F90 F90FLAGS LIBS
  CHEASE_F90=gfortran CHEASE_MACHINE="$MACHINE" make -j"$JOBS" chease
) >>"$LOG" 2>&1 || die "build failed; see $LOG"
[[ -s "$BUILD_DIR/chease" ]] || die "chease was not produced at $BUILD_DIR/chease (see $LOG)"

# --- install ------------------------------------------------------------------
mkdir -p "$PREFIX/bin"
cp -f "$BUILD_DIR/chease" "$PREFIX/bin/chease"
chmod +x "$PREFIX/bin/chease"
note "installed bin/chease into $PREFIX"

# --- manifest -----------------------------------------------------------------
# Values reach Python through the environment and the heredoc is quoted, so no
# shell text is ever parsed as Python source. Interpolating them broke on the
# first quoted path in `git status --porcelain` ("""...name"""" is a
# SyntaxError) -- after bin/ was installed and before the manifest existed.
VAFT_MANIFEST_PREFIX="$PREFIX" \
VAFT_MANIFEST_SOURCE="$SOURCE" \
VAFT_MANIFEST_REVISION="$REVISION" \
VAFT_MANIFEST_DESCRIBED="$DESCRIBED" \
VAFT_MANIFEST_BRANCH="$BRANCH" \
VAFT_MANIFEST_REMOTE="$REMOTE" \
VAFT_MANIFEST_DIRTY_DIFF_SHA="$DIRTY_DIFF_SHA" \
VAFT_MANIFEST_BUILD_DIR="$BUILD_DIR" \
VAFT_MANIFEST_MAKE_COMMAND="$MAKE_COMMAND" \
VAFT_MANIFEST_MACHINE="$MACHINE" \
VAFT_MANIFEST_FC_PATH="$FC_PATH" \
VAFT_MANIFEST_FC_VERSION="$FC_VERSION" \
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
executable = prefix / "bin" / "chease"
record = {
    "code": "chease",
    "installer": "install/install_chease.sh",
    "prefix": str(prefix),
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
    "chease_machine": env["MACHINE"],
    "compiler": {"fortran": env["FC_PATH"], "version": env["FC_VERSION"]},
    "platform": env["PLATFORM"],
    "host": platform.node(),
    "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "log": env["LOG"],
    "executables": {
        "chease": {"path": str(executable), "sha256": sha(executable), "size": executable.stat().st_size}
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("wrote", sys.argv[1])
EOF

# --- acceptance ---------------------------------------------------------------
# check_chease.py refines a packaged equilibrium and compares q, pressure and
# current against it, which is a stronger acceptance than anything upstream
# ships. The binary is installed either way so a failure can be examined.
if ((!SKIP_TESTS)); then
  note "verifying with install/check_chease.py"
  CHEASEHOME="$PREFIX" "$PYTHON" "$SCRIPT_DIR/check_chease.py" --source "$SOURCE" --prefix "$PREFIX" || true
fi
printf '\nPoint VAFT at this build:\n  export CHEASEHOME=%s\n' "$PREFIX"

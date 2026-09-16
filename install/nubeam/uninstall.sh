#!/usr/bin/env bash
# Remove what install/nubeam/macos.sh (or linux.sh) generated, and nothing else.
#
# Usage:
#   bash install/nubeam/uninstall.sh --nubeam-root PATH [--dry-run]
#
# The POSIX NUBEAM recipes name this script in three places -- their usage text,
# the header of every Make.local they generate, and the error raised when one of
# them is re-run over an existing installation -- so a build that stops halfway
# leaves the operator pointed here.
#
# What it removes is decided by `<root>/.nubeam-install-manifest`, which the
# installer writes before it generates anything, not by a list kept here. Two
# rules keep that from reaching further than it should:
#
#   * Every path must sit inside the NUBEAM tree. The installer generates only
#     there, so a manifest entry that points outside it is a corrupted manifest
#     rather than an instruction, and this refuses it.
#   * A generated config is removed only if it still carries the marker the
#     installer wrote into it. If you edited `share/Make.local` by hand, or
#     replaced it with your own, it stays and this says so -- the file is yours
#     the moment you change it.
#
# The NUBEAM source itself is never touched: no `git` command appears here, and
# the tree's own files are not the installer's to remove.
set -euo pipefail
IFS=$'\n\t'

NUBEAM_ROOT="${NUBEAM_SOURCE_DIR:-}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash install/nubeam/uninstall.sh --nubeam-root PATH [--dry-run]

  --nubeam-root PATH   the NUBEAM source tree an installer was run against
                       (or set NUBEAM_SOURCE_DIR)
  --dry-run            list what would be removed and remove nothing
  -h, --help

Removes the installation prefix (<root>/local), the generated build directory,
the downloaded NTCC sources and the configs the installer wrote, then the
manifest itself. Running it twice is not an error: the second run reports that
there is nothing recorded and exits 0.
EOF
}

die() { printf 'uninstall.sh: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

while (($#)); do
  case "$1" in
    --nubeam-root) (($# >= 2)) || die '--nubeam-root needs a path'; NUBEAM_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$NUBEAM_ROOT" ]] || { usage >&2; die "--nubeam-root is required (or set NUBEAM_SOURCE_DIR)"; }
ROOT_DIR="$(cd "$NUBEAM_ROOT" 2>/dev/null && pwd -P)" || die "NUBEAM tree does not exist: $NUBEAM_ROOT"
MANIFEST="$ROOT_DIR/.nubeam-install-manifest"

# A second run is a no-op rather than a failure: the common case is an operator
# who is not sure whether the first one finished.
if [[ ! -e "$MANIFEST" ]]; then
  note "no installation manifest at $MANIFEST; nothing recorded to remove"
  exit 0
fi

is_child_of_root() {
  case "$1" in
    "$ROOT_DIR"/*) return 0 ;;
    *) return 1 ;;
  esac
}

# A tree built on one machine and copied to another carries a manifest full of
# the first machine's paths -- a NUBEAM tree built on macOS and rsynced to a
# Linux server records /Users/... entries that do not exist here. Those are the
# same relative paths under a different root, so they can be rebased onto this
# one.
#
# Which root, though, is not something to infer. The longest common prefix of
# the entries is the root only when they do not all share a subdirectory; when
# they do -- two entries under <root>/local, say -- it lands a level too deep,
# and rebasing then maps recorded paths onto *different* real paths that are
# still inside the tree, where the outside-the-tree refusal cannot see them.
# Deleting <root>/bin because the manifest said <old>/local/bin is exactly the
# kind of mistake this script exists to not make.
#
# So the installer records the root (`root\t<path>`), and that is used when
# present. For a manifest written before that -- macos.sh still writes none --
# fall back to the common prefix, but only accept it when its last component
# matches this tree's, which is what a relocated copy of the same tree looks
# like. Anything else is refused rather than guessed at.
recorded_root() {
  local declared common="" path
  declared="$(awk -F'\t' '$1 == "root" { print $2; exit }' "$MANIFEST")"
  if [[ -n "$declared" ]]; then
    printf '%s' "$declared"
    return 0
  fi
  while IFS=$'\t' read -r kind path; do
    [[ -n "${path:-}" && "$kind" != "root" ]] || continue
    if [[ -z "$common" ]]; then
      common="$path"
      continue
    fi
    while [[ "${path#"$common"}" == "$path" ]]; do
      common="${common%/*}"
      [[ -n "$common" ]] || return 1
    done
  done < "$MANIFEST"
  [[ -n "$common" ]] || return 1
  # The guard: a relocated tree keeps its own name.
  [[ "${common##*/}" == "${ROOT_DIR##*/}" ]] || {
    printf 'the manifest records no root, and its entries share only %s, whose name does not match this tree (%s).\n' "$common" "$ROOT_DIR" >&2
    printf 'Refusing to guess which prefix they were written under; remove the installation by hand, or delete %s.\n' "$MANIFEST" >&2
    return 2
  }
  printf '%s' "$common"
}

RECORDED_ROOT="$(recorded_root)" || {
  status=$?
  ((status == 2)) && exit 1
  RECORDED_ROOT=""
}
REBASED=0
if [[ -n "$RECORDED_ROOT" && "$RECORDED_ROOT" != "$ROOT_DIR" ]]; then
  note "this manifest was written for $RECORDED_ROOT; the tree is now $ROOT_DIR"
  note "reading its entries relative to the tree they are in"
  REBASED=1
fi

rebase() {
  if ((REBASED)); then
    printf '%s' "$ROOT_DIR${1#"$RECORDED_ROOT"}"
  else
    printf '%s' "$1"
  fi
}

#: The marker the installers write. The alternation covers the spelling used
#: before these recipes moved from external/nubeam into install/nubeam, so an
#: installation made by the older script is still recognised as ours.
MARKER='Generated by (VAFT (install|external)/nubeam|.*/install\.sh)'

removed=0
kept=0
while IFS=$'\t' read -r kind path; do
  [[ -n "${kind:-}" && -n "${path:-}" ]] || continue
  [[ "$kind" != "root" ]] || continue
  path="$(rebase "$path")"
  is_child_of_root "$path" || die "refusing path outside source tree: $path"
  case "$kind" in
    managed_dir)
      if [[ -d "$path" ]]; then
        if ((DRY_RUN)); then
          note "would remove directory $path"
        else
          note "removing directory $path"
          rm -rf "$path"
        fi
        removed=$((removed + 1))
      fi
      ;;
    generated_config)
      if [[ -f "$path" ]]; then
        if grep -qE "$MARKER" "$path"; then
          if ((DRY_RUN)); then
            note "would remove generated config $path"
          else
            note "removing generated config $path"
            rm -f "$path"
          fi
          removed=$((removed + 1))
        else
          note "keeping $path: it no longer carries the installer's marker, so it is yours"
          kept=$((kept + 1))
        fi
      fi
      ;;
    *)
      die "unrecognised manifest entry '$kind' in $MANIFEST"
      ;;
  esac
done < "$MANIFEST"

if ((DRY_RUN)); then
  note "would remove the manifest $MANIFEST"
  note "dry run: nothing was removed ($removed entries would go, $kept kept)"
  exit 0
fi

rm -f "$MANIFEST"
note "removed $removed recorded entries and the manifest"
if ((kept)); then
  note "$kept file(s) were left in place because you had changed them"
fi

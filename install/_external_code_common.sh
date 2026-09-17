# shellcheck shell=bash
# Prefix ownership for install_chease.sh, install_gpec.sh and install_efit.sh.
#
# Sourced, never run. The caller defines die() and note() and has
# `set -euo pipefail` in force.
#
# The rule these functions enforce: --uninstall may remove only what the
# installer can show it put there. A manifest inside the prefix is not such
# evidence on its own, because the installer writes that manifest into whatever
# directory it is given: `--prefix ~/.local` followed by
# `--uninstall --prefix ~/.local` used to satisfy the check and then
# `rm -rf ~/.local`. So
#
#   * the install refuses a directory that already holds something and carries
#     no ownership record for the same code;
#   * the ownership record says whether the installer created the directory;
#   * the uninstall removes the files the manifest lists, this installer's own
#     build logs and the two records, and then the directory itself only when
#     the installer created it and nothing else is left in it.
#
# Nothing here removes a directory recursively.

VAFT_EXTERNAL_MANIFEST_NAME="vaft-external-install.json"
#: Written before the build starts, so a build that fails leaves a prefix the
#: next run recognises as its own (the JSON manifest is written only after a
#: successful build). Tab-separated `key<TAB>value`, like .nubeam-install-manifest.
VAFT_EXTERNAL_MARKER_NAME=".vaft-external-prefix"

# vaft_external_canonical_path PATH
# The physical absolute spelling of PATH, which need not exist yet: the longest
# existing ancestor is resolved with `pwd -P` and the missing tail is appended.
# A `..` in the missing tail cannot be resolved physically, so it is refused
# (status 1) rather than guessed at. Without this, `/tmp/../<checkout>/x`
# compares unequal to the checkout it is inside.
vaft_external_canonical_path() {
  local path="$1" tail="" head name
  [[ -n "$path" ]] || return 1
  [[ "$path" == /* ]] || path="$PWD/$path"
  head="$path"
  while [[ ! -d "$head" ]]; do
    name="${head##*/}"
    head="${head%/*}"
    [[ -n "$head" ]] || head="/"
    case "$name" in
      ''|.) ;;
      ..) return 1 ;;
      *) tail="/$name$tail" ;;
    esac
  done
  head="$(cd "$head" && pwd -P)" || return 1
  [[ "$head" != "/" ]] || head=""
  if [[ -z "$head$tail" ]]; then
    printf '/\n'
  else
    printf '%s\n' "$head$tail"
  fi
}

# vaft_external_is_inside PATH ROOT -- both canonical. True for ROOT itself.
vaft_external_is_inside() {
  case "$1/" in
    "$2"/*) return 0 ;;
  esac
  return 1
}

# vaft_external_marker_field MARKER KEY
vaft_external_marker_field() {
  local key value
  while IFS=$'\t' read -r key value; do
    if [[ "$key" == "$2" ]]; then
      printf '%s\n' "$value"
      return 0
    fi
  done <"$1"
  return 0
}

# vaft_external_prefix_owner PREFIX PYTHON
# Prints the code that owns PREFIX, or nothing when no record is there. The
# marker wins; a manifest alone is what a VAFT 0.7.0 install left behind.
vaft_external_prefix_owner() {
  local prefix="$1" python="$2"
  if [[ -f "$prefix/$VAFT_EXTERNAL_MARKER_NAME" ]]; then
    vaft_external_marker_field "$prefix/$VAFT_EXTERNAL_MARKER_NAME" code
  elif [[ -f "$prefix/$VAFT_EXTERNAL_MANIFEST_NAME" ]]; then
    "$python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("code", ""))' \
      "$prefix/$VAFT_EXTERNAL_MANIFEST_NAME" || die "cannot read $prefix/$VAFT_EXTERNAL_MANIFEST_NAME"
  fi
}

# vaft_external_claim_prefix PREFIX CODE PYTHON
# Creates PREFIX if need be, writes the marker, and sets PREFIX_CREATED to 1
# when this installer (now or on an earlier run) created the directory.
vaft_external_claim_prefix() {
  local prefix="$1" code="$2" python="$3" owner
  PREFIX_CREATED=0
  if [[ -e "$prefix" || -L "$prefix" ]]; then
    [[ -d "$prefix" ]] || die "the install prefix exists and is not a directory: $prefix"
    owner="$(vaft_external_prefix_owner "$prefix" "$python")"
    if [[ -n "$owner" ]]; then
      [[ "$owner" == "$code" ]] || die "$prefix holds a VAFT install of '$owner', not '$code'. Every external code needs a prefix of its own, because each keeps its record under the same name there."
      if [[ -f "$prefix/$VAFT_EXTERNAL_MARKER_NAME" ]]; then
        [[ "$(vaft_external_marker_field "$prefix/$VAFT_EXTERNAL_MARKER_NAME" prefix_created)" != 1 ]] || PREFIX_CREATED=1
      fi
    elif [[ -n "$(ls -A "$prefix")" ]]; then
      die "the install prefix already exists, is not empty and was not created by this script: $prefix. --uninstall must be able to tell its own files from yours, so install into a directory of its own, e.g. --prefix $prefix/vaft-$code"
    fi
  else
    mkdir -p "$prefix"
    PREFIX_CREATED=1
  fi
  printf 'code\t%s\nprefix_created\t%s\n' "$code" "$PREFIX_CREATED" >"$prefix/$VAFT_EXTERNAL_MARKER_NAME"
}

# vaft_external_uninstall_prefix PREFIX CODE PYTHON
vaft_external_uninstall_prefix() {
  local prefix="$1" code="$2" python="$3"
  local marker="$prefix/$VAFT_EXTERNAL_MARKER_NAME" manifest="$prefix/$VAFT_EXTERNAL_MANIFEST_NAME"
  local owner created=0 listed="" relative log directory
  owner="$(vaft_external_prefix_owner "$prefix" "$python")"
  [[ -n "$owner" ]] || die "no $VAFT_EXTERNAL_MANIFEST_NAME under $prefix; nothing this script installed is there to remove"
  [[ "$owner" == "$code" ]] || die "$prefix holds a VAFT install of '$owner', not '$code'; refusing to touch it"
  if [[ -f "$marker" ]]; then
    [[ "$(vaft_external_marker_field "$marker" prefix_created)" != 1 ]] || created=1
  fi

  if [[ -f "$manifest" ]]; then
    # `installed_files` is relative to the prefix. A 0.7.0 manifest has only
    # `executables`, whose paths were spelled from its own `prefix` string.
    listed="$("$python" -c '
import json, sys
from pathlib import PurePath
record = json.load(open(sys.argv[1], encoding="utf-8"))
files = record.get("installed_files")
if files is None:
    files = []
    for entry in (record.get("executables") or {}).values():
        try:
            files.append(str(PurePath(entry["path"]).relative_to(record["prefix"])))
        except (KeyError, TypeError, ValueError):
            pass
print("\n".join(files))
' "$manifest")" || die "cannot read $manifest"
  fi
  while IFS= read -r relative; do
    [[ -n "$relative" ]] || continue
    case "/$relative/" in
      //*|*/../*|*/./*) die "refusing a manifest entry that is not a plain path under the prefix: $relative" ;;
    esac
    if [[ -f "$prefix/$relative" || -L "$prefix/$relative" ]]; then
      rm -f -- "$prefix/$relative"
      note "removed $prefix/$relative"
    fi
  done <<<"$listed"

  for log in "$prefix/logs/$code-build-"*.log; do
    [[ -f "$log" ]] || continue
    rm -f -- "$log"
  done
  rm -f -- "$manifest" "$marker"
  for directory in "$prefix/bin" "$prefix/logs"; do
    [[ ! -d "$directory" ]] || rmdir "$directory" 2>/dev/null ||
      note "left $directory: it holds files this script did not install"
  done

  if ((created)); then
    if rmdir "$prefix" 2>/dev/null; then
      note "removed prefix $prefix"
    else
      note "left $prefix: it holds files this script did not install"
    fi
  else
    note "left the directory $prefix itself: this script has no record of creating it"
  fi
}

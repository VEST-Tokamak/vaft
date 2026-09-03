#!/usr/bin/env bash
# VAFT uninstall for Linux, macOS and Windows (WSL2).
#
# One entry point rather than three: removal is identical on every POSIX
# platform, so there is no per-platform wrapper to write.
#
#   bash install/uninstall.sh            # list what will go, then confirm
#   bash install/uninstall.sh --dry-run  # print the plan, change nothing
#   bash install/uninstall.sh --yes      # skip the confirmation
#
# On native Windows use install/uninstall_windows_native.ps1 instead.
set -euo pipefail

# shellcheck disable=SC2034  # consumed by install/_common.sh
case "$(uname -s 2>/dev/null || echo unknown)" in
    Darwin) VAFT_PLATFORM_LABEL="macOS" ;;
    Linux) VAFT_PLATFORM_LABEL="Linux" ;;
    *) VAFT_PLATFORM_LABEL="POSIX" ;;
esac

# shellcheck source=install/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

vaft_uninstall_main "$@"

#!/usr/bin/env bash
# VAFT bootstrap for macOS (Apple silicon).
#
# Intel Macs are not supported: imas_core publishes no Intel macOS wheel and
# no source distribution, so the dependency cannot be resolved there.
#
#   bash install/macos.sh              # create/update the environment
#   bash install/macos.sh --check-only # verify only, change nothing
#
# Prerequisites (install these yourself first): Git (Xcode command line tools
# provide it) and Miniconda.
set -euo pipefail

# shellcheck disable=SC2034  # consumed by install/_common.sh
VAFT_PLATFORM_LABEL="macOS"
# shellcheck source=install/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [ "$(uname -s)" != "Darwin" ]; then
    printf 'Note: this is the macOS script but the system reports %s.\n' "$(uname -s)"
    printf 'Use install/linux.sh or install/windows_wsl.sh instead.\n'
fi

vaft_bootstrap_main "$@"

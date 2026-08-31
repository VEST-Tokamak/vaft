#!/usr/bin/env bash
# VAFT bootstrap for Windows Subsystem for Linux (WSL2).
#
#   bash install/windows_wsl.sh              # create/update the environment
#   bash install/windows_wsl.sh --check-only # verify only, change nothing
#
# This path is optional. VAFT is fully supported on native Windows through
# install/windows_native.ps1; WSL2 is for students who prefer a Linux shell or
# who will later build the external Fortran codes.
#
# Prerequisites (install these yourself first, inside the WSL distribution):
# Git and Linux Miniconda. Do not point this script at a Conda installed on the
# Windows side.
set -euo pipefail

# shellcheck disable=SC2034  # consumed by install/_common.sh
VAFT_PLATFORM_LABEL="Windows (WSL2)"
# shellcheck source=install/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

case "${VAFT_REPOSITORY_ROOT}" in
    /mnt/[a-z]/*)
        printf 'Warning: the checkout lives on a Windows drive (%s).\n' "${VAFT_REPOSITORY_ROOT}"
        printf 'File access across the WSL boundary is slow and Git file modes behave\n'
        printf 'differently. Cloning into the Linux filesystem (for example ~/git/vaft)\n'
        printf 'is strongly recommended.\n\n'
        ;;
esac

vaft_bootstrap_main "$@"

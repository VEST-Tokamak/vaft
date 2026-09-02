#!/usr/bin/env bash
# VAFT bootstrap for Linux. Ubuntu is the validated reference distribution.
#
#   bash install/linux.sh              # create/update the environment
#   bash install/linux.sh --check-only # verify only, change nothing
#
# Prerequisites (install these yourself first): Git and Miniconda.
set -euo pipefail

# shellcheck disable=SC2034  # consumed by install/_common.sh
VAFT_PLATFORM_LABEL="Linux"
# shellcheck source=install/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

vaft_bootstrap_main "$@"

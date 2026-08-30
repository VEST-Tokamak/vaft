# shellcheck shell=bash
# Shared POSIX bootstrap logic for the VAFT course environment.
#
# This file is sourced by install/linux.sh, install/macos.sh and
# install/windows_wsl.sh. Those wrappers only set VAFT_PLATFORM_LABEL and add
# any preflight their operating system needs; every step below is shared.
#
# The bootstrap is idempotent and strictly non-destructive: it never runs
# `git stash`, `git reset`, `git clean` or `git checkout`, never writes or reads
# credentials, and never modifies a Conda environment other than `vaft`.

set -euo pipefail

VAFT_ENV_NAME="vaft"
VAFT_KERNEL_NAME="vaft"
VAFT_KERNEL_DISPLAY_NAME="Python (vaft)"

VAFT_REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAFT_PLATFORM_LABEL="${VAFT_PLATFORM_LABEL:-POSIX}"

vaft_summary_lines=()
vaft_failed=0

vaft_record() {
    # vaft_record STATUS NAME [DETAIL]
    local status="$1" name="$2" detail="${3:-}"
    vaft_summary_lines+=("[${status}] ${name}")
    if [ "${status}" = "FAIL" ]; then
        vaft_failed=1
    fi
    if [ -n "${detail}" ]; then
        printf '[%s] %s: %s\n' "${status}" "${name}" "${detail}"
    else
        printf '[%s] %s\n' "${status}" "${name}"
    fi
}

vaft_die() {
    printf '\n[FAIL] %s\n' "$1" >&2
    exit 1
}

vaft_detect_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        vaft_die "Conda was not found on PATH.
Install Miniconda first, reopen your shell, then rerun this script.
See install/README.md for the download link and per-platform instructions.
This script deliberately does not install Conda for you."
    fi
    vaft_record PASS "Conda" "$(conda --version)"
}

vaft_environment_exists() {
    # Parsed without a helper interpreter: the outer shell is not guaranteed to
    # have a usable `python` on PATH, only `conda`.
    #
    # awk decides on its own rather than piping into `grep -q`: under
    # `set -o pipefail`, a `grep -q` that exits on the first match can leave the
    # upstream process killed by SIGPIPE, which would make the pipeline report
    # failure and silently recreate an environment that already exists.
    conda env list | awk -v name="${VAFT_ENV_NAME}" '$1 == name { found = 1 } END { exit found ? 0 : 1 }'
}

vaft_create_or_update_environment() {
    local specification="${VAFT_REPOSITORY_ROOT}/environment.yml"
    [ -f "${specification}" ] || vaft_die "Missing ${specification}."

    if vaft_environment_exists; then
        printf 'Updating the existing `%s` environment ...\n' "${VAFT_ENV_NAME}"
        conda env update --name "${VAFT_ENV_NAME}" --file "${specification}" --prune
        vaft_record PASS "vaft environment" "updated in place"
    else
        printf 'Creating the `%s` environment ...\n' "${VAFT_ENV_NAME}"
        conda env create --name "${VAFT_ENV_NAME}" --file "${specification}"
        vaft_record PASS "vaft environment" "created"
    fi
}

vaft_run() {
    # Run a command inside the `vaft` environment without mutating the caller's shell.
    conda run --name "${VAFT_ENV_NAME}" --no-capture-output "$@"
}

vaft_report_python() {
    vaft_record PASS "Python" "$(vaft_run python -c 'import platform, sys; print(platform.python_version(), sys.executable)')"
}

vaft_install_editable() {
    printf 'Installing VAFT in editable mode from %s ...\n' "${VAFT_REPOSITORY_ROOT}"
    ( cd "${VAFT_REPOSITORY_ROOT}" && vaft_run python -m pip install -e . )
    vaft_record PASS "editable VAFT installation" "${VAFT_REPOSITORY_ROOT}"
}

vaft_verify_import_location() {
    if vaft_run python -c '
import sys
from pathlib import Path
import vaft
root = Path(sys.argv[1]).resolve()
located = Path(vaft.__file__).resolve()
if root not in located.parents:
    sys.stderr.write(f"vaft resolves to {located}, outside {root}\n")
    raise SystemExit(1)
print(located)
' "${VAFT_REPOSITORY_ROOT}"; then
        vaft_record PASS "VAFT resolves to this checkout"
    else
        vaft_record FAIL "VAFT resolves to this checkout" \
            "an unrelated installed copy is shadowing ${VAFT_REPOSITORY_ROOT}"
    fi
}

vaft_register_kernel() {
    # `--name vaft` overwrites any existing spec of the same name, so repeated
    # runs replace the kernel in place instead of accumulating duplicates.
    vaft_run python -m ipykernel install --user \
        --name "${VAFT_KERNEL_NAME}" \
        --display-name "${VAFT_KERNEL_DISPLAY_NAME}" >/dev/null
    # Counted inside the environment so the check never depends on an outer
    # interpreter, and so a duplicate registration is caught rather than assumed away.
    if vaft_run python -c 'import json, subprocess, sys
payload = subprocess.run(
    [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
    capture_output=True, text=True, check=True,
).stdout
names = list(json.loads(payload).get("kernelspecs", {}))
sys.exit(0 if names.count(sys.argv[1]) == 1 else 1)
' "${VAFT_KERNEL_NAME}"; then
        vaft_record PASS "${VAFT_KERNEL_DISPLAY_NAME} kernel"
    else
        vaft_record FAIL "${VAFT_KERNEL_DISPLAY_NAME} kernel" \
            "expected exactly one kernelspec named ${VAFT_KERNEL_NAME}"
    fi
}

vaft_report_hsds_client() {
    if vaft_run python -c 'import h5pyd' >/dev/null 2>&1; then
        vaft_record PASS "HSDS client"
    else
        vaft_record FAIL "HSDS client" "h5pyd did not import"
    fi
}

vaft_report_jupyterlab() {
    if vaft_run python -c 'import jupyterlab' >/dev/null 2>&1; then
        vaft_record PASS "JupyterLab"
    else
        vaft_record FAIL "JupyterLab" "jupyterlab did not import"
    fi
}

vaft_print_summary() {
    printf '\nVAFT bootstrap (%s)\n' "${VAFT_PLATFORM_LABEL}"
    printf -- '--------------\n'
    if [ "${#vaft_summary_lines[@]}" -gt 0 ]; then
        printf '%s\n' "${vaft_summary_lines[@]}"
    fi
    cat <<'NEXT'

Next:
  1. Run `hsconfigure` if your HSDS credentials are not configured yet.
     This script never asks for, stores, or transmits your credentials.
  2. Run `conda run -n vaft python install/check_vaft_environment.py`.
  3. Run `conda activate vaft && jupyter lab`, and choose the "Python (vaft)" kernel.

This script changed only: the `vaft` Conda environment, an editable VAFT
installation inside it, and the user-level "Python (vaft)" Jupyter kernelspec.
It did not modify your repository checkout or any other Conda environment.
NEXT
    return "${vaft_failed}"
}

vaft_run_checker() {
    vaft_run python "${VAFT_REPOSITORY_ROOT}/install/check_vaft_environment.py"
}

vaft_bootstrap_main() {
    local check_only=0
    for argument in "$@"; do
        case "${argument}" in
            --check-only) check_only=1 ;;
            -h|--help)
                cat <<'USAGE'
Usage: bootstrap script [--check-only]

  (no arguments)  create or update the `vaft` environment, install VAFT in
                  editable mode, and register the "Python (vaft)" kernel
  --check-only    run install/check_vaft_environment.py and change nothing
USAGE
                return 0
                ;;
            *) vaft_die "Unknown option: ${argument}" ;;
        esac
    done

    printf 'VAFT bootstrap (%s)\nRepository: %s\n\n' \
        "${VAFT_PLATFORM_LABEL}" "${VAFT_REPOSITORY_ROOT}"

    vaft_detect_conda
    if [ "${check_only}" -eq 1 ]; then
        vaft_run_checker
        return $?
    fi

    vaft_create_or_update_environment
    vaft_report_python
    vaft_install_editable
    vaft_verify_import_location
    vaft_report_hsds_client
    vaft_report_jupyterlab
    vaft_register_kernel
    vaft_print_summary
}

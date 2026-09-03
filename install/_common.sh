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
        # Deliberately not --prune: that removes anything in the environment
        # that environment.yml does not mention, which would silently delete
        # packages a student installed themselves.
        conda env update --name "${VAFT_ENV_NAME}" --file "${specification}"
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
    if ! ( cd "${VAFT_REPOSITORY_ROOT}" && vaft_run python -m pip install -e . ); then
        printf '\n' >&2
        if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "x86_64" ]; then
            vaft_die "The editable installation failed on an Intel Mac.
VAFT depends on imas_core, which publishes wheels only for Apple silicon,
Linux x86_64 and Windows -- there is no Intel macOS wheel and no source
distribution, so pip cannot resolve it on this machine.
Use an Apple silicon Mac, a Linux machine, or WSL2. See install/README.md."
        fi
        vaft_die "The editable installation failed. Read the pip output above:
it names the dependency that could not be installed. Rerunning this script is
safe once the cause is fixed."
    fi
    vaft_record PASS "editable VAFT installation" "${VAFT_REPOSITORY_ROOT}"
}

vaft_register_kernel() {
    # `--name vaft` overwrites any existing spec of the same name, so repeated
    # runs replace the kernel in place instead of accumulating duplicates.
    # Whether exactly one survives is then confirmed by the checker.
    vaft_run python -m ipykernel install --user \
        --name "${VAFT_KERNEL_NAME}" \
        --display-name "${VAFT_KERNEL_DISPLAY_NAME}" >/dev/null
    vaft_record PASS "${VAFT_KERNEL_DISPLAY_NAME} kernel" "registered"
}

vaft_print_summary() {
    printf '\nVAFT bootstrap (%s): what changed\n' "${VAFT_PLATFORM_LABEL}"
    printf -- '--------------\n'
    if [ "${#vaft_summary_lines[@]}" -gt 0 ]; then
        printf '%s\n' "${vaft_summary_lines[@]}"
    fi
    cat <<'NEXT'

This script changed only: the `vaft` Conda environment, an editable VAFT
installation inside it, and the user-level "Python (vaft)" Jupyter kernelspec.
It did not modify your repository checkout or any other Conda environment.

Next:
  1. Run `hsconfigure` if your HSDS credentials are not configured yet.
     This script never asks for, stores, or transmits your credentials.
  2. Run `conda activate vaft && jupyter lab`, and choose the "Python (vaft)" kernel.
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
        if ! vaft_environment_exists; then
            vaft_die "The \`${VAFT_ENV_NAME}\` environment does not exist yet.
Run this script without --check-only to create it."
        fi
        vaft_run_checker
        return $?
    fi

    vaft_create_or_update_environment
    vaft_report_python
    vaft_install_editable
    vaft_register_kernel
    vaft_print_summary

    # Verification lives in one place. The checker reports every environment
    # property with its own remediation, so the bootstrap does not reimplement
    # those probes -- and its exit status becomes the bootstrap's.
    printf '\nVerifying the environment ...\n\n'
    vaft_run_checker
}

# ---------------------------------------------------------------------------
# Uninstall
#
# Reverses exactly what the bootstrap above created, and nothing else. The
# ordering matters: the kernelspec is removed through `conda run -n vaft`, so
# it has to go before the environment that provides that interpreter.
#
# `~/.hscfg` is deliberately left alone. The bootstrap never writes it --
# `hsconfigure` does, run by the student -- and it holds HSDS credentials.
# ---------------------------------------------------------------------------

vaft_dry_run=0
vaft_assume_yes=0
vaft_keep_build_artifacts=0

# What the editable install leaves in the checkout, and the only thing here a
# reinstall would inherit. Deliberately *not* `build/` or `dist/`: the
# bootstrap never creates those -- `python -m build` does, per RELEASING.md --
# and deleting a maintainer's release artifacts is not this script's business.
#
# Explicit paths, never `git clean`: the tooling in install/ promises it runs
# no destructive Git command, and that promise still applies here.
vaft_build_artifacts=("vaft.egg-info")

vaft_kernelspec_candidates() {
    # Where Jupyter keeps user-level kernelspecs, without needing Jupyter to
    # tell us -- by the time the environment is gone there is no interpreter
    # left to ask. Only used as a fallback sweep; the supported path is
    # `jupyter kernelspec remove`.
    vaft_kernelspec_paths=()
    if [ -n "${JUPYTER_DATA_DIR:-}" ]; then
        vaft_kernelspec_paths+=("${JUPYTER_DATA_DIR}/kernels")
    fi
    if [ -n "${XDG_DATA_HOME:-}" ]; then
        vaft_kernelspec_paths+=("${XDG_DATA_HOME}/jupyter/kernels")
    fi
    vaft_kernelspec_paths+=(
        "${HOME}/Library/Jupyter/kernels"
        "${HOME}/.local/share/jupyter/kernels"
    )
}

vaft_kernel_is_present() {
    vaft_kernelspec_candidates
    local directory
    for directory in "${vaft_kernelspec_paths[@]}"; do
        if [ -d "${directory}/${VAFT_KERNEL_NAME}" ]; then
            return 0
        fi
    done
    return 1
}

vaft_assert_environment_is_not_active() {
    # Removal is ordered kernel-first, because the kernelspec can only be
    # removed through the environment's own interpreter. That ordering means a
    # `conda env remove` which refuses part-way would leave a working
    # environment with no kernel -- strictly worse than not having started.
    #
    # Conda refuses to remove the environment the current shell has activated
    # ("You must deactivate the existing environment before you can remove
    # it"), and that is the one case we can see coming. Stop before touching
    # anything.
    local active="${CONDA_DEFAULT_ENV:-}"
    if [ -z "${active}" ] && [ -n "${CONDA_PREFIX:-}" ]; then
        active="$(basename "${CONDA_PREFIX}")"
    fi
    if [ "${active}" = "${VAFT_ENV_NAME}" ]; then
        vaft_die "The \`${VAFT_ENV_NAME}\` environment is active in this shell, and Conda
refuses to remove an environment you are standing in.
Run \`conda deactivate\` first, then rerun this script.
Nothing has been removed."
    fi
}

vaft_remove_kernel() {
    local removed=0

    # The supported route, while there is still an interpreter to run it. A
    # broken or Jupyter-less environment must not abort the uninstall, so the
    # failure is swallowed and the directory sweep below picks up the slack.
    if vaft_environment_exists; then
        if vaft_run python -m jupyter kernelspec remove -f "${VAFT_KERNEL_NAME}" \
            >/dev/null 2>&1; then
            removed=1
        fi
    fi

    vaft_kernelspec_candidates
    local directory
    for directory in "${vaft_kernelspec_paths[@]}"; do
        if [ -d "${directory}/${VAFT_KERNEL_NAME}" ]; then
            rm -rf "${directory:?}/${VAFT_KERNEL_NAME:?}"
            removed=1
        fi
    done

    if [ "${removed}" -eq 1 ]; then
        vaft_record PASS "${VAFT_KERNEL_DISPLAY_NAME} kernel" "removed"
    else
        vaft_record SKIP "${VAFT_KERNEL_DISPLAY_NAME} kernel" "not registered"
    fi
}

vaft_remove_environment() {
    if ! vaft_environment_exists; then
        vaft_record SKIP "vaft environment" "not present"
        return 0
    fi
    printf 'Removing the `%s` environment ...\n' "${VAFT_ENV_NAME}"
    # --name pins the removal to the exact name. No prefix or pattern match,
    # so environments like `vaft-np2-test` are never in scope.
    if ! conda env remove --name "${VAFT_ENV_NAME}" --yes; then
        vaft_record FAIL "vaft environment" "conda env remove failed"
        return 0
    fi
    vaft_record PASS "vaft environment" "removed"
}

vaft_present_build_artifacts() {
    vaft_present_artifacts=()
    local artifact
    for artifact in "${vaft_build_artifacts[@]}"; do
        if [ -e "${VAFT_REPOSITORY_ROOT}/${artifact}" ]; then
            vaft_present_artifacts+=("${artifact}")
        fi
    done
}

vaft_remove_build_artifacts() {
    if [ "${vaft_keep_build_artifacts}" -eq 1 ]; then
        vaft_record SKIP "build artifacts" "kept (--keep-build-artifacts)"
        return 0
    fi
    vaft_present_build_artifacts
    if [ "${#vaft_present_artifacts[@]}" -eq 0 ]; then
        vaft_record SKIP "build artifacts" "none in the checkout"
        return 0
    fi
    local artifact
    for artifact in "${vaft_present_artifacts[@]}"; do
        rm -rf "${VAFT_REPOSITORY_ROOT:?}/${artifact}"
    done
    vaft_record PASS "build artifacts" "removed ${vaft_present_artifacts[*]}"
}

vaft_print_removal_plan() {
    printf 'This will remove:\n'
    if vaft_environment_exists; then
        printf '  - the `%s` Conda environment, and the editable VAFT installation in it\n' \
            "${VAFT_ENV_NAME}"
    else
        printf '  - (the `%s` Conda environment is already absent)\n' "${VAFT_ENV_NAME}"
    fi
    if vaft_kernel_is_present; then
        printf '  - the user-level "%s" Jupyter kernelspec\n' "${VAFT_KERNEL_DISPLAY_NAME}"
    else
        printf '  - (the "%s" kernelspec is already absent)\n' "${VAFT_KERNEL_DISPLAY_NAME}"
    fi
    if [ "${vaft_keep_build_artifacts}" -eq 1 ]; then
        printf '  - (build artifacts kept: --keep-build-artifacts)\n'
    else
        vaft_present_build_artifacts
        if [ "${#vaft_present_artifacts[@]}" -gt 0 ]; then
            printf '  - build artifacts in %s: %s\n' \
                "${VAFT_REPOSITORY_ROOT}" "${vaft_present_artifacts[*]}"
        else
            printf '  - (no build artifacts in %s)\n' "${VAFT_REPOSITORY_ROOT}"
        fi
    fi
    cat <<'KEEP'

It will not touch: your repository checkout, any other Conda environment, or
`~/.hscfg` -- that file holds your HSDS credentials and this script never
created it.
KEEP
}

vaft_confirm_removal() {
    if [ "${vaft_assume_yes}" -eq 1 ]; then
        return 0
    fi
    if [ ! -t 0 ]; then
        vaft_die "Refusing to remove anything without confirmation.
No terminal is attached, so this script cannot ask. Rerun with --yes to
confirm non-interactively, or --dry-run to see what would be removed."
    fi
    printf '\nRemove these? [y/N] '
    local reply
    read -r reply
    case "${reply}" in
        y|Y|yes|YES|Yes) return 0 ;;
        *) return 1 ;;
    esac
}

vaft_print_uninstall_summary() {
    printf '\nVAFT uninstall (%s): what changed\n' "${VAFT_PLATFORM_LABEL}"
    printf -- '--------------\n'
    if [ "${#vaft_summary_lines[@]}" -gt 0 ]; then
        printf '%s\n' "${vaft_summary_lines[@]}"
    fi
    cat <<'NEXT'

Your checkout, every other Conda environment and `~/.hscfg` are untouched.
Reinstall at any time with the bootstrap script for your platform; it will
create the environment from scratch, exactly as it did the first time.
NEXT
    return "${vaft_failed}"
}

vaft_uninstall_main() {
    for argument in "$@"; do
        case "${argument}" in
            --yes|-y) vaft_assume_yes=1 ;;
            --dry-run) vaft_dry_run=1 ;;
            --keep-build-artifacts) vaft_keep_build_artifacts=1 ;;
            -h|--help)
                cat <<'USAGE'
Usage: uninstall script [--yes] [--dry-run] [--keep-build-artifacts]

  (no arguments)          list what would be removed, then ask for confirmation
  --yes, -y               skip the confirmation (for CI and scripting)
  --dry-run               print the plan and change nothing
  --keep-build-artifacts  leave vaft.egg-info/ in place

Removes the `vaft` Conda environment, the editable VAFT installation inside
it, and the user-level "Python (vaft)" Jupyter kernelspec. Never removes
`~/.hscfg`, your checkout, or any other Conda environment.
USAGE
                return 0
                ;;
            *) vaft_die "Unknown option: ${argument}" ;;
        esac
    done

    printf 'VAFT uninstall (%s)\nRepository: %s\n\n' \
        "${VAFT_PLATFORM_LABEL}" "${VAFT_REPOSITORY_ROOT}"

    vaft_detect_conda
    vaft_assert_environment_is_not_active
    vaft_print_removal_plan

    if [ "${vaft_dry_run}" -eq 1 ]; then
        printf '\nDry run: nothing was removed.\n'
        return 0
    fi

    if ! vaft_confirm_removal; then
        printf 'Aborted. Nothing was removed.\n'
        return 0
    fi

    printf '\n'
    # Kernel first: removing it goes through the environment's own interpreter.
    vaft_remove_kernel
    vaft_remove_environment
    vaft_remove_build_artifacts
    vaft_print_uninstall_summary
}

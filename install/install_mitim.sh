#!/usr/bin/env bash
# Install MITIM-fusion into its own isolated Python environment for VAFT (#1588).
#
# Usage:
#   bash install/install_mitim.sh [--prefix PATH] [--version 5.3.0] [--python python3.12] [--conda]
#   bash install/install_mitim.sh --check-only [--prefix PATH]
#
# MITIM (https://github.com/pabloprf/MITIM-fusion, MIT licence) needs Python
# 3.10-3.12 and pulls tensorflow, botorch and torch, so it gets an environment of
# its own and VAFT never imports it: vaft.code.mitim runs it through that
# environment's interpreter. This script creates the environment, installs the
# pinned release tag (never a moving branch) editable from a clone of that tag,
# runs the availability probe from
# the MITIM side, and writes vaft-external-install.json beside it. VAFT never
# clones or updates MITIM at run time.
#
# Afterwards point VAFT at the interpreter it prints:
#   export VAFT_MITIM_PYTHON=<prefix>/bin/python        (venv)
#   export VAFT_MITIM_PYTHON=<conda env>/bin/python     (--conda)
# GACODE is not installed here: MITIM runs the build $GACODEHOME names, the same
# one VAFT's own TGLF/NEO adapters use.
set -euo pipefail
IFS=$'\n\t'

VERSION="5.3.0"
PREFIX="${HOME}/.local/share/vaft/external/mitim"
PYTHON_BIN="python3.12"
USE_CONDA=0
CHECK_ONLY=0
REPOSITORY="https://github.com/pabloprf/MITIM-fusion"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --conda) USE_CONDA=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

INTERPRETER="${PREFIX}/bin/python"
# Never let a ~/.local package satisfy (or be uninstalled by) this environment's pip.
export PYTHONNOUSERSITE=1

probe() {
  # MITIM reads $MITIM_CONFIG while importing; give the check a throwaway one.
  local scratch
  scratch="$(mktemp -d)"
  printf '{"preferences": {"verbose_level": "1"}, "local": {"machine": "local", "scratch": "%s/"}}' \
    "${scratch}" > "${scratch}/config.json"
  MITIM_CONFIG="${scratch}/config.json" "${INTERPRETER}" - <<'PY'
import json, platform, sys
import mitim_tools
import mitim_modules.portals.PORTALSmain  # noqa: F401  (PORTALS must import)
print(json.dumps({"mitim_version": mitim_tools.__version__,
                  "python": sys.executable, "python_version": platform.python_version()}))
PY
}

if [[ "${CHECK_ONLY}" == 1 ]]; then
  probe
  exit 0
fi

if [[ -e "${INTERPRETER}" ]]; then
  echo "reusing the environment at ${PREFIX}"
elif [[ "${USE_CONDA}" == 1 ]]; then
  conda create -y -q -p "${PREFIX}" "python=${PYTHON_BIN#python}" pip
else
  "${PYTHON_BIN}" -m venv "${PREFIX}"
fi

"${INTERPRETER}" -m pip install -q --upgrade pip
# MITIM must run from its repository checkout: __mitimroot__ (two levels above
# mitim_tools) has to hold templates/ (input.neo.controls, ...), which a wheel does
# not ship. So the pinned tag is cloned here, at install time, and installed
# editable. VAFT never clones or updates it at run time.
SOURCE="${PREFIX}/MITIM-fusion"
if [[ ! -d "${SOURCE}/.git" ]]; then
  git clone -q --depth 1 --branch "v${VERSION}" "${REPOSITORY}" "${SOURCE}"
fi
"${INTERPRETER}" -m pip install -e "${SOURCE}"
# PORTALS (mitim_tools.gacode_tools.utils) imports fortranformat, which MITIM 5.3.0
# does not declare. Found on tdst once the user site was excluded (#1588).
"${INTERPRETER}" -m pip install fortranformat

REPORT="$(probe)"
echo "${REPORT}"
INSTALLED="$("${INTERPRETER}" -c 'import mitim_tools; print(mitim_tools.__version__)')"
if [[ "${INSTALLED}" != "${VERSION}" ]]; then
  echo "installed MITIM ${INSTALLED}, expected ${VERSION}" >&2
  exit 1
fi

"${INTERPRETER}" - "${PREFIX}" "${VERSION}" "${REPOSITORY}" <<'PY'
import json, sys, datetime, subprocess
prefix, version, repository = sys.argv[1:4]
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
record = {
    "code": "mitim",
    "version": version,
    "source": f"{repository}@v{version}",
    "interpreter": sys.executable,
    "installed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "pip_freeze": freeze.splitlines(),
}
with open(f"{prefix}/vaft-external-install.json", "w") as handle:
    json.dump(record, handle, indent=1)
PY

echo
echo "MITIM ${VERSION} installed. Point VAFT at it:"
echo "  export VAFT_MITIM_PYTHON=${INTERPRETER}"

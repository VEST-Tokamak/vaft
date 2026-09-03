"""Assert that nothing the VAFT bootstrap creates survived the uninstaller.

Run from the repository root, on a machine where the `vaft` environment is
supposed to be gone -- so it deliberately uses the ambient interpreter rather
than `conda run -n vaft python`, which would be a contradiction in terms.

Used twice by .github/workflows/bootstrap-ci.yml, once after each uninstall in
the install -> uninstall -> install -> uninstall cycle. The second pass is the
one that matters: if the first uninstall left state behind, the second install
took the update branch instead of creating the environment from scratch, and
the platform's real first-install path was never exercised.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ENVIRONMENT_NAME = "vaft"
KERNEL_NAME = "vaft"
# Only what the bootstrap itself creates. `build/` and `dist/` come from
# `python -m build`, which the bootstrap never runs.
BUILD_ARTIFACTS = ("vaft.egg-info",)

failures: list[str] = []


def conda_environment_names() -> list[str]:
    # Resolved rather than invoked by bare name: on Windows conda is usually
    # `condabin\conda.bat`, and CreateProcess appends only `.exe` when it
    # searches PATH -- it never consults PATHEXT. shutil.which does.
    conda = shutil.which("conda")
    if conda is None:
        sys.exit("[FAIL] conda is not on PATH, so the environment cannot be checked")
    listing = subprocess.run(
        [conda, "env", "list"], capture_output=True, text=True, check=True
    ).stdout
    names = []
    for line in listing.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        names.append(line.split()[0])
    return names


def kernelspec_roots() -> list[Path]:
    roots = []
    if os.environ.get("JUPYTER_DATA_DIR"):
        roots.append(Path(os.environ["JUPYTER_DATA_DIR"]) / "kernels")
    if os.environ.get("XDG_DATA_HOME"):
        roots.append(Path(os.environ["XDG_DATA_HOME"]) / "jupyter" / "kernels")
    if os.environ.get("APPDATA"):
        roots.append(Path(os.environ["APPDATA"]) / "jupyter" / "kernels")
    home = Path.home()
    roots.append(home / "Library" / "Jupyter" / "kernels")
    roots.append(home / ".local" / "share" / "jupyter" / "kernels")
    return roots


names = conda_environment_names()
print(f"conda environments: {names}")
if ENVIRONMENT_NAME in names:
    failures.append(f"the `{ENVIRONMENT_NAME}` environment survived the uninstall")

for root in kernelspec_roots():
    spec = root / KERNEL_NAME
    if spec.exists():
        failures.append(f"the `{KERNEL_NAME}` kernelspec survived at {spec}")

repository_root = Path(__file__).resolve().parents[2]
for artifact in BUILD_ARTIFACTS:
    path = repository_root / artifact
    if path.exists():
        failures.append(f"{artifact} survived the uninstall at {path}")

if failures:
    for failure in failures:
        print(f"[FAIL] {failure}")
    sys.exit(1)

print("[PASS] the machine is back to a pre-install state")

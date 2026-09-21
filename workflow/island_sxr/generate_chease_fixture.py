"""Regenerate the CHEASE-refined Solov'ev equilibrium the #886 island notebook compares.

    PYTHONPATH=$PWD CHEASE=/path/to/chease python workflow/island_sxr/generate_chease_fixture.py

The notebook ``analytic_island_model_and_synthetic_response_model.ipynb`` puts
the same magnetic island on two equilibria: an analytic VEST-sized Solov'ev
model, and that model after a CHEASE fixed-boundary refinement. The first is
built in the notebook from :func:`vaft.process.equilibrium.solovev_example`;
the second needs the CHEASE binary, so its output is kept as a fixture and the
notebook runs offline. This script is the only way that fixture is made.

What is stored beside the g-file is enough to check it later: the Solov'ev
parameters (so the notebook and ``test/test_island_chease_fixture.py`` rebuild
the same input), the CHEASE configuration, the binary's hash and source
revision, and the VAFT commit.

A path-run script imports whichever ``vaft`` the environment's editable
install points at, which in a worktree is usually not this checkout; the
script asserts it imported the tree it lives in.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPOSITORY / "test" / "data" / "island_sxr"
GEQDSK_NAME = "solovev_vest_chease.geqdsk"
PROVENANCE_NAME = "solovev_vest_chease.json"

#: The analytic input, passed to ``solovev_example``. The toroidal field puts
#: q = 2 near psi_N = 0.37 and q = 3 near 0.84, so both resonances the notebook
#: uses lie inside the plasma.
SOLOVEV_PARAMETERS = {
    "topology": "limited",
    "toroidal_field": 0.13,
    "a_parameter": 0.5,
    "resolution": 129,
}

#: CHEASE settings: the defaults, with the output box matched to the input grid
#: so the two equilibria are compared on the same number of nodes.
CHEASE_SETTINGS = {"nw": 129, "create_plot": False}


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    return result.stdout.strip()


def build_input():
    """The Solov'ev equilibrium with its q profile filled, as CHEASE's input needs."""
    import numpy as np

    from vaft.process.equilibrium import calculate_q_profile_from_psi, solovev_example

    eq = solovev_example(**SOLOVEV_PARAMETERS)
    psi_n = (eq.psi_1d - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    q = calculate_q_profile_from_psi(
        eq.psi, eq.r, eq.z, (eq.psi_1d, eq.f), eq.psi_axis, eq.psi_boundary,
        np.clip(psi_n, 0.01, 0.99), axis_rz=eq.magnetic_axis, cocos=eq.convention.cocos,
    )
    return dataclasses.replace(eq, q=np.asarray(q, dtype=float))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--keep-workdir", type=Path, default=None,
                        help="Keep CHEASE's working directory here instead of a temporary one.")
    args = parser.parse_args(argv)

    import vaft

    if not Path(vaft.__file__).resolve().is_relative_to(REPOSITORY):
        sys.exit(f"imported vaft from {vaft.__file__}, not from {REPOSITORY}; "
                 "set PYTHONPATH to this checkout")
    from vaft.code.chease import CHEASEConfig, refine_equilibrium
    from vaft.data.eqdsk import from_equilibrium

    executable = os.environ.get("CHEASE")
    if not executable:
        sys.exit("set CHEASE to the chease executable")
    executable = Path(executable).resolve()

    workdir = args.keep_workdir or Path(tempfile.mkdtemp(prefix="island-chease-"))
    workdir.mkdir(parents=True, exist_ok=True)
    config = CHEASEConfig(executable=str(executable), workdir=workdir, **CHEASE_SETTINGS)
    result = refine_equilibrium(from_equilibrium(build_input()), config)
    if not result.ok:
        sys.exit(f"CHEASE failed (return code {result.returncode}); see {workdir}")

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / GEQDSK_NAME
    shutil.copyfile(result.refined_geqdsk, target)
    provenance = {
        "schema": 1,
        "purpose": "CHEASE fixed-boundary refinement of the analytic Solov'ev equilibrium "
                   "that notebooks/analytic_island_model_and_synthetic_response_model.ipynb "
                   "compares against (#886)",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "workflow/island_sxr/generate_chease_fixture.py",
        "solovev_example": SOLOVEV_PARAMETERS,
        "chease_config": {key: value for key, value in CHEASE_SETTINGS.items()},
        "chease_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "chease_source_revision": _git("rev-parse", "--short", "HEAD", cwd=executable.parent) or None,
        "vaft_commit": _git("rev-parse", "--short", "HEAD", cwd=REPOSITORY) or None,
        "geqdsk_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (args.output / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {target} and {args.output / PROVENANCE_NAME}")
    if args.keep_workdir is None:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

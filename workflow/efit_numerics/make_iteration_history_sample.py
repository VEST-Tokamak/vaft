"""Regenerate the packaged EFIT iteration histories of shot 39915 (#1038).

    PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install \\
        python workflow/efit_numerics/make_iteration_history_sample.py --workdir /scratch/ih

``notebooks/convergence_study_of_efit.ipynb`` inspects a saved history rather
than running EFIT, so it needs one to ship. This writes two, from the same
k-files, differing in ``NXITER`` and in the ``MXITER`` cap that moves with it:

- ``39915_nxiter1.json`` -- the routine configuration (EFIT's default
  ``NXITER=1``, ``MXITER=100``);
- ``39915_nxiter3.json`` -- ``NXITER=3`` with ``MXITER = 514 // 3``, the largest
  cap that cannot overrun EFIT's compiled-in 515-entry iteration arrays (#171).

The input is the packaged sample's own pipeline product
(``vaft/data/samples/39915/source/pipeline-until-efit.json.gz``), whose
equilibrium already carries the k-file namelists, with ``TABLE_DIR`` pointed at
the packaged 129x129 table. Every slice is run: ``prepare_efit_inputs`` walks
the constraint slices by position, so a subset would write one slice's data
under another's name.

The written files hold no machine paths: the m-file and log references are
reduced to file names and the table directory to its repository path, so the
files are the same wherever they were produced.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
PRODUCT = REPOSITORY / "vaft/data/samples/39915/source/pipeline-until-efit.json.gz"
OUTPUT = REPOSITORY / "vaft/data/efit/iteration_history"
SHOT = 39915
#: EFIT indexes its per-iteration arrays by the cumulative counter against a
#: compiled-in 515 with no clamp (#171).
ITERATION_ARRAY_BOUND = 515
CASES = {"nxiter1": None, "nxiter3": 3}


def _portable(value: Any) -> Any:
    """``value`` with absolute paths reduced to what is meaningful off this machine."""
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        path = Path(value)
        try:
            return str(path.resolve().relative_to(REPOSITORY))
        except ValueError:
            return path.name
    return value


def run_case(ods, workdir: Path, inner_iterations: int | None):
    from vaft.code.efit.config import EFITNumericsConfig
    from vaft.code.efit.magnetic import EFITConfig, prepare_efit_inputs, run_efit

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    numerics = EFITNumericsConfig(
        inner_iterations=inner_iterations,
        max_iterations=(ITERATION_ARRAY_BOUND - 1) // inner_iterations if inner_iterations else 100,
    )
    times = [float(value) for value in ods["equilibrium.time"]]
    config = EFITConfig(
        workdir=workdir, shot=SHOT, times=times, args=("129",),
        numerics=numerics, iteration_history="summary",
    )
    inputs = prepare_efit_inputs(copy.deepcopy(ods), config)
    result = run_efit(inputs, config)
    if result.iteration_history is None:
        raise RuntimeError(f"EFIT did not run: {result.status} {result.reason}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path, help="scratch directory for the EFIT runs")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--efit-home", default=None)
    args = parser.parse_args(argv)
    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())

    from omas import load_omas_json

    import vaft
    from vaft.code.efit import EFITIterationHistory
    from vaft.data.resources import data_path

    print(f"vaft from {vaft.__file__}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as scratch:
        product = Path(scratch) / "product.json"
        product.write_bytes(gzip.decompress(PRODUCT.read_bytes()))
        ods = load_omas_json(str(product), consistency_check=False)
    # Point the k-file writer at the packaged table, as ab_efit_table does.
    # This edits the input cache the k-files are written from; the product
    # itself is never written back.
    table = str(Path(data_path("efit")).resolve()) + "/"
    parameters = ods["equilibrium.code.parameters"]
    for index in range(len(ods["equilibrium.time"])):
        parameters[f"time_slice.{index}.IN1.TABLE_DIR"] = table
        parameters[f"time_slice.{index}.IN1.INPUT_DIR"] = table

    args.output.mkdir(parents=True, exist_ok=True)
    for name, inner in CASES.items():
        result = run_case(ods, args.workdir / name, inner)
        history = EFITIterationHistory.from_dict(_portable(result.iteration_history.to_dict()))
        destination = history.write_json(args.output / f"{SHOT}_{name}.json")
        counts = [item.iterations_n for item in history]
        print(f"{destination}: {len(history)} slices, iterations {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

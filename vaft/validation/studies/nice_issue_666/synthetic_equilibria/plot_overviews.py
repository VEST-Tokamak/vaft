"""Collect NICE equilibrium ODS artifacts and render canonical overviews.

The machine context (wall) comes from the repository-only
``vaft/data/samples/41672/source/pipeline-until-efit.json.gz``, which does not
ship in the wheel, so its path is a required argument::

    python -m vaft.validation.studies.nice_issue_666.synthetic_equilibria.plot_overviews \
        --source vaft/data/samples/41672/source/pipeline-until-efit.json.gz \
        --native /tmp/nice331-solovev-20260914 --output .
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Rendering goes through vaft.plot; this script only saves figures, so it
# selects a non-interactive backend instead of importing pyplot itself.
os.environ.setdefault("MPLBACKEND", "Agg")

from vaft.code.nice import collect_nice_outputs
from vaft.omas import load, plot_equilibrium_overview, save


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="pipeline-until-efit.json.gz of shot 41672 (repository sample, not packaged)")
    parser.add_argument("--native", type=Path, default=Path("/tmp/nice331-solovev-20260914"),
                        help="root of the native NICE run directories (full_default, tokamaker_full_default)")
    parser.add_argument("--output", type=Path, default=Path.cwd(), help="where the ODS and PNG files are written")
    args = parser.parse_args(argv)

    machine = load(args.source)
    cases = {
        "solovev": args.native / "full_default",
        "tokamaker": args.native / "tokamaker_full_default",
    }
    for name, case in cases.items():
        result = collect_nice_outputs(case)
        if result.ods is None or not result.converged:
            raise RuntimeError(f"{name}: no converged NICE equilibrium ODS")
        ods = result.ods
        # Native NICE tables contain the equilibrium only. Add immutable machine
        # context for the canonical overview without altering equilibrium data.
        ods["wall"] = machine["wall"]
        ods["dataset_description.data_entry.pulse"] = 41672
        ods_path = args.output / f"nice_{name}_equilibrium_331ms.json.gz"
        png_path = args.output / f"nice_{name}_equilibrium_overview_331ms.png"
        save(ods, ods_path)
        figure, _axes = plot_equilibrium_overview(ods, time_slice=0, show=False)
        figure.savefig(png_path, dpi=180, bbox_inches="tight")
        print(f"{name}: {ods_path} {png_path}")


if __name__ == "__main__":
    main()

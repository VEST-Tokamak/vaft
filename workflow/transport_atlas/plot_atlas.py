"""Reference maps of the transport atlas (issue #1427): a thin CLI over vaft.plot.

The renderers are :func:`vaft.plot.transport_atlas.transport_atlas_scatter` and
:func:`vaft.plot.transport_atlas.transport_atlas_mode_branch`. This script reads
``atlas.csv`` and writes the reference figures:

* ``drive_space``: a/L_Te against a/L_ne, coloured by f_e. #1427 names a/L_Ti
  against a/L_Te, but under the #1414 Ti = Te policy a/L_Ti *is* a/L_Te and that
  map collapses onto the diagonal. ``ion_drive_space`` is drawn only from rows whose
  ``ti_lineage`` is not a fixed ratio of Te (lane K's inferred Ti).
* ``response_space``: Q_i/Q_GB against Q_e/Q_GB, coloured by the ion-scale gamma_max.
* ``mode_branch``: the drive plane coloured by the direction of the ion-scale mode.

    python plot_atlas.py --atlas ~/runs/campaign/atlas/transport/atlas.csv --out figs/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

REFERENCE = {
    "drive_space": ("a_over_lte", "a_over_lne", "f_e"),
    "response_space": ("qi_gb", "qe_gb", "gamma_max_ion_scale"),
}

#: Drawn only from rows whose Ti is not a fixed multiple of Te.
ION_DRIVE = ("a_over_lti", "a_over_lte", "f_e")


def independent_ti(table):
    """Rows whose Ti was not set as a ratio of Te (contract v1 ``ti_lineage``)."""
    lineage = table["ti_lineage"].astype(str)
    return ~(lineage.eq("ti_eq_te_assumed") | lineage.str.match(r"^ti_te_.*_(assumed|caller)"))


def main(argv: Optional[list[str]] = None) -> int:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import pandas as pd

    from vaft.plot.transport_atlas import transport_atlas_mode_branch, transport_atlas_scatter

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    table = pd.read_csv(args.atlas)
    table = table[table["tglf_status"] == "solved"]
    config = str(table["tglf_config"].iloc[0]) if len(table) else ""
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (x, y, c) in REFERENCE.items():
        fig, ax = transport_atlas_scatter(table, x, y, c, symlog=(name == "response_space"))
        ax.set_title(f"{name.replace('_', ' ')}: {config}, Tier A ({ax.vaft_drawn} surfaces)", fontsize=9)
        fig.savefig(args.out / f"{name}.png", dpi=150)
        print(name, ax.vaft_drawn)
    fig, ax = transport_atlas_mode_branch(table)
    counts = ax.vaft_counts
    ax.set_title(f"ion-scale mode direction: {config} ({counts['electron']} / {counts['ion']} surfaces)",
                 fontsize=9)
    fig.savefig(args.out / "mode_branch.png", dpi=150)
    print("mode_branch", counts)
    independent = table[independent_ti(table)]
    if len(independent):
        fig, ax = transport_atlas_scatter(independent, *ION_DRIVE)
        ax.set_title(f"ion drive space: Ti not a ratio of Te ({ax.vaft_drawn} surfaces)", fontsize=9)
        fig.savefig(args.out / "ion_drive_space.png", dpi=150)
        print("ion_drive_space", ax.vaft_drawn)
    else:
        print("ion_drive_space skipped: every row's Ti is a fixed ratio of Te")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

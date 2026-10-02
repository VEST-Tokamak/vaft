"""Reference maps of the transport atlas (issue #1427): a thin CLI over vaft.plot.

The renderers are :func:`vaft.plot.transport_atlas.transport_atlas_scatter` and
:func:`vaft.plot.transport_atlas.transport_atlas_mode_branch`. This script reads
``atlas.csv`` and writes the reference figures:

Axes are named "x, y" below.

* ``drive_space``: x = a/L_ne, y = a/L_Te, coloured by f_e; the same plane as
  ``mode_branch``. #1427 names (a/L_Ti, a/L_Te), but under the #1414 Ti = Te policy
  a/L_Ti *is* a/L_Te and that map collapses onto the diagonal. ``ion_drive_space``
  (x = a/L_Ti, y = a/L_Te) is drawn only from rows whose Ti is not a ratio of Te.
* ``response_space``: x = Q_i/Q_GB, y = Q_e/Q_GB (#1427), coloured by the ion-scale
  gamma_max.
* ``mode_branch``: x = a/L_ne, y = a/L_Te, coloured by the direction of the growing
  ion-scale mode.

    python plot_atlas.py --atlas ~/runs/campaign/atlas/transport/atlas.csv --out figs/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

REFERENCE = {
    "drive_space": ("a_over_lne", "a_over_lte", "f_e"),
    "response_space": ("qi_gb", "qe_gb", "gamma_max_ion_scale"),
}

#: Drawn only from rows whose Ti is not a fixed multiple of Te.
ION_DRIVE = ("a_over_lti", "a_over_lte", "f_e")


def independent_ti(table):
    """Rows whose Ti is not a fixed ratio of Te.

    Judged by the value, not the spelling: a ratio-derived Ti always carries its
    ``ti_te_ratio``, whatever status word its lineage ends in, and a row with no
    resolved lineage is not independent either.
    """
    import pandas as pd

    ratio = pd.to_numeric(table["ti_te_ratio"], errors="coerce")
    lineage = table["ti_lineage"]
    return ratio.isna() & lineage.notna() & ~lineage.astype(str).isin(("", "unresolved", "nan"))


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

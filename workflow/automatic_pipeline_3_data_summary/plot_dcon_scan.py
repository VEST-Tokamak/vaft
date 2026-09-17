#!/usr/bin/env python3
"""Operating-space scatter over a DCON scan, coloured by time or by stability.

The harvest this reads -- `vaft.code.gpec.read_dcon_scan` -- lives in the
package and is unit-tested there; only the drawing lives here. That split is
deliberate: a continuous colour channel is the one thing `vaft.plot`'s view
models cannot express (a `Series` carries no per-point colour, and the model
hierarchy is scheduled for retirement in the `vaft.view` work), and a scan is
cross-shot while the stage plots are per-shot -- so there is no `ValidationPlot`
slot for this figure at any granularity. `workflow/` is outside the package and
is where matplotlib may be used directly, so one honest scatter goes here rather
than a binned approximation of it going into the typed layer.

Usage:
    python plot_dcon_scan.py --workdir /srv/vest.filedb/public/39915/gpec \
        --shot 39915 --output stability_operating_space.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vaft.code.gpec import read_dcon_scan

#: The axes of the empirical kink / double-tearing operating space. Both are
#: DCON's own numbers for the equilibrium it was given, so the diagram shows
#: where the solved cases actually sat rather than where a separate
#: reconstruction thought they did.
DEFAULT_X = "q95"
DEFAULT_Y = "li3"


def _finite(rows, *names):
    """Rows carrying a finite float for every named column."""
    keep = []
    for row in rows:
        values = [row.get(name) for name in names]
        if any(value is None for value in values):
            continue
        if all(np.isfinite(float(value)) for value in values):
            keep.append(row)
    return keep


def plot_operating_space(rows, *, x=DEFAULT_X, y=DEFAULT_Y, colour="time_ms", output=None):
    """Scatter ``y`` against ``x``, one point per DCON run, coloured by ``colour``."""
    required = (x, y, "time_ms") if colour == "time_ms" else (x, y)
    usable = _finite(rows, *required)
    if not usable:
        # Name every column the filter used, not just the axes: a scan is
        # rejected outright when its run directories carry no numeric time
        # label, and blaming the axes sends the reader to the two columns that
        # were fine.
        named = ", ".join(repr(name) for name in required)
        raise SystemExit(f"no run in this scan carries finite values for all of: {named}")

    figure, axes = plt.subplots(figsize=(7.0, 5.5))
    x_values = np.array([float(row[x]) for row in usable])
    y_values = np.array([float(row[y]) for row in usable])

    if colour == "time_ms":
        points = axes.scatter(
            x_values, y_values,
            c=np.array([float(row["time_ms"]) for row in usable]),
            cmap="viridis", edgecolors="k", linewidths=0.4, alpha=0.85,
        )
        figure.colorbar(points, ax=axes, label="time [ms]")
    else:
        # `stable_free_boundary` is a tri-state: True, False, or None where DCON
        # computed no total-energy eigenvalue at all (`vac_flag=false`). Drawing
        # the third as a distinct class keeps "not computed" from reading as
        # "unstable".
        classes = {
            True: ("stable", "tab:green", "o"),
            False: ("unstable", "tab:red", "X"),
            None: ("not computed", "tab:gray", "s"),
        }
        for verdict, (label, color, marker) in classes.items():
            selected = [row for row in usable if row.get("stable_free_boundary") is verdict]
            if not selected:
                continue
            axes.scatter(
                [float(row[x]) for row in selected],
                [float(row[y]) for row in selected],
                color=color, marker=marker, label=label,
                edgecolors="k", linewidths=0.4, alpha=0.85,
            )
        axes.legend()

    shots = {row["shot"] for row in usable if row.get("shot") is not None}
    subject = f"shot {shots.pop()}" if len(shots) == 1 else f"{len(shots)} shots"
    axes.set_xlabel(x)
    axes.set_ylabel(y)
    axes.set_title(f"DCON operating space — {subject}, {len(usable)} runs")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()

    if output is not None:
        figure.savefig(output, dpi=300)
        print(f"wrote {output}")
    return figure, axes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path,
                        help="GPEC-suite case directory (<time>/<module>/nn=<mode>/).")
    parser.add_argument("--shot", type=int, default=None, help="Shot number for the rows.")
    parser.add_argument("--modes", default="1", help="Comma-separated toroidal mode numbers.")
    parser.add_argument("--x", default=DEFAULT_X, help="Column on the x axis.")
    parser.add_argument("--y", default=DEFAULT_Y, help="Column on the y axis.")
    parser.add_argument("--colour", default="time_ms", choices=("time_ms", "stability"),
                        help="Continuous time colouring, or the stability verdict.")
    parser.add_argument("--output", type=Path, default=Path("stability_operating_space.png"))
    args = parser.parse_args()

    modes = tuple(int(item) for item in str(args.modes).split(",") if item.strip())
    rows = read_dcon_scan(args.workdir, modes=modes, shot=args.shot)
    if not rows:
        raise SystemExit(f"no DCON run found under {args.workdir}")
    plot_operating_space(rows, x=args.x, y=args.y, colour=args.colour, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

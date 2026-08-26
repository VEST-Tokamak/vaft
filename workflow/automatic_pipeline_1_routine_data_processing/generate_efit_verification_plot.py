#!/usr/bin/env python3
"""Render the finalized-ODS EFIT verification overview."""

from __future__ import annotations

import argparse
from pathlib import Path

from omas import load_omas_json

from vaft.omas import plot_equilibrium_overview_verification
from vaft.plot import save_figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time-slice", default=0, type=int)
    args = parser.parse_args()

    ods = load_omas_json(str(args.input), consistency_check=False)
    if "equilibrium.time_slice" not in ods or not len(ods["equilibrium.time_slice"]):
        raise ValueError("finalized EFIT ODS contains no accepted equilibrium slices")
    if args.time_slice < 0 or args.time_slice >= len(ods["equilibrium.time_slice"]):
        raise IndexError(
            f"time slice {args.time_slice} is outside the finalized ODS range"
        )

    figure, _axes = plot_equilibrium_overview_verification(
        ods,
        time_slice=args.time_slice,
        show=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

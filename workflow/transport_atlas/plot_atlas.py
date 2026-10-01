"""Scatter maps over the transport atlas (issue #1427, reference plots).

``scatter`` puts any atlas column on x, y and colour, with an optional row filter. The
two reference maps of #1427 are built from it:

* drive space: a/L_Te against a/L_ne, coloured by the heat-channel fraction f_e.
  #1427 names a/L_Ti against a/L_Te, but under the #1414 Ti = Te policy a/L_Ti *is*
  a/L_Te and that map collapses onto the diagonal. ``ion_drive_space`` draws it only
  for rows whose ``ti_lineage`` is not a ratio policy (lane K's inferred Ti);
* response space: Q_i/Q_GB against Q_e/Q_GB, coloured by the ion-scale gamma_max.

``mode_branch`` splits a/L_Te against a/L_ne by the sign of the real frequency of
the fastest-growing ion-scale mode (``ky rho_s <= 1``). In TGLF a negative frequency
is the ion diamagnetic direction (``tglf/src/tglf_max.f90``). Marker size follows
``|Q_tot|/Q_GB``. The two colours are frequency directions, not ITG/TEM labels.

Marker shape separates the EFIT lineages; filled markers are ``good`` slices and open
ones ``admissible``. These are maps of model predictions under the atlas's declared
assumptions, not experimental operating boundaries.

    python plot_atlas.py --atlas ~/runs/campaign/atlas/transport/atlas.csv --out figs/
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Callable, Optional

MARKERS = {"magnetics": "o", "electron_kinetic": "s"}

REFERENCE = {
    "drive_space": ("a_over_lte", "a_over_lne", "f_e"),
    "response_space": ("qi_gb", "qe_gb", "gamma_max_ion_scale"),
}

#: Drawn only from rows whose Ti is not a fixed multiple of Te.
ION_DRIVE = ("a_over_lti", "a_over_lte", "f_e")


def _independent_ti(row: dict) -> bool:
    lineage = str(row.get("ti_lineage") or "")
    return not (lineage.startswith(("assumed_ti_te_", "inferred_ti_te_")) or "_ti_te_" in lineage)


def _label(column: str) -> str:
    """The column's mathematical symbol and unit, from the atlas schema's single source."""
    import importlib.util
    import sys

    name = "transport_atlas_build"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("build_atlas.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.axis_label(column)


def read_atlas(path: str | Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def scatter(rows, x: str, y: str, color: Optional[str] = None, *,
            where: Optional[Callable[[dict], bool]] = None, ax=None, symlog: bool = False):
    """Scatter two atlas columns, coloured by a third. Rows missing any of them are skipped.

    Returns the axes and the number of points drawn.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
    drawn = 0
    mappable = None
    # The colour scale spans only what is drawn: a filtered-out row must not stretch it.
    shown = [r for r in rows if (where is None or where(r))
             and _number(r.get(x)) is not None and _number(r.get(y)) is not None]
    values = [_number(r.get(color)) for r in shown] if color else []
    finite = [v for v in values if v is not None]
    vmin, vmax = (min(finite), max(finite)) if finite else (None, None)
    for lineage, marker in MARKERS.items():
        for label, filled in (("good", True), ("admissible", False)):
            picked = [r for r in rows if r.get("efit_lineage") == lineage and r.get("efit_quality") == label
                      and (where is None or where(r))]
            pts = [(_number(r.get(x)), _number(r.get(y)), _number(r.get(color)) if color else None)
                   for r in picked]
            pts = [p for p in pts if p[0] is not None and p[1] is not None and (not color or p[2] is not None)]
            if not pts:
                continue
            xs, ys, cs = zip(*pts)
            kwargs = dict(marker=marker, s=26, linewidths=0.8,
                          label=f"{lineage}, {label} ({len(pts)})")
            if color:
                if filled:
                    mappable = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=vmin, vmax=vmax, **kwargs)
                else:
                    import matplotlib as mpl

                    cmap = mpl.colormaps["viridis"]
                    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                    mappable = ax.scatter(xs, ys, facecolors="none",
                                          edgecolors=[cmap(norm(c)) for c in cs], **kwargs)
            else:
                ax.scatter(xs, ys, facecolors=None if filled else "none", **kwargs)
            drawn += len(pts)
    ax.set_xlabel(_label(x))
    ax.set_ylabel(_label(y))
    if symlog:
        ax.set_xscale("symlog", linthresh=1e-2)
        ax.set_yscale("symlog", linthresh=1e-2)
    if color and mappable is not None:
        import matplotlib as mpl

        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
        ax.figure.colorbar(sm, ax=ax, label=_label(color))
    ax.legend(fontsize=7, loc="best")
    return ax, drawn


def mode_branch(rows, ax=None):
    """a/L_Te against a/L_ne, coloured by the frequency direction of the ion-scale mode.

    Returns the axes and the number of points drawn in each direction.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
    counts = {}
    for name, sign, colour in ((r"electron direction ($\omega_r > 0$)", 1, "tab:blue"),
                               (r"ion direction ($\omega_r < 0$)", -1, "tab:red")):
        pts = []
        for r in rows:
            w = _number(r.get("omega_at_gamma_max_ion_scale"))
            x, y, q = (_number(r.get(k)) for k in ("a_over_lne", "a_over_lte", "q_tot_gb"))
            if None in (w, x, y) or w == 0 or (w > 0) != (sign > 0):
                continue
            pts.append((x, y, 6 + 10 * math.log10(1 + abs(q or 0.0)), r.get("efit_lineage")))
        counts[name] = len(pts)
        for lineage, marker in MARKERS.items():
            sub = [p for p in pts if p[3] == lineage]
            if sub:
                xs, ys, ss, _ = zip(*sub)
                ax.scatter(xs, ys, s=ss, marker=marker, facecolors="none", edgecolors=colour,
                           linewidths=0.8, label=f"{name}, {lineage} ({len(sub)})")
    ax.set_xlabel(_label("a_over_lne"))
    ax.set_ylabel(_label("a_over_lte"))
    ax.text(0.99, 0.01, r"$\omega_r$ of the fastest-growing mode with $k_y\rho_s\leq 1$;"
            "\n" r"marker area $\propto \log_{10}(1+|Q_e+Q_i|/Q_{\mathrm{GB}})$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.35")
    ax.legend(fontsize=6.5, loc="upper left")
    return ax, counts


def main(argv: Optional[list[str]] = None) -> int:
    import matplotlib

    matplotlib.use("Agg", force=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [r for r in read_atlas(args.atlas) if r.get("tglf_status") == "solved"]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (x, y, c) in REFERENCE.items():
        ax, drawn = scatter(rows, x, y, c, symlog=(name == "response_space"))
        config = rows[0].get("tglf_config", "") if rows else ""
        ax.set_title(f"{name.replace('_', ' ')}: {config}, Tier A ({drawn} surfaces)", fontsize=9)
        ax.figure.savefig(args.out / f"{name}.png", dpi=150)
        print(name, drawn)
    ax, counts = mode_branch(rows)
    config = rows[0].get("tglf_config", "") if rows else ""
    plus, minus = counts.values()
    ax.set_title(f"ion-scale mode direction: {config} ({plus} / {minus} surfaces)", fontsize=9)
    ax.figure.savefig(args.out / "mode_branch.png", dpi=150)
    print("mode_branch", counts)
    independent = [r for r in rows if _independent_ti(r)]
    if independent:
        ax, drawn = scatter(independent, *ION_DRIVE)
        ax.set_title(f"ion drive space: Ti not a ratio of Te ({drawn} surfaces)", fontsize=9)
        ax.figure.savefig(args.out / "ion_drive_space.png", dpi=150)
        print("ion_drive_space", drawn)
    else:
        print("ion_drive_space skipped: every row's Ti is a fixed ratio of Te")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

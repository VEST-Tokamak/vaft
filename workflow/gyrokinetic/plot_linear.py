"""Per-surface linear spectra: CGYRO vs TGLF on the same input vs TGLF SAT0-3 (#1354).

Reads ``linear.csv`` from ``build_linear.py`` and, with ``--sat-runs``, the #1482 TGLF
run tree (``<shot>-<ms>-<lineage>/tglf-sat<n>-<field>/<shot>/<lineage>/<ms>/r<r/a>/``),
whose spectra are parsed by Lane T's own reader
(:func:`vaft.code.gacode.tglf.outputs.collect_tglf_outputs` on that branch). Writes one
figure per (state, surface, field model) with :func:`vaft.plot.gyrokinetics.plot_linear_spectrum`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np

_TGLF_FIELD = {"es": "es", "em-aperp": "em-bper", "em-aperp-bpar": "em-bper-bpar"}


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(path: Path) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["shot"]), float(row["time_efit_s"]), row["efit_lineage"],
                   float(row["r_over_a"]), row["field_model"])
            groups.setdefault(key, []).append(row)
    return groups


def sat_spectra(root: Optional[Path], key: tuple) -> dict[str, dict]:
    """``{"TGLF SAT<n>": {"ky", "gamma", "omega"}}`` for one surface, when on disk."""
    if root is None:
        return {}
    from vaft.code.gacode.tglf.outputs import collect_tglf_outputs

    shot, time, lineage, r_over_a, field = key
    ms = int(round(time * 1000))
    spectra = {}
    for sat in range(4):
        directory = (root / f"{shot}-{ms}-{lineage}" / f"tglf-sat{sat}-{_TGLF_FIELD[field]}"
                     / str(shot) / lineage / f"{ms:05d}" / f"r{r_over_a:.2f}")
        outputs = collect_tglf_outputs(directory) if directory.is_dir() else None
        growth = None if outputs is None else getattr(outputs, "growth_rate", None)
        if growth is None or outputs.ky_spectrum is None:
            continue
        entry = {
            "ky": np.asarray(outputs.ky_spectrum, dtype=float),
            "gamma": np.asarray(growth, dtype=float),
            "omega": np.asarray(outputs.frequency, dtype=float),
        }
        # The linear eigenvalues do not depend on the saturation rule within a rule
        # family, so identical spectra are merged under one label instead of being
        # drawn on top of each other (SAT0/1 and SAT2/3 coincide on #1482's runs).
        for label, other in spectra.items():
            if all(np.array_equal(entry[k], other[k], equal_nan=True) for k in entry):
                spectra[label + f"/{sat}"] = spectra.pop(label)
                break
        else:
            spectra[f"TGLF SAT{sat}"] = entry
    return spectra


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--linear", type=Path, required=True, help="linear.csv")
    parser.add_argument("--sat-runs", type=Path, help="#1482 transport_sensitivity/runs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    from vaft.plot.gyrokinetics import plot_linear_spectrum
    from vaft.plot.style import save_figure

    args.out.mkdir(parents=True, exist_ok=True)
    count = 0
    for key, rows in sorted(load_rows(args.linear).items()):
        rows = sorted(rows, key=lambda r: _float(r["ky"]))
        cgyro = {
            "ky": np.array([_float(r["ky"]) for r in rows]),
            "gamma": np.array([_float(r["cgyro_gamma"]) for r in rows]),
            "omega": np.array([_float(r["cgyro_omega"]) for r in rows]),
            "converged": np.array([r["cgyro_qualified"] == "True" for r in rows]),
        }
        references = {"TGLF linear (same ky)": {
            "ky": cgyro["ky"],
            "gamma": np.array([_float(r["tglf_gamma"]) for r in rows]),
            "omega": np.array([_float(r["tglf_omega"]) for r in rows]),
        }}
        references.update(sat_spectra(args.sat_runs, key))
        shot, time, lineage, r_over_a, field = key
        title = f"{shot} @ {time:.3f} s ({lineage}), r/a={r_over_a:.2f}, {field}"
        figure, _ = plot_linear_spectrum(cgyro, references=references, title=title)
        name = f"{shot}_{int(round(time * 1000))}_{lineage}_r{r_over_a:.2f}_{field}.png"
        save_figure(figure, args.out / name, dpi=150)
        count += 1
    print(f"{count} figures in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

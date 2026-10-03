"""Saturated nonlinear CGYRO flux vs TGLF SAT0-3 on the same surface (#1354).

Reads one nonlinear run directory (``run_nonlinear.py``), forms the gyro-Bohm energy
flux time traces -- total, ions (all ion species) and electrons, summed over fields and
toroidal modes -- and averages them over a saturated window. The window is either given
(``--window t0 t1``, in ``a/c_s``) or, by default, the last half of the run, which is
recorded as an automatic choice rather than a judgement of saturation. The window's
first and second halves are averaged separately as well: when they disagree by more than
``--drift-tolerance``, the run is flagged ``not_stationary``.

CGYRO's gyro-Bohm unit is TGLF's (``Q_GB = n_e T_e c_s (rho_s/a)^2``, deuterium ``c_s``,
``rho_s`` from ``B_unit``), so ``Q/Q_GB`` compares directly with #1482's ``q_tot_gb``.

The locality QA (:func:`vaft.code.gacode.cgyro.locality_report`) is evaluated on the
same window: ``rho*``, ``L_x/a``, the measured radial correlation length ``l_corr`` and
``epsilon_local = l_corr / min(L_Ti, L_Te, L_n, L_q)``, with box and locality verdicts.

Writes ``nonlinear.json`` and ``flux_trace.png`` next to the run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _sat_reference(path: Optional[Path], record: dict, field: str) -> dict[int, float]:
    if path is None:
        return {}
    tglf_field = {"es": "es", "em-aperp": "em-bper"}[field]
    out: dict[int, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (int(row["shot"]) == int(record["shot"])
                    and abs(float(row["time_efit_s"]) - float(record["time_efit_s"])) < 6e-4
                    and row["efit_lineage"] == record["efit_lineage"]
                    and abs(float(row["r_over_a"]) - float(record["r_over_a"])) < 1e-3
                    and row["field_model"] == tglf_field and row.get("q_tot_gb")):
                out[int(row["sat_rule"])] = float(row["q_tot_gb"])
    return out


def _locality(run_dir: Path, outputs, window) -> dict:
    """Locality QA from the saved local-input summary (``run_nonlinear.py``)."""
    from types import SimpleNamespace

    from vaft.code.gacode.cgyro import locality_report

    path = run_dir / "local_summary.json"
    if not path.is_file():
        return {"reason": "no local_summary.json (run predates the locality QA)"}
    data = json.loads(path.read_text(encoding="utf-8"))
    norm = data.get("normalisation")
    local = SimpleNamespace(
        r_over_a=data["r_over_a"], geometry=data["geometry"],
        species={k: np.asarray(v) for k, v in data["species"].items()},
        normalisation=None if norm is None else SimpleNamespace(**norm))
    return locality_report(local, outputs, window=tuple(window))


def _record(run_dir: Path) -> dict:
    """``record.json``, or -- for a run killed before writing one -- the state key read
    back from the directory layout ``<shot>/<lineage>/<ms>/r<r/a>/<field>/<kind>``."""
    path = run_dir / "record.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    parts = run_dir.resolve().parts
    shot, lineage, ms, surface, field = parts[-6], parts[-5], parts[-4], parts[-3], parts[-2]
    return {"shot": int(shot), "efit_lineage": lineage, "time_efit_s": int(ms) / 1000.0,
            "r_over_a": float(surface.lstrip("r")), "field_model": field,
            "record": "reconstructed from the directory (run stopped before record.json)"}


def _mean(t: np.ndarray, y: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, t) / (t[-1] - t[0]))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="nonlinear run directory")
    parser.add_argument("--sensitivity", type=Path, help="#1482 sensitivity.csv")
    parser.add_argument("--window", type=float, nargs=2)
    parser.add_argument("--drift-tolerance", type=float, default=0.2)
    args = parser.parse_args(argv)

    from vaft.code.gacode.cgyro import collect_cgyro_outputs

    record = _record(args.run)
    run = collect_cgyro_outputs(args.run)
    if run is None or run.flux is None or run.time is None:
        raise SystemExit(f"{args.run}: no flux record")
    trace = run.flux_time_trace()              # (species, moment, time)
    t = np.asarray(run.time, dtype=float)[: trace.shape[-1]]
    z = np.asarray(run.equilibrium["species"]["z"], dtype=float)
    energy = trace[:, 1, :]
    q_e = energy[z < 0].sum(axis=0)
    q_i = energy[z > 0].sum(axis=0)
    q_tot = q_e + q_i

    if args.window:
        window, choice = tuple(args.window), "given"
    else:
        window, choice = (0.5 * float(t[-1]), float(t[-1])), "auto_last_half"
    mask = (t >= window[0]) & (t <= window[1])
    if np.count_nonzero(mask) < 4:
        raise SystemExit(f"window {window} holds fewer than 4 samples")
    tw = t[mask]
    half = len(tw) // 2
    averages = {name: _mean(tw, series[mask]) for name, series in
                (("q_tot_gb", q_tot), ("q_i_gb", q_i), ("q_e_gb", q_e))}
    first = _mean(tw[: half + 1], q_tot[mask][: half + 1])
    second = _mean(tw[half:], q_tot[mask][half:])
    drift = abs(second - first) / max(abs(averages["q_tot_gb"]), 1e-30)
    sat = _sat_reference(args.sensitivity, record, record["field_model"])
    locality = _locality(args.run, run, window)

    summary = {
        **{k: record[k] for k in ("shot", "time_efit_s", "efit_lineage", "r_over_a", "field_model")},
        "sim_time_end": float(t[-1]), "window": list(window), "window_choice": choice,
        **averages,
        "q_tot_std_in_window": float(np.std(q_tot[mask])),
        "half_window_drift": drift,
        "stationary": bool(drift <= args.drift_tolerance),
        "tglf_q_tot_gb": {f"sat{k}": v for k, v in sorted(sat.items())},
        "cgyro_over_tglf": {f"sat{k}": averages["q_tot_gb"] / v for k, v in sorted(sat.items()) if v},
        "resolution": (record.get("provenance") or {}).get("resolution"),
        # Box adequacy and the local approximation, judged on the same window as the
        # flux: a local result is quoted together with how local it actually is.
        "locality": locality,
    }
    (args.run / "nonlinear.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    from vaft.plot.gyrokinetics import plot_flux_trace
    from vaft.plot.style import save_figure

    figure, _ = plot_flux_trace(
        t, {"Q_tot": q_tot, "Q_i": q_i, "Q_e": q_e}, window=window,
        references={f"TGLF SAT{k}": v for k, v in sorted(sat.items())},
        title=(f"CGYRO nonlinear {record['shot']} r/a={record['r_over_a']:.2f} "
               f"{record['field_model']}"))
    save_figure(figure, args.run / "flux_trace.png", dpi=150)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

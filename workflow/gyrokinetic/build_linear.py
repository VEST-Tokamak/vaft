"""Build the linear CGYRO-vs-TGLF product from a ``run_linear.py`` tree (#1354).

Reads every ``record.json`` under ``--runs`` and writes, to ``--out``:

``linear.csv``
    One row per (state, surface, field model, ky): CGYRO ``gamma``/``omega`` (ion
    direction negative), the TGLF linear eigenvalue at the same ky and input, and their
    ratio and branch agreement.
``linear_summary.csv``
    One row per (state, surface, field model): the ky of maximum growth, ``gamma_max``
    and the branch there for both codes, plus -- when ``--sensitivity`` is given -- the
    #1482 SAT0-3 ``Q_tot/Q_GB`` on the same surface for the nonlinear step to pick from.
``schema.json``
    Column definitions, units and conventions.

Only qualified CGYRO points (linear converged) enter ``gamma_max``; unconverged points
are kept in ``linear.csv`` with ``cgyro_qualified = false``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = {
    "product": "vaft gyrokinetic_linear (Lane Y, #1354)",
    "conventions": {
        "ky": "k_y rho_s, rho_s = c_s/(e B_unit/m_D) (GACODE)",
        "gamma, omega": "c_s/a, c_s = sqrt(T_e/m_D)",
        "omega sign": "ion diamagnetic direction negative (TGLF's convention); CGYRO's "
                      "native sign is converted using the ion direction CGYRO reports",
        "branch": "'ion' if omega < 0 else 'electron'",
        "state key": "(shot, time_efit_s, efit_lineage), Lane T #1453 provisional",
        "inputs": "Ti=Te (#1414), H+/C6+ Z_eff 2, no ExB shear; CGYRO input renamed "
                  "from the TGLF local input, so both codes see one surface",
    },
    "linear.csv": {
        "shot": "int", "time_efit_s": "s", "efit_lineage": "str", "r_over_a": "-",
        "field_model": "es | em-aperp (CGYRO N_FIELD 1/2; TGLF USE_BPER)",
        "ky": "k_y rho_s",
        "cgyro_gamma": "c_s/a", "cgyro_omega": "c_s/a, ion negative",
        "cgyro_qualified": "linear frequency converged below FREQ_TOL",
        "cgyro_status": "converged | max_time (not converged at MAX_TIME, typically "
                        "marginal) | decayed (field underflowed: strongly damped, stable) "
                        "| failed",
        "cgyro_exit": "CGYRO EXIT line", "cgyro_sim_time": "a/c_s at the end",
        "tglf_gamma": "c_s/a, most unstable TGLF mode at this ky",
        "tglf_omega": "c_s/a, ion negative",
        "gamma_ratio": "cgyro_gamma / tglf_gamma",
        "branch_agree": "both codes' leading mode in the same diamagnetic direction",
    },
    "linear_summary.csv": {
        "ky_max_cgyro": "ky of max qualified CGYRO gamma",
        "gamma_max_cgyro": "c_s/a", "branch_cgyro": "at ky_max_cgyro",
        "ky_max_tglf": "ky of max TGLF linear gamma on the same ky grid",
        "gamma_max_tglf": "c_s/a", "branch_tglf": "at ky_max_tglf",
        "gamma_max_ratio": "gamma_max_cgyro / gamma_max_tglf",
        "n_qualified": "qualified CGYRO points of n_ky",
        "q_tot_gb_sat0..3": "#1482 Q_tot/Q_GB (column q_tot_gb) on this surface and field model",
        "gamma_max_tglf_sat0..3, ky_max_tglf_sat0..3": "#1482 TGLF transport-run spectrum peak, "
                                                        "its own ky grid",
    },
}


def _records(root: Path) -> Iterable[dict]:
    for path in sorted(root.rglob("record.json")):
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue


def _branch(omega: Optional[float]) -> Optional[str]:
    if omega is None or not math.isfinite(omega):
        return None
    return "ion" if omega < 0 else "electron"


def _cgyro_status(record: dict) -> Optional[str]:
    """converged | max_time | decayed (stable, amplitude underflow) | failed | None."""
    if not record:
        return None
    if record.get("status") == "decayed":
        return "decayed"
    if record.get("status") != "solved":
        return "failed"
    return "converged" if record.get("qualified") else "max_time"


def _key(record: dict) -> tuple:
    return (int(record["shot"]), round(float(record["time_efit_s"]), 4),
            str(record["efit_lineage"]), round(float(record["r_over_a"]), 3),
            str(record["field_model"]))


def build_rows(root: Path) -> list[dict]:
    cgyro: dict[tuple, dict] = {}
    tglf: dict[tuple, dict] = {}
    for record in _records(root):
        if "field_model" not in record:
            continue
        key = (*_key(record), round(float(record["ky"]), 4))
        (cgyro if record.get("code") == "cgyro" else tglf)[key] = record
    rows = []
    for key in sorted(set(cgyro) | set(tglf)):
        c = cgyro.get(key, {})
        t = tglf.get(key, {})
        eigenvalues = t.get("eigenvalues") if t.get("status") == "solved" else None
        tglf_omega, tglf_gamma = (eigenvalues or [[None, None]])[0]
        c_gamma, c_omega = c.get("gamma"), c.get("omega_ion_negative")
        ratio = (c_gamma / tglf_gamma) if (c_gamma is not None and tglf_gamma) else None
        rows.append({
            "shot": key[0], "time_efit_s": key[1], "efit_lineage": key[2],
            "r_over_a": key[3], "field_model": key[4], "ky": key[5],
            "cgyro_gamma": c_gamma, "cgyro_omega": c_omega,
            "cgyro_qualified": bool(c.get("qualified", False)),
            "cgyro_status": _cgyro_status(c),
            "cgyro_exit": c.get("exit_message"), "cgyro_sim_time": c.get("sim_time"),
            "tglf_gamma": tglf_gamma, "tglf_omega": tglf_omega,
            "gamma_ratio": ratio,
            "branch_agree": (None if _branch(c_omega) is None or _branch(tglf_omega) is None
                             else _branch(c_omega) == _branch(tglf_omega)),
        })
    return rows


def _sensitivity(path: Optional[Path]) -> dict[tuple, dict[int, dict[str, float]]]:
    """``{(shot, time, lineage, r/a, field): {sat: {...}}}`` from #1482's ``sensitivity.csv``.

    Columns as Lane T writes them (``build_sensitivity.py``): ``sat_rule``,
    ``field_model`` (``es``/``em-bper``), ``q_tot_gb`` and the TGLF spectrum peak
    ``gamma_max``/``ky_at_gamma_max``/``omega_at_gamma_max``. TGLF's ``em-bper`` is
    CGYRO's ``em-aperp`` (A_parallel only).
    """
    if path is None:
        return {}
    field_map = {"es": "es", "em-bper": "em-aperp", "em-bper-bpar": "em-aperp-bpar"}
    table: dict[tuple, dict[int, dict[str, float]]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            field = field_map.get(row.get("field_model", ""))
            if field is None or row.get("status") not in (None, "", "solved"):
                continue

            def number(name: str) -> Optional[float]:
                try:
                    return float(row[name])
                except (KeyError, TypeError, ValueError):
                    return None

            key = (int(row["shot"]), round(float(row["time_efit_s"]), 4), row["efit_lineage"],
                   round(float(row["r_over_a"]), 3), field)
            table.setdefault(key, {})[int(row["sat_rule"])] = {
                "q_tot_gb": number("q_tot_gb"),
                "gamma_max": number("gamma_max"),
                "ky_at_gamma_max": number("ky_at_gamma_max"),
                "omega_at_gamma_max": number("omega_at_gamma_max"),
            }
    return table


def build_summary(rows: list[dict], sensitivity: dict) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["shot"], row["time_efit_s"], row["efit_lineage"], row["r_over_a"],
               row["field_model"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, members in sorted(groups.items()):
        qualified = [r for r in members if r["cgyro_qualified"] and r["cgyro_gamma"] is not None]
        tglf_rows = [r for r in members if r["tglf_gamma"] is not None]
        best_c = max(qualified, key=lambda r: r["cgyro_gamma"], default=None)
        best_t = max(tglf_rows, key=lambda r: r["tglf_gamma"], default=None)
        entry = {
            "shot": key[0], "time_efit_s": key[1], "efit_lineage": key[2],
            "r_over_a": key[3], "field_model": key[4],
            "n_ky": len(members), "n_qualified": len(qualified),
            "ky_max_cgyro": None if best_c is None else best_c["ky"],
            "gamma_max_cgyro": None if best_c is None else best_c["cgyro_gamma"],
            "branch_cgyro": None if best_c is None else _branch(best_c["cgyro_omega"]),
            "ky_max_tglf": None if best_t is None else best_t["ky"],
            "gamma_max_tglf": None if best_t is None else best_t["tglf_gamma"],
            "branch_tglf": None if best_t is None else _branch(best_t["tglf_omega"]),
        }
        entry["gamma_max_ratio"] = (
            entry["gamma_max_cgyro"] / entry["gamma_max_tglf"]
            if entry["gamma_max_cgyro"] is not None and entry["gamma_max_tglf"] else None)
        for sat, values in sorted(sensitivity.get(key, {}).items()):
            entry[f"q_tot_gb_sat{sat}"] = values["q_tot_gb"]
            entry[f"gamma_max_tglf_sat{sat}"] = values["gamma_max"]
            entry[f"ky_max_tglf_sat{sat}"] = values["ky_at_gamma_max"]
        summary.append(entry)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, help="#1482 sensitivity.csv")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.runs)
    summary = build_summary(rows, _sensitivity(args.sensitivity))
    _write_csv(args.out / "linear.csv", rows)
    _write_csv(args.out / "linear_summary.csv", summary)
    (args.out / "schema.json").write_text(json.dumps(SCHEMA, indent=1), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "surfaces": len(summary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

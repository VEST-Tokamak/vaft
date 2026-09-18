"""When has a VEST EFIT reconstruction stopped changing? (#924)

    python workflow/efit_numerics/convergence_study.py --output /scratch/conv \\
        --tables-generated-129 /tables/129 --tables-generated-257 /tables/257 \\
        --table /scratch/conv/efit_convergence_study.json

VEST stops on ``ERRMIN = 1e-2`` (#171), and neither number EFIT reports can say
whether that is converged:

- the chi-square is the prescribed vessel current on stock EFIT (#918) and
  holds no magnetic information under legacy sigma (#891);
- ``terror``/``cerror`` -- stored by VAFT as
  ``convergence.grad_shafranov_deviation_value`` -- is the iteration increment
  ``max|psi - psi_previous| / |delta psi|``, not a test of the Grad-Shafranov
  equation.

So this measures convergence the way a converged answer is defined: the
equilibrium stops moving, and its own force balance, evaluated independently
from the g-file (:func:`vaft.process.equilibrium.grad_shafranov_residual`),
stops improving.

Design
------

``NXITER`` x ``ERRMIN`` on the routine 129x129 grid, a fixed handful of slices,
everything else routine.  ``MXITER`` is ``(515 - 1) // NXITER``: EFIT indexes
its per-iteration arrays by the cumulative counter against a compiled-in 515
with no clamp (#171), so that is the largest cap that cannot overrun them.

Two references per slice, because they answer different questions:

- **iteration reference** -- the tightest setting that stopped on its
  criterion on the 129 grid.  A setting's distance from it is iteration error.
- **grid reference** -- the same on 257x257.  Grid error is the distance
  between 129 and 257 at the same setting, both on tables generated together;
  the packaged 129 table is not bit-identical to a regenerated one, so that
  difference is measured separately rather than assumed away.

The recommendation is judged on iteration error only: the grid is not a
setting being chosen here.

Run it only on an EFIT whose reported Ip chi-square matches its solve (#918).
On stock EFIT every ``NXITER > 1`` stop test compares against a
vessel-inflated ``saiold`` and the scan measures that defect instead.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time as _clock
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
SEED_STUDY = REPOSITORY / "workflow" / "efit_numerics" / "seed_basin.py"
DOMAIN_STUDY = REPOSITORY / "workflow" / "efit_numerics" / "domain_grid.py"
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_convergence_study.json"

#: Ramp-up (or first plasma slice), peak Ip, ramp-down on each reference shot,
#: picked from the defect-free NXITER=1 run of #171.  41672 @ 331 ms is also
#: #664's reference slice.
SLICES: dict[int, tuple[float, ...]] = {
    39915: (0.315, 0.320, 0.327),
    41524: (0.327, 0.331, 0.334),
    41672: (0.321, 0.331, 0.342),
}
INNER_ITERATIONS = (1, 3, 5, 10)
ERROR_MINIMUM = (1.0e-2, 1.0e-3, 1.0e-4)
#: EFIT's compiled per-iteration array bound (#171).
ITERATION_ARRAY_BOUND = 515
ROUTINE_GRID = 129
FINE_GRID = 257
#: The settings run on the generated 129/257 pair: the whole cross.  A run is
#: about a second, and the tightest setting does not converge on every slice
#: (NXITER = 10, ERRMIN = 1e-4 fails in `bound` on 41672 @ 321 and 331), so a
#: short list could leave a slice with no grid pair at all.
FINE_SETTINGS = tuple((n, e) for n in INNER_ITERATIONS for e in ERROR_MINIMUM)
#: Which Green table a case ran on.  The packaged 129 table is the routine
#: one; the grid comparison uses a 129 and a 257 table from the same generator
#: on the same machine, so that nothing but the grid differs between a pair.
#: A table regenerated on another EFUND build is not bit-identical to the
#: packaged one (record by record, at most 2e-9 relative), and the packaged
#: 129 cases double as the control that this does not matter: on the 2026-09-18
#: run every packaged/generated 129 pair agreed to the digits EFIT writes.
PACKAGED = "packaged"
GENERATED = "generated"

#: #924's initial tolerances on iteration error.  Part of the result, not a
#: constant of nature: the summary reports every distance, so a reader can
#: re-judge against their own.
TOLERANCES = {
    "lcfs_rms_mm": 1.0,
    "betap_relative": 5.0e-3,
    "li_relative": 5.0e-3,
    "gs_edge_over_floor": 1.10,
}

#: Normalized flux beyond which a residual counts as edge, and within which core.
CORE_PSI_NORM = 0.5
EDGE_PSI_NORM = 0.8
MU0 = 4.0e-7 * np.pi
PROFILE_POINTS = 101


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


def max_iterations_for(inner_iterations: int, bound: int = ITERATION_ARRAY_BOUND) -> int:
    """The largest ``MXITER`` whose cumulative counter cannot reach ``bound``."""
    inner = int(inner_iterations)
    if inner < 1:
        raise ValueError("inner_iterations must be at least 1")
    return (int(bound) - 1) // inner


def case_name(grid: int, inner: int, error_minimum: float, table: str = PACKAGED) -> str:
    suffix = "" if table == PACKAGED else "gen"
    return f"g{grid}{suffix}_nx{inner}_err{error_minimum:.0e}"


def configurations(
    inner: Sequence[int] = INNER_ITERATIONS,
    error_minimum: Sequence[float] = ERROR_MINIMUM,
    fine: Sequence[tuple[int, float]] = FINE_SETTINGS,
) -> list[dict[str, Any]]:
    """Every case to run: the routine-table cross first, then the grid pairs.

    Each grid-reference setting runs on the generated 129 table and the
    generated 257 table, so their difference is the grid and nothing else; the
    same setting on the packaged table measures what regenerating the table
    alone does.
    """
    plan = [
        {"grid": ROUTINE_GRID, "table": PACKAGED, "inner_iterations": n, "error_minimum": e}
        for n in inner
        for e in error_minimum
    ]
    plan += [
        {"grid": grid, "table": GENERATED, "inner_iterations": n, "error_minimum": e}
        for n, e in fine
        for grid in (ROUTINE_GRID, FINE_GRID)
    ]
    for case in plan:
        case["max_iterations"] = max_iterations_for(case["inner_iterations"])
        case["name"] = case_name(case["grid"], case["inner_iterations"], case["error_minimum"], case["table"])
    return plan


def select_times(available: Iterable[float], wanted: Iterable[float], *, tolerance: float = 5.0e-7) -> list[float]:
    """The wanted slices, matched to the shot's own time base **by value**.

    Returns the matched values from ``available`` in ascending time.  A wanted
    time with no match raises rather than silently shrinking the study.
    """
    base = np.asarray(list(available), dtype=float)
    chosen = []
    for time in sorted(float(t) for t in wanted):
        hits = np.flatnonzero(np.abs(base - time) <= tolerance)
        if hits.size == 0:
            raise ValueError(f"no slice at {time:.6f} s in the shot's time base")
        chosen.append(float(base[hits[0]]))
    return chosen


# --------------------------------------------------------------------------
# What one equilibrium looks like
# --------------------------------------------------------------------------


def _grid(mapping: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    nw, nh = int(mapping["NW"]), int(mapping["NH"])
    r = float(mapping["RLEFT"]) + np.linspace(0.0, float(mapping["RDIM"]), nw)
    z = float(mapping["ZMID"]) - float(mapping["ZDIM"]) / 2.0 + np.linspace(0.0, float(mapping["ZDIM"]), nh)
    return r, z


def _psi_1d(mapping: Mapping[str, Any]) -> np.ndarray:
    size = np.asarray(mapping["PPRIME"]).size
    return np.linspace(float(mapping["SIMAG"]), float(mapping["SIBRY"]), size)


def grad_shafranov_metrics(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """The independent force-balance residual of one g-file, whole/core/edge.

    ``whole`` is the RMS of ``delta_star - source`` over every plasma point
    over the RMS of the source; ``core`` and ``edge`` are medians of the
    per-surface relative residual inside ``CORE_PSI_NORM`` and beyond
    ``EDGE_PSI_NORM``.  These are the #171 definitions, kept so the two
    studies stay comparable.
    """
    from vaft.process.equilibrium import grad_shafranov_residual

    r, z = _grid(mapping)
    result = grad_shafranov_residual(
        np.asarray(mapping["PSIRZ"], dtype=float), r, z,
        psi_1d=_psi_1d(mapping),
        pprime=np.asarray(mapping["PPRIME"], dtype=float),
        ffprime=np.asarray(mapping["FFPRIM"], dtype=float),
        psi_axis=float(mapping["SIMAG"]),
        psi_boundary=float(mapping["SIBRY"]),
        boundary_r=np.asarray(mapping["RBBBS"], dtype=float),
        boundary_z=np.asarray(mapping["ZBBBS"], dtype=float),
    )
    relative = np.asarray(result.relative, dtype=float)
    psi_norm = np.asarray(result.psi_norm, dtype=float)
    difference = result.delta_star[result.mask] - result.source[result.mask]
    whole = float(np.sqrt(np.mean(difference**2)) / result.scale)
    core = relative[psi_norm < CORE_PSI_NORM]
    edge = relative[psi_norm >= EDGE_PSI_NORM]
    return {
        "whole": whole,
        "core": float(np.nanmedian(core)) if np.isfinite(core).any() else float("nan"),
        "edge": float(np.nanmedian(edge)) if np.isfinite(edge).any() else float("nan"),
        "profile": {
            "psi_norm": psi_norm.tolist(),
            "relative": [float(v) if np.isfinite(v) else None for v in relative],
        },
    }


def _segment_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Distance from each point to the closed polyline ``polygon``."""
    start = polygon
    end = np.roll(polygon, -1, axis=0)
    edge = end - start
    length2 = np.einsum("ij,ij->i", edge, edge)
    length2 = np.where(length2 > 0.0, length2, 1.0)
    offset = points[:, None, :] - start[None, :, :]
    t = np.clip(np.einsum("pij,ij->pi", offset, edge) / length2, 0.0, 1.0)
    nearest = start[None, :, :] + t[..., None] * edge[None, :, :]
    return np.sqrt(((points[:, None, :] - nearest) ** 2).sum(-1)).min(axis=1)


def boundary_distance(a_r, a_z, b_r, b_z) -> dict[str, float]:
    """RMS and Hausdorff distance between two closed boundaries [m].

    Point-to-segment, both directions, so neither the sampling of one boundary
    nor which one is called the reference changes the answer.
    """
    a = np.c_[np.asarray(a_r, dtype=float), np.asarray(a_z, dtype=float)]
    b = np.c_[np.asarray(b_r, dtype=float), np.asarray(b_z, dtype=float)]
    ab = _segment_distances(a, b)
    ba = _segment_distances(b, a)
    both = np.concatenate([ab, ba])
    return {"rms": float(np.sqrt(np.mean(both**2))), "hausdorff": float(max(ab.max(), ba.max()))}


def _profiles(mapping: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """p, FF', q on a common normalized-flux abscissa, and j_phi on the midplane."""
    grid = np.linspace(0.0, 1.0, PROFILE_POINTS)
    out = {}
    for key, name in (("PRES", "p"), ("FFPRIM", "ffprime"), ("QPSI", "q")):
        values = np.asarray(mapping[key], dtype=float)
        out[name] = np.interp(grid, np.linspace(0.0, 1.0, values.size), values)
    return out


def midplane_current(mapping: Mapping[str, Any], radii: np.ndarray) -> np.ndarray:
    """Toroidal current density on Z = Z_axis at ``radii``, from the flux functions [A/m^2].

    ``R p'(psi) + FF'(psi) / (mu0 R)`` evaluated where the g-file's own flux
    puts each radius.  NaN outside the plasma, so a comparison uses only the
    chord both equilibria share.
    """
    from scipy.interpolate import RectBivariateSpline

    r, z = _grid(mapping)
    psi = np.asarray(mapping["PSIRZ"], dtype=float)
    spline = RectBivariateSpline(r, z, psi)
    axis, boundary = float(mapping["SIMAG"]), float(mapping["SIBRY"])
    flux = spline(radii, np.full_like(radii, float(mapping["ZMAXIS"])), grid=False)
    psi_norm = (flux - axis) / (boundary - axis)
    pprime = np.asarray(mapping["PPRIME"], dtype=float)
    ffprime = np.asarray(mapping["FFPRIM"], dtype=float)
    abscissa = np.linspace(0.0, 1.0, pprime.size)
    current = radii * np.interp(psi_norm, abscissa, pprime) + np.interp(
        psi_norm, abscissa, ffprime
    ) / (MU0 * radii)
    return np.where((psi_norm >= 0.0) & (psi_norm <= 1.0), current, np.nan)


def _relative_rms(value: np.ndarray, reference: np.ndarray) -> float:
    keep = np.isfinite(value) & np.isfinite(reference)
    if not keep.any():
        return float("nan")
    scale = np.sqrt(np.mean(reference[keep] ** 2))
    if scale == 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((value[keep] - reference[keep]) ** 2)) / scale)


SCALARS = ("betap", "li", "q95", "wmhd", "area", "volume", "rm", "zm", "terror", "ipmhd")


def compare(record: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    """One equilibrium against its reference on the same slice."""
    a, b = record["geqdsk"], reference["geqdsk"]
    distance = boundary_distance(a["RBBBS"], a["ZBBBS"], b["RBBBS"], b["ZBBBS"])
    out: dict[str, float] = {
        "lcfs_rms_mm": 1.0e3 * distance["rms"],
        "lcfs_hausdorff_mm": 1.0e3 * distance["hausdorff"],
        "axis_shift_mm": 1.0e3 * float(
            np.hypot(float(a["RMAXIS"]) - float(b["RMAXIS"]), float(a["ZMAXIS"]) - float(b["ZMAXIS"]))
        ),
    }
    for name in ("betap", "li", "q95", "wmhd", "area", "volume"):
        mine, theirs = record["scalars"].get(name), reference["scalars"].get(name)
        if mine is None or theirs is None or not theirs:
            out[f"{name}_relative"] = float("nan")
        else:
            out[f"{name}_relative"] = float(abs(mine - theirs) / abs(theirs))
    mine_p, ref_p = _profiles(a), _profiles(b)
    for name in ("p", "ffprime", "q"):
        out[f"{name}_profile_relative"] = _relative_rms(mine_p[name], ref_p[name])
    r_ref, _ = _grid(b)
    out["jphi_midplane_relative"] = _relative_rms(midplane_current(a, r_ref), midplane_current(b, r_ref))
    return out


# --------------------------------------------------------------------------
# References, distances, the recommendation
# --------------------------------------------------------------------------


def converged(record: Mapping[str, Any]) -> bool:
    """Stopped on its own criterion and left an equilibrium to measure."""
    return (
        record.get("geqdsk") is not None
        and not record.get("collapsed")
        and record.get("exit_path") == "iconvr=2"
    )


def tightness(case: Mapping[str, Any]) -> tuple[float, int]:
    """Sort key: smaller ERRMIN first, then more inner iterations."""
    return (float(case["error_minimum"]), -int(case["inner_iterations"]))


def _on(record: Mapping[str, Any], grid: int, table: str) -> bool:
    return record["case"]["grid"] == grid and record["case"].get("table", PACKAGED) == table


def choose_reference(
    records: Sequence[Mapping[str, Any]], grid: int, table: str = PACKAGED
) -> Mapping[str, Any] | None:
    """The tightest converged run of one slice on ``grid`` and ``table``, or None."""
    candidates = [r for r in records if _on(r, grid, table) and converged(r)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: tightness(r["case"]))[0]


def recommend(
    rows: Sequence[Mapping[str, Any]],
    tolerances: Mapping[str, float] = TOLERANCES,
) -> dict[str, Any]:
    """The cheapest routine-grid setting whose iteration error is within tolerance on every slice.

    ``rows`` are per (slice, case) records carrying ``case``, ``converged``,
    ``seconds``, ``iteration_error`` (distances to the 129 reference) and
    ``gs_edge_over_floor``.  A setting that fails to converge on any slice is
    not eligible.  Cheapest is total serial seconds over the slices.
    """
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not _on(row, ROUTINE_GRID, PACKAGED):
            continue
        by_case.setdefault(row["case"]["name"], []).append(row)
    verdicts = {}
    for name, members in by_case.items():
        failures = []
        for row in members:
            if not row["converged"]:
                failures.append(f"{row['shot']}@{row['time_ms']}: did not stop on its criterion")
                continue
            error = row["iteration_error"] or {}
            for key in ("lcfs_rms_mm", "betap_relative", "li_relative"):
                value = error.get(key, float("nan"))
                if not np.isfinite(value) or value > tolerances[key]:
                    failures.append(f"{row['shot']}@{row['time_ms']}: {key}={value:.3g}")
            ratio = row.get("gs_edge_over_floor", float("nan"))
            if not np.isfinite(ratio) or ratio > tolerances["gs_edge_over_floor"]:
                failures.append(f"{row['shot']}@{row['time_ms']}: gs_edge_over_floor={ratio:.3g}")
        verdicts[name] = {
            "case": members[0]["case"],
            "slices": len(members),
            "seconds": float(sum(row["seconds"] for row in members)),
            "within_tolerance": not failures,
            "failures": failures,
        }
    eligible = [v for v in verdicts.values() if v["within_tolerance"]]
    best = min(eligible, key=lambda v: v["seconds"]) if eligible else None
    return {
        "tolerances": dict(tolerances),
        "recommended": best["case"]["name"] if best else None,
        "cases": verdicts,
    }


def same_setting_pair(
    records: Sequence[Mapping[str, Any]], first: tuple[int, str], second: tuple[int, str]
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    """The tightest setting at which both (grid, table) runs converged."""
    def key(record):
        return (record["case"]["inner_iterations"], record["case"]["error_minimum"])

    a = {key(r): r for r in records if _on(r, *first) and converged(r)}
    b = {key(r): r for r in records if _on(r, *second) and converged(r)}
    shared = sorted(set(a) & set(b), key=lambda k: (k[1], -k[0]))
    return (a[shared[0]], b[shared[0]]) if shared else None


def _gs_summary(record: Mapping[str, Any] | None) -> dict[str, float] | None:
    if record is None or record.get("gs") is None:
        return None
    return {k: record["gs"][k] for k in ("whole", "core", "edge")}


def analyse(slices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-slice references, paired distances and the recommendation."""
    rows = []
    references = {}
    for block in slices:
        records = block["records"]
        coarse = choose_reference(records, ROUTINE_GRID, PACKAGED)
        fine = choose_reference(records, FINE_GRID, GENERATED)
        grid_pair = same_setting_pair(records, (ROUTINE_GRID, GENERATED), (FINE_GRID, GENERATED))
        table_pair = same_setting_pair(records, (ROUTINE_GRID, PACKAGED), (ROUTINE_GRID, GENERATED))
        key = f"{block['shot']}@{block['time_ms']}"
        references[key] = {
            "iteration_reference": coarse["case"]["name"] if coarse else None,
            "grid_reference": fine["case"]["name"] if fine else None,
            "gs_floor_129": _gs_summary(coarse),
            # Grid error: generated 129 against generated 257, same setting.
            "discretization": (
                {"setting": grid_pair[0]["case"]["name"], **compare(grid_pair[0], grid_pair[1]),
                 "gs_129": _gs_summary(grid_pair[0]), "gs_257": _gs_summary(grid_pair[1])}
                if grid_pair else None
            ),
            # Table-build error: packaged 129 against generated 129, same setting.
            "table_regeneration": (
                {"setting": table_pair[0]["case"]["name"], **compare(table_pair[0], table_pair[1])}
                if table_pair else None
            ),
        }
        floor = coarse["gs"]["edge"] if coarse else float("nan")
        for record in records:
            ok = converged(record)
            produced = record.get("geqdsk") is not None
            row = {
                "shot": block["shot"],
                "time_ms": block["time_ms"],
                "case": record["case"],
                "converged": ok,
                "outcome": record.get("outcome"),
                "exit_path": record.get("exit_path"),
                "iterations_n": record.get("iterations_n"),
                "seconds": record["seconds"],
                "terror": record["scalars"].get("terror") if produced else None,
                "cerror_final": record.get("cerror_final"),
                "gs": {k: record["gs"][k] for k in ("whole", "core", "edge")} if produced else None,
                "gs_edge_over_floor": (
                    record["gs"]["edge"] / floor if produced and np.isfinite(floor) and floor else float("nan")
                ),
                "iteration_error": (
                    compare(record, coarse) if produced and coarse and _on(record, ROUTINE_GRID, PACKAGED) else None
                ),
                "total_error": compare(record, fine) if produced and fine else None,
            }
            rows.append(row)
    return {"references": references, "rows": rows, "recommendation": recommend(rows)}


# --------------------------------------------------------------------------
# Running EFIT
# --------------------------------------------------------------------------


def _read_slice(workdir: Path, shot: int, time_ms: int) -> dict[str, Any]:
    """The g/a/m files one slice left behind, found by the time in their names."""
    from vaft.code.efit.slice_name import file_name_microseconds
    from vaft.data import read_aeqdsk, read_geqdsk
    from vaft.data.meqdsk import read_meqdsk

    found: dict[str, Path] = {}
    for path in workdir.iterdir():
        kind = path.name[:1]
        if kind not in "gam" or not path.name[1:].startswith(f"0{shot}."):
            continue
        micro = file_name_microseconds(path.name)
        if micro is not None and round(micro / 1000) == time_ms:
            found[kind] = path
    out: dict[str, Any] = {"geqdsk": None, "scalars": {}, "gs": None, "cerror": None, "cerror_final": None}
    if "g" in found:
        mapping = dict(read_geqdsk(found["g"]).mapping)
        out["geqdsk"] = mapping
        try:
            out["gs"] = grad_shafranov_metrics(mapping)
        except Exception as error:  # a degenerate equilibrium is a result, not a crash
            out["gs"] = {"whole": float("nan"), "core": float("nan"), "edge": float("nan"),
                         "profile": None, "error": repr(error)}
    if "a" in found:
        scalars = read_aeqdsk(found["a"]).scalars
        out["scalars"] = {k: float(scalars[k]) for k in SCALARS if k in scalars}
    if "m" in found:
        variable = read_meqdsk(found["m"]).variables.get("cerror")
        if variable is not None:
            history = np.asarray(variable.data, dtype=float).reshape(-1)
            history = history[np.isfinite(history) & (history > 0.0)]
            out["cerror"] = history.tolist()
            out["cerror_final"] = float(history[-1]) if history.size else None
    return out


def prepare_constraints(shot: int, product: Path, times: Sequence[float], *, workdir: Path, tables: str,
                        tstep: float, average_window: float, seed_study):
    """Constraints for the chosen slices only.

    ``prepare_efit_inputs`` overwrites ``equilibrium.time`` with the config's
    times and then walks slices by position, so handing it a subset of a
    longer constraints ODS writes one slice's data under another's name.  The
    constraints are therefore built for exactly the chosen slices.
    """
    from omas import load_omas_json
    from vaft.code.efit import generate_constraints_ods
    from vaft.omas.vacuum_magnetics import quality_gate
    from vaft.validation.efit_channels import decide_efit_channels, efit_probe_count

    baseline = _module(REPOSITORY / "workflow" / "efit_numerics" / "baseline_termination.py",
                       "baseline_termination")
    wrapper = _module(
        REPOSITORY / "workflow" / "automatic_pipeline_1_routine_data_processing" / "generate_constraints_ods.py",
        "generate_constraints_ods_wrapper",
    )
    ods = seed_study.load_product(product)
    window_times, _ = wrapper._select_times(ods, "auto", tstep, None, None)
    chosen = np.asarray(select_times(window_times, times), dtype=float)

    source = copy.deepcopy(ods)
    if "equilibrium" in source:
        del source["equilibrium"]
    source["equilibrium.time"] = chosen
    gated, _ = quality_gate(source, window=(float(window_times[0]), float(window_times[-1])))
    decisions = decide_efit_channels(gated, chosen, nbprobe=efit_probe_count(gated))
    workdir.mkdir(parents=True, exist_ok=True)
    generate_constraints_ods(
        gated, shot, str(workdir), tables, chosen,
        list(baseline.DEFAULT_UNCERTAINTY), list(baseline.DEFAULT_WEIGHTING),
        decisions=decisions, average_window=average_window,
    )
    built = load_omas_json(str(workdir / f"{shot}_constraints.json"), consistency_check=False)
    stored = np.asarray(built["equilibrium.time"], dtype=float)
    if stored.shape != chosen.shape or not np.allclose(stored, chosen, atol=5e-7):
        raise RuntimeError(f"{shot}: constraints carry {stored.tolist()}, asked for {chosen.tolist()}")
    return built, chosen


def _scientific(case: Mapping[str, Any]):
    from vaft.code.efit.config import EFITScientificConfig

    scientific = EFITScientificConfig()
    numerics = replace(
        scientific.numerics,
        inner_iterations=int(case["inner_iterations"]),
        error_minimum=float(case["error_minimum"]),
        max_iterations=int(case["max_iterations"]),
    )
    return replace(scientific, numerics=numerics)


def run_case(built, *, shot: int, times: Sequence[float], case: Mapping[str, Any], workdir: Path,
             efit: str) -> dict[str, Any]:
    """One configuration over the chosen slices of one shot, serially."""
    import shutil

    from vaft.code.efit.magnetic import EFITConfig, prepare_efit_inputs, run_efit
    from vaft.code.efit.termination import parse_slices

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    scientific = _scientific(case)
    config = EFITConfig(
        executable=efit, workdir=workdir, shot=shot, times=list(times), args=(str(case["grid"]),),
        profile=scientific.profile, initialization=scientific.initialization,
        numerics=scientific.numerics, constraints=scientific.constraints,
    )
    inputs = prepare_efit_inputs(copy.deepcopy(built), config)
    started = _clock.perf_counter()
    result = run_efit(inputs, config)
    seconds = _clock.perf_counter() - started
    log = workdir / "run_efit.out"
    text = log.read_text(errors="replace") if log.is_file() else (result.stdout or "")
    parsed = {int(item["time_ms"]): item for item in parse_slices(text)}

    records = []
    for time in times:
        time_ms = int(round(float(time) * 1000.0))
        entry = parsed.get(time_ms, {})
        record = {
            "case": dict(case),
            "time_ms": time_ms,
            # The library parser's classification, so this study and #171's
            # count the same thing: `iconvr=2`, `iterations_exhausted` or
            # `solver_error`; None means the log never mentions the slice.
            "exit_path": entry.get("exit_path"),
            "iterations_n": entry.get("iterations_n"),
            "collapsed": bool(entry.get("collapsed")),
            "solver_errors": entry.get("solver_errors", []),
            # The driver calls EFIT once per slice, so this is the slice's own
            # serial wall time; were several slices passed, it is their mean.
            "seconds": seconds / max(1, len(times)),
        }
        record.update(_read_slice(workdir, shot, time_ms))
        record["outcome"] = (
            "produced" if record["geqdsk"] is not None
            else "solver_error" if record["exit_path"] == "solver_error"
            else "collapsed" if record["collapsed"]
            else "no_output"
        )
        records.append(record)
    return {"seconds": seconds, "returncode": result.returncode, "records": records}


def _strip(record: Mapping[str, Any]) -> dict[str, Any]:
    """A record without the full g-file, for the stored table."""
    return {k: v for k, v in record.items() if k != "geqdsk"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE,
                        help="summary JSON; point it into --output on a shared checkout")
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--tables-generated-129", default=None,
                        help="129x129 table generated beside --tables-generated-257, with its EFUND manifest")
    parser.add_argument("--tables-generated-257", default=None,
                        help="257x257 table directory with its EFUND manifest")
    parser.add_argument("--shots", default=None, help="comma-separated subset of the study's shots")
    parser.add_argument("--cases", default=None, help="comma-separated case names (dry runs)")
    parser.add_argument("--tstep", type=float, default=0.001)
    parser.add_argument("--average-window", type=float, default=0.0005)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())
    seed_study = _module(SEED_STUDY, "seed_basin")
    domain = _module(DOMAIN_STUDY, "domain_grid")
    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    resolved = resolve_toolchain()
    efit = resolved.get("efit")
    if efit is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2

    tables = {(ROUTINE_GRID, PACKAGED): str(Path(data_path("efit")).resolve()) + "/"}
    plan = configurations()
    generated = {ROUTINE_GRID: args.tables_generated_129, FINE_GRID: args.tables_generated_257}
    if all(generated.values()):
        for grid, directory in generated.items():
            resolved_dir = Path(directory).expanduser().resolve()
            domain.verify_table(resolved_dir, {"grid": (grid, grid), "domain": domain.ROUTINE_DOMAIN})
            tables[(grid, GENERATED)] = str(resolved_dir) + "/"
    else:
        plan = [case for case in plan if case["table"] == PACKAGED]
        print("no generated 129/257 table pair: the discretization floor will not be measured",
              file=sys.stderr)
    if args.cases:
        wanted = set(args.cases.split(","))
        plan = [case for case in plan if case["name"] in wanted]

    install_record = Path(os.environ.get("EFITHOME", "")) / "vaft-external-install.json"
    reference = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }
    shots = [int(v) for v in args.shots.split(",")] if args.shots else sorted(SLICES)

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    blocks: list[dict[str, Any]] = []
    for shot in shots:
        # One EFIT call per slice.  Run back to back in one call, a slice
        # starts from wherever its predecessor left the solver, so a setting
        # that breaks one slice would also be charged for the next.  Separate
        # calls give every slice the same start under every setting, and a
        # serial wall time of its own.  The constraints carry TABLE_DIR, so
        # each (slice, table) gets its own; building them costs seconds.
        for time in SLICES[shot]:
            built_by_table = {}
            for grid, table in sorted({(case["grid"], case["table"]) for case in plan}):
                built_by_table[(grid, table)] = prepare_constraints(
                    shot, Path(data_path(products[shot])), [time],
                    workdir=output / f"shot_{shot}" / f"t{round(time * 1000):05d}" / f"constraints_{grid}_{table}",
                    tables=tables[(grid, table)],
                    tstep=args.tstep, average_window=args.average_window, seed_study=seed_study,
                )
            records = []
            for case in plan:
                built, chosen = built_by_table[(case["grid"], case["table"])]
                run = run_case(built, shot=shot, times=chosen, case=case,
                               workdir=output / f"shot_{shot}" / f"t{round(time * 1000):05d}" / case["name"],
                               efit=str(efit))
                record = run["records"][0]
                records.append(record)
                print(f"{shot}@{record['time_ms']} {case['name']}: {record['exit_path'] or record['outcome']} "
                      f"{record['iterations_n']} it in {run['seconds']:.1f} s", flush=True)
            blocks.append({"shot": shot, "time_ms": records[0]["time_ms"], "records": records})

    analysis = analyse(blocks)
    payload = {
        "schema_version": SCHEMA,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved),
        "efit_install": json.loads(install_record.read_text()) if install_record.is_file() else None,
        "tables": {f"{grid}/{table}": v for (grid, table), v in tables.items()},
        "slices": {str(k): list(v) for k, v in SLICES.items()},
        "plan": plan,
        "analysis": analysis,
        "records": [
            {"shot": b["shot"], "time_ms": b["time_ms"], "records": [_strip(r) for r in b["records"]]}
            for b in blocks
        ],
    }
    destination = args.table.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_json_default) + "\n",
                           encoding="utf-8")
    print(f"wrote {destination}")
    print(f"recommended: {analysis['recommendation']['recommended']}")
    return 0


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    raise SystemExit(main())

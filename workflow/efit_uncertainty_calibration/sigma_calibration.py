"""At what uncertainty scale are the magnetics informative and the fit convergent? (#891)

    python workflow/efit_uncertainty_calibration/sigma_calibration.py \\
        --output /scratch/sigma --table /scratch/sigma/efit_sigma_calibration.json

Under ``uncertainty_mode = "standard_deviation"`` (the contract #921 made mean
what EFIT fits against) the magnetic sigma is ``ConstraintErrors``' relative
error times ``|measured|`` -- 1 % on the probes -- while the residual the model
reaches is 4-12 %.  Nothing converges (#924, 0/81), even once the probes the
array review condemns are out (#977).  This scans the sigma, one family at a
time and both together, and reports the **range** over which the fit
converges, the magnetics are fitted to their stated sigma and are not thrown
away, and the #924 vertical drift is gone.  It does not pick the rung with the
smallest chi-square: the sigma is the chi-square's denominator, so a wide
enough sigma always wins that contest by removing the information.

Contract
--------

- ``standard_deviation`` sigma, binary weights, the diagnostics stage's
  gating as shipped (recorded faults + the array review), NXITER 1, the
  packaged 129 table, the nine #924 slices, ERRMIN 1e-2 and 1e-4.
- Ladder: the ``bpol_probe`` sigma alone, the ``flux_loop`` sigma alone, and
  both together, each multiplied by 1, 2, 4, 8, 16, 32 through
  ``uncertainty_scales`` (which divides the submitted sigma).  Ip, the PF
  currents and the diamagnetic flux keep their sigma.
- Floor: ``sigma_i = max(sigma_i, f * median |m|)`` over the family's fitted
  channels at that slice, ``f`` 0 or 2 %, written into a copy of the
  constraints before the k-file -- a near-zero reading otherwise gets a
  near-zero sigma.  The multiplier then applies to the floored sigma.
- The diamagnetic flux is held inactive (its sigma scaled by 1e8): under
  ``standard_deviation`` it becomes a strong constraint and crashes the fit
  once the magnetics are widened (#1027); it gets its own axis (#386).
- Profile basis (1,1) and (2,1): the routine (2,2) basis does not settle once
  the magnetics carry weight -- two of its directions trade off indefinitely
  (#1027) -- while these converge numerically.  Two bases, so the range's
  dependence on the basis is visible; each basis is judged on its own.
- EFIT's chi-square target ``SAICON`` is ``N + 3 sqrt(2N)`` per slice, N the
  fitted probes, loops, Ip and PF currents: the legacy 80 is below the
  expected chi-square of a statistical sigma, so a converged fit could never
  take the ``iconvr = 2`` exit (#1027).  The legacy reference keeps 80.
- One legacy-weight run per basis is the **reference branch** for
  displacement and continuity, nothing more.

Initialization
--------------

Every rung starts from the same cold state: the seed ellipse of the default
initialization, one EFIT call per slice in an emptied workdir, no g-file or
restart input.  There is no continuation along the ladder and no start from
the legacy solution: a narrow sigma that converges only because a wider one
found the branch first would be measuring path dependence, not the sigma.
Every run records an initialization fingerprint and the driver refuses a run
whose fingerprint differs from the first run's.  A cold-versus-warm start
comparison is a separate study.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
CONVERGENCE_STUDY = REPOSITORY / "workflow" / "efit_numerics" / "convergence_study.py"
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_sigma_calibration.json"

MULTIPLIERS = (1, 2, 4, 8, 16, 32)
FLOORS = (0.0, 0.02)
FAMILIES = ("bpol_probe", "flux_loop")
#: Which families each ladder moves together.
LADDERS = (("bpol_probe",), ("flux_loop",), ("bpol_probe", "flux_loop"))
ERROR_MINIMA = (1.0e-2, 1.0e-4)
#: (KPPCUR, KFFCUR): profile bases the magnetics can determine (#1027).
BASES = ((1, 1), (2, 1))
#: Scale that makes the diamagnetic flux inactive under standard_deviation (#1027, #386).
DIAMAGNETIC_INACTIVE_SCALE = 1.0e-8
#: Constraint families counted in N for the chi-square target.
FITTED_FAMILIES = ("bpol_probe", "flux_loop", "ip", "pf_current")
MAX_ITERATIONS = 514  # (515 - 1) // NXITER, NXITER = 1 (#171)

#: The operating-range rule (#891).  Part of the result, stated with it.
CRITERIA = {
    "all_converge_at": 1.0e-4,
    "reduced_chi2_range": (0.5, 5.0),
    "residual_over_narrowest": 1.5,
    "max_median_drift_mm": 5.0,
}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def rung_name(multipliers: Mapping[str, float], floor: float, basis: Sequence[int] = (1, 1)) -> str:
    return (f"p{basis[0]}f{basis[1]}_probe_x{multipliers['bpol_probe']:g}_loop_x{multipliers['flux_loop']:g}"
            f"_floor{100 * floor:g}pct")


def rungs(
    multipliers: Sequence[float] = MULTIPLIERS,
    floors: Sequence[float] = FLOORS,
    ladders: Sequence[Sequence[str]] = LADDERS,
    bases: Sequence[Sequence[int]] = BASES,
) -> list[dict[str, Any]]:
    """Every distinct rung, per basis: the baseline first, then each ladder in order, per floor.

    The point where a ladder sits at x1 is the same rung whichever ladder it
    belongs to, so it appears once, tagged with every ladder it serves.
    """
    out: dict[tuple, dict[str, Any]] = {}
    points = [
        (tuple(basis), float(floor), ladder, float(m))
        for basis in bases for floor in floors for ladder in ladders for m in multipliers
    ]
    for basis, floor, ladder, m in points:
        values = {family: (m if family in ladder else 1.0) for family in FAMILIES}
        key = (basis, values["bpol_probe"], values["flux_loop"], floor)
        rung = out.setdefault(key, {
            "name": rung_name(values, floor, basis),
            "uncertainty_mode": "standard_deviation",
            "basis": [int(basis[0]), int(basis[1])],
            "multipliers": values,
            "floor": floor,
            "ladders": [],
        })
        label = "+".join(ladder)
        if label not in rung["ladders"]:
            rung["ladders"].append(label)
    return list(out.values())


def legacy_reference(basis: Sequence[int] = (1, 1)) -> dict[str, Any]:
    return {
        "name": f"p{basis[0]}f{basis[1]}_legacy_reference",
        "uncertainty_mode": "legacy_weight",
        "basis": [int(basis[0]), int(basis[1])],
        "multipliers": {family: 1.0 for family in FAMILIES},
        "floor": 0.0,
        "ladders": [],
    }


def uncertainty_scales_for(rung: Mapping[str, Any]) -> dict[str, float]:
    """``uncertainty_scales`` divides the sigma, so a multiplier m is a scale 1/m.

    The diamagnetic flux is made inactive on every standard_deviation rung.
    """
    scales = {family: 1.0 / float(rung["multipliers"][family]) for family in FAMILIES}
    scales["diamagnetic_flux"] = DIAMAGNETIC_INACTIVE_SCALE
    return scales


def fitted_constraint_count(ods, slice_index: int = 0, families: Iterable[str] = FITTED_FAMILIES) -> int:
    """Constraints with a positive weight at one slice, over ``families``.

    ``ip`` and other scalar families are a single node; arrays count members.
    """
    root = f"equilibrium.time_slice.{slice_index}.constraints"
    count = 0
    for family in families:
        path = f"{root}.{family}"
        if path not in ods:
            continue
        node = ods[path]
        members = [f"{path}.{j}" for j in range(len(node))] if f"{path}.weight" not in ods else [path]
        for member in members:
            try:
                if float(ods[f"{member}.weight"]) > 0.0:
                    count += 1
            except Exception:
                continue
    return count


def chi_squared_target(fitted: int) -> float:
    """``N + 3 sqrt(2N)``: three standard deviations above a statistical chi-square's mean."""
    n = float(fitted)
    return n + 3.0 * np.sqrt(2.0 * n) if n > 0 else float("nan")


def apply_sigma_floor(ods, fraction: float, families: Iterable[str] = FAMILIES) -> list[dict[str, Any]]:
    """Raise each fitted channel's ``measured_error_upper`` to ``fraction`` x its family's median |m|.

    Per slice and per family, over the channels with a positive weight.
    Mutates ``ods`` (pass a copy) and returns, per slice and family, the floor
    and how many channels it raised; ``fraction == 0`` changes nothing.
    """
    changes: list[dict[str, Any]] = []
    if not fraction:
        return changes
    for index in range(len(ods["equilibrium.time_slice"])):
        root = f"equilibrium.time_slice.{index}.constraints"
        for family in families:
            path = f"{root}.{family}"
            if path not in ods:
                continue
            channels = []
            for j in range(len(ods[path])):
                node = f"{path}.{j}"
                try:
                    weight = float(ods[f"{node}.weight"])
                    measured = float(ods[f"{node}.measured"])
                except Exception:
                    continue
                if weight > 0.0 and np.isfinite(measured):
                    channels.append((j, measured))
            if not channels:
                continue
            floor = float(fraction) * float(np.median([abs(m) for _j, m in channels]))
            raised = 0
            for j, _m in channels:
                key = f"{path}.{j}.measured_error_upper"
                current = float(ods[key]) if key in ods else float("nan")
                if not np.isfinite(current) or abs(current) < floor:
                    ods[key] = floor
                    raised += 1
            changes.append({"slice": index, "family": family, "floor": floor, "raised": raised,
                            "fitted": len(channels)})
    return changes


# --------------------------------------------------------------------------
# Summary and the operating range
# --------------------------------------------------------------------------


def _median(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def summarize(records: Sequence[Mapping[str, Any]], rung_list: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per rung: convergence, fit against sigma, fit in the data's units, drift."""
    by_rung: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_rung.setdefault(record["rung"], []).append(record)
    rows = []
    for rung in rung_list:
        members = by_rung.get(rung["name"], [])
        tight = [r for r in members if r["error_minimum"] == CRITERIA["all_converge_at"]]
        converged = [r for r in tight if r["converged"]]
        slices = {(r["shot"], r["time_ms"]) for r in members}

        def fit(key):
            return _median((r.get("fit") or {}).get(key) for r in converged)

        rows.append({
            "rung": rung["name"],
            "uncertainty_mode": rung["uncertainty_mode"],
            "basis": list(rung.get("basis", [])),
            "multipliers": dict(rung["multipliers"]),
            "floor": rung["floor"],
            "ladders": list(rung["ladders"]),
            "slices": len(slices),
            "converged_tight": len(converged),
            "converged_loose": sum(1 for r in members if r["error_minimum"] != CRITERIA["all_converge_at"] and r["converged"]),
            "probe_reduced_chi2": fit("probe_reduced_chi2"),
            "loop_reduced_chi2": fit("loop_reduced_chi2"),
            "probe_residual": fit("probe_median_relative"),
            "loop_residual": fit("loop_median_relative"),
            "probe_sigma": fit("probe_sigma_median"),
            "median_abs_drift_mm": _median(abs(r["drift"]["dz_mm"]) for r in tight if r.get("drift")),
            "median_lcfs_to_reference_mm": _median(
                r["to_reference"]["lcfs_rms_mm"] for r in converged if r.get("to_reference")
            ),
            "median_iterations_tight": _median(r["iterations_n"] for r in tight),
        })
    return rows


def operating_range(rows: Sequence[Mapping[str, Any]], criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    """The rungs that satisfy every criterion, and why each other rung does not.

    (i) the tight fit converges on every slice; (ii) both magnetic families'
    reduced chi-square lie in the stated range; (iii) the probe residual is
    within the stated factor of its value at the narrowest converging sigma --
    a wider sigma must not buy convergence by giving the magnetics up; (iv)
    the median vertical drift between the loose and the tight stop is within
    the stated bound.
    """
    scan = [row for row in rows if row["uncertainty_mode"] == "standard_deviation"]
    low, high = criteria["reduced_chi2_range"]
    verdicts: dict[str, list[str]] = {}
    narrowest_by_basis: dict[str, str | None] = {}
    # Each profile basis is its own calibration: the residual a wide sigma is
    # compared with is the narrowest converging rung *of the same basis*.
    for basis in sorted({tuple(row.get("basis") or ()) for row in scan}):
        members = [row for row in scan if tuple(row.get("basis") or ()) == basis]
        verdicts.update(_basis_verdicts(members, criteria, low, high, narrowest_by_basis, basis))
    return {
        "criteria": {k: (list(v) if isinstance(v, tuple) else v) for k, v in criteria.items()},
        "narrowest_converging": narrowest_by_basis,
        "range": [name for name, failures in verdicts.items() if not failures],
        "failures": verdicts,
    }


def _basis_verdicts(scan, criteria, low, high, narrowest_by_basis, basis) -> dict[str, list[str]]:
    converging = [row for row in scan if row["slices"] and row["converged_tight"] == row["slices"]]
    narrowest = min(
        converging,
        key=lambda row: (row["multipliers"]["bpol_probe"] * row["multipliers"]["flux_loop"], row["floor"]),
        default=None,
    )
    narrowest_by_basis["p{}f{}".format(*basis) if basis else "unspecified"] = narrowest["rung"] if narrowest else None
    reference_residual = narrowest["probe_residual"] if narrowest else float("nan")
    verdicts: dict[str, list[str]] = {}
    for row in scan:
        failures = []
        if not row["slices"] or row["converged_tight"] < row["slices"]:
            failures.append(f"converges on {row['converged_tight']}/{row['slices']} slices")
        for family in ("probe", "loop"):
            value = row[f"{family}_reduced_chi2"]
            if not (np.isfinite(value) and low <= value <= high):
                failures.append(f"{family} reduced chi2 {value:.3g} outside [{low}, {high}]")
        if not np.isfinite(reference_residual):
            failures.append("no rung converges everywhere, so no narrowest residual to compare with")
        elif not (np.isfinite(row["probe_residual"])
                  and row["probe_residual"] <= criteria["residual_over_narrowest"] * reference_residual):
            failures.append(
                f"probe residual {row['probe_residual']:.3g} over "
                f"{criteria['residual_over_narrowest']}x the narrowest converging {reference_residual:.3g}"
            )
        drift = row["median_abs_drift_mm"]
        if not (np.isfinite(drift) and drift <= criteria["max_median_drift_mm"]):
            failures.append(f"median |drift| {drift:.3g} mm over {criteria['max_median_drift_mm']} mm")
        verdicts[row["rung"]] = failures
    return verdicts


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------


def require_same_start(baseline: str | None, fingerprint: str | None, label: str) -> str | None:
    """The baseline fingerprint, set by the first run; any later run must match it."""
    if fingerprint is None:
        raise RuntimeError(f"{label}: no initialization fingerprint recorded")
    if baseline is not None and fingerprint != baseline:
        raise RuntimeError(
            f"{label}: initialization fingerprint {fingerprint} differs from the baseline "
            f"{baseline}; every rung must cold-start from the same state"
        )
    return fingerprint if baseline is None else baseline


def _strip(record: Mapping[str, Any]) -> dict[str, Any]:
    keep = dict(record)
    keep.pop("geqdsk", None)
    keep.pop("cerror", None)
    if isinstance(keep.get("gs"), dict):
        keep["gs"] = {k: v for k, v in keep["gs"].items() if k != "profile"}
    return keep


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--shots", default=None, help="comma-separated subset of the study's shots")
    parser.add_argument("--rungs", default=None, help="comma-separated rung names (dry runs); the legacy reference always runs")
    parser.add_argument("--tstep", type=float, default=0.001)
    parser.add_argument("--average-window", type=float, default=0.0005)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())
    study = _module(CONVERGENCE_STUDY, "convergence_study")
    seed_study = study._module(study.SEED_STUDY, "seed_basin")
    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    resolved = resolve_toolchain()
    efit = resolved.get("efit")
    if efit is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2

    references = [legacy_reference(basis) for basis in BASES]
    plan = references + rungs()
    if args.rungs:
        wanted = set(args.rungs.split(","))
        plan = references + [rung for rung in plan[len(references):] if rung["name"] in wanted]
    tables = str(Path(data_path("efit")).resolve()) + "/"
    reference = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }
    shots = [int(v) for v in args.shots.split(",")] if args.shots else sorted(study.SLICES)
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)

    baseline_fingerprint: str | None = None
    records: list[dict[str, Any]] = []
    floors_applied: dict[str, list[dict[str, Any]]] = {}
    for shot in shots:
        for time in study.SLICES[shot]:
            tag = f"t{round(time * 1000):05d}"
            built, chosen = study.prepare_constraints(
                shot, Path(data_path(products[shot])), [time],
                workdir=output / f"shot_{shot}" / tag / "constraints", tables=tables,
                tstep=args.tstep, average_window=args.average_window, seed_study=seed_study,
            )
            geqdsk: dict[tuple[str, float], Any] = {}
            fitted = fitted_constraint_count(built)
            target = chi_squared_target(fitted)
            for rung in plan:
                ods = copy.deepcopy(built)
                changed = apply_sigma_floor(ods, rung["floor"])
                if changed:
                    floors_applied.setdefault(rung["name"], []).extend(
                        {**c, "shot": shot, "time_ms": round(time * 1000)} for c in changed
                    )
                scales = uncertainty_scales_for(rung) if rung["uncertainty_mode"] == "standard_deviation" else None
                for error_minimum in ERROR_MINIMA:
                    case = {
                        "grid": study.ROUTINE_GRID, "table": study.PACKAGED, "inner_iterations": 1,
                        "error_minimum": error_minimum, "max_iterations": MAX_ITERATIONS,
                        "kppcur": rung["basis"][0], "kffcur": rung["basis"][1],
                        # The legacy reference keeps EFIT's own target (80).
                        "chi_squared_target": target if rung["uncertainty_mode"] == "standard_deviation" else None,
                        "name": f"{rung['name']}_err{error_minimum:.0e}",
                    }
                    run = study.run_case(
                        ods, shot=shot, times=chosen, case=case,
                        workdir=output / f"shot_{shot}" / tag / case["name"], efit=str(efit),
                        uncertainty_mode=rung["uncertainty_mode"], uncertainty_scales=scales,
                    )
                    record = run["records"][0]
                    baseline_fingerprint = require_same_start(
                        baseline_fingerprint, (record.get("initialization") or {}).get("sha256"),
                        f"{shot}@{record['time_ms']} {case['name']}",
                    )
                    record["converged"] = study.converged(record)
                    geqdsk[(rung["name"], error_minimum)] = record.get("geqdsk") if record["converged"] else None
                    record.update({"rung": rung["name"], "shot": shot, "error_minimum": error_minimum,
                                   "basis": rung["basis"], "fitted_constraints": fitted,
                                   "chi_squared_target": case["chi_squared_target"]})
                    records.append(record)
                    print(f"{shot}@{record['time_ms']} {case['name']}: {record['exit_path'] or record['outcome']} "
                          f"{record['iterations_n']} it", flush=True)
            # Drift (loose -> tight, same rung) and distance to the legacy reference branch.
            for record in records[-2 * len(plan):]:
                ref_name = "p{}f{}_legacy_reference".format(*record["basis"])
                ref = geqdsk.get((ref_name, 1.0e-4)) or geqdsk.get((ref_name, 1.0e-2))
                loose = geqdsk.get((record["rung"], 1.0e-2))
                tight = geqdsk.get((record["rung"], 1.0e-4))
                if record["error_minimum"] == 1.0e-4 and loose is not None and tight is not None:
                    record["drift"] = study.rigid_vertical_shift(loose, tight)
                own = geqdsk.get((record["rung"], record["error_minimum"]))
                if own is not None and ref is not None:
                    shift = study.rigid_vertical_shift(own, ref)
                    record["to_reference"] = {
                        "lcfs_rms_mm": shift["lcfs_rms_mm"],
                        "dz_mm": shift["dz_mm"],
                        "axis_dz_mm": 1e3 * (float(own["ZMAXIS"]) - float(ref["ZMAXIS"])),
                    }

    rows = summarize(records, plan)
    verdict = operating_range(rows)
    install = Path(os.environ.get("EFITHOME", "")) / "vaft-external-install.json"
    payload = {
        "schema_version": SCHEMA,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved),
        "efit_install": json.loads(install.read_text()) if install.is_file() else None,
        "tables": tables,
        "slices": {str(k): list(v) for k, v in study.SLICES.items()},
        "initialization_fingerprint": baseline_fingerprint,
        "rungs": plan,
        "floors_applied": floors_applied,
        "summary": rows,
        "operating_range": verdict,
        "records": [_strip(r) for r in records],
    }
    destination = args.table.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True, default=study._json_default) + "\n",
                           encoding="utf-8")
    print(f"wrote {destination}")
    print(f"operating range: {verdict['range']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

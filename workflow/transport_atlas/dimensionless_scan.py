"""Local TGLF sensitivities in the dimensionless coordinates of the confinement scaling.

The coordinates (rho_*, beta, nu_*, q) are those of the confinement-time scaling fit in
``workflow/confinement_scaling``. Each solved surface of the TGLF SAT-rule sensitivity product is rerun with one dimensionless
parameter moved by a factor ``1 +- delta`` and the others held fixed. Only the inputs
that carry the parameter are changed, and they move together:

=========  ===========================================================================
nu_star    ``XNUE x f``
beta       ``BETAE x f`` and ``P_PRIME_LOC x f`` (the pressure-gradient drive scales
           with beta at fixed gradients)
q          ``Q_LOC x f``; ``Q_PRIME_LOC x f^2`` (magnetic shear fixed);
           ``P_PRIME_LOC x f`` (beta fixed); ``XNUE / f`` (nu_star fixed)
=========  ===========================================================================

The scalings follow from how ``vaft.code.gacode.tglf.inputs`` builds these inputs
(``q_prime_loc = (q/r)^2 s``, ``p_prime_loc = (|q|/r)(-beta*/8 pi)``). ``nu_star`` is
taken proportional to q, as in the Verdoolaege form the confinement scaling fits
(``vaft.formula.equilibrium.nu_star_from_n_T_B_R_epsilon_kappa_I``).

Held fixed in every scan: ``DEBYE`` (lambda_De / rho_s, a fifth dimensionless parameter
outside the confinement-scaling set, ~beta^-1/4 at fixed rho_*, nu_*; negligible at TGLF's ky range)
and ``VEXB_SHEAR`` (zero in the atlas inputs; a real diamagnetic-level ExB shear in
a/c_s units would scale with rho_* and break ``S_rho_star = 0``).

The sensitivity is the centred log difference

    S_x = [ln Q(+) - ln Q(-)] / [ln(1 + delta) - ln(1 - delta)]

taken on the gyro-Bohm flux. Two kinds of sensitivity are written separately:

* ``gb`` is ``Q / Q_GB``. TGLF is a local model, so ``Q / Q_GB`` does not depend on
  rho_* at all; ``S_rho_star = 0`` follows from the model's structure and needs no
  run.
* ``dim`` is ``Q / (n T c_s) = rho_*^2 x Q / Q_GB``, the flux in physical units at a
  fixed ``n T c_s`` scale. Here ``S_rho_star = 2`` and the other indices equal the
  ``gb`` ones.

**This is a consistency test against the confinement scaling, not the identity
``S_x^{tau_E} = -S_x^Q``.** Suppose transport were local, the same at every radius,
and the profile *shape* (a/L) stayed fixed while the parameters moved (a diffusive
response, Q linear in the gradient). Then ``Omega_i tau_E ~ rho_*^-3 / G(nu_*, beta, q)``
would give ``mu_rho = -3`` and ``mu_x = -S_x`` for the others.
VEST departs from every one of those assumptions:
- stiffness: if Q ~ grad^alpha at fixed power, ``mu_x ~ -S_x / alpha`` with alpha >> 1
  for stiff transport, and fixed-gradient slopes near threshold carry the
  threshold-shift amplification;
- the power deposition differs between surfaces;
- the edge is outside the surfaces scanned;
- the gradients are fixed rather than flux-matched;
- T_i = T_e is assumed.

So ``summary.csv`` sets the TGLF numbers *beside* the confinement scaling's fitted indices with their
errors. It does not test them for equality.

Usage (run inside one allocation; GACODE from ``$GACODE_ROOT``)::

    python dimensionless_scan.py run   --runs ~/runs/transport/v1/sensitivity --out ~/runs/transport/v1/dimensionless --workers 16
    python dimensionless_scan.py build --out ~/runs/transport/v1/dimensionless [--confinement <confinement-scaling product dir>]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

#: Input -> exponent of the factor f = 1 +- delta, per dimensionless parameter.
SCANS: dict[str, dict[str, int]] = {
    "nu_star": {"XNUE": 1},
    "beta": {"BETAE": 1, "P_PRIME_LOC": 1},
    "q": {"Q_LOC": 1, "Q_PRIME_LOC": 2, "P_PRIME_LOC": 1, "XNUE": -1},
}
SIGNS = {"plus": 1.0, "minus": -1.0}
DELTA = 0.1
QUANTITIES = ("qe", "qi", "q_tot")
#: The confinement scaling's names for the indices of Omega_i tau_E = C rho*^mu_rho beta^mu_beta nu*^mu_nu q^mu_q.
INDEX = {"rho_star": "mu_rho", "beta": "mu_beta", "nu_star": "mu_nu", "q": "mu_q"}
#: The confinement scaling's direct dimensionless fit with all four indices
#: (``workflow/confinement_scaling/extensions.py``).
DIRECT_FIT = "dimensionless:rho,beta,nu,q"


def perturbed(parameters: dict[str, Any], scan: str, sign: float, delta: float = DELTA) -> dict[str, Any]:
    """``parameters`` with ``scan`` moved by ``1 + sign * delta`` (module docstring)."""
    factor = 1.0 + sign * delta
    out = dict(parameters)
    for key, power in SCANS[scan].items():
        if key not in out:
            raise KeyError(f"input.tglf has no {key}; cannot scan {scan}")
        out[key] = float(out[key]) * factor**power
    return out


def sensitivity(plus: Optional[float], minus: Optional[float], delta: float = DELTA) -> Optional[float]:
    """Centred log-difference; ``None`` unless both fluxes are finite and positive."""
    if plus is None or minus is None or not (plus > 0 and minus > 0):
        return None
    if not (math.isfinite(plus) and math.isfinite(minus)):
        return None
    return (math.log(plus) - math.log(minus)) / (math.log1p(delta) - math.log1p(-delta))


def _fluxes(native: Any) -> dict[str, float]:
    energy = list(native.gbflux["energy"])
    return {"qe": float(energy[0]), "qi": float(sum(energy[1:])), "q_tot": float(sum(energy))}


# --------------------------------------------------------------------------- run
def _cases(runs: Path, configs: list[str]):
    """(config, state.json, r_over_a, native surface dir) for every solved surface."""
    for config in configs:
        for state_path in sorted(runs.glob(f"*/{config}/*/*/*/state.json")):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for surface in state["surfaces"]:
                if surface.get("status") != "solved":
                    continue
                directory = state_path.parent / f"r{float(surface['r_over_a']):.2f}"
                if (directory / "input.tglf").is_file():
                    yield config, state, float(surface["r_over_a"]), directory


def _run_one(directory: Path, parameters: dict[str, Any], gacode: dict[str, str]) -> dict[str, Any]:
    from vaft.code.gacode.tglf import TGLFConfig
    from vaft.code.gacode.tglf.inputs import TGLFInputs, write_input_tglf
    from vaft.code.gacode.tglf.runner import run_tglf

    record_path = directory / "record.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("parameters") == parameters and record.get("status") == "solved"
                and record.get("gacode") == gacode):
            return record          # resumable: same input and build, already solved
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    path = write_input_tglf(parameters, directory / "input.tglf")
    config = TGLFConfig(**gacode)
    result = run_tglf(TGLFInputs(workdir=directory, input_tglf=path, parameters=parameters), config,
                      check=False)
    native = result.outputs_native
    solved = bool(result.ok and native is not None and native.gbflux)
    record = {"parameters": parameters, "gacode": gacode, "status": "solved" if solved else "failed",
              "runtime_status": result.runtime_status, "returncode": result.returncode,
              "fluxes_gb": _fluxes(native) if solved else None,
              "version": None if native is None else native.version}
    record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def run(args: argparse.Namespace) -> int:
    from vaft.code.mitim.compare import read_input_tglf

    configs = args.configs or [f"tglf-sat{s}-em-bper" for s in range(4)]
    gacode = {key: value for key, value in (("home", args.gacode_home), ("platform", args.gacode_platform))
              if value}
    jobs = []
    for config, state, r, native_dir in _cases(args.runs, configs):
        base = read_input_tglf(native_dir / "input.tglf")
        leaf = (args.out / "runs" / config / str(state["shot"]) / state["efit_lineage"]
                / f"{round(state['time_efit_s'] * 1000):05d}" / f"r{r:.2f}")
        meta = {"config": config, "shot": state["shot"], "time_efit_s": state["time_efit_s"],
                "efit_lineage": state["efit_lineage"], "efit_quality": state["efit_quality"],
                "state_identity": state["state_identity"], "r_over_a": r,
                "sat_rule": (state.get("tglf_parameters") or {}).get("sat_rule"),
                "native_dir": str(native_dir), "delta": args.delta}
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "case.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        # A rerun of the unperturbed input checks the scan reproduces the stored run.
        jobs.append((leaf / "base", base))
        for scan in SCANS:
            for name, sign in SIGNS.items():
                jobs.append((leaf / f"{scan}_{name}", perturbed(base, scan, sign, args.delta)))
    print(json.dumps({"runs": len(jobs)}), flush=True)
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, directory, parameters, gacode): directory
                   for directory, parameters in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                solved = future.result().get("status") == "solved"
            except Exception as error:  # one broken case must not abort the batch
                print(json.dumps({"case": str(futures[future]), "error": f"{type(error).__name__}: {error}"}),
                      flush=True)
                solved = False
            failed += 0 if solved else 1
    (args.out / "scan.json").write_text(json.dumps({"delta": args.delta, "configs": configs,
                                                    "scans": SCANS, "runs": len(jobs),
                                                    "failed": failed}, indent=1), encoding="utf-8")
    print(json.dumps({"runs": len(jobs), "failed": failed}), flush=True)
    return 0


# --------------------------------------------------------------------------- build
ROW_COLUMNS = ["shot", "time_efit_s", "efit_lineage", "efit_quality", "r_over_a", "config", "sat_rule",
               "quantity", "base_gb", "stored_gb", "base_reproduction", "status", "reason"]
for _x in ("rho_star", *SCANS):
    ROW_COLUMNS += [f"S_gb_{_x}", f"S_dim_{_x}"]
for _x in SCANS:
    ROW_COLUMNS += [f"S_gb_{_x}_plus_side", f"S_gb_{_x}_minus_side"]

SUMMARY_COLUMNS = ["config", "sat_rule", "quantity", "index", "n_surfaces", "tglf_mu_median",
                   "tglf_mu_q25", "tglf_mu_q75", "fit548_direct", "fit548_direct_se",
                   "fit548_completed", "fit548_completed_se", "note"]


def _record(directory: Path) -> Optional[dict]:
    path = directory / "record.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build_rows(out: Path, delta: float) -> list[dict[str, Any]]:
    from vaft.code.gacode.tglf.outputs import TglfOutputs, collect_tglf_outputs

    rows = []
    for case_path in sorted((out / "runs").glob("*/*/*/*/r*/case.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        delta = float(case.get("delta", delta))     # each leaf's own delta, not the latest run's
        leaf = case_path.parent
        records = {d.name: _record(d) for d in leaf.iterdir() if d.is_dir()}
        native_dir = Path(case["native_dir"])
        stored = (TglfOutputs.read_json(native_dir / "outputs.json")
                  if (native_dir / "outputs.json").is_file() else collect_tglf_outputs(native_dir))
        stored_flux = _fluxes(stored) if stored is not None and stored.gbflux else {}
        base = records.get("base") or {}
        for quantity in QUANTITIES:
            row = {key: case[key] for key in ("shot", "time_efit_s", "efit_lineage", "efit_quality",
                                              "r_over_a", "config", "sat_rule")}
            row["quantity"] = quantity
            row["stored_gb"] = stored_flux.get(quantity)
            if base.get("status") != "solved":
                row.update(status="failed", reason="base rerun did not solve")
                rows.append(row)
                continue
            q0 = base["fluxes_gb"][quantity]
            row["base_gb"] = q0
            if row["stored_gb"] is not None:
                row["base_reproduction"] = abs(q0 - row["stored_gb"]) / max(abs(row["stored_gb"]), 1e-3)
            row["S_gb_rho_star"], row["S_dim_rho_star"] = 0.0, 2.0
            reasons = []
            for scan in SCANS:
                plus, minus = records.get(f"{scan}_plus") or {}, records.get(f"{scan}_minus") or {}
                fp = plus["fluxes_gb"][quantity] if plus.get("status") == "solved" else None
                fm = minus["fluxes_gb"][quantity] if minus.get("status") == "solved" else None
                value = sensitivity(fp, fm, delta)
                row[f"S_gb_{scan}"] = row[f"S_dim_{scan}"] = value
                if value is None:
                    reasons.append(f"{scan}: " + ("run failed" if fp is None or fm is None
                                                  else "flux not positive on both sides"))
                # One-sided slopes: a large difference means the response is not
                # log-linear over +-delta, and the centred value is an average.
                if fp is not None and q0 > 0 and fp > 0:
                    row[f"S_gb_{scan}_plus_side"] = (math.log(fp) - math.log(q0)) / math.log1p(delta)
                if fm is not None and q0 > 0 and fm > 0:
                    row[f"S_gb_{scan}_minus_side"] = (math.log(q0) - math.log(fm)) / -math.log1p(-delta)
            row["status"] = "ok" if not reasons else "partial"
            row["reason"] = "; ".join(reasons) or None
            rows.append(row)
    return rows


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def reference_indices(confinement: Optional[Path]) -> dict[str, dict[str, Optional[float]]]:
    """The confinement scaling's indices: the direct dimensionless fit and the completed free fit (primary:A)."""
    found: dict[str, dict[str, Optional[float]]] = {name: {} for name in INDEX}
    if confinement is None:
        return found
    terms = {"rho_star": "rho_star", "beta_t": "beta", "nu_star": "nu_star", "q_cyl": "q"}
    coefficients = confinement / "extensions" / "coefficients.csv"
    if coefficients.is_file():
        for row in csv.DictReader(open(coefficients, encoding="utf-8")):
            # Only the full four-index fit: extensions.py also writes reduced fits
            # (q left out, rho* and q only) under the same prefix.
            if row["fit"] == DIRECT_FIT and row["term"] in terms:
                found[terms[row["term"]]].update(direct=float(row["coef"]), direct_se=float(row["se_cluster"]))
    closures = confinement / "closures" / "closures.csv"
    if closures.is_file():
        for row in csv.DictReader(open(closures, encoding="utf-8")):
            if row["data"] == "primary:A" and row["model"] == "free":
                for name, mu in INDEX.items():
                    value, se = row.get(f"{mu}_completed"), row.get(f"{mu}_completed_se")
                    found[name].update(completed=float(value) if value else None,
                                       completed_se=float(se) if se else None)
    return found


def build_summary(rows: list[dict[str, Any]], references: dict) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(r["config"], r["sat_rule"], r["quantity"]) for r in rows}, key=str)
    for config, sat, quantity in keys:
        group = [r for r in rows if (r["config"], r["sat_rule"], r["quantity"]) == (config, sat, quantity)]
        for name in ("rho_star", *SCANS):
            # Consistency mapping (module docstring): mu_x = -S_x for local transport at
            # fixed profile shape, and mu_rho = -3 by gyro-Bohm construction.
            values = [-3.0 if name == "rho_star" else -r[f"S_gb_{name}"]
                      for r in group if r.get(f"S_gb_{name}") is not None]
            ref = references.get(name, {})
            out.append({"config": config, "sat_rule": sat, "quantity": quantity, "index": INDEX[name],
                        "n_surfaces": len(values), "tglf_mu_median": _percentile(values, 0.5),
                        "tglf_mu_q25": _percentile(values, 0.25), "tglf_mu_q75": _percentile(values, 0.75),
                        "fit548_direct": ref.get("direct"), "fit548_direct_se": ref.get("direct_se"),
                        "fit548_completed": ref.get("completed"),
                        "fit548_completed_se": ref.get("completed_se"),
                        "note": ("gyro-Bohm by construction of a local model, not a TGLF result"
                                 if name == "rho_star" else "consistency test at fixed profile shape, "
                                                            "not S^tau = -S^Q")})
    return out


def _write(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: "" if row.get(c) is None else row.get(c) for c in columns})


def build(args: argparse.Namespace) -> int:
    scan = json.loads((args.out / "scan.json").read_text(encoding="utf-8"))
    rows = build_rows(args.out, float(scan["delta"]))
    if not rows:
        raise SystemExit(f"no scan cases under {args.out / 'runs'}")
    summary = build_summary(rows, reference_indices(args.confinement))
    _write(args.out / "dimensionless_sensitivity.csv", rows, ROW_COLUMNS)
    _write(args.out / "summary.csv", summary, SUMMARY_COLUMNS)
    schema = {
        "description": "Local TGLF sensitivities in the #548 dimensionless coordinates (#1900 item 4).",
        "delta": scan["delta"], "scans": scan["scans"],
        "S_gb": "d ln(Q/Q_GB) / d ln x; S_gb_rho_star = 0 by locality (no run)",
        "S_dim": "d ln(Q/(n T c_s)) / d ln x = S_gb, except S_dim_rho_star = 2",
        "consistency_mapping": "mu_x = -S_gb_x and mu_rho = -3 hold only for local, radially uniform "
                               "transport at fixed profile shape (diffusive response); stiff transport "
                               "divides them by the stiffness exponent. summary.csv sets them beside #548, "
                               "it does not equate them",
        "fit548_direct": "Omega_i tau_E = C rho*^a beta^b nu*^c q^d on the Thomson subset (extensions.py)",
        "fit548_completed": "Kadomtsev-completed indices of the free engineering fit (closures.py, primary:A)",
        "base_reproduction": "|base rerun - stored run| / max(|stored|, 1e-3) in Q_GB",
    }
    (args.out / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "ok": sum(r.get("status") == "ok" for r in rows)}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--runs", type=Path, required=True, help="#1482 sensitivity root (tglf-sat*/ trees)")
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--configs", nargs="*", help="default: tglf-sat0..3-em-bper")
    runner.add_argument("--delta", type=float, default=DELTA)
    runner.add_argument("--workers", type=int, default=4)
    runner.add_argument("--gacode-home")
    runner.add_argument("--gacode-platform")
    builder = sub.add_parser("build")
    builder.add_argument("--out", type=Path, required=True)
    builder.add_argument("--confinement", type=Path, help="#548 product root (closures/, extensions/)")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else build(args)


if __name__ == "__main__":
    sys.exit(main())

"""TGLF saturation-rule and electromagnetic sensitivity tables (issue #1482).

The input is the trees ``run_tglf.py`` writes, one ``tglf-sat<n>-<field>/`` per
configuration, for the same states. No solver runs here. The output is a product
separate from the transport atlas:

    sensitivity.csv        one row per (shot, time_efit_s, efit_lineage, r_over_a, tglf_config)
    sensitivity_pairs.csv  one row per (shot, time_efit_s, efit_lineage, r_over_a):
                           spread across SAT rules at a fixed field model, and the
                           EM-vs-ES change at each SAT rule
    schema.json

No configuration is preferred. Every row names its full settings, and a difference
is reported as a number, never as a regime label. ``Delta(A, B) = (A - B) /
max(|A|, |B|, eps)`` with ``eps = 1e-3`` gyro-Bohm units, so the measure lies in
[-2, 2], and a pair that both sit below eps reads as about 0, not as noise blown up.

The k_y peak: ``write_tglf_sum_flux_spectrum`` (tglf_inout.f90) writes entry k as the
log-trapezoid integral of the flux over the interval [ky_{k-1}, ky_k] (with ky_0 = 0),
``dky0*Q_{k-1} + dky1*Q_k`` where ``dky0 + dky1 = ky_k - ky_{k-1}``. Dividing an entry
by its interval width gives the interval's mean spectral density, so the peak is
reported at the midpoint of the interval with the largest
``|sum over species and fields of entry_k| / (ky_k - ky_{k-1})``; the signed sum is
taken first, as for ``ky_q_mean``. The result does not depend on how the grid is spaced.
That weighting is only formed when ``KYGRID_MODEL != 0``. For ``KYGRID_MODEL = 0`` (the
linear grid ``ky_k = k * ky_1``, tglf_kygrid.f90) TGLF leaves ``dky0 = 0, dky1 = ky_1``,
so entry k is ``ky_1 * Q_k``: a right-endpoint rectangle that samples the flux AT
``ky_k``. The peak is then ``ky_k`` of the largest ``|entry_k| / ky_1``. Every other
value of ``KYGRID_MODEL`` takes the ``.ne.0`` branch, so the trapezoid rule applies. The
grid model is read from the run's recorded ``tglf_parameters`` (``extra_parameters``,
default 1); a value that is not an integer leaves ``ky_q_peak`` empty.

    python build_sensitivity.py --runs ~/runs/transport/sensitivity --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

EPS = 1e-3
QUANTITIES = ("qe_gb", "qi_gb", "gamma_e_gb", "q_tot_gb")
FIELD_MODELS = ("es", "em-bper", "em-bper-bpar")

SCHEMA = {
    "shot": ("-", "key", "VEST shot number"),
    "time_efit_s": ("s", "key", "equilibrium slice time"),
    "efit_lineage": ("-", "key", "magnetics | electron_kinetic"),
    "r_over_a": ("-", "key", "surface, GACODE rmin/rmin[-1]"),
    "tglf_config": ("-", "key", "tglf-sat<n>-<field>: discovery label; full settings in the columns below"),
    "sat_rule": ("-", "provenance", "TGLF SAT_RULE"),
    "field_model": ("-", "provenance", "es | em-bper | em-bper-bpar"),
    "use_bper": ("-", "provenance", "TGLF USE_BPER"),
    "use_bpar": ("-", "provenance", "TGLF USE_BPAR"),
    "tglf_parameters": ("-", "provenance", "every physics setting of the run, JSON"),
    "efit_quality": ("-", "label", "#1331 slice label"),
    "ti_lineage": ("-", "label", "how Ti was resolved"),
    "state_identity": ("-", "provenance", "sha256 of the resolved upstream state (equal across configurations)"),
    "tglf_run_identity": ("-", "provenance", "sha256 of state + settings + surface + revision"),
    "gacode_revision": ("-", "provenance", "GACODE tree used"),
    "status": ("-", "provenance", "solved | failed | not_ready | missing_outputs (native tree pruned)"),
    "qe_gb": ("Q_GB", "tglf_predicted", "electron energy flux"),
    "qi_gb": ("Q_GB", "tglf_predicted", "ion energy flux summed over ions"),
    "gamma_e_gb": ("Gamma_GB", "tglf_predicted", "electron particle flux"),
    "q_tot_gb": ("Q_GB", "tglf_predicted", "qe_gb + qi_gb"),
    "qe_over_qi": ("-", "tglf_predicted", "qe_gb / qi_gb; empty when |qi_gb| < eps"),
    "gamma_max": ("c_s/a", "tglf_predicted", "largest growth rate over ky and modes"),
    "ky_at_gamma_max": ("-", "tglf_predicted", "ky rho_s at gamma_max"),
    "omega_at_gamma_max": ("c_s/a", "tglf_predicted", "real frequency at gamma_max (unlabelled sign)"),
    "gamma_max_ion_scale": ("c_s/a", "tglf_predicted", "largest growth rate at ky rho_s <= 1"),
    "ky_at_gamma_max_ion_scale": ("-", "tglf_predicted", "ky rho_s at gamma_max_ion_scale"),
    "omega_at_gamma_max_ion_scale": ("c_s/a", "tglf_predicted", "real frequency at gamma_max_ion_scale"),
    "n_unstable_ky": ("-", "tglf_predicted", "ky points with a growing mode"),
    "ky_q_mean": ("-", "tglf_predicted", "energy-flux-weighted mean ky rho_s"),
    "ky_q_peak": ("-", "tglf_predicted", "midpoint of the ky interval with the largest mean |Q| density (module docstring)"),
    "f_em": ("-", "tglf_predicted", "share of sum|Q| in magnetic fields; empty for es"),
}

PAIR_SCHEMA = {
    "shot": ("-", "key", "VEST shot number"),
    "time_efit_s": ("s", "key", "equilibrium slice time"),
    "efit_lineage": ("-", "key", "magnetics | electron_kinetic"),
    "r_over_a": ("-", "key", "surface"),
    "efit_quality": ("-", "label", "#1331 slice label"),
    "n_configs": ("-", "provenance", "distinct solved configurations at this surface"),
}
for _q in QUANTITIES:
    for _field in FIELD_MODELS:
        PAIR_SCHEMA[f"{_q}_sat_spread_{_field}"] = (
            "-", "model_derived",
            f"max over SAT-rule pairs of |Delta({_q})| at field model {_field}")
    for _sat in range(4):
        PAIR_SCHEMA[f"{_q}_em_vs_es_sat{_sat}"] = (
            "-", "model_derived", f"Delta({_q}) of em-bper against es at SAT_RULE={_sat}")


def _atlas_module():
    name = "transport_atlas_build"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("build_atlas.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def delta(a: Optional[float], b: Optional[float], eps: float = EPS) -> Optional[float]:
    """``(a - b) / max(|a|, |b|, eps)``, or ``None`` when either side is missing or not finite."""
    a, b = _finite(a), _finite(b)
    if a is None or b is None:
        return None
    return (a - b) / max(abs(a), abs(b), eps)


def ky_grid_model(parameters: dict) -> Optional[int]:
    """TGLF ``KYGRID_MODEL`` of a run from its recorded ``tglf_parameters`` (default 1).

    ``extra_parameters`` is recorded verbatim and upper-cased only when the input file
    is written, so the key is matched without regard to case.
    """
    extra = parameters.get("extra_parameters") or {}
    for key, value in extra.items():
        if str(key).upper() == "KYGRID_MODEL":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return 1


def ky_peak(native: Any, parameters: Optional[dict] = None) -> Optional[float]:
    """ky rho_s of the largest |Q| spectral density, by the run's grid model (module docstring)."""
    ky = getattr(native, "ky_spectrum", None)
    spectrum = getattr(native, "sum_flux_spectrum", None)
    if ky is None or spectrum is None or ky.size < 2 or spectrum.shape[2] != ky.size:
        return None
    model = ky_grid_model(parameters or {})
    if model is None:
        return None
    ky = np.asarray(ky, dtype=float)
    entries = np.abs(np.nan_to_num(spectrum[..., 1]).sum(axis=(0, 1)))
    if model == 0:
        # dky0 = 0, dky1 = ky_1 for every k: entry_k = ky_1 * Q_k, sampled at the node.
        if not ky[0] > 0 or not np.any(entries > 0):
            return None
        return float(ky[int(np.argmax(entries))])
    lower = np.concatenate(([0.0], ky[:-1]))
    width = ky - lower
    density = np.divide(entries, width, out=np.zeros_like(entries), where=width > 0)
    if not np.any(density > 0):
        return None
    k = int(np.argmax(density))
    return float(0.5 * (lower[k] + ky[k]))


def field_model(parameters: dict) -> str:
    bper, bpar = bool(parameters.get("use_bper")), bool(parameters.get("use_bpar"))
    return "em-bper-bpar" if (bper and bpar) else "em-bper" if bper else "es" if not bpar else "es-bpar"


def build_rows(runs: Path) -> list[dict[str, Any]]:
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    from vaft.process.transport_state import EFIT_QUALITIES

    atlas = _atlas_module()
    rows = []
    seen: dict[tuple, Path] = {}
    for path in sorted(Path(runs).glob("**/tglf-sat*/*/*/*/state.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        parameters = state.get("tglf_parameters") or {}
        label = state.get("tglf_config")
        expected = f"tglf-sat{parameters.get('sat_rule')}-{field_model(parameters)}"
        if parameters.get("sat_rule") not in (0, 1, 2, 3) or field_model(parameters) not in FIELD_MODELS:
            raise ValueError(f"{path}: (SAT_RULE, field model) = ({parameters.get('sat_rule')}, "
                             f"{field_model(parameters)}) is not in the sensitivity space")
        if label != expected:
            raise ValueError(f"{path}: tglf_config {label!r} does not name its settings ({expected})")
        if state["efit_quality"] not in EFIT_QUALITIES:
            raise ValueError(f"{path}: efit_quality {state['efit_quality']!r} is not good/admissible")
        key = (label, state["state_identity"])
        if key in seen:
            # Two copies of one configuration on one state (a re-run beside an old tree)
            # would otherwise be merged silently, last copy winning.
            raise ValueError(f"{label} on one state appears twice: {seen[key]} and {path}")
        seen[key] = path
        for surface in state["surfaces"]:
            r = float(surface["r_over_a"])
            row = {
                "shot": state["shot"], "time_efit_s": state["time_efit_s"],
                "efit_lineage": state["efit_lineage"], "r_over_a": r,
                "tglf_config": state.get("tglf_config"),
                "sat_rule": parameters.get("sat_rule"), "field_model": field_model(parameters),
                "use_bper": parameters.get("use_bper"), "use_bpar": parameters.get("use_bpar"),
                "tglf_parameters": json.dumps(parameters, sort_keys=True),
                "efit_quality": state["efit_quality"], "ti_lineage": state.get("ti_lineage"),
                "state_identity": state["state_identity"],
                "tglf_run_identity": surface.get("run_identity"),
                "gacode_revision": state.get("gacode_revision"),
                "status": surface.get("status"),
            }
            outputs = path.parent / f"r{r:.2f}" / "outputs.json"
            if surface.get("status") == "solved" and not outputs.is_file():
                row["status"] = "missing_outputs"  # pruned native tree: reported, not fatal
            if row["status"] == "solved":
                native = TglfOutputs.read_json(outputs)
                descriptors = atlas.spectral_descriptors(native)
                row.update({k: v for k, v in descriptors.items() if k in SCHEMA})
                qe, qi = descriptors.get("qe_gb"), descriptors.get("qi_gb")
                if qe is not None and qi is not None and abs(qi) >= EPS:
                    row["qe_over_qi"] = qe / qi
                row["ky_q_peak"] = ky_peak(native, parameters)
            rows.append(row)
    return rows


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["status"] == "solved":
            key = (row["shot"], row["time_efit_s"], row["efit_lineage"], round(row["r_over_a"], 4))
            groups.setdefault(key, []).append(row)
    out = []
    for (shot, t, lineage, r), group in sorted(groups.items()):
        identities = {g["state_identity"] for g in group}
        if len(identities) != 1:
            raise ValueError(f"{shot} {t} {lineage}: configurations were run on different states")
        by: dict[tuple, dict] = {}
        for g in group:
            config = (g["sat_rule"], g["field_model"])
            if config in by:
                raise ValueError(f"{shot} {t} {lineage} r/a {r}: configuration {config} appears twice")
            by[config] = g
        pair = {"shot": shot, "time_efit_s": t, "efit_lineage": lineage, "r_over_a": r,
                "efit_quality": group[0]["efit_quality"], "n_configs": len(by)}

        def value(config, q):
            return _finite(by[config].get(q)) if config in by else None

        for q in QUANTITIES:
            for field in FIELD_MODELS:
                values = [v for s in range(4) if (v := value((s, field), q)) is not None]
                deltas = [abs(d) for a, b in itertools.combinations(values, 2)
                          if (d := delta(a, b)) is not None]
                pair[f"{q}_sat_spread_{field}"] = max(deltas) if deltas else None
            for sat in range(4):
                pair[f"{q}_em_vs_es_sat{sat}"] = delta(value((sat, "em-bper"), q), value((sat, "es"), q))
        out.append(pair)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True, help="root holding tglf-sat*/ trees")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    atlas = _atlas_module()
    rows = build_rows(args.runs)
    if not rows:
        raise SystemExit(f"no tglf-sat*/ state.json under {args.runs}")
    pairs = build_pairs(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    atlas._write_csv(args.out / "sensitivity.csv", rows, SCHEMA)
    atlas._write_csv(args.out / "sensitivity_pairs.csv", pairs, PAIR_SCHEMA)
    configs = sorted({r["tglf_config"] for r in rows})
    schema = {
        "description": "TGLF SAT-rule x field-model sensitivity (#1482) on representative "
                       "#1331 Tier A states. Model output, no preferred configuration.",
        "row_key": ["shot", "time_efit_s", "efit_lineage", "r_over_a", "tglf_config"],
        "pair_row_key": ["shot", "time_efit_s", "efit_lineage", "r_over_a"],
        "delta": "Delta(A, B) = (A - B) / max(|A|, |B|, eps), eps = 1e-3 gyro-Bohm units",
        "configurations": configs,
        "columns": {k: {"unit": u, "category": c, "definition": d} for k, (u, c, d) in SCHEMA.items()},
        "pair_columns": {k: {"unit": u, "category": c, "definition": d}
                         for k, (u, c, d) in PAIR_SCHEMA.items()},
        "counts": {"rows": len(rows), "surfaces": len(pairs),
                   "states": len({(r["shot"], r["time_efit_s"], r["efit_lineage"]) for r in rows})},
    }
    (args.out / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    print(json.dumps({"configs": configs, **schema["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

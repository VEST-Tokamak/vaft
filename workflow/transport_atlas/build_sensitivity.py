"""TGLF saturation-rule and electromagnetic sensitivity tables.

The input is the trees ``run_tglf.py`` writes, one ``tglf-sat<n>-<field>/`` per
configuration, for the same states. No solver runs here. The output is a product
separate from the transport atlas:

    sensitivity.csv        one row per (shot, time_efit_s, efit_lineage, r_over_a, tglf_config)
    sensitivity_pairs.csv  one row per (shot, time_efit_s, efit_lineage, r_over_a):
                           spread across SAT rules at a fixed field model, and the
                           EM-vs-ES change at each SAT rule
    sat_pairs.csv          one row per (shot, time_efit_s, efit_lineage, r_over_a,
                           field_model, pair) for the controlled SAT pairs
    linear_spectra.csv     growth rate and frequency per (row, ky, mode)
    schema.json

Controlled SAT pairs. TGLF's start-up presets (tglf_startup.f90) set the
linear model from SAT_RULE, so the pairs compare different things:

* ``sat0-sat1``, *saturation-model sensitivity*. The linear physics is shared only
  when the input has ``UNITS=GYRO``: SAT0 forces GYRO, SAT1 keeps the input's units.
* ``sat2-sat3``, *quasilinear/saturation-closure sensitivity*. SAT3 also applies
  a per-mode amplitude factor (QLA), so this is not a pure intensity comparison.
* ``sat1-sat2``, *combined model comparison*. XNU_MODEL, WDIA_TRAPPED and the ky
  grid change together with the saturation rule. Growth-rate and frequency
  spectra are reported side by side (``linear_spectra.csv``). A change in
  ``gamma_max`` is not used as a share of the flux change.

Shared linear physics is measured for every pair row, not assumed from the SAT
rules: ky grid, eigenvalues and QL weights must agree within
``spectral.LINEAR_RTOL`` (``linear_identical``). Only then, and only when both runs'
flux decompositions rebuild TGLF's ``sum_flux_spectrum``
(``vaft.code.gacode.tglf.spectral``), is the change split exactly into an intensity
term and an amplitude-factor term. The split is evaluated at the first run's factor,
so it depends on the order.

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
    "status": ("-", "provenance", "solved | failed | not_ready | not_run | missing_outputs (native tree pruned; "
                                  "not_run: ready, but the batch recorded no result for it)"),
    "runtime_status": ("-", "provenance", "launcher status of the surface's run (completed | timeout | error | ...); "
                                          "empty when it was not run"),
    "tglf_reason": ("-", "provenance", "readiness reason, runtime status, or not_run when not solved (as the atlas)"),
    "n_errors": ("-", "provenance", "number of solver/launcher error lines recorded for the surface"),
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
    "native_source": ("-", "provenance", "outputs.json | native_files (re-parsed out.tglf.* of a pre-#1591 record)"),
    "units_effective": ("-", "provenance", "UNITS TGLF ran with after its presets (scalar_saturation_parameters)"),
    "xnu_model_effective": ("-", "provenance", "XNU_MODEL TGLF ran with after its presets"),
    "wdia_trapped_effective": ("-", "provenance", "WDIA_TRAPPED after the presets: 1 for SAT_RULE 2/3, else 0 "
                                                  "(tglf_startup.f90; TGLF does not echo it)"),
    "nky": ("-", "provenance", "ky points of the run"),
    "nmodes": ("-", "provenance", "modes per ky of the run"),
    "flux_reconstruction": ("-", "provenance", "validated | reconstruction_mismatch | unavailable: whether QL weight "
                                               "x intensity x QLA rebuilds sum_flux_spectrum (spectral.py)"),
    "flux_reconstruction_error": ("-", "provenance", "max relative rebuild error over flux quantities"),
}

#: (sat_a, sat_b, comparison) of the controlled pairs; module docstring.
SAT_PAIRS = ((0, 1, "saturation_model"), (2, 3, "quasilinear_saturation_closure"), (1, 2, "combined_model"))
SPLIT_CHANNELS = {"qe_gb": (0, 1), "qi_gb": (slice(1, None), 1), "gamma_e_gb": (0, 0)}

SAT_PAIR_SCHEMA = {
    "shot": ("-", "key", "VEST shot number"),
    "time_efit_s": ("s", "key", "equilibrium slice time"),
    "efit_lineage": ("-", "key", "magnetics | electron_kinetic"),
    "r_over_a": ("-", "key", "surface"),
    "field_model": ("-", "key", "es | em-bper | em-bper-bpar (equal on both sides)"),
    "pair": ("-", "key", "sat<a>-sat<b>"),
    "comparison": ("-", "label", "saturation_model | quasilinear_saturation_closure | combined_model "
                                 "(module docstring; not a regime label)"),
    "efit_quality": ("-", "label", "#1331 slice label"),
    "linear_identical": ("-", "model_derived", "ky grid, eigenvalues and QL weights agree within LINEAR_RTOL"),
    "linear_max_rel_diff": ("-", "model_derived", "their largest relative difference; empty when the grids differ "
                                                  "in shape"),
    "split_status": ("-", "provenance", "split | linear_differs | unavailable (ClosureSplit.status; also "
                                        "unavailable when the two runs use different KYGRID_MODELs, which leaves "
                                        "linear_identical empty, or when electrons are adiabatic)"),
}
for _q in QUANTITIES:
    SAT_PAIR_SCHEMA[f"{_q}_a"] = ("GB", "tglf_predicted", f"{_q} of sat_a")
    SAT_PAIR_SCHEMA[f"{_q}_b"] = ("GB", "tglf_predicted", f"{_q} of sat_b")
    SAT_PAIR_SCHEMA[f"delta_{_q}"] = ("-", "model_derived", f"Delta({_q}_b, {_q}_a)")
for _q in SPLIT_CHANNELS:
    SAT_PAIR_SCHEMA[f"{_q}_change"] = ("GB", "model_derived", f"{_q}_b - {_q}_a from the validated decompositions; "
                                                             "empty unless split_status = split")
    SAT_PAIR_SCHEMA[f"{_q}_intensity_term"] = ("GB", "model_derived", "sum W (phi_b - phi_a) QLA_a")
    SAT_PAIR_SCHEMA[f"{_q}_qla_term"] = ("GB", "model_derived", "sum W phi_b (QLA_b - QLA_a)")
for _q in ("gamma_max", "ky_at_gamma_max", "omega_at_gamma_max"):
    SAT_PAIR_SCHEMA[f"{_q}_a"] = SCHEMA[_q][:2] + (f"{_q} of sat_a (reported, not attributed)",)
    SAT_PAIR_SCHEMA[f"{_q}_b"] = SCHEMA[_q][:2] + (f"{_q} of sat_b (reported, not attributed)",)

SPECTRA_SCHEMA = {
    "shot": ("-", "key", "VEST shot number"),
    "time_efit_s": ("s", "key", "equilibrium slice time"),
    "efit_lineage": ("-", "key", "magnetics | electron_kinetic"),
    "r_over_a": ("-", "key", "surface"),
    "tglf_config": ("-", "key", "tglf-sat<n>-<field>"),
    "ky_index": ("-", "key", "1-based ky point"),
    "mode": ("-", "key", "1-based mode"),
    "sat_rule": ("-", "provenance", "TGLF SAT_RULE"),
    "units_effective": ("-", "provenance", "UNITS after presets: CGYRO moves the ky points, not ky's units"),
    "ky": ("-", "tglf_predicted", "ky rho_s"),
    "gamma": ("c_s/a", "tglf_predicted", "growth rate"),
    "omega": ("c_s/a", "tglf_predicted", "real frequency (unlabelled sign)"),
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


def effective_controls(native: Any) -> dict[str, Any]:
    """The linear-model controls TGLF ran with, from the run's own outputs (SCHEMA)."""
    out: dict[str, Any] = {}
    parameters = getattr(native, "saturation_parameters", None) or {}
    if "SAT_RULE" in parameters:
        out["units_effective"] = parameters.get("UNITS")
        out["xnu_model_effective"] = parameters.get("XNU_MODEL")
        # tglf_startup.f90 (USE_PRESETS hard-coded true) zeroes WDIA_TRAPPED for every
        # SAT rule before setting it to 1 for SAT2/3, so no input value survives.
        out["wdia_trapped_effective"] = 1.0 if int(parameters["SAT_RULE"]) in (2, 3) else 0.0
    ky = getattr(native, "ky_spectrum", None)
    eigen = getattr(native, "eigenvalue_spectrum", None)
    if ky is not None:
        out["nky"] = int(np.size(ky))
    if eigen is not None and np.ndim(eigen) == 2:
        out["nmodes"] = int(np.shape(eigen)[1] // 2)
    return out


def load_native(outputs: Path) -> tuple[Any, str]:
    """A surface's native container, and where it came from.

    ``outputs.json`` written before the spectral parsers existed has no
    spectra. When the run's own ``out.tglf.*`` files are still beside it, they are
    parsed again instead: the same files, read with the current parser, with no
    solver rerun.
    """
    from vaft.code.gacode.tglf.outputs import TglfOutputs, collect_tglf_outputs

    stored = TglfOutputs.read_json(outputs)
    if stored.ql_flux_spectrum is None and (outputs.parent / "out.tglf.gbflux").is_file():
        native = collect_tglf_outputs(outputs.parent)
        if native is not None:
            return native, "native_files"
    return stored, "outputs.json"


def build_rows(runs: Path) -> list[dict[str, Any]]:
    from vaft.code.gacode.tglf.spectral import flux_decomposition

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
                "runtime_status": surface.get("runtime_status"),
                "n_errors": len(surface.get("errors") or []),
            }
            outputs = path.parent / f"r{r:.2f}" / "outputs.json"
            if surface.get("status") == "solved" and not outputs.is_file():
                row["status"] = "missing_outputs"  # pruned native tree: reported, not fatal
            # Why a surface is not solved, with the atlas's vocabulary (build_atlas.py):
            # a solver timeout (returned as a failed result), a launcher error and a readiness
            # refusal are different things.
            if row["status"] == "solved":
                row["tglf_reason"] = None
            elif row["status"] == "failed":
                row["tglf_reason"] = surface.get("runtime_status")
            elif row["status"] in ("not_run", "missing_outputs"):
                row["tglf_reason"] = row["status"]
            else:
                row["tglf_reason"] = surface.get("readiness")
            if row["status"] == "solved":
                native, row["native_source"] = load_native(outputs)
                descriptors = atlas.spectral_descriptors(native)
                row.update({k: v for k, v in descriptors.items() if k in SCHEMA})
                qe, qi = descriptors.get("qe_gb"), descriptors.get("qi_gb")
                if qe is not None and qi is not None and abs(qi) >= EPS:
                    row["qe_over_qi"] = qe / qi
                row["ky_q_peak"] = ky_peak(native, parameters)
                reported = (native.saturation_parameters or {}).get("SAT_RULE")
                if reported is not None and int(reported) != parameters.get("sat_rule"):
                    raise ValueError(f"{outputs}: TGLF ran SAT_RULE {reported}, the state says "
                                     f"{parameters.get('sat_rule')}")
                row.update(effective_controls(native))
                model = ky_grid_model(parameters)
                if model is not None:
                    decomposition = flux_decomposition(native, kygrid_model=model)
                    row["flux_reconstruction"] = decomposition.status
                    row["flux_reconstruction_error"] = decomposition.relative_error
                row["_native"] = native            # for the pair tables; not a CSV column
            rows.append(row)
    return rows


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tolerance = _atlas_module().SURFACE_TOLERANCE
    # Surfaces are joined like the atlas does: nearest within SURFACE_TOLERANCE, and the
    # pair row carries the rows' own r_over_a so the two tables join on the key text.
    groups: list[tuple[tuple, float, list[dict]]] = []   # ((shot, t, lineage), r, rows)
    for row in sorted((r for r in rows if r["status"] == "solved"),
                      key=lambda r: (r["shot"], r["time_efit_s"], r["efit_lineage"], r["r_over_a"])):
        state = (row["shot"], row["time_efit_s"], row["efit_lineage"])
        if groups and groups[-1][0] == state and abs(row["r_over_a"] - groups[-1][1]) <= tolerance:
            groups[-1][2].append(row)
        else:
            groups.append((state, row["r_over_a"], [row]))
    out = []
    for (shot, t, lineage), r, group in groups:
        identities = {g["state_identity"] for g in group}
        if len(identities) != 1:
            raise ValueError(f"{shot} {t} {lineage}: configurations were run on different states")
        qualities = {g["efit_quality"] for g in group}
        if len(qualities) != 1:
            # Equal state identities with different labels: the slice was re-labelled
            # between two configuration runs, and the pair cannot name one label.
            raise ValueError(f"{shot} {t} {lineage}: one state carries efit_quality {sorted(qualities)}")
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


def build_sat_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per controlled SAT pair at one surface and field model (module docstring)."""
    from vaft.code.gacode.tglf.spectral import closure_split

    tolerance = _atlas_module().SURFACE_TOLERANCE
    solved = [r for r in rows if r["status"] == "solved" and "_native" in r]
    out = []
    for a in solved:
        for sat_a, sat_b, comparison in SAT_PAIRS:
            if a["sat_rule"] != sat_a:
                continue
            matches = [b for b in solved if b["sat_rule"] == sat_b and b["field_model"] == a["field_model"]
                       and b["state_identity"] == a["state_identity"]
                       and abs(b["r_over_a"] - a["r_over_a"]) <= tolerance]
            if not matches:
                continue
            if len(matches) > 1:   # build_rows refuses a configuration twice on one state
                raise ValueError(f"{a['shot']} {a['time_efit_s']} {a['efit_lineage']} r/a {a['r_over_a']}: "
                                 f"sat{sat_b} {a['field_model']} matches {len(matches)} surfaces")
            (b,) = matches
            pair = {"shot": a["shot"], "time_efit_s": a["time_efit_s"], "efit_lineage": a["efit_lineage"],
                    "r_over_a": a["r_over_a"], "field_model": a["field_model"],
                    "pair": f"sat{sat_a}-sat{sat_b}", "comparison": comparison,
                    "efit_quality": a["efit_quality"]}
            for q in QUANTITIES:
                pair[f"{q}_a"], pair[f"{q}_b"] = a.get(q), b.get(q)
                pair[f"delta_{q}"] = delta(b.get(q), a.get(q))
            for q in ("gamma_max", "ky_at_gamma_max", "omega_at_gamma_max"):
                pair[f"{q}_a"], pair[f"{q}_b"] = a.get(q), b.get(q)
            # Each side's own grid model; a pair across grid models is not split.
            models = {ky_grid_model(json.loads(r["tglf_parameters"])) for r in (a, b)}
            if len(models) != 1 or None in models:
                pair["split_status"] = "unavailable"
            else:
                split = closure_split(a["_native"], b["_native"], kygrid_model=models.pop())
                pair["split_status"] = split.status
                pair["linear_max_rel_diff"] = split.linear_max_rel_diff
                pair["linear_identical"] = split.status != "linear_differs"
                # The channels index species 0 as the electrons; with adiabatic electrons
                # (spectral species 2..ns) that slot is an ion, so nothing is split.
                kinetic_electrons = all((getattr(r["_native"], "spectral_species", None) or (0,))[0] == 1
                                        for r in (a, b))
                if split.status == "split" and not kinetic_electrons:
                    pair["split_status"] = "unavailable"
                if pair["split_status"] == "split":
                    for q, (species, quantity) in SPLIT_CHANNELS.items():
                        for suffix, values in (("change", split.total), ("intensity_term", split.intensity_term),
                                               ("qla_term", split.qla_term)):
                            pair[f"{q}_{suffix}"] = float(np.sum(values[species, :, quantity]))
            out.append(pair)
    return out


def build_linear_spectra(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Growth rate and frequency per (row, ky, mode): the linear side of every comparison."""
    out = []
    for row in rows:
        if row["status"] != "solved" or "_native" not in row:
            continue
        native = row["_native"]
        ky, gamma, omega = native.ky_spectrum, native.growth_rate, native.frequency
        if ky is None or gamma is None or omega is None or gamma.shape[0] != np.size(ky):
            continue
        for k in range(gamma.shape[0]):
            for m in range(gamma.shape[1]):
                out.append({"shot": row["shot"], "time_efit_s": row["time_efit_s"],
                            "efit_lineage": row["efit_lineage"], "r_over_a": row["r_over_a"],
                            "tglf_config": row["tglf_config"], "ky_index": k + 1, "mode": m + 1,
                            "sat_rule": row["sat_rule"], "units_effective": row.get("units_effective"),
                            "ky": _finite(ky[k]), "gamma": _finite(gamma[k, m]),
                            "omega": _finite(omega[k, m])})
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
    sat_pairs = build_sat_pairs(rows)
    spectra = build_linear_spectra(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    atlas._write_csv(args.out / "sensitivity.csv", rows, SCHEMA)
    atlas._write_csv(args.out / "sensitivity_pairs.csv", pairs, PAIR_SCHEMA)
    atlas._write_csv(args.out / "sat_pairs.csv", sat_pairs, SAT_PAIR_SCHEMA)
    atlas._write_csv(args.out / "linear_spectra.csv", spectra, SPECTRA_SCHEMA)
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
        "sat_pair_row_key": ["shot", "time_efit_s", "efit_lineage", "r_over_a", "field_model", "pair"],
        "sat_pair_interpretation": {
            "saturation_model": "SAT0 vs SAT1: saturation-model sensitivity; shares linear physics only "
                                "at UNITS=GYRO (linear_identical says whether it did)",
            "quasilinear_saturation_closure": "SAT2 vs SAT3: quasilinear/saturation-closure sensitivity, "
                                              "including SAT3's per-mode amplitude factor; not a pure "
                                              "saturation-intensity decomposition",
            "combined_model": "SAT1 vs SAT2: linear presets and saturation change together; gamma/omega "
                              "spectra are reported (linear_spectra.csv) but not used to attribute the flux change",
        },
        "sat_pair_columns": {k: {"unit": u, "category": c, "definition": d}
                             for k, (u, c, d) in SAT_PAIR_SCHEMA.items()},
        "linear_spectra_columns": {k: {"unit": u, "category": c, "definition": d}
                                   for k, (u, c, d) in SPECTRA_SCHEMA.items()},
        "counts": {"rows": len(rows), "surfaces": len(pairs), "sat_pairs": len(sat_pairs),
                   "sat_pairs_split": sum(p.get("split_status") == "split" for p in sat_pairs),
                   "flux_reconstruction": {s: sum(r.get("flux_reconstruction") == s for r in rows)
                                           for s in ("validated", "reconstruction_mismatch", "unavailable")},
                   "states": len({(r["shot"], r["time_efit_s"], r["efit_lineage"]) for r in rows})},
    }
    (args.out / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    print(json.dumps({"configs": configs, **schema["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

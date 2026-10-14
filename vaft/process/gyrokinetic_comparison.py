"""Four independent scientific comparison axes, reusable outside rendering."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ComparisonAssessment",
    "assess_comparison",
    "local_result_signature",
    "prepare_diagnostic_traces",
    "select_time",
]


@dataclass(frozen=True)
class ComparisonAssessment:
    intent: str
    axes: Mapping[str, Mapping[str, Any]]

    @property
    def arithmetic_compatible(self):
        return self.axes["display"]["status"] == "matched"


def _missing_evidence(value):
    if value is None:
        return True
    if isinstance(value, Mapping):
        return any(_missing_evidence(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_missing_evidence(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return not np.isfinite(value)
    return False


def assess_comparison(
    left, right, *, intent="exploratory_overlay", controlled=(), rtol=1e-3
):
    """Assess signatures grouped by display/state/model/qualification.

    Parameters
    ----------
    left : mapping
        Left result evidence grouped by display/state/model/qualification [-].
    right : mapping
        Right result evidence with the same axis/key vocabulary [-].
    intent : str, optional
        Matched baseline, controlled sensitivity or exploratory overlay [-].
    controlled : sequence of str, optional
        Explicit controlled changes named axis.key; display mismatches cannot be controlled [-].
    rtol : float, optional
        Relative numerical tolerance for compatible scalar/array evidence [-].

    Returns
    -------
    ComparisonAssessment
        Independent per-axis status, reasons and supporting evidence [-].

    Applicability
    -------------
    Machine-independent. Requires the declared physical definitions, source identity and units.
    Matching display conventions alone does not establish physical equivalence or convergence.

    Notes
    -----
    Assess signatures grouped by display/state/model/qualification.

    Each axis is a mapping of named evidence. Missing keys/None are unverified.
    Controlled variations must be named 'axis.key', never inferred from intent.

    Provenance
    ----------
    .. [issue] #2030 four-axis comparison and time-selection contract; #2028 diagnostic ownership.
    """
    if intent not in (
        "matched_baseline",
        "controlled_sensitivity",
        "exploratory_overlay",
    ):
        raise ValueError("unknown comparison intent")
    result = {}
    for axis in ("display", "state", "model", "qualification"):
        a, b = left.get(axis, {}) or {}, right.get(axis, {}) or {}
        reasons = []
        evidence = {}
        unknown = not a or not b
        mismatch = False
        changed = False
        for key in sorted(set(a) | set(b)):
            x, y = a.get(key), b.get(key)
            evidence[key] = {"left": x, "right": y}
            if _missing_evidence(x) or _missing_evidence(y):
                unknown = True
                reasons.append(f"{key}: missing evidence")
                continue
            try:
                equal = bool(np.allclose(x, y, rtol=rtol, atol=0, equal_nan=False))
            except (TypeError, ValueError):
                equal = x == y
            if not equal:
                if f"{axis}.{key}" in controlled and axis != "display":
                    changed = True
                    reasons.append(f"{key}: controlled difference")
                else:
                    mismatch = True
                    reasons.append(f"{key}: differs")
            if axis == "qualification" and (
                x in ("unverified", "insufficient_data")
                or y in ("unverified", "insufficient_data")
            ):
                unknown = True
                reasons.append(f"{key}: qualification unverified")
        status = (
            "incompatible"
            if mismatch
            else "unverified"
            if unknown
            else "controlled_difference"
            if changed
            else "matched"
        )
        result[axis] = {"status": status, "reasons": reasons, "evidence": evidence}
    return ComparisonAssessment(intent, result)


def select_time(times, requested=None, *, tolerance=None):
    """Nearest finite time and evidence; no implicit scientific tolerance.

    Parameters
    ----------
    times : array-like
        Stored finite, distinct experimental timestamps [s].
    requested : float, optional
        Requested experimental time, or first stored time when omitted [s].
    tolerance : float, optional
        Explicit maximum time mismatch; absence leaves scientific alignment unverified [s].

    Returns
    -------
    dict
        Stored index, requested/selected time, difference and alignment status [s; index -].

    Applicability
    -------------
    Machine-independent. Requires the declared physical definitions, source identity and units.
    Matching display conventions alone does not establish physical equivalence or convergence.

    Notes
    -----
    Nearest finite time and evidence; no implicit scientific tolerance.

    Provenance
    ----------
    .. [issue] #2030 four-axis comparison and time-selection contract; #2028 diagnostic ownership.
    """
    t = np.asarray(times, float)
    if (
        t.ndim != 1
        or not len(t)
        or not np.isfinite(t).all()
        or len(np.unique(t)) != len(t)
    ):
        raise ValueError("unavailable or ambiguous time stamps")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("time tolerance must be finite and nonnegative")
    target = float(t[0] if requested is None else requested)
    if not np.isfinite(target):
        raise ValueError("requested time must be finite")
    distances = np.abs(t - target)
    i = int(np.argmin(distances))
    delta = float(t[i] - target)
    if (
        np.count_nonzero(
            np.isclose(
                distances,
                distances[i],
                rtol=0,
                atol=4 * np.finfo(float).eps * max(1, abs(target)),
            )
        )
        > 1
    ):
        raise ValueError("ambiguous equidistant time slices")
    if tolerance is not None and abs(delta) > tolerance:
        raise ValueError("selected experimental time outside tolerance")
    return {
        "index": i,
        "requested_time": target,
        "selected_time": float(t[i]),
        "delta_time": delta,
        "tolerance": tolerance,
        "status": "matched" if tolerance is not None else "unverified",
    }


def prepare_diagnostic_traces(diagnostics, frame_times):
    """Prepare causal, gap-broken traces; renderers never interpolate or reduce.

    Parameters
    ----------
    diagnostics : mapping
        Labelled traces with time, values, unit and optional maximum gap [declared trace units].
    frame_times : array-like
        Simulation timestamps of the selected field frames [trace time units].

    Returns
    -------
    dict
        Causal per-frame traces, broken gaps and fixed display limits [input trace units].

    Applicability
    -------------
    Machine-independent. Requires the declared physical definitions, source identity and units.
    Matching display conventions alone does not establish physical equivalence or convergence.

    Notes
    -----
    Prepare causal, gap-broken traces; renderers never interpolate or reduce.

    Provenance
    ----------
    .. [issue] #2030 four-axis comparison and time-selection contract; #2028 diagnostic ownership.
    """
    result = {}
    for label, trace in diagnostics.items():
        t = np.asarray(trace["time"], float)
        y = np.asarray(trace["values"], float)
        if (
            t.ndim != 1
            or y.shape != t.shape
            or not len(t)
            or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)
        ):
            raise ValueError(f"invalid diagnostic trace {label}")
        gap = trace.get(
            "max_gap", 3 * float(np.median(np.diff(t))) if len(t) > 1 else np.inf
        )
        if gap <= 0:
            raise ValueError("diagnostic max_gap must be positive")
        frames = []
        for instant in frame_times:
            n = int(np.searchsorted(t, instant, side="right"))
            x = t[:n]
            v = y[:n].copy()
            # Break each segment ending after a gap; no interpolation across it.
            if n > 1:
                v[1:][np.diff(x) > gap] = np.nan
            frames.append((x, v))
        finite = y[np.isfinite(y)]
        low, high = (
            (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        )
        pad = max((high - low) * 0.05, abs(high) * 0.01, 1e-12)
        result[label] = {
            "frames": frames,
            "limits": (low - pad, high + pad),
            "unit": trace.get("unit", "unspecified"),
            "max_gap": gap,
        }
    return result


def local_result_signature(obj):
    """Non-mutating IMAS signature with explicit missing physical/provenance evidence.

    Parameters
    ----------
    obj : object
        IMAS/OMAS entry supporting the shared non-mutating dotted-path accessor [-].

    Returns
    -------
    dict
        Display, physical-state, model and qualification evidence with unknowns kept as None [-].

    Applicability
    -------------
    Machine-independent. Requires the declared physical definitions, source identity and units.
    Matching display conventions alone does not establish physical equivalence or convergence.

    Notes
    -----
    Non-mutating IMAS signature with explicit missing physical/provenance evidence.

    Provenance
    ----------
    .. [issue] #2030 four-axis comparison and time-selection contract; #2028 diagnostic ownership.
    """
    import xml.etree.ElementTree as ET

    from vaft.code.gacode.result_identity import parameters, plain
    from vaft.ods_access import path_count, path_value

    base = "gyrokinetics_local"

    def get(path):
        return plain(path_value(obj, f"{base}.{path}", None))

    raw = get("code.parameters")
    record = parameters(raw) or {}

    def parameter(name):
        if isinstance(raw, dict):
            # Walk OMAS-decoded dictionaries without turning unknown into a match.
            def walk(node):
                if isinstance(node, dict):
                    if name in node:
                        return node[name]
                    for value in node.values():
                        found = walk(value)
                        if found is not None:
                            return found
                return None

            return walk(raw)
        try:
            node = ET.fromstring(str(raw)).find(".//" + name)
            return node.text if node is not None else None
        except ET.ParseError:
            return None

    species = [
        {
            key: get(f"species.{i}.{key}")
            for key in (
                "charge_norm",
                "mass_norm",
                "density_norm",
                "temperature_norm",
                "density_log_gradient_norm",
                "temperature_log_gradient_norm",
            )
        }
        for i in range(path_count(obj, f"{base}.species"))
    ]
    state = {
        key: get(path)
        for key, path in {
            "radius": "flux_surface.r_minor_norm",
            "q": "flux_surface.q",
            "shear": "flux_surface.magnetic_shear_r_minor",
            "elongation": "flux_surface.elongation",
            "beta": "species_all.beta_reference",
            "flow": "species_all.velocity_tor_norm",
            "ExB_shear": "species_all.shearing_rate_norm",
            "electron_temperature_eV": "normalizing_quantities.t_e",
            "electron_density": "normalizing_quantities.n_e",
        }.items()
    }
    state["species"] = species or None
    state["state_identity"] = (
        record.get("state_identity")
        if record.get("state_evidence") == "explicit"
        else None
    )
    state["experimental_time"] = get("time")
    run = (record.get("local_runs") or [{}])[0]
    native = run.get("native_input") or {}
    state["effective_collisionality"] = native.get("NU_EE", native.get("XNUE"))
    iv_modes = []
    for wave in range(path_count(obj, f"{base}.linear.wavevector")):
        for mode in range(
            path_count(obj, f"{base}.linear.wavevector.{wave}.eigenmode")
        ):
            prefix = f"linear.wavevector.{wave}.eigenmode.{mode}"
            if get(prefix + ".initial_value_run") == 1:
                iv_modes.append(get(prefix + ".growth_rate_tolerance") is not None)
    linear_evidence = (
        (
            "all_retained_initial_value_modes_converged"
            if all(iv_modes)
            else "contains_unqualified_initial_value_modes"
        )
        if iv_modes
        else None
    )
    return {
        "display": {
            "normalization": parameter("normalisation"),
            "frequency_sign": parameter("frequency_sign_convention"),
        },
        "state": state,
        "model": {
            "solver": get("code.name"),
            "effective_configuration": record.get("configuration"),
            "A_parallel": get("model.include_a_field_parallel"),
            "B_parallel": get("model.include_b_field_parallel"),
        },
        "qualification": {
            "linear_frequency_evidence": linear_evidence,
            "numerical_convergence": None,
            "stationarity": None,
            "locality": parameter("locality_verdict"),
        },
    }

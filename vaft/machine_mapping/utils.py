"""Shared orchestration helpers for `vaft.machine_mapping`."""

from __future__ import annotations

import os
import datetime
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

# Dotted-path access lives in `vaft.ods_access` (issue #118), a leaf module with
# no VAFT dependencies, so layers that must not import the database -- notably
# `vaft.validation` -- can reach the same non-mutating primitives this module
# uses. Re-exported here because these names have always been imported from
# `vaft.machine_mapping.utils`.
from vaft.ods_access import get_path, path_count, path_exists, path_value, set_path
from vaft.database import raw as raw_db
from vaft.process.signal_processing import process_signal as process_signal_impl


DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR = (
    1.0e-4,
    1.0e-4,
    5.0e-2,
    3.0e-2,
    1.0e-2,
    1.0e-1,
    1.0e-2,
    1.0e-1,
    1.0e-2,
)

DEFAULT_CONSTRAINT_UNCERTAINTIES = {
    "pf_active_current": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[0],
    "tf_b_field_tor_vacuum_r": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[1],
    "magnetics_ip": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[2],
    "magnetics_diamagnetic_flux": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[3],
    "magnetics_bpol_inboard": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[4],
    "magnetics_bpol_side": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[5],
    "magnetics_bpol_outboard": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[6],
    "magnetics_flux_loop_inboard": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[7],
    "magnetics_flux_loop_outboard": DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR[8],
}


def package_data_path(filename: str) -> str:
    """Return an absolute path to a file shipped with `vaft.machine_mapping`."""
    return os.path.join(os.path.dirname(__file__), filename)


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Resolve the default on-disk data root for donor-style assets."""
    if data_root is not None:
        return Path(data_root)
    return Path(__file__).resolve().parents[1] / "data"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {} if data is None else data


def _resolve_info_file_path(filename: str | None) -> str:
    if filename is None:
        return package_data_path("vest.yaml")

    candidate = Path(filename)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())

    packaged = Path(package_data_path(filename))
    if packaged.exists():
        return str(packaged)

    return package_data_path("vest.yaml")


def _deep_merge(base: Any, override: Any) -> Any:
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        return override
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class VestConfigurationError(ValueError):
    """Raised when VEST machine-mapping configuration is invalid."""


def _revision_bound(value: Any, *, name: str, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise VestConfigurationError(f"{context}: {name} must be an integer shot number")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise VestConfigurationError(f"{context}: {name} must be an integer shot number") from exc


def _match_revision(
    revisions: Sequence[Mapping[str, Any]] | None,
    shot: int,
    *,
    context: str,
) -> tuple[Mapping[str, Any] | None, int | None, dict[str, int | None] | None]:
    """Validate a revision table and return the single entry matching ``shot``.

    Returns ``(revision, index, bounds)``, or ``(None, None, None)`` when the
    shot falls outside every declared era. Shared by `resolve_shot_revisions`
    and its provenance-reporting counterpart so overlap validation cannot
    drift between them.
    """
    if revisions is None:
        return None, None, None
    if not isinstance(revisions, Sequence) or isinstance(revisions, (str, bytes)):
        raise VestConfigurationError(f"{context}: revisions must be a list")

    parsed_revisions: list[tuple[Mapping[str, Any], int | None, int | None]] = []
    matching: list[tuple[Mapping[str, Any], int, int | None, int | None]] = []
    numeric_shot = int(shot)
    for index, revision in enumerate(revisions):
        revision_context = f"{context} revision {index}"
        if not isinstance(revision, Mapping):
            raise VestConfigurationError(f"{revision_context}: entry must be a mapping")
        first = _revision_bound(revision.get("from_shot"), name="from_shot", context=revision_context)
        last = _revision_bound(revision.get("to_shot"), name="to_shot", context=revision_context)
        if first is None and last is None:
            raise VestConfigurationError(
                f"{revision_context}: at least one of from_shot or to_shot is required"
            )
        if first is not None and last is not None and first > last:
            raise VestConfigurationError(f"{revision_context}: from_shot must not exceed to_shot")
        parsed_revisions.append((revision, first, last))
        if (first is None or numeric_shot >= first) and (last is None or numeric_shot <= last):
            matching.append((revision, index, first, last))

    for index, (_, first, last) in enumerate(parsed_revisions):
        for other_index, (_, other_first, other_last) in enumerate(parsed_revisions[index + 1 :], index + 1):
            lower = max(
                float("-inf") if first is None else first,
                float("-inf") if other_first is None else other_first,
            )
            upper = min(
                float("inf") if last is None else last,
                float("inf") if other_last is None else other_last,
            )
            if lower <= upper:
                raise VestConfigurationError(
                    f"{context}: revisions {index} and {other_index} overlap"
                )

    if len(matching) > 1:
        raise VestConfigurationError(f"{context}: overlapping revisions apply to shot {numeric_shot}")
    if not matching:
        return None, None, None
    revision, index, first, last = matching[0]
    return revision, index, {"from_shot": first, "to_shot": last}


def resolve_shot_revisions(
    base: Mapping[str, Any],
    revisions: Sequence[Mapping[str, Any]] | None,
    shot: int,
    *,
    context: str,
) -> dict[str, Any]:
    """Merge the one unambiguous configuration revision applicable to ``shot``.

    Bounds are inclusive. A revision needs at least one bound; unrestricted
    defaults belong in ``base``. This helper is intentionally also usable for
    nested processing eras such as plasma-current baseline windows.
    """
    resolved, _provenance = resolve_shot_revisions_with_provenance(
        base, revisions, shot, context=context
    )
    return resolved


def resolve_shot_revisions_with_provenance(
    base: Mapping[str, Any],
    revisions: Sequence[Mapping[str, Any]] | None,
    shot: int,
    *,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Like `resolve_shot_revisions`, but also report which era was applied.

    The second return value records the matched revision's index and its
    inclusive shot bounds (or ``None`` for both when the shot falls back to
    ``base``), so a shot's effective processing era is recoverable without
    re-deriving it from scattered conditionals (issue #195).
    """
    resolved = _deep_merge({}, dict(base))
    revision, index, bounds = _match_revision(revisions, shot, context=context)
    provenance: dict[str, Any] = {
        "context": context,
        "revision_index": index,
        "revision_bounds": bounds,
    }
    if revision is not None:
        override = {
            key: value
            for key, value in revision.items()
            if key not in {"from_shot", "to_shot"}
        }
        resolved = _deep_merge(resolved, override)
    return resolved, provenance


def _required_number(mapping: Mapping[str, Any], key: str, *, context: str) -> float:
    if key not in mapping:
        raise VestConfigurationError(f"{context}: missing required parameter {key!r}")
    try:
        return float(mapping[key])
    except (TypeError, ValueError) as exc:
        raise VestConfigurationError(f"{context}: parameter {key!r} must be numeric") from exc


def validate_calibration(calibration: Mapping[str, Any]) -> None:
    """Validate a declarative, named VEST calibration definition."""
    if not isinstance(calibration, Mapping):
        raise VestConfigurationError("calibration must be a mapping")
    calibration_type = calibration.get("type")
    context = f"calibration {calibration_type!r}"
    if calibration_type == "linear":
        operation = calibration.get("operation")
        if operation not in {"multiply", "divide"}:
            raise VestConfigurationError(f"{context}: operation must be 'multiply' or 'divide'")
        factor = _required_number(calibration, "factor", context=context)
        if operation == "divide" and factor == 0:
            raise VestConfigurationError(f"{context}: factor must be non-zero for division")
        return
    if calibration_type == "exponential_pressure":
        for key in ("scale", "slope", "offset", "base"):
            _required_number(calibration, key, context=context)
        if float(calibration["base"]) <= 0:
            raise VestConfigurationError(f"{context}: base must be positive")
        return
    if calibration_type == "logarithmic_power":
        for key in ("scale", "input_offset", "slope", "exponent_offset", "base"):
            _required_number(calibration, key, context=context)
        if float(calibration["base"]) <= 0 or float(calibration["slope"]) == 0:
            raise VestConfigurationError(f"{context}: base must be positive and slope non-zero")
        return
    raise VestConfigurationError(f"Unsupported VEST calibration type {calibration_type!r}")


def calibrate_vest_signal(data: Any, calibration: Mapping[str, Any]) -> np.ndarray:
    """Apply one named calibration to a raw VEST waveform."""
    validate_calibration(calibration)
    values = np.asarray(data, dtype=float)
    calibration_type = calibration["type"]
    if calibration_type == "linear":
        factor = float(calibration["factor"])
        return values * factor if calibration["operation"] == "multiply" else values / factor
    if calibration_type == "exponential_pressure":
        return float(calibration["scale"]) * float(calibration["base"]) ** (
            float(calibration["slope"]) * values + float(calibration["offset"])
        )
    return float(calibration["scale"]) * float(calibration["base"]) ** (
        (values - float(calibration["input_offset"])) / float(calibration["slope"])
        + float(calibration["exponent_offset"])
    )


def resolve_vest_diagnostic(
    shot: int,
    diagnostic: str,
    *,
    info_file: str | None = None,
    with_provenance: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Return the effective, validated canonical configuration for one diagnostic.

    With ``with_provenance=True`` the return value becomes
    ``(config, provenance)``, where ``provenance`` records which top-level
    revision was applied. The default single-value return is unchanged for
    existing callers.
    """
    content = load_yaml(_resolve_info_file_path(info_file))
    defaults = content.get("0") or content.get(0) or {}
    diagnostics = defaults.get("diagnostics", {}) if isinstance(defaults, Mapping) else {}
    if not isinstance(diagnostics, Mapping) or diagnostic not in diagnostics:
        raise VestConfigurationError(f"No canonical VEST diagnostic configuration for {diagnostic!r}")
    config = diagnostics[diagnostic]
    if not isinstance(config, Mapping):
        raise VestConfigurationError(f"VEST diagnostic {diagnostic!r} must be a mapping")
    base = {key: value for key, value in config.items() if key != "revisions"}
    resolved, provenance = resolve_shot_revisions_with_provenance(
        base, config.get("revisions"), int(shot), context=f"VEST diagnostic {diagnostic!r}"
    )
    calibration = resolved.get("calibration")
    if calibration is not None:
        validate_calibration(calibration)
    if with_provenance:
        return resolved, provenance
    return resolved


DIAGNOSTICS_TIME_POLICIES_KEY = "diagnostics_time_policies"


@dataclass(frozen=True)
class DiagnosticsTimePolicy:
    """One named temporal coverage for a mapped diagnostics component.

    Windows are half-open -- ``tstart <= t < tend`` -- on a uniform ``dt``
    grid.  The same convention applies to every policy, so a component's
    coverage is fully described by these three numbers plus the policy name.
    """

    name: str
    tstart: float
    tend: float
    dt: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.name,
            "tstart": self.tstart,
            "tend": self.tend,
            "dt": self.dt,
        }


class DiagnosticsTimePolicyTable(dict):
    """``component -> DiagnosticsTimePolicy`` with an explicit missing-key error.

    A component that reaches the mapping stage without a configured policy is a
    configuration bug, not a ``KeyError`` to be caught somewhere downstream.

    ``windows`` keeps every configured window (including ones no component
    currently uses) and ``default`` is the window the stage's own
    ``tstart``/``tend``/``dt`` arguments retune, so the manifest can report the
    whole policy document rather than only the components that were mapped.
    """

    def __init__(
        self,
        *args: Any,
        windows: Mapping[str, "DiagnosticsTimePolicy"] | None = None,
        default: "DiagnosticsTimePolicy | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.windows: Dict[str, DiagnosticsTimePolicy] = dict(windows or {})
        self.default = default

    def __missing__(self, key: Any) -> "DiagnosticsTimePolicy":
        known = ", ".join(sorted(str(name) for name in self)) or "<none>"
        raise VestConfigurationError(
            f"No diagnostics time policy is configured for component {key!r}; "
            f"configured components: {known}"
        )


def _diagnostics_time_window(
    name: Any, window: Any, *, context: str
) -> DiagnosticsTimePolicy:
    if not isinstance(window, Mapping):
        raise VestConfigurationError(f"{context}: window {name!r} must be a mapping")
    window_context = f"{context}: window {name!r}"
    policy = DiagnosticsTimePolicy(
        name=str(name),
        tstart=_required_number(window, "tstart", context=window_context),
        tend=_required_number(window, "tend", context=window_context),
        dt=_required_number(window, "dt", context=window_context),
    )
    if not all(np.isfinite([policy.tstart, policy.tend, policy.dt])):
        raise VestConfigurationError(f"{window_context}: tstart, tend, and dt must be finite")
    if policy.tend <= policy.tstart:
        raise VestConfigurationError(f"{window_context}: tend must be greater than tstart")
    if policy.dt <= 0.0:
        raise VestConfigurationError(
            f"{window_context}: dt must be positive; a native timebase is an explicit mapper mode"
        )
    return policy


def resolve_diagnostics_time_policies(
    *,
    analysis_override: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    info_file: str | None = None,
) -> DiagnosticsTimePolicyTable:
    """Return the effective per-component diagnostics time policy (issue #244).

    The processed diagnostics product does not share one temporal coverage.
    Equilibrium magnetics intentionally use a short analysis window, while TF,
    barometry, and EC power must retain the full discharge history; this
    resolves which named window each component uses.

    ``analysis_override`` retunes the *default* window only -- that is what the
    stage's long-standing ``tstart``/``tend``/``dt`` arguments have always
    meant.  ``overrides`` is deep-merged over the whole configured document and
    can add windows or re-point components.
    """
    content = load_yaml(_resolve_info_file_path(info_file))
    document = content.get(DIAGNOSTICS_TIME_POLICIES_KEY)
    context = DIAGNOSTICS_TIME_POLICIES_KEY
    if not isinstance(document, Mapping):
        raise VestConfigurationError(
            f"VEST configuration defines no {DIAGNOSTICS_TIME_POLICIES_KEY!r} section"
        )
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise VestConfigurationError(f"{context}: overrides must be a mapping")
        document = _deep_merge(document, overrides)

    default_name = document.get("default")
    if not isinstance(default_name, str) or not default_name:
        raise VestConfigurationError(f"{context}: 'default' must name a configured window")

    raw_windows = document.get("windows")
    if not isinstance(raw_windows, Mapping) or not raw_windows:
        raise VestConfigurationError(f"{context}: 'windows' must be a non-empty mapping")
    # Checked before the override is applied: otherwise a `default` naming a
    # window that does not exist would be materialized out of the override's
    # own tstart/tend/dt, and the "not configured" guard below could never
    # fire for the callers that always pass them (the Snakemake rule does).
    if default_name not in raw_windows:
        raise VestConfigurationError(
            f"{context}: default window {default_name!r} is not configured; "
            f"configured windows: {', '.join(sorted(str(n) for n in raw_windows))}"
        )
    if analysis_override:
        merged = dict(raw_windows.get(default_name) or {})
        merged.update(
            {
                key: value
                for key, value in analysis_override.items()
                if key in ("tstart", "tend", "dt") and value is not None
            }
        )
        raw_windows = dict(raw_windows)
        raw_windows[default_name] = merged
    windows = {
        str(name): _diagnostics_time_window(name, window, context=context)
        for name, window in raw_windows.items()
    }
    if default_name not in windows:
        raise VestConfigurationError(
            f"{context}: default window {default_name!r} is not configured"
        )

    components = document.get("components")
    if not isinstance(components, Mapping):
        raise VestConfigurationError(f"{context}: 'components' must be a mapping")
    table = DiagnosticsTimePolicyTable(
        windows=windows, default=windows[default_name]
    )
    for component, window_name in components.items():
        if not isinstance(window_name, str) or window_name not in windows:
            raise VestConfigurationError(
                f"{context}: component {str(component)!r} names unknown window "
                f"{window_name!r}; configured windows: {', '.join(sorted(windows))}"
            )
        table[str(component)] = windows[window_name]
    return table


PLASMA_TIMING_KEY = "plasma_timing"

#: The rule blocks of ``plasma_timing`` whose keys are keyword arguments of
#: :func:`vaft.process.onset.active_window`.
_PLASMA_TIMING_RULE_BLOCKS = ("h_alpha", "ip")


@dataclass(frozen=True)
class PlasmaTimingPolicy:
    """The configured plasma-timing policy (issue #409).

    ``window`` is the shared plasma-analysis range every consumer searches
    inside; ``reference_lead_s`` the stretch before it that serves as the
    baseline reference.  ``h_alpha``, ``ip`` and each ``lines[label]`` entry
    are keyword arguments of :func:`vaft.process.onset.active_window`, keyed
    by the signal they were tuned on; ``usability`` and ``agreement`` are the
    channel checks and cross-check tolerances ``vaft.omas.plasma_timing``
    applies.  The generic detectors carry none of these numbers.
    """

    window: DiagnosticsTimePolicy
    reference_lead_s: float
    h_alpha: Mapping[str, Any]
    lines: Mapping[str, Mapping[str, Any]]
    ip: Mapping[str, Any]
    usability: Mapping[str, float]
    agreement: Mapping[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window.as_dict(),
            "reference_lead_s": self.reference_lead_s,
            "h_alpha": dict(self.h_alpha),
            "lines": {label: dict(rule) for label, rule in self.lines.items()},
            "ip": dict(self.ip),
            "usability": dict(self.usability),
            "agreement": dict(self.agreement),
        }


def _active_window_rule(rule: Any, *, context: str) -> Dict[str, Any]:
    """Validate one rule block against ``active_window``'s keyword parameters."""
    import inspect

    from vaft.process.onset import active_window

    if not isinstance(rule, Mapping) or not rule:
        raise VestConfigurationError(f"{context}: must be a non-empty mapping")
    parameters = inspect.signature(active_window).parameters
    allowed = {
        name for name, p in parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    unknown = sorted(set(rule) - allowed)
    if unknown:
        raise VestConfigurationError(
            f"{context}: {', '.join(repr(k) for k in unknown)} "
            f"{'is' if len(unknown) == 1 else 'are'} not a keyword argument of "
            f"vaft.process.onset.active_window"
        )
    out: Dict[str, Any] = {}
    for key, value in rule.items():
        if isinstance(value, bool) or value is None:
            out[key] = value
        else:
            number = _required_number(rule, key, context=context)
            if not np.isfinite(number):
                raise VestConfigurationError(f"{context}: parameter {key!r} must be finite")
            out[key] = int(number) if key in ("prefilter_samples", "order") else number
    return out


def _number_block(block: Any, keys: Sequence[str], *, context: str) -> Dict[str, float]:
    if not isinstance(block, Mapping):
        raise VestConfigurationError(f"{context}: must be a mapping")
    out = {key: _required_number(block, key, context=context) for key in keys}
    for key, value in out.items():
        if not np.isfinite(value) or value < 0.0:
            raise VestConfigurationError(f"{context}: parameter {key!r} must be finite and >= 0")
    return out


PLASMA_TIMING_USABILITY_KEYS = (
    "min_samples",
    "min_finite_fraction",
    "rail_level",
    "rail_fraction",
    "max_railed_fraction",
    "min_baseline_mad",
    "min_valid_fraction",
)
PLASMA_TIMING_AGREEMENT_KEYS = ("onset_tolerance_s", "lag_tolerance_s", "offset_tolerance_s")


def resolve_plasma_timing_policy(*, info_file: str | None = None) -> PlasmaTimingPolicy:
    """Return the configured plasma-timing policy (issue #409).

    ``window`` names one of the ``diagnostics_time_policies`` windows -- the
    shared plasma-analysis range -- and is resolved without any analysis
    override, so the stage's ``tstart``/``tend`` arguments never move it.
    Every rule block is checked against ``active_window``'s signature here,
    so a misspelled parameter is a configuration error at load time rather
    than a ``TypeError`` inside a detector.
    """
    content = load_yaml(_resolve_info_file_path(info_file))
    document = content.get(PLASMA_TIMING_KEY)
    context = PLASMA_TIMING_KEY
    if not isinstance(document, Mapping):
        raise VestConfigurationError(
            f"VEST configuration defines no {PLASMA_TIMING_KEY!r} section"
        )
    window_name = document.get("window")
    windows = resolve_diagnostics_time_policies(info_file=info_file).windows
    if not isinstance(window_name, str) or window_name not in windows:
        raise VestConfigurationError(
            f"{context}: 'window' must name a configured diagnostics time window; "
            f"configured windows: {', '.join(sorted(windows))}"
        )
    lead = _required_number(document, "reference_lead_s", context=context)
    if not np.isfinite(lead) or lead < 0.0:
        raise VestConfigurationError(f"{context}: 'reference_lead_s' must be finite and >= 0")
    rules = {
        name: _active_window_rule(document.get(name), context=f"{context}: {name}")
        for name in _PLASMA_TIMING_RULE_BLOCKS
    }
    raw_lines = document.get("lines", {})
    if raw_lines is None:
        raw_lines = {}
    if not isinstance(raw_lines, Mapping):
        raise VestConfigurationError(f"{context}: 'lines' must be a mapping of label -> rule")
    lines = {
        str(label): _active_window_rule(rule, context=f"{context}: lines[{label!r}]")
        for label, rule in raw_lines.items()
    }
    return PlasmaTimingPolicy(
        window=windows[window_name],
        reference_lead_s=lead,
        h_alpha=rules["h_alpha"],
        lines=lines,
        ip=rules["ip"],
        usability=_number_block(
            document.get("usability"), PLASMA_TIMING_USABILITY_KEYS, context=f"{context}: usability"
        ),
        agreement=_number_block(
            document.get("agreement"), PLASMA_TIMING_AGREEMENT_KEYS, context=f"{context}: agreement"
        ),
    )


def build_window_time_axis(
    source_time: Any,
    tstart: float,
    tend: float,
    dt: float,
) -> np.ndarray:
    """Build the half-open target grid for one window, clipped to real coverage.

    The result never extrapolates: it is confined to
    ``[max(tstart, source[0]), min(tend, source[-1]))``.  A component whose
    acquisition is shorter than its configured window therefore realizes a
    narrower span rather than a fabricated one, and the caller can report that
    clipping honestly.
    """
    source = np.asarray(source_time, dtype=float).reshape(-1)
    tstart, tend, dt = float(tstart), float(tend), float(dt)
    if not all(np.isfinite([tstart, tend, dt])):
        raise VestConfigurationError("Time window tstart, tend, and dt must be finite")
    if tend <= tstart:
        raise VestConfigurationError("Time window tend must be greater than tstart")
    if dt <= 0.0:
        raise VestConfigurationError(
            "Time window dt must be positive; a native timebase is an explicit mapper mode"
        )
    if source.size == 0:
        raise VestConfigurationError("Cannot build a time axis from an empty source timebase")
    start = max(tstart, float(source[0]))
    end = min(tend, float(source[-1]))
    if end <= start:
        raise VestConfigurationError(
            f"Requested window [{tstart}, {tend}) does not overlap the source coverage "
            f"[{float(source[0])}, {float(source[-1])}]"
        )
    axis = np.arange(start, end, dt, dtype=float)
    if axis.size == 0:
        raise VestConfigurationError(
            f"Requested window [{tstart}, {tend}) produces an empty grid at dt={dt}"
        )
    return axis


def _normalize_shot_key(source: Any) -> str:
    try:
        return str(int(source))
    except Exception:
        return str(source)


def raw_database_info(file: str, shot: int, key: str) -> Dict[str, Dict[str, Any]]:
    """
    Return channel metadata for a system key from a YAML source.

    The returned structure is normalized as
    ``{"labels": {...}, "fields": {...}, "gains": {...}}``
    where channel indices are string keys.
    """
    info_file = _resolve_info_file_path(file)
    content = load_yaml(info_file)

    shot_key = _normalize_shot_key(shot)
    default_block = content.get("0") or content.get(0) or content.get("static") or {}
    shot_block = content.get(shot_key, {})
    merged_block = _deep_merge(default_block, shot_block)

    key_block = merged_block.get(key, {})
    if not isinstance(key_block, Mapping):
        raise ValueError(f"No valid mapping found for key '{key}' in '{info_file}'")

    labels: Dict[str, Any] = {}
    fields: Dict[str, Any] = {}
    gains: Dict[str, Any] = {}

    for channel_index, channel_info in key_block.items():
        channel_key = str(channel_index)
        if not isinstance(channel_info, Mapping):
            continue
        label = channel_info.get("label", channel_info.get("name", channel_key))
        field = channel_info.get("field")
        gain = channel_info.get("gain", 1.0)
        labels[channel_key] = label
        fields[channel_key] = field
        gains[channel_key] = gain

    if not fields:
        raise ValueError(f"No channel definitions for key '{key}' in '{info_file}'")

    return {"labels": labels, "fields": fields, "gains": gains}


def load_raw_data(
    source: str, field: str | int, options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw data by delegating DB access to `vaft.database.raw`."""
    if options is None:
        options = {}

    source_type = options.get("source_type", "shot")

    if source_type == "shot":
        try:
            numeric_field = int(field)
            loaded = raw_db.load(int(source), numeric_field)
        except (TypeError, ValueError):
            loaded = raw_db.vest_load_by_name(int(source), str(field))
        return raw_db.require_signal(
            loaded,
            shot=int(source),
            field=field,
            signal_name=str(field),
        )

    file_format = options.get("file_format", "mat")
    if file_format != "mat":
        raise ValueError(f"Unsupported file format: {file_format}")

    from scipy.io import loadmat

    mat_data = loadmat(source)
    time = np.asarray(mat_data.get("time", np.array([]))).reshape(-1)
    data = np.asarray(mat_data.get(str(field), np.array([]))).reshape(-1)
    return time, data


def process_signal(
    time: np.ndarray, data: np.ndarray, options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Deprecated wrapper for :func:`vaft.process.process_signal`."""
    warnings.warn(
        "vaft.machine_mapping.utils.process_signal() is deprecated; use "
        "vaft.process.process_signal().",
        DeprecationWarning,
        stacklevel=2,
    )
    return process_signal_impl(time, data, options)


def get_diagnostic_info(
    source: str, diagnostic_type: str, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if options is None:
        options = {}

    source_type = options.get("source_type", "shot")
    info_file = _resolve_info_file_path(options.get("info_file"))
    info = load_yaml(info_file)

    if source_type == "shot":
        shot_key = _normalize_shot_key(source)
        shot_block = info.get(shot_key)
        default_block = info.get("0") or info.get(0) or info.get("static") or info
        block = shot_block or default_block
    else:
        block = info

    if diagnostic_type not in block:
        raise ValueError(
            f"No information found for diagnostic '{diagnostic_type}' in '{info_file}'"
        )

    return block[diagnostic_type]


def get_static_info(
    source: str, diagnostic_type: str, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if options is None:
        options = {}

    source_type = options.get("source_type", "shot")
    info_file = _resolve_info_file_path(options.get("info_file"))
    info = load_yaml(info_file)

    def pick_block() -> Dict[str, Any]:
        if source_type != "shot":
            return info
        shot_key = _normalize_shot_key(source)
        return info.get(shot_key) or info.get("0") or info.get(0) or info

    block = pick_block()

    if isinstance(block, dict) and "static" in block:
        static = block.get("static") or {}
        if diagnostic_type not in static:
            raise ValueError(
                f"No static information for '{diagnostic_type}' in '{info_file}'"
            )
        return static[diagnostic_type]

    if diagnostic_type in block:
        return block[diagnostic_type]

    for _, group in (block or {}).items():
        if isinstance(group, dict) and diagnostic_type in group:
            return group[diagnostic_type]

    raise ValueError(f"No static information found for '{diagnostic_type}' in '{info_file}'")


def process_static_geometry(ods: Any, diagnostic_type: str, static_info: Dict[str, Any]) -> None:
    if "geometry" not in static_info:
        return

    geometry = static_info["geometry"]

    if diagnostic_type == "flux_loop":
        for i, loop in enumerate(geometry.get("loops", [])):
            set_path(ods, f"flux_loop.loop.{i}.position.r", loop.get("r", 0.0))
            set_path(ods, f"flux_loop.loop.{i}.position.z", loop.get("z", 0.0))
            set_path(ods, f"flux_loop.loop.{i}.position.phi", loop.get("phi", 0.0))
            set_path(ods, f"flux_loop.loop.{i}.area", loop.get("area", 0.0))

    elif diagnostic_type == "b_field_pol_probe":
        for i, probe in enumerate(geometry.get("probes", [])):
            set_path(ods, f"b_field_pol_probe.probe.{i}.position.r", probe.get("r", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.position.z", probe.get("z", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.position.phi", probe.get("phi", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.orientation.r", probe.get("orientation_r", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.orientation.z", probe.get("orientation_z", 0.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.orientation.phi", probe.get("orientation_phi", 0.0))

    elif diagnostic_type == "rogowski_coil":
        # Issue #215: these legacy helpers write a top-level
        # `rogowski_coil.coil.<i>.*` path that is not the canonical IMAS
        # location (`magnetics.rogowski_coil.<i>.*`) and does not match the DD
        # geometry convention (`position[:]` contour, `turns_per_metre`).
        # Refuse rather than silently populating the wrong structure; the
        # canonical mapping lives in vaft.machine_mapping.magnetics.
        raise VestConfigurationError(
            "Legacy rogowski_coil static helpers write the non-canonical "
            "'rogowski_coil.coil.*' path; use "
            "vaft.machine_mapping.magnetics._map_rogowski_coils, which writes "
            "magnetics.rogowski_coil.* (issue #215)"
        )


def process_static_channels(ods: Any, diagnostic_type: str, static_info: Dict[str, Any]) -> None:
    if "channels" not in static_info:
        return

    channels = static_info["channels"]

    if diagnostic_type == "flux_loop":
        for i, channel in enumerate(channels):
            set_path(ods, f"flux_loop.loop.{i}.name", channel.get("name", f"FL{i}"))
            set_path(ods, f"flux_loop.loop.{i}.gain", channel.get("gain", 1.0))
            set_path(ods, f"flux_loop.loop.{i}.offset", channel.get("offset", 0.0))
            set_path(
                ods,
                f"flux_loop.loop.{i}.calibration_factor",
                channel.get("calibration_factor", 1.0),
            )

    elif diagnostic_type == "b_field_pol_probe":
        for i, channel in enumerate(channels):
            set_path(ods, f"b_field_pol_probe.probe.{i}.name", channel.get("name", f"BP{i}"))
            set_path(ods, f"b_field_pol_probe.probe.{i}.gain", channel.get("gain", 1.0))
            set_path(ods, f"b_field_pol_probe.probe.{i}.offset", channel.get("offset", 0.0))
            set_path(
                ods,
                f"b_field_pol_probe.probe.{i}.calibration_factor",
                channel.get("calibration_factor", 1.0),
            )

    elif diagnostic_type == "rogowski_coil":
        # Issue #215: these legacy helpers write a top-level
        # `rogowski_coil.coil.<i>.*` path that is not the canonical IMAS
        # location (`magnetics.rogowski_coil.<i>.*`) and does not match the DD
        # geometry convention (`position[:]` contour, `turns_per_metre`).
        # Refuse rather than silently populating the wrong structure; the
        # canonical mapping lives in vaft.machine_mapping.magnetics.
        raise VestConfigurationError(
            "Legacy rogowski_coil static helpers write the non-canonical "
            "'rogowski_coil.coil.*' path; use "
            "vaft.machine_mapping.magnetics._map_rogowski_coils, which writes "
            "magnetics.rogowski_coil.* (issue #215)"
        )


def get_metadata(source: str, options: dict | None = None) -> dict[str, Any]:
    if options is None:
        options = {}

    source_type = options.get("source_type", "shot")
    metadata_type = options.get("metadata_type", "all")
    _ = metadata_type  # Reserved for future metadata filtering.

    if source_type == "shot":
        return {
            "shot": int(source),
            "timestamp": datetime.datetime.now().isoformat(),
            "source_type": "shot",
        }
    return {
        "file": source,
        "timestamp": datetime.datetime.now().isoformat(),
        "source_type": "file",
    }


def _scaled_uncertainty(values: Any, relative_error: float):
    array = np.abs(float(relative_error) * np.asarray(values, dtype=float))
    if array.ndim == 0:
        return float(array)
    return array


def _annotate_series(
    ods: Any,
    base_path: str,
    relative_error: float,
    time_source_path: str | None = None,
) -> None:
    data_path = f"{base_path}.data"
    if not path_exists(ods, data_path):
        return
    if time_source_path is not None and path_exists(ods, time_source_path):
        set_path(ods, f"{base_path}.time", get_path(ods, time_source_path))
    set_path(ods, f"{base_path}.data_error_upper", _scaled_uncertainty(get_path(ods, data_path), relative_error))


@lru_cache(maxsize=1)
def _load_magnetics_channel_groups() -> dict[str, list[int]]:
    geometry_path = resolve_data_root() / "geometry" / "VEST_MagneticsGeometry_Full_ver_2302.yaml"
    with open(geometry_path, "r", encoding="utf-8") as handle:
        channels = yaml.safe_load(handle)["channels"]

    groups = {
        "magnetics_bpol_inboard": [],
        "magnetics_bpol_side": [],
        "magnetics_bpol_outboard": [],
        "magnetics_flux_loop_inboard": [],
        "magnetics_flux_loop_outboard": [],
    }

    flux_index = 0
    probe_index = 0
    for channel in channels:
        radial = float(channel["r"])
        vertical = float(channel["z"])
        if channel["kind"] == "flux_loop":
            if radial < 0.15:
                groups["magnetics_flux_loop_inboard"].append(flux_index)
            elif radial > 0.5:
                groups["magnetics_flux_loop_outboard"].append(flux_index)
            flux_index += 1
            continue

        if radial < 0.09:
            groups["magnetics_bpol_inboard"].append(probe_index)
        elif abs(vertical) > 0.8:
            groups["magnetics_bpol_side"].append(probe_index)
        elif radial > 0.795:
            groups["magnetics_bpol_outboard"].append(probe_index)
        probe_index += 1

    return groups


def normalize_constraint_uncertainties(
    uncertainty: Sequence[float] | Mapping[str, float] | None = None,
) -> dict[str, float]:
    if uncertainty is None:
        return dict(DEFAULT_CONSTRAINT_UNCERTAINTIES)

    if isinstance(uncertainty, Mapping):
        normalized = dict(DEFAULT_CONSTRAINT_UNCERTAINTIES)
        unknown = set(uncertainty) - set(normalized)
        if unknown:
            unknown_text = ", ".join(sorted(str(item) for item in unknown))
            raise KeyError(f"Unknown uncertainty keys: {unknown_text}")
        normalized.update({key: float(value) for key, value in uncertainty.items()})
        return normalized

    values = tuple(float(value) for value in uncertainty)
    if len(values) != len(DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR):
        raise ValueError(
            "Constraint uncertainty vector must contain "
            f"{len(DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR)} entries"
        )

    return dict(zip(DEFAULT_CONSTRAINT_UNCERTAINTIES.keys(), values))


def apply_pf_active_current_uncertainties(ods: Any, relative_error: float | None = None) -> None:
    if not path_exists(ods, "pf_active.coil"):
        return
    if relative_error is None:
        relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["pf_active_current"]
    for coil_index, _ in enumerate(get_path(ods, "pf_active.coil")):
        _annotate_series(
            ods,
            f"pf_active.coil.{coil_index}.current",
            relative_error,
            time_source_path="pf_active.time",
        )


def apply_tf_uncertainties(ods: Any, relative_error: float | None = None) -> None:
    if relative_error is None:
        relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["tf_b_field_tor_vacuum_r"]
    _annotate_series(
        ods,
        "tf.b_field_tor_vacuum_r",
        relative_error,
        time_source_path="tf.time",
    )


def apply_magnetics_uncertainties(
    ods: Any,
    *,
    ip_relative_error: float | None = None,
    diamagnetic_flux_relative_error: float | None = None,
    bpol_inboard_relative_error: float | None = None,
    bpol_side_relative_error: float | None = None,
    bpol_outboard_relative_error: float | None = None,
    flux_loop_inboard_relative_error: float | None = None,
    flux_loop_outboard_relative_error: float | None = None,
    fl_correct_coeff: Sequence[float] | None = None,
) -> None:
    groups = _load_magnetics_channel_groups()

    if ip_relative_error is None:
        ip_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_ip"]
    if diamagnetic_flux_relative_error is None:
        diamagnetic_flux_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_diamagnetic_flux"]
    if bpol_inboard_relative_error is None:
        bpol_inboard_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_bpol_inboard"]
    if bpol_side_relative_error is None:
        bpol_side_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_bpol_side"]
    if bpol_outboard_relative_error is None:
        bpol_outboard_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_bpol_outboard"]
    if flux_loop_inboard_relative_error is None:
        flux_loop_inboard_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_flux_loop_inboard"]
    if flux_loop_outboard_relative_error is None:
        flux_loop_outboard_relative_error = DEFAULT_CONSTRAINT_UNCERTAINTIES["magnetics_flux_loop_outboard"]

    _annotate_series(ods, "magnetics.ip.0", ip_relative_error)
    _annotate_series(
        ods,
        "magnetics.diamagnetic_flux.0",
        diamagnetic_flux_relative_error,
        time_source_path="magnetics.time",
    )

    if fl_correct_coeff is not None and path_exists(ods, "magnetics.flux_loop"):
        for flux_index, coeff in enumerate(fl_correct_coeff):
            flux_data_path = f"magnetics.flux_loop.{flux_index}.flux.data"
            if not path_exists(ods, flux_data_path):
                continue
            set_path(ods, flux_data_path, np.asarray(get_path(ods, flux_data_path), dtype=float) / float(coeff))

    for probe_index in groups["magnetics_bpol_inboard"]:
        _annotate_series(
            ods,
            f"magnetics.b_field_pol_probe.{probe_index}.field",
            bpol_inboard_relative_error,
            time_source_path="magnetics.time",
        )
    for probe_index in groups["magnetics_bpol_side"]:
        _annotate_series(
            ods,
            f"magnetics.b_field_pol_probe.{probe_index}.field",
            bpol_side_relative_error,
            time_source_path="magnetics.time",
        )
    for probe_index in groups["magnetics_bpol_outboard"]:
        _annotate_series(
            ods,
            f"magnetics.b_field_pol_probe.{probe_index}.field",
            bpol_outboard_relative_error,
            time_source_path="magnetics.time",
        )

    for flux_index in groups["magnetics_flux_loop_inboard"]:
        _annotate_series(
            ods,
            f"magnetics.flux_loop.{flux_index}.flux",
            flux_loop_inboard_relative_error,
            time_source_path="magnetics.time",
        )
    for flux_index in groups["magnetics_flux_loop_outboard"]:
        _annotate_series(
            ods,
            f"magnetics.flux_loop.{flux_index}.flux",
            flux_loop_outboard_relative_error,
            time_source_path="magnetics.time",
        )


def apply_default_constraint_uncertainties(
    ods: Any,
    uncertainty: Sequence[float] | Mapping[str, float] | None = None,
    *,
    fl_correct_coeff: Sequence[float] | None = None,
) -> None:
    normalized = normalize_constraint_uncertainties(uncertainty)
    apply_pf_active_current_uncertainties(ods, normalized["pf_active_current"])
    apply_tf_uncertainties(ods, normalized["tf_b_field_tor_vacuum_r"])
    apply_magnetics_uncertainties(
        ods,
        ip_relative_error=normalized["magnetics_ip"],
        diamagnetic_flux_relative_error=normalized["magnetics_diamagnetic_flux"],
        bpol_inboard_relative_error=normalized["magnetics_bpol_inboard"],
        bpol_side_relative_error=normalized["magnetics_bpol_side"],
        bpol_outboard_relative_error=normalized["magnetics_bpol_outboard"],
        flux_loop_inboard_relative_error=normalized["magnetics_flux_loop_inboard"],
        flux_loop_outboard_relative_error=normalized["magnetics_flux_loop_outboard"],
        fl_correct_coeff=fl_correct_coeff,
    )


__all__ = [
    "DEFAULT_CONSTRAINT_UNCERTAINTIES",
    "DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR",
    "DIAGNOSTICS_TIME_POLICIES_KEY",
    "DiagnosticsTimePolicy",
    "DiagnosticsTimePolicyTable",
    "VestConfigurationError",
    "apply_default_constraint_uncertainties",
    "apply_magnetics_uncertainties",
    "apply_pf_active_current_uncertainties",
    "apply_tf_uncertainties",
    "build_window_time_axis",
    "get_diagnostic_info",
    "get_metadata",
    "get_path",
    "get_static_info",
    "load_raw_data",
    "raw_database_info",
    "load_yaml",
    "normalize_constraint_uncertainties",
    "package_data_path",
    "path_count",
    "path_exists",
    "path_value",
    "process_signal",
    "process_static_channels",
    "process_static_geometry",
    "resolve_data_root",
    "resolve_diagnostics_time_policies",
    "set_path",
]

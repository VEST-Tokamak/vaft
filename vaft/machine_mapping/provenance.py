"""Effective VEST processing provenance for one shot (issue #195).

For a requested shot it must be possible to determine which processing
revision actually applies -- calibration era, baseline era, FL10
compensation mode, PF gain era, PF6 saturation-repair policy, and the
equilibrium-magnetics acquisition era -- without reading several unrelated
`if shot ...` statements. This module is that single query point, and the
record it returns is what later EFIT/VFIT comparison work (#73) needs in
order to say which preprocessing produced a given result.
"""

from __future__ import annotations

from typing import Any

from .magnetics import UNSUPPORTED_MAGNETICS_GEOMETRY_SHOTS
from .utils import resolve_shot_revisions_with_provenance, resolve_vest_diagnostic

__all__ = ["vest_processing_provenance"]


def _nested_provenance(
    processing: dict[str, Any], name: str, shot: int, *, context: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = processing[name]
    return resolve_shot_revisions_with_provenance(
        {key: value for key, value in item.items() if key != "revisions"},
        item.get("revisions"),
        shot,
        context=context,
    )


def _plasma_current_provenance(shot: int) -> dict[str, Any]:
    config, top_level = resolve_vest_diagnostic(shot, "plasma_current", with_provenance=True)
    processing = config["processing"]

    reference, reference_prov = _nested_provenance(
        processing, "reference", shot, context="plasma_current reference"
    )
    baseline, baseline_prov = _nested_provenance(
        processing, "baseline", shot, context="plasma_current baseline"
    )
    sign, sign_prov = _nested_provenance(
        processing, "sign", shot, context="plasma_current sign"
    )

    return {
        "source_field": int(config["source"]["field"]),
        "calibration": {
            "factor": float(config["calibration"]["factor"]),
            "operation": str(config["calibration"]["operation"]),
            "revision": top_level,
        },
        "baseline": {
            "analysis_window": (
                int(baseline["analysis_start"]),
                int(baseline["analysis_end"]),
            ),
            "lookback": int(baseline["lookback"]),
            "revision": baseline_prov,
        },
        "reference": {
            "mode": str(reference.get("mode", "subtract")),
            # Renamed from `mutual_inductance` in issue #214: the divisor
            # must be in ohms for V/X to be a current. The donor calls it a
            # mutual inductance; see vest.yaml for that unresolved conflict.
            "effective_resistance_ohm": float(reference["effective_resistance_ohm"]),
            "flux_gain": float(reference["flux_gain"]),
            "compensation_enabled": str(reference.get("mode", "subtract")) != "disabled",
            "revision": reference_prov,
        },
        "sign": {"multiply": float(sign["multiply"]), "revision": sign_prov},
    }


def _pf_active_provenance(shot: int) -> dict[str, Any]:
    config, top_level = resolve_vest_diagnostic(shot, "pf_active", with_provenance=True)
    processing = config["processing"]
    repair = processing.get("saturation_repair") or {}
    return {
        "coil_gains": {
            int(index): float(gain) for index, gain in processing["coil_gains"].items()
        },
        "coil_gains_revision": top_level,
        "saturation_repair": {
            int(index): {
                "value": float(policy["value"]),
                "tolerance": float(policy["tolerance"]),
            }
            for index, policy in repair.items()
        },
    }


def _equilibrium_magnetics_provenance(shot: int) -> dict[str, Any]:
    config = resolve_vest_diagnostic(shot, "equilibrium_magnetics")
    window, window_prov = _nested_provenance(
        config["processing"], "window", shot, context="equilibrium_magnetics window"
    )
    flux_window = window.get("flux_baseline_window")
    flux_samples = window.get("flux_baseline_samples")
    return {
        "daq_mode": str(window["daq_mode"]),
        "output_index_window": (int(window["index_start"]), int(window["index_end"])),
        "probe_baseline_end": int(window["probe_baseline_end"]),
        "flux_baseline_window": (
            None if flux_window is None else (float(flux_window[0]), float(flux_window[1]))
        ),
        "flux_baseline_samples": None if flux_samples is None else int(flux_samples),
        "revision": window_prov,
        "geometry_supported": int(shot) not in UNSUPPORTED_MAGNETICS_GEOMETRY_SHOTS,
        "required_geometry_version": UNSUPPORTED_MAGNETICS_GEOMETRY_SHOTS.get(int(shot)),
    }


def vest_processing_provenance(shot: int) -> dict[str, Any]:
    """Return the effective processing era for *shot*, per diagnostic.

    Pure configuration resolution -- no raw data is loaded, so this is safe
    to call for reporting, manifests, or comparison metadata.
    """
    numeric_shot = int(shot)
    return {
        "shot": numeric_shot,
        "plasma_current": _plasma_current_provenance(numeric_shot),
        "pf_active": _pf_active_provenance(numeric_shot),
        "equilibrium_magnetics": _equilibrium_magnetics_provenance(numeric_shot),
    }

"""Effective VEST processing provenance for one shot.

For a requested shot it must be possible to determine which processing
revision actually applies -- calibration era, baseline era, FL10
compensation mode, PF gain era, PF6 saturation-repair policy, and the
equilibrium-magnetics acquisition era -- without reading several unrelated
`if shot ...` statements. This module is that single query point, and the
record it returns is what an EFIT/VFIT comparison needs in
order to say which preprocessing produced a given result.
"""

from __future__ import annotations

from typing import Any

from .magnetics import (
    UNSUPPORTED_MAGNETICS_GEOMETRY_SHOTS,
    known_magnetics_faults,
    magnetics_wiring_for_shot,
)
from .effective_config import (
    configuration_fingerprint,
    impa_shot_era,
    resolve_vest_processing_config,
    vest_configuration_source,
)
from .magnetics_registry import REGISTRY_CONFIG_PATH, registry_fingerprint
from .utils import (
    resolve_diagnostics_time_policies,
    resolve_plasma_timing_policy,
    resolve_shot_revisions_with_provenance,
    resolve_vest_diagnostic,
)

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


def _plasma_current_provenance(shot: int, info_file: str | None = None) -> dict[str, Any]:
    config, top_level = resolve_vest_diagnostic(
        shot, "plasma_current", info_file=info_file, with_provenance=True
    )
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
            # An effective resistance, not the donor's `mutual_inductance`:
            # the divisor must be in ohms for V/X to be a current. The donor calls it a
            # mutual inductance; see vest.yaml for that unresolved conflict.
            "effective_resistance_ohm": float(reference["effective_resistance_ohm"]),
            "flux_gain": float(reference["flux_gain"]),
            "compensation_enabled": str(reference.get("mode", "subtract")) != "disabled",
            "revision": reference_prov,
        },
        "sign": {"multiply": float(sign["multiply"]), "revision": sign_prov},
    }


def _pf_active_provenance(shot: int, info_file: str | None = None) -> dict[str, Any]:
    config, top_level = resolve_vest_diagnostic(
        shot, "pf_active", info_file=info_file, with_provenance=True
    )
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


def _equilibrium_magnetics_provenance(shot: int, info_file: str | None = None) -> dict[str, Any]:
    config = resolve_vest_diagnostic(shot, "equilibrium_magnetics", info_file=info_file)
    window, window_prov = _nested_provenance(
        config["processing"], "window", shot, context="equilibrium_magnetics window"
    )
    flux_window = window.get("flux_baseline_window")
    flux_samples = window.get("flux_baseline_samples")
    wiring = magnetics_wiring_for_shot(int(shot))
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
        # Which raw field feeds each probe position, and which positions are
        # recorded as broken, for this shot.
        "wiring": {
            "layout": wiring.layout,
            "probe_overrides": [dict(entry) for entry in wiring.overrides],
            "revision": dict(wiring.provenance),
        },
        "known_faults": [
            {"kind": kind, "index": index, "reason": reason}
            for (kind, index), reason in sorted(known_magnetics_faults(int(shot)).items())
        ],
        # The wiring and fault resolvers read the packaged vest.yaml only; say
        # so rather than let a campaign file appear to have supplied them.
        "wiring_and_faults_source": "vaft/machine_mapping/vest.yaml",
        # The effective sensor definitions this shot resolved to (the vest.yaml
        # sensor registry):
        # where they live and the fingerprint of the resolved records.
        "sensor_registry": {
            "source": REGISTRY_CONFIG_PATH,
            "fingerprint": registry_fingerprint(int(shot)),
        },
    }


def _core_profiles_provenance(shot: int) -> dict[str, Any]:
    from .core_profiles import vest_core_profiles_policy

    policy = vest_core_profiles_policy(shot)
    return {
        "coordinate": policy.coordinate,
        "ti_te_ratio": {
            "value": policy.ti_te_ratio,
            "sigma": policy.ti_te_ratio_sigma,
            "status": policy.ti_te_ratio_status,
        },
        "impurities": {
            species: {"fraction": policy.impurity_fractions[species], "status": policy.impurity_status[species]}
            for species in policy.impurity_species
        },
        "impurity_model": None if policy.impurity_model is None else {
            "status": policy.impurity_model["status"],
            "target_zeff": policy.impurity_model["target_zeff"],
            "species": [
                {key: item[key] for key in ("element", "charge_state", "mass", "relative_density")}
                for item in policy.impurity_model["species"]
            ],
        },
        "revision": policy.provenance["revision"],
        "source": policy.source,
    }


def _effective_summary(shot: int, diagnostic: str, info_file: str | None) -> dict[str, Any]:
    """Fingerprint, applied revisions and effective values of one diagnostic."""
    resolved = resolve_vest_processing_config(shot, diagnostic, info_file=info_file)
    record = resolved.as_record()
    return {
        "fingerprint": record["fingerprint"],
        "revisions": record["provenance"]["revisions"],
        "effective": record["effective"],
    }


def _impa_provenance(shot: int, info_file: str | None) -> dict[str, Any]:
    from .impa import _resolve_gain

    resolved = resolve_vest_processing_config(shot, "impa", info_file=info_file)
    calibration = resolved.effective_dict().get("calibration") or {}
    era, era_provenance = impa_shot_era(calibration, shot)
    return {
        "gain": _resolve_gain(calibration, shot),
        "era": {
            "revision_index": era_provenance["revision_index"],
            "revision_bounds": era_provenance["revision_bounds"],
            "channels": None if era is None else era.get("channels"),
            "bz_channels": None if era is None else era.get("bz_channels"),
        },
        "processing": dict(resolved.effective_dict().get("processing") or {}),
        "fingerprint": resolved.fingerprint,
    }


def _time_policies_provenance(info_file: str | None) -> dict[str, Any]:
    policies = resolve_diagnostics_time_policies(info_file=info_file)
    timing = resolve_plasma_timing_policy(info_file=info_file)
    effective = {
        "default": policies.default.name if policies.default is not None else None,
        "windows": {name: policy.as_dict() for name, policy in sorted(policies.windows.items())},
        "components": {name: policy.name for name, policy in sorted(policies.items())},
    }
    return {
        "diagnostics_time_policies": {**effective, "fingerprint": configuration_fingerprint(effective)},
        "plasma_timing": {
            **timing.as_dict(),
            "fingerprint": configuration_fingerprint(timing.as_dict()),
        },
    }


def vest_processing_provenance(shot: int, *, info_file: str | None = None) -> dict[str, Any]:
    """Return the effective processing era for *shot*, per diagnostic.

    Pure configuration resolution -- no raw data is loaded, so this is safe
    to call for reporting, manifests, or comparison metadata.

    ``info_file`` selects the configuration file exactly as every resolver
    does (``None``: the packaged vest.yaml; a missing explicit path raises).
    The record names that file and its sha256 (``source``), and carries the
    TF, IMPA and time-policy resolutions and per-diagnostic fingerprints.
    The equilibrium-magnetics wiring and known faults are always the
    packaged file's (``wiring_and_faults_source``).
    """
    numeric_shot = int(shot)
    return {
        "shot": numeric_shot,
        "source": vest_configuration_source(info_file),
        "plasma_current": _plasma_current_provenance(numeric_shot, info_file),
        "pf_active": _pf_active_provenance(numeric_shot, info_file),
        "equilibrium_magnetics": _equilibrium_magnetics_provenance(numeric_shot, info_file),
        "tf": _effective_summary(numeric_shot, "tf", info_file),
        "impa": _impa_provenance(numeric_shot, info_file),
        "time_policies": _time_policies_provenance(info_file),
        "core_profiles": _core_profiles_provenance(numeric_shot),
        "fingerprints": {
            name: resolve_vest_processing_config(numeric_shot, name, info_file=info_file).fingerprint
            for name in ("plasma_current", "pf_active", "equilibrium_magnetics", "tf", "impa")
        },
    }

"""Magnetic diagnostic extraction for NICE, independent of EFIT k-files."""

from __future__ import annotations

from typing import Any
from dataclasses import replace

import numpy as np

from .config import NiceConfig, NiceDiagnostic


def _value_at(ods: Any, base: str, time: float) -> float:
    try:
        data = np.asarray(ods[f"{base}.data"], float)
    except Exception:
        return float("nan")
    if data.ndim != 1 or not data.size:
        return float("nan")
    try:
        times = np.asarray(ods[f"{base}.time"], float)
        if times.ndim != 1 or not times.size:
            raise ValueError
    except Exception:
        times = np.asarray(ods["magnetics.time"], float)
    if data.size != times.size:
        return float("nan")
    return float(np.interp(time, times, data))


def _uncertainty_at(ods: Any, base: str, time: float, default: float) -> float:
    for leaf in ("data_error_upper", "data_error_lower"):
        try:
            values = np.abs(np.asarray(ods[f"{base}.{leaf}"], float))
            if values.size == 1:
                value = float(values[0])
            else:
                try:
                    times = np.asarray(ods[f"{base}.time"], float)
                except Exception:
                    times = np.asarray(ods["magnetics.time"], float)
                value = float(np.interp(time, times, values))
            if np.isfinite(value) and value > 0:
                return value
        except Exception:
            pass
    return float(default)


def _identity(node: Any, fallback: str) -> str:
    for key in ("identifier", "name"):
        try:
            if key not in node:
                continue
            value = str(node[key])
            if value:
                return value
        except Exception:
            pass
    return fallback


def _explicitly_disabled(config: NiceConfig, path: str, identifier: str) -> bool:
    disabled = set(config.disabled_channels)
    return path in disabled or identifier in disabled


def diagnostics_from_ods(
    ods: Any, time: float, config: NiceConfig
) -> tuple[NiceDiagnostic, ...]:
    """Return a complete, provenance-ready channel table for one slice."""
    result: list[NiceDiagnostic] = []

    ip_path = "magnetics.ip.0"
    if config.include_plasma_current:
        value = _value_at(ods, ip_path, time)
        identifier = _identity(ods[ip_path], "Ip")
        enabled = bool(
            np.isfinite(value) and not _explicitly_disabled(config, ip_path, identifier)
        )
        result.append(
            NiceDiagnostic(
                "plasma_current",
                ip_path,
                identifier,
                {},
                value,
                _uncertainty_at(ods, ip_path, time, config.default_ip_uncertainty),
                config.default_ip_uncertainty,
                enabled,
                "" if enabled else "missing/non-finite or explicitly disabled",
            )
        )

    for family, signal, default, include in (
        (
            "bpol_probe",
            "field",
            config.default_bpol_uncertainty,
            config.include_bpol_probes,
        ),
        (
            "flux_loop",
            "flux",
            config.default_flux_uncertainty,
            config.include_flux_loops,
        ),
    ):
        ods_family = "b_field_pol_probe" if family == "bpol_probe" else "flux_loop"
        if not include:
            continue
        try:
            count = len(ods[f"magnetics.{ods_family}"])
        except Exception:
            count = 0
        for index in range(count):
            root = f"magnetics.{ods_family}.{index}"
            base = f"{root}.{signal}"
            node = ods[root]
            identifier = _identity(node, f"{family}:{index}")
            if family == "bpol_probe":
                geometry = {
                    "r": float(node["position.r"]),
                    "z": float(node["position.z"]),
                    "phi": float(node["position.phi"])
                    if "position.phi" in node
                    else 0.0,
                    "poloidal_angle": float(node["poloidal_angle"])
                    if "poloidal_angle" in node
                    else 0.0,
                }
            else:
                geometry = {"positions": []}
                for pos in range(len(node["position"])):
                    p = node[f"position.{pos}"]
                    geometry["positions"].append(
                        {
                            "r": float(p["r"]),
                            "z": float(p["z"]),
                            "phi": float(p["phi"]) if "phi" in p else 0.0,
                        }
                    )
            value = _value_at(ods, base, time)
            uncertainty = _uncertainty_at(ods, base, time, default)
            enabled = bool(
                np.isfinite(value)
                and not _explicitly_disabled(config, root, identifier)
            )
            result.append(
                NiceDiagnostic(
                    family,
                    root,
                    identifier,
                    geometry,
                    value,
                    uncertainty,
                    uncertainty,
                    enabled,
                    "" if enabled else "missing/non-finite or explicitly disabled",
                )
            )
    if config.include_diamagnetic_flux:
        root = "magnetics.diamagnetic_flux.0"
        value = _value_at(ods, root, time)
        result.append(
            NiceDiagnostic(
                "diamagnetic_flux",
                root,
                _identity(ods[root], "diamagnetic_flux"),
                {},
                value,
                _uncertainty_at(
                    ods, root, time, config.default_diamagnetic_uncertainty
                ),
                config.default_diamagnetic_uncertainty,
                False,
                "unsupported by the NICE 1.0 standalone reconstruction objective",
            )
        )
    result = [
        replace(
            d,
            original_value=d.value,
            conditioned_value=d.value,
            original_uncertainty=d.uncertainty,
            original_enabled=d.enabled,
        )
        for d in result
    ]
    if config.diagnostic_source == "magnetics":
        return tuple(result)
    if config.diagnostic_source != "equilibrium_constraints":
        raise ValueError(
            "diagnostic_source must be magnetics or equilibrium_constraints"
        )
    times = np.asarray(ods["equilibrium.time"], float)
    matches = np.flatnonzero(np.isclose(times, time, rtol=0, atol=1e-9))
    if len(matches) != 1:
        raise ValueError(f"No unique exact-time equilibrium constraints at {time}s")
    constraints = ods[f"equilibrium.time_slice.{int(matches[0])}.constraints"]
    conditioned = []
    for d in result:
        if d.family == "diamagnetic_flux":
            conditioned.append(d)
            continue
        family = {
            "plasma_current": "ip",
            "bpol_probe": "bpol_probe",
            "flux_loop": "flux_loop",
        }[d.family]
        key = family if family == "ip" else f"{family}.{d.ods_path.rsplit('.', 1)[1]}"
        if key not in constraints:
            conditioned.append(
                replace(
                    d,
                    enabled=False,
                    reason="absent from shared equilibrium constraints",
                )
            )
            continue
        c = constraints[key]
        if (
            d.family != "plasma_current"
            and "source" in c
            and str(c["source"]) != d.identifier
        ):
            raise ValueError(
                f"Shared constraint source mismatch for {d.ods_path}: {c['source']} != {d.identifier}"
            )
        value = float(c["measured"])
        sigma = abs(float(c["measured_error_upper"]))
        weight = float(c["weight"])
        enabled = (
            np.isfinite(value)
            and np.isfinite(sigma)
            and sigma > 0
            and weight > 0
            and not _explicitly_disabled(config, d.ods_path, d.identifier)
        )
        conditioned.append(
            replace(
                d,
                value=value,
                conditioned_value=value,
                uncertainty=sigma,
                normalization=sigma,
                weight=weight,
                enabled=enabled,
                reason="" if enabled else "disabled or invalid shared constraint",
            )
        )
    return tuple(conditioned)

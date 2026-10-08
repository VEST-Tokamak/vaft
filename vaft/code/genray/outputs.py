"""Read GENRAY's ``genray.nc`` and map the rays into the IMAS ``waves`` IDS.

GENRAY writes CGS: lengths in cm, power in erg/s, fields in gauss. Everything
converts to SI here. Only quantities GENRAY computes are written; the spot,
phase and electric-field nodes of ``beam_tracing`` stay absent (a ray tracer
has no beam width or phase front), and nothing is zero-filled.

Two definitions worth stating:

* ``beam.length`` is the arc length along the ray, integrated from the ray
  positions. GENRAY's ``ws`` is the *poloidal* distance, which the DD's
  "curvilinear length" is not.
* ``beam.electrons.power`` is the power absorbed from the ray so far,
  ``delpwr[0] - delpwr``. The run is electrons-only (``nbulk = 1``), so every
  damping channel GENRAY applies is electron damping. GENRAY also lowers
  ``delpwr`` by ``refl_loss`` at each reflection, which is not absorption, so
  the node is written only when ``refl_loss`` is zero (GENRAY's default).

A run is *complete* only when every total is present and at least one ray was
actually traced into the plasma: GENRAY exits 0 and writes ``genray.nc`` even
when no ray reaches the plasma (``nrayelt`` 0, status -1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: CGS -> SI.
_CM = 1.0e-2
_ERG_PER_S = 1.0e-7
_SPEED_OF_LIGHT = 2.99792458e8

#: Why GENRAY stopped a ray (``iray_status_nc``), as ``netcdfr3d.f`` documents it.
RAY_STOP_REASONS = {
    -1: "never entered the plasma (dinit_1ray found no boundary crossing)",
    1: "poloidal distance exceeded poldist_mx",
    2: "remaining power fell below delpwrmn",
    3: "reflection count reached ireflm",
    4: "reached the toroidal limiter boundary",
    5: "integration steps exceeded maxsteps_rk",
    6: "refractive index exceeded cnmax",
    7: "upper-hybrid condition in the cold dispersion (id = 1, 2)",
    8: "group velocity exceeded 1.1 c",
    9: "Hamiltonian residual exceeded toll_hamilt",
    10: "LSC approach stop",
    12: "ray left the computational domain",
    13: "Runge-Kutta step fell below 1e-11",
}

#: IMAS waves ``coherent_wave.identifier.type`` for electron-cyclotron waves.
EC_WAVE_TYPE = {"index": 1, "name": "EC", "description": "Wave field for electron cyclotron heating and current drive"}


def _text(variable: Any) -> str:
    raw = np.asarray(variable[:])
    if raw.dtype.kind in "SU":
        return b"".join(np.atleast_1d(raw).astype("S1")).decode("ascii", "replace").strip()
    return str(raw)


def read_genray_netcdf(path: str | Path) -> dict[str, Any]:
    """The per-ray arrays and totals of ``genray.nc``, in SI, trimmed to each ray's length."""
    import netCDF4

    with netCDF4.Dataset(str(path)) as data:
        var = data.variables
        counts = np.asarray(var["nrayelt"][:], dtype=int).reshape(-1)
        frequency = float(var["freqcy"][:])
        omega_over_c = 2.0 * np.pi * frequency / _SPEED_OF_LIGHT

        def rays(name: str, scale: float = 1.0) -> list[np.ndarray]:
            values = np.asarray(var[name][:], dtype=float)
            return [values[i, : counts[i]] * scale for i in range(counts.size)]

        r, z, phi = rays("wr", _CM), rays("wz", _CM), rays("wphi")
        power = rays("delpwr", _ERG_PER_S)
        n_r, n_z, n_phi = rays("wn_r"), rays("wn_z"), rays("wn_phi")
        parsed = {
            "version": _text(var["version"]) if "version" in var else "",
            "frequency_hz": frequency,
            "ioxm": int(var["ioxm"][:]) if "ioxm" in var else None,
            "n_rays": int(counts.size),
            "rays": [],
            "power_injected_w": float(var["power_inj_total"][:]) * _ERG_PER_S if "power_inj_total" in var else None,
            "power_absorbed_w": float(var["power_total"][:]) * _ERG_PER_S if "power_total" in var else None,
            "power_absorbed_electrons_w": float(var["powtot_e"][:]) * _ERG_PER_S if "powtot_e" in var else None,
            "power_absorbed_at_reflections_w": (
                float(var["w_tot_pow_absorb_at_refl_nc"][:]) * _ERG_PER_S if "w_tot_pow_absorb_at_refl_nc" in var else None
            ),
            "refl_loss": float(var["refl_loss"][:]) if "refl_loss" in var else None,
            "stop_status": (
                np.asarray(var["iray_status_nc"][:], dtype=int).reshape(-1).tolist() if "iray_status_nc" in var else None
            ),
            # GENRAY carries the ray through vacuum to the plasma boundary and
            # integrates from there: the first traced point is here, not at
            # the launcher.
            "start_r_m": np.asarray(var["r_starting"][:], dtype=float).reshape(-1).tolist() if "r_starting" in var else None,
            "start_z_m": np.asarray(var["z_starting"][:], dtype=float).reshape(-1).tolist() if "z_starting" in var else None,
        }
        if parsed["stop_status"] is not None:
            parsed["stop_reasons"] = [RAY_STOP_REASONS.get(code, f"code {code}") for code in parsed["stop_status"]]
        status = parsed["stop_status"] or [None] * counts.size
        traced = [bool(counts[i] > 1 and (status[i] if i < len(status) else None) != -1) for i in range(counts.size)]
        parsed["traced"] = traced
        missing = [
            name for name, value in (
                ("power_inj_total", parsed["power_injected_w"]),
                ("power_total", parsed["power_absorbed_w"]),
                ("powtot_e", parsed["power_absorbed_electrons_w"]),
                ("iray_status_nc", parsed["stop_status"]),
            ) if value is None
        ]
        parsed["missing_totals"] = missing
        parsed["complete"] = bool(any(traced) and not missing)
        for i in range(counts.size):
            x = r[i] * np.cos(phi[i])
            y = r[i] * np.sin(phi[i])
            step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z[i]) ** 2)
            parsed["rays"].append({
                "traced": traced[i],
                "length": np.concatenate([[0.0], np.cumsum(step)]),
                "r": r[i], "z": z[i], "phi": phi[i],
                "power": power[i],
                "n_r": n_r[i], "n_z": n_z[i], "n_phi": n_phi[i],
                "n_parallel": rays("wnpar")[i], "n_perpendicular": rays("wnper")[i],
                "k_scale": omega_over_c,
            })
    return parsed


def genray_to_waves(
    parsed: Mapping[str, Any],
    ods: Any,
    *,
    time: float,
    beam_name: str = "",
    provenance: Mapping[str, Any] | None = None,
    coherent_wave_index: int = 0,
) -> Any:
    """Write parsed GENRAY rays into ``waves.coherent_wave[index].beam_tracing[0]``.

    ``time`` is the physical time the inputs were taken at [s].
    """
    if not parsed.get("complete"):
        raise ValueError(
            "GENRAY output is incomplete: "
            + (f"missing {', '.join(parsed.get('missing_totals') or [])}; " if parsed.get("missing_totals") else "")
            + f"traced rays {sum(parsed.get('traced') or [])}/{parsed.get('n_rays')}"
        )
    existing = ods["waves.time"] if "waves" in ods and "time" in ods["waves"] else None
    if existing is not None and not np.allclose(np.asarray(existing, dtype=float), [float(time)]):
        raise ValueError(
            f"waves already holds t = {np.asarray(existing).tolist()} s; write a run at t = {time:g} s "
            "into a separate ODS"
        )
    base = f"waves.coherent_wave.{int(coherent_wave_index)}"
    if "waves" in ods and "coherent_wave" in ods["waves"] and int(coherent_wave_index) in ods["waves.coherent_wave"]:
        # Replace, never merge: a previous run's extra rays must not survive.
        del ods["waves.coherent_wave"][int(coherent_wave_index)]
    ods["waves.ids_properties.homogeneous_time"] = 1
    ods["waves.time"] = np.array([float(time)])
    ods[f"{base}.identifier.type.index"] = EC_WAVE_TYPE["index"]
    ods[f"{base}.identifier.type.name"] = EC_WAVE_TYPE["name"]
    ods[f"{base}.identifier.type.description"] = EC_WAVE_TYPE["description"]
    if beam_name:
        ods[f"{base}.identifier.antenna_name"] = beam_name
    ods[f"{base}.wave_solver_type.index"] = 1
    ods[f"{base}.wave_solver_type.name"] = "Beam/ray tracing"

    tracing = f"{base}.beam_tracing.0"
    ods[f"{tracing}.time"] = float(time)
    refl_loss = parsed.get("refl_loss") or 0.0
    rays = [ray for ray in parsed["rays"] if ray.get("traced", True)]
    for index, ray in enumerate(rays):
        beam = f"{tracing}.beam.{index}"
        power = np.asarray(ray["power"], dtype=float)
        ods[f"{beam}.power_initial"] = float(power[0])
        ods[f"{beam}.length"] = np.asarray(ray["length"], dtype=float)
        ods[f"{beam}.position.r"] = np.asarray(ray["r"], dtype=float)
        ods[f"{beam}.position.z"] = np.asarray(ray["z"], dtype=float)
        ods[f"{beam}.position.phi"] = np.asarray(ray["phi"], dtype=float)
        scale = float(ray["k_scale"])
        ods[f"{beam}.wave_vector.k_r"] = np.asarray(ray["n_r"], dtype=float) * scale
        ods[f"{beam}.wave_vector.k_z"] = np.asarray(ray["n_z"], dtype=float) * scale
        ods[f"{beam}.wave_vector.k_tor"] = np.asarray(ray["n_phi"], dtype=float) * scale
        ods[f"{beam}.wave_vector.n_parallel"] = np.asarray(ray["n_parallel"], dtype=float)
        ods[f"{beam}.wave_vector.n_perpendicular"] = np.asarray(ray["n_perpendicular"], dtype=float)
        if refl_loss == 0.0:
            ods[f"{beam}.electrons.power"] = power[0] - power

    global_ = f"{base}.global_quantities.0"
    ods[f"{global_}.time"] = float(time)
    ods[f"{global_}.frequency"] = float(parsed["frequency_hz"])
    if parsed.get("power_absorbed_w") is not None:
        ods[f"{global_}.power"] = float(parsed["power_absorbed_w"])
    if parsed.get("power_absorbed_electrons_w") is not None:
        ods[f"{global_}.electrons.power_thermal"] = float(parsed["power_absorbed_electrons_w"])

    record = {
        "genray_version": parsed.get("version", ""),
        "ioxm": parsed.get("ioxm"),
        "power_injected_w": parsed.get("power_injected_w"),
        "power_absorbed_at_reflections_w": parsed.get("power_absorbed_at_reflections_w"),
        "refl_loss": parsed.get("refl_loss"),
        "ray_stop_status": parsed.get("stop_status"),
        "ray_stop_reasons": parsed.get("stop_reasons"),
        "rays_traced": parsed.get("traced"),
        "rays_written": "only rays that entered the plasma, in GENRAY order",
        "first_traced_point": (
            "the plasma boundary: GENRAY does not trace the vacuum segment from the launcher, so "
            "position/length start where the ray enters the plasma"
        ),
        "length": "arc length integrated from ray positions (GENRAY ws is poloidal distance)",
        "electrons.power": (
            "delpwr[0] - delpwr, cumulative, electrons-only run"
            if refl_loss == 0.0 else "not written: refl_loss > 0 lowers delpwr without absorption"
        ),
        "inputs": dict(provenance or {}),
    }
    ods["waves.code.name"] = "GENRAY"
    if parsed.get("version"):
        ods["waves.code.version"] = str(parsed["version"])
    ods["waves.code.parameters"] = json.dumps(record, default=float, sort_keys=True)
    return ods


def collect_genray_outputs(workdir: str | Path) -> Path | None:
    """The ``genray.nc`` GENRAY wrote into ``workdir``, if any."""
    candidate = Path(workdir) / "genray.nc"
    return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None


__all__ = ["EC_WAVE_TYPE", "RAY_STOP_REASONS", "collect_genray_outputs", "genray_to_waves", "read_genray_netcdf"]

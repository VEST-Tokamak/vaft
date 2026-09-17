"""One discharge's startup, as a handful of numbers (issue #888).

:func:`startup_summary` gathers what the startup tutorial reads off a shot --
when the plasma formed and on what evidence, how large the current got, the
fill it broke down in, and the vacuum field at that instant -- into one flat
``dict``. Every entry is a reading of an existing evaluator rather than a new
definition:

* timing: :func:`vaft.omas.plasma_timing.plasma_timing` and
  :func:`vaft.omas.plasma_features.ip_peak`;
* prefill: :func:`vaft.omas.process_wrapper.compute_prefill_pressure_ods`
  (the median of the gauge before the onset, as the ``lloyd_margin`` map);
* point proxies: :func:`vaft.omas.process_wrapper.compute_startup_proxies_ods`;
* null area and Lloyd margin: :func:`compute_vacuum_field_map` and
  :func:`compute_connection_length_map_ods` on the tutorial's 33x33 grid, with
  the definitions tutorial session 02 uses in its cells;
* ECR radius: :func:`vaft.formula.startup.electron_cyclotron_resonance_radius`.

An input a product does not carry makes its entries ``None`` -- never an
exception -- and ``unavailable`` names the entry and the reason. The caller's
ODS is not modified: the vacuum evaluators solve and store vessel currents, so
they run on a private copy of the IDS they read.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np

__all__ = ["NULL_FIELD_THRESHOLD_T", "startup_summary"]

#: A grid point is part of the field null when ``|B_p|`` is below this [T]
#: (5 G), the threshold tutorial session 02 plots the null area against.
NULL_FIELD_THRESHOLD_T = 5e-4

#: IDS the vacuum evaluators read; copied so their writes stay private.
_VACUUM_ROOTS = ("pf_active", "pf_passive", "tf", "wall", "barometry", "dataset_description")

#: Half-width of the window the launched EC power is read over [s].
_EC_POWER_HALF_WINDOW_S = 1e-3


def _private_copy(ods: Any) -> Any:
    from omas import ODS

    from vaft.ods_access import path_exists

    private = ODS(consistency_check=False)
    for root in _VACUUM_ROOTS:
        if path_exists(ods, root):
            private[root] = copy.deepcopy(ods[root])
    return private


def _float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _ec_power(ods: Any, instant: float) -> float | None:
    """Median launched EC power over ``instant +- 1 ms`` [W], ``None`` if absent."""
    from vaft.ods_access import path_value

    power = path_value(ods, "ec_launchers.beam.0.power_launched.data")
    if power is None:
        return None
    stamps = path_value(ods, "ec_launchers.beam.0.power_launched.time")
    if stamps is None:
        stamps = path_value(ods, "ec_launchers.time")
    power = np.asarray(power, dtype=float).ravel()
    if stamps is None or np.asarray(stamps).size != power.size:
        raise ValueError("ec_launchers.beam.0.power_launched has no matching time base")
    stamps = np.asarray(stamps, dtype=float).ravel()
    window = power[np.abs(stamps - instant) <= _EC_POWER_HALF_WINDOW_S]
    if window.size == 0 or not np.isfinite(window).any():
        raise ValueError(
            "ec_launchers.beam.0.power_launched has no valid sample within "
            f"{_EC_POWER_HALF_WINDOW_S * 1e3:g} ms of breakdown"
        )
    return float(np.nanmedian(window))


def startup_summary(
    ods: Any,
    *,
    reference_rz: tuple[float, float] = (0.4, 0.0),
    ec_frequency_Hz: float = 2.45e9,
    resolution: int = 33,
) -> dict[str, Any]:
    """The startup of one discharge, summarised at its breakdown onset.

    Args:
        ods: any ODS (``vaft.database.load`` output, the packaged sample).
        reference_rz: point the loop voltage, vertical field and decay index
            are read at [m].
        ec_frequency_Hz: ECRH frequency the resonance radius is computed for [Hz].
        resolution: points per axis of the vacuum map and connection-length
            trace the null area and the Lloyd margin are counted on [-]; 33 is
            what tutorial session 02 uses.

    Returns:
        A flat ``dict``:

        * ``shot``: ``dataset_description.data_entry.pulse``.
        * ``t_breakdown`` [s], ``onset_source``: the plasma-timing onset and the
          source that decided it (``h_alpha_*`` or ``ip_principal``).
        * ``ip_peak_A`` [A], ``t_ip_peak`` [s]: the representative current peak.
        * ``pressure_Pa`` [Pa]: prefill, median of the gauge before the onset.
        * ``v_loop_V`` [V], ``b_z_T`` [T], ``decay_index`` [-]: vacuum proxies at
          ``reference_rz`` at the PF sample nearest the onset; ``reference_rz``.
        * ``null_area_fraction`` [-]: share of the grid points inside limiter
          unit 0 with ``|B_p| < 5 G``.
        * ``lloyd_threshold_fraction`` [-]: share of the traced interior points
          where a Lloyd threshold exists (``A p L > 1``);
          ``lloyd_margin_fraction`` [-]: share of those where ``|E_phi|`` clears
          it (margin >= 1) -- the two numbers session 02 prints.
        * ``r_ecr_m`` [m]: fundamental ECR radius at the onset;
          ``ec_frequency_Hz``.
        * ``ec_power_W`` [W]: NaN-aware median of
          ``ec_launchers.beam.0.power_launched`` over onset +- 1 ms; ``None``
          when the product has no EC launcher.
        * ``unavailable``: ``{entry: reason}`` for every entry left ``None``
          because an input was missing or an evaluator refused.
    """
    from vaft.ods_access import path_value

    from .plasma_features import ip_peak
    from .plasma_timing import plasma_timing
    from .process_wrapper import (
        compute_connection_length_map_ods,
        compute_prefill_pressure_ods,
        compute_startup_proxies_ods,
        compute_vacuum_field_map,
        _limiter_outline,
        _inside_limiter,
        _vacuum_toroidal_product,
    )

    out: dict[str, Any] = {
        "shot": None,
        "t_breakdown": None,
        "onset_source": None,
        "ip_peak_A": None,
        "t_ip_peak": None,
        "pressure_Pa": None,
        "reference_rz": (float(reference_rz[0]), float(reference_rz[1])),
        "v_loop_V": None,
        "b_z_T": None,
        "decay_index": None,
        "null_area_fraction": None,
        "lloyd_threshold_fraction": None,
        "lloyd_margin_fraction": None,
        "r_ecr_m": None,
        "ec_frequency_Hz": float(ec_frequency_Hz),
        "ec_power_W": None,
        "unavailable": {},
    }
    missing = out["unavailable"]

    def _skip(keys, error):
        for key in keys:
            missing[key] = f"{type(error).__name__}: {error}"

    shot = path_value(ods, "dataset_description.data_entry.pulse")
    if shot is not None:
        out["shot"] = int(shot)
    else:
        missing["shot"] = "dataset_description.data_entry.pulse is absent"

    timing = None
    try:
        timing = plasma_timing(ods)
    except Exception as error:  # noqa: BLE001 - any product, degrade not raise
        _skip(("t_breakdown", "onset_source", "ip_peak_A", "t_ip_peak"), error)
    if timing is not None:
        if timing.found:
            out["t_breakdown"] = float(timing.onset)
            out["onset_source"] = timing.source
        else:
            reason = timing.fallback_reason or "no plasma timing"
            for key in ("t_breakdown", "onset_source"):
                missing[key] = reason
        try:
            feature = ip_peak(ods, timing=timing)
            if feature.found:
                out["ip_peak_A"] = _float(feature.value)
                out["t_ip_peak"] = _float(feature.time)
            else:
                reason = feature.reason or ", ".join(feature.flags) or "no peak"
                missing["ip_peak_A"] = missing["t_ip_peak"] = reason
        except Exception as error:  # noqa: BLE001
            _skip(("ip_peak_A", "t_ip_peak"), error)

    t_breakdown = out["t_breakdown"]
    instant_keys = (
        "pressure_Pa", "v_loop_V", "b_z_T", "decay_index", "null_area_fraction",
        "lloyd_threshold_fraction", "lloyd_margin_fraction", "r_ecr_m", "ec_power_W",
    )
    if t_breakdown is None:
        for key in instant_keys:
            missing.setdefault(key, "no breakdown onset to evaluate at")
        return out

    try:
        out["ec_power_W"] = _ec_power(ods, t_breakdown)
        if out["ec_power_W"] is None:
            missing["ec_power_W"] = "ec_launchers.beam.0.power_launched is absent"
    except Exception as error:  # noqa: BLE001
        _skip(("ec_power_W",), error)

    private = _private_copy(ods)

    try:
        out["pressure_Pa"] = compute_prefill_pressure_ods(private, before=t_breakdown)
    except Exception as error:  # noqa: BLE001
        _skip(("pressure_Pa",), error)

    try:
        product = _vacuum_toroidal_product(private, t_breakdown)
        from vaft.formula.startup import electron_cyclotron_resonance_radius

        out["r_ecr_m"] = _float(electron_cyclotron_resonance_radius(product, float(ec_frequency_Hz)))
    except Exception as error:  # noqa: BLE001
        _skip(("r_ecr_m",), error)

    try:
        proxies = compute_startup_proxies_ods(private, rz=out["reference_rz"])
        index = int(np.argmin(np.abs(proxies["time"] - t_breakdown)))
        out["v_loop_V"] = _float(proxies["v_loop"][index])
        out["b_z_T"] = _float(proxies["b_z"][index])
        out["decay_index"] = _float(proxies["decay_index"][index])
        for key in ("v_loop_V", "b_z_T", "decay_index"):
            if out[key] is None:
                missing[key] = "non-finite at the onset sample"
    except Exception as error:  # noqa: BLE001
        _skip(("v_loop_V", "b_z_T", "decay_index"), error)
        # Without the vacuum field nothing below can be computed.
        _skip(("null_area_fraction", "lloyd_threshold_fraction", "lloyd_margin_fraction"), error)
        return out

    field = None
    try:
        from vaft.formula.equilibrium import poloidal_field_magnitude

        field = compute_vacuum_field_map(private, time=t_breakdown, resolution=int(resolution))
        wall_r, wall_z = _limiter_outline(private)
        if wall_r is None:
            raise ValueError("wall.description_2d.0.limiter.unit.0.outline is absent")
        mesh_r, mesh_z = np.meshgrid(field["r"], field["z"], indexing="ij")
        inside = _inside_limiter(wall_r, wall_z, mesh_r, mesh_z)
        b_p = poloidal_field_magnitude(field["b_r"], field["b_z"])
        out["null_area_fraction"] = float(np.mean(b_p[inside] < NULL_FIELD_THRESHOLD_T))
    except Exception as error:  # noqa: BLE001
        _skip(("null_area_fraction",), error)

    try:
        if field is None:
            raise ValueError("the vacuum field map is unavailable")
        if out["pressure_Pa"] is None:
            raise ValueError("no prefill pressure (barometry absent)")
        from vaft.formula.equilibrium import toroidal_electric_field
        from vaft.formula.startup import breakdown_margin, lloyd_breakdown_field

        traced = compute_connection_length_map_ods(
            private, time=t_breakdown, resolution=int(resolution)
        )
        interior = ~np.asarray(traced["outside"], dtype=bool)
        e_local = np.abs(toroidal_electric_field(field["r"][:, None], field["dpsi_dt"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            threshold = lloyd_breakdown_field(out["pressure_Pa"], traced["length_m"])
        margin = breakdown_margin(e_local, threshold)
        has_threshold = interior & np.isfinite(margin)
        out["lloyd_threshold_fraction"] = (
            float(has_threshold.sum() / interior.sum()) if interior.any() else None
        )
        out["lloyd_margin_fraction"] = (
            float(np.mean(margin[has_threshold] >= 1.0)) if has_threshold.any() else 0.0
        )
    except Exception as error:  # noqa: BLE001
        _skip(("lloyd_threshold_fraction", "lloyd_margin_fraction"), error)

    return out

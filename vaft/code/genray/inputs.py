"""Materialise a GENRAY EC case from IMAS ``ec_launchers``, ``equilibrium`` and ``core_profiles``.

GENRAY is driven through ``genray.dat``, whose units are mixed: frequency
[GHz], density [1e19 m^-3], temperature [keV], power [MW], lengths [m] and
angles [degree]. Every conversion happens here, once, next to the name it
feeds.

Radial coordinate: the profiles are tabulated on sqrt(psi_N), GENRAY's
``indexrho = 4``, which GENRAY evaluates as ``sqrt((psi - psimag) / (psilim -
psimag))`` from the same g-file. psi_N is built from ``core_profiles`` grid
psi against the equilibrium's own axis and boundary flux, so the ODS psi
convention cancels. ``rho_tor_norm`` is deliberately not used: packaged VAFT
equilibria store a sqrt(psi_N) proxy under that name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import GENRAYConfig, GENRAYInputs

#: File names inside the case directory. GENRAY opens ``genray.dat`` first.
GENRAY_INPUT_NAME = "genray.dat"
EQDSK_NAME = "equilib.dat"
PROVENANCE_NAME = "vaft_genray_inputs.json"

#: Largest gap between the profile data and the axis / boundary tolerated
#: before the adapter refuses rather than extrapolates, in sqrt(psi_N).
PROFILE_COVERAGE_TOLERANCE = 0.02


# --------------------------------------------------------------------------- #
# launch geometry
# --------------------------------------------------------------------------- #


def eccone_angles(steering_angle_pol: float, steering_angle_tor: float) -> tuple[float, float]:
    """GENRAY ``(alfast, betast)`` [degree] from the IMAS steering angles [rad].

    IMAS DD 3.41: ``angle_pol = atan2(-k_Z, -k_R)``, ``angle_tor = arcsin(k_phi/k)``,
    so the unit wave vector is ``k_R = -cos(pol) cos(tor)``, ``k_phi = sin(tor)``,
    ``k_Z = -sin(pol) cos(tor)``.

    GENRAY ``raypatt='genray'`` (``cone_ec.f``) launches along
    ``(cos(betast) cos(alfast + phist), cos(betast) sin(alfast + phist), sin(betast))``
    in Cartesian coordinates with ``phist`` the launcher's toroidal angle, so
    ``alfast`` is measured from ``+e_R`` toward ``+e_phi`` and ``betast`` is the
    elevation above the ``Z = const`` plane. Both frames are right-handed with
    ``Z`` up, the same sense as IMAS phi. A beam aimed at the axis is
    ``(180, 0)``.
    """
    pol = float(steering_angle_pol)
    tor = float(steering_angle_tor)
    k_r = -math.cos(pol) * math.cos(tor)
    k_phi = math.sin(tor)
    k_z = -math.sin(pol) * math.cos(tor)
    alfast = math.degrees(math.atan2(k_phi, k_r)) % 360.0
    betast = math.degrees(math.asin(max(-1.0, min(1.0, k_z))))
    return alfast, betast + 0.0


# --------------------------------------------------------------------------- #
# time selection
# --------------------------------------------------------------------------- #


def _nearest_index(times: Any, time: float, tolerance: float, what: str) -> int:
    """Index of the sample nearest ``time``; refuses one farther than ``tolerance``."""
    stamps = np.asarray(times, dtype=float).reshape(-1)
    finite = np.isfinite(stamps)
    if not finite.any():
        raise ValueError(f"{what} has no finite time base")
    distance = np.where(finite, np.abs(stamps - float(time)), np.inf)
    index = int(np.argmin(distance))
    if distance[index] > tolerance:
        raise ValueError(
            f"{what} has no sample within {tolerance:g} s of t = {time:g} s "
            f"(nearest is {stamps[index]:g} s); pass a matching time or widen time_tolerance"
        )
    return index


def _get(ods: Any, path: str) -> Any:
    """``ods[path]`` or ``None``, without materialising a missing path."""
    node = ods
    for part in path.split("."):
        try:
            key = int(part) if part.isdigit() else part
            if key not in node:
                return None
            node = node[key]
        except (TypeError, KeyError, IndexError):
            return None
    return node


def _require(ods: Any, path: str, what: str) -> Any:
    value = _get(ods, path)
    if value is None:
        raise ValueError(f"{what}: {path} is missing")
    return value


def _time_base(ods: Any, path: str, what: str) -> Any:
    """``path`` or, in a homogeneous-time ``ec_launchers``, the IDS-level ``ec_launchers.time``."""
    stamps = _get(ods, path)
    if stamps is not None:
        return stamps
    if _get(ods, "ec_launchers.ids_properties.homogeneous_time") == 1:
        return _require(ods, "ec_launchers.time", what)
    raise ValueError(f"{what}: {path} is missing")


def _signal_at(ods: Any, base: str, time: float, tolerance: float, what: str) -> tuple[float, float]:
    """``(value, sample_time)`` of a DD signal ``base.data`` on its time base."""
    data = np.asarray(_require(ods, f"{base}.data", what), dtype=float).reshape(-1)
    stamps = _time_base(ods, f"{base}.time", what)
    index = _nearest_index(stamps, time, tolerance, what)
    return float(data[index]), float(np.asarray(stamps, dtype=float).reshape(-1)[index])


# --------------------------------------------------------------------------- #
# the three IDS
# --------------------------------------------------------------------------- #


def _launcher(ods: Any, config: GENRAYConfig) -> dict[str, Any]:
    beam = f"ec_launchers.beam.{int(config.beam_index)}"
    what = f"ec_launchers beam {config.beam_index}"
    if _get(ods, beam) is None:
        raise ValueError(f"{what} is missing")
    tol = float(config.time_tolerance)
    geometry_index = _nearest_index(_time_base(ods, f"{beam}.time", what), config.time, tol, f"{what} geometry")

    def at(path: str) -> float:
        value = float(np.asarray(_require(ods, f"{beam}.{path}", what), dtype=float).reshape(-1)[geometry_index])
        if not math.isfinite(value):
            raise ValueError(f"{what}: {path} is not finite at the requested time")
        return value

    r, z, phi = at("launching_position.r"), at("launching_position.z"), at("launching_position.phi")
    pol, tor = at("steering_angle_pol"), at("steering_angle_tor")
    frequency, frequency_time = _signal_at(ods, f"{beam}.frequency", config.time, tol, f"{what} frequency")
    if not frequency > 0.0:
        raise ValueError(f"{what}: frequency must be positive")

    if config.power_w is not None:
        power, power_source = float(config.power_w), "config.power_w"
    else:
        power, power_time = _signal_at(ods, f"{beam}.power_launched", config.time, tol, f"{what} power_launched")
        if not (math.isfinite(power) and power > 0.0):
            raise ValueError(
                f"{what}: power_launched at t = {power_time:g} s is {power!r}; pass power_w explicitly"
            )
        power_source = f"{beam}.power_launched at t = {power_time:g} s"

    alfast, betast = eccone_angles(pol, tor)
    return {
        "name": str(_get(ods, f"{beam}.name") or ""),
        "identifier": str(_get(ods, f"{beam}.identifier") or ""),
        "r": r,
        "z": z,
        "phi": phi,
        "steering_angle_pol": pol,
        "steering_angle_tor": tor,
        "alfast_deg": alfast,
        "betast_deg": betast,
        "frequency_hz": frequency,
        "frequency_time": frequency_time,
        "power_w": power,
        "power_source": power_source,
    }


def _equilibrium_index(ods: Any, config: GENRAYConfig) -> int:
    times = _get(ods, "equilibrium.time")
    if times is None:
        count = len(_require(ods, "equilibrium.time_slice", "equilibrium"))
        times = [float(_require(ods, f"equilibrium.time_slice.{i}.time", "equilibrium")) for i in range(count)]
    return _nearest_index(times, config.time, float(config.time_tolerance), "equilibrium")


def _profiles(ods: Any, config: GENRAYConfig, equilibrium_index: int) -> dict[str, Any]:
    index = _nearest_index(
        _require(ods, "core_profiles.time", "core_profiles"), config.time, float(config.time_tolerance), "core_profiles"
    )
    base = f"core_profiles.profiles_1d.{index}"
    what = "core_profiles"
    psi = np.asarray(_require(ods, f"{base}.grid.psi", f"{what} (grid.psi is required; rho_tor_norm is not trusted)"), dtype=float)
    slice_ = f"equilibrium.time_slice.{equilibrium_index}.global_quantities"
    psi_axis = float(_require(ods, f"{slice_}.psi_axis", "equilibrium"))
    psi_boundary = float(_require(ods, f"{slice_}.psi_boundary", "equilibrium"))
    if psi_boundary == psi_axis:
        raise ValueError("equilibrium psi_boundary equals psi_axis")
    # psi_N cancels the ODS psi convention only if both IDS share it. Where
    # core_profiles states its own axis/boundary flux, they must agree.
    for name, reference in (("psi_magnetic_axis", psi_axis), ("psi_boundary", psi_boundary)):
        own = _get(ods, f"{base}.grid.{name}")
        if own is not None and abs(float(own) - reference) > 0.01 * abs(psi_boundary - psi_axis):
            raise ValueError(
                f"{what}: grid.{name} = {float(own):g} disagrees with equilibrium ({reference:g}); "
                "the two IDS do not share a psi convention"
            )
    psi_n = (psi - psi_axis) / (psi_boundary - psi_axis)

    # The cold dispersion sees every electron: the DD's total `density` first.
    density = _get(ods, f"{base}.electrons.density")
    density_source = f"{base}.electrons.density"
    if density is None:
        density = _require(ods, f"{base}.electrons.density_thermal", what)
        density_source = f"{base}.electrons.density_thermal"
    density = np.asarray(density, dtype=float)
    temperature = np.asarray(_require(ods, f"{base}.electrons.temperature", what), dtype=float)

    usable = np.isfinite(psi_n) & np.isfinite(density) & np.isfinite(temperature) & (psi_n >= -1e-9)
    if usable.sum() < 3:
        raise ValueError(f"{what}: fewer than three finite electron samples")
    rho = np.sqrt(np.clip(psi_n[usable], 0.0, None))
    order = np.argsort(rho)
    rho, density, temperature = rho[order], density[usable][order], temperature[usable][order]
    if rho[0] > PROFILE_COVERAGE_TOLERANCE or rho[-1] < 1.0 - PROFILE_COVERAGE_TOLERANCE:
        raise ValueError(
            f"{what}: electron profiles span sqrt(psi_N) = [{rho[0]:.3f}, {rho[-1]:.3f}]; "
            f"GENRAY needs [0, 1] within {PROFILE_COVERAGE_TOLERANCE}, and VAFT does not extrapolate"
        )
    if np.any(density < 0.0):
        raise ValueError(f"{what}: negative electron density")

    grid = np.linspace(0.0, 1.0, int(config.n_rho))
    ne = np.interp(grid, rho, density)
    te_ev = np.interp(grid, rho, temperature)
    floored = 0
    if np.any(te_ev <= 0.0):
        if config.minimum_temperature_ev is None:
            raise ValueError(
                f"{what}: electron temperature reaches {te_ev.min():g} eV (at sqrt(psi_N) = "
                f"{grid[np.argmin(te_ev)]:.3f}); GENRAY needs Te > 0. Pass minimum_temperature_ev "
                "to floor it explicitly"
            )
    if config.minimum_temperature_ev is not None:
        floor = float(config.minimum_temperature_ev)
        floored = int(np.count_nonzero(te_ev < floor))
        te_ev = np.maximum(te_ev, floor)
    te_kev = te_ev / 1000.0

    zeff_data = _get(ods, f"{base}.zeff")
    zeff_problem = None
    if zeff_data is not None:
        zeff_all = np.asarray(zeff_data, dtype=float).reshape(-1)
        if zeff_all.shape != psi.shape:
            zeff_problem = f"zeff has {zeff_all.size} points, grid.psi has {psi.size}"
        elif not np.all(np.isfinite(zeff_all[usable])):
            zeff_problem = "zeff is not finite everywhere"
        elif np.any(zeff_all[usable] < 1.0):
            zeff_problem = f"zeff falls to {zeff_all[usable].min():g} (< 1)"
    if zeff_data is not None and zeff_problem is None:
        zeff = np.interp(grid, rho, np.asarray(zeff_data, dtype=float).reshape(-1)[usable][order])
        zeff_source = f"{base}.zeff"
    elif config.zeff is not None:
        zeff = np.full(grid.shape, float(config.zeff))
        zeff_source = "config.zeff (uniform)" + (f"; core_profiles zeff rejected: {zeff_problem}" if zeff_problem else "")
    else:
        raise ValueError(f"{what}: " + (zeff_problem or "no zeff profile") + "; pass zeff explicitly")

    return {
        "time": float(np.asarray(_require(ods, "core_profiles.time", what), dtype=float).reshape(-1)[index]),
        "rho_sqrt_psi_n": grid,
        "ne_m3": ne,
        "te_kev": te_kev,
        "zeff": zeff,
        "zeff_source": zeff_source,
        "density_source": density_source,
        "data_span_sqrt_psi_n": [float(rho[0]), float(rho[-1])],
        "temperature_floor_ev": config.minimum_temperature_ev,
        "temperature_points_floored": floored,
    }


# --------------------------------------------------------------------------- #
# the namelist file
# --------------------------------------------------------------------------- #


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10e}".replace("e", "d")
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(_format(item) for item in np.asarray(value).tolist())
    raise TypeError(f"cannot write {type(value).__name__} into a namelist")


def render_namelists(groups: Mapping[str, Mapping[str, Any]]) -> str:
    """Deterministic ``&group ... &end`` text, in the order given."""
    lines = ["! Written by vaft.code.genray; units follow genray.dat (GHz, 1e19 m^-3, keV, MW, m, degree)."]
    for group, entries in groups.items():
        lines.append(f" &{group}")
        for name, value in entries.items():
            lines.append(f" {name}={_format(value)}")
        lines.append(" &end")
    return "\n".join(lines) + "\n"


def genray_namelists(launcher: Mapping[str, Any], profiles: Mapping[str, Any], config: GENRAYConfig) -> dict[str, dict[str, Any]]:
    """The namelist groups VAFT sets; everything else keeps GENRAY's ``default_in``."""
    n = int(config.n_rho)
    groups: dict[str, dict[str, Any]] = {
        "genr": {"mnemonic": "genray", "rayop": "netcdf", "dielectric_op": "disabled",
                 "r0x": 1.0, "b0": 1.0, "outdat": "zrn.dat", "stat": "new"},
        "tokamak": {"eqdskin": EQDSK_NAME, "indexrho": 4, "ipsi": 1, "ionetwo": 1,
                    "ieffic": 3, "psifactr": 0.999, "deltripl": 0.0, "nloop": 24, "i_ripple": 1},
        "wave": {"frqncy": launcher["frequency_hz"] / 1.0e9, "ioxm": config.ioxm,
                 "ireflm": int(config.max_reflections), "jwave": int(config.harmonic),
                 "istart": 1, "delpwrmn": 1.0e-4, "ibw": 0, "i_vgr_ini": 1, "poldist_mx": 1.0e5},
        "scatnper": {"iscat": 0},
        "dispers": {"ib": 1, "id": 2, "iherm": 1, "iabsorp": 1, "iswitch": 0, "del_y": 1.0e-3,
                    "jy_d": 1, "idswitch": 2, "iabswitch": 1, "n_relt_harm": 5, "n_relt_intgr": 50,
                    "iflux": 2, "i_im_nperp": 1, "i_geom_optic": 1, "ray_direction": 1.0},
        "numercl": {"irkmeth": 2, "ndim1": 6, "isolv": 1, "idif": 1, "nrelt": 1000,
                    "prmt1": 0.0, "prmt2": 9.999e5, "prmt3": 1.0e-4, "prmt4": 1.0e-5, "prmt6": 1.0e-3,
                    "icorrect": 1, "iout3d": "enable", "maxsteps_rk": int(config.max_steps)},
        "output": {"iwcntr": 0, "iwopen": 1, "iwj": 1, "itools": 0, "i_plot_b": 0, "i_plot_d": 0},
        "plasma": {"ndens": n, "nbulk": 1, "izeff": 2, "idens": 1, "temp_scale(1)": 1.0, "den_scale(1)": 1.0},
        "species": {"charge(1)": 1.0, "dmas(1)": 1.0},
        # Central ray only (na1 = 0): alpha1/alpha2 describe a cone GENRAY does
        # not launch, and no beam width is claimed.
        "eccone": {"powtot": launcher["power_w"] / 1.0e6, "raypatt": "genray", "ncone": 1,
                   "zst": launcher["z"], "rst": launcher["r"], "phist": math.degrees(launcher["phi"]),
                   "alfast": launcher["alfast_deg"], "betast": launcher["betast_deg"],
                   "alpha1": 1.0, "alpha2": 0.0, "na1": 0, "na2": 1},
        "dentab": {"prof": np.asarray(profiles["ne_m3"]) / 1.0e19},
        "temtab": {"prof": np.asarray(profiles["te_kev"])},
        "tpoptab": {"prof": np.ones(n)},
        "vflowtab": {"prof": np.zeros(n)},
        "zeftab": {"zeff1": np.asarray(profiles["zeff"])},
    }
    for group, entries in config.namelist_overrides.items():
        groups.setdefault(str(group), {}).update(dict(entries))
    return groups


def prepare_genray_inputs(ods: Any, config: GENRAYConfig, workdir: str | Path | None = None) -> GENRAYInputs:
    """Write ``genray.dat``, ``equilib.dat`` and the provenance record for one case.

    Raises ``ValueError`` for anything the three IDS cannot establish at
    ``config.time`` -- a missing launcher field, a time mismatch beyond
    ``config.time_tolerance``, profiles that do not span the plasma, a
    non-positive launched power -- rather than filling it in.
    """
    from vaft.data.eqdsk import from_omas, write_geqdsk

    directory = Path(config.workdir if workdir is None else workdir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    launcher = _launcher(ods, config)
    equilibrium_index = _equilibrium_index(ods, config)
    profiles = _profiles(ods, config, equilibrium_index)

    eqdsk = write_geqdsk(from_omas(ods, equilibrium_index), directory / EQDSK_NAME)
    groups = genray_namelists(launcher, profiles, config)
    genray_in = directory / GENRAY_INPUT_NAME
    genray_in.write_text(render_namelists(groups), encoding="utf-8")

    equilibrium_time = _get(ods, "equilibrium.time")
    provenance = {
        "adapter": "vaft.code.genray",
        "requested_time_s": float(config.time),
        "time_tolerance_s": float(config.time_tolerance),
        "equilibrium_index": equilibrium_index,
        "equilibrium_time_s": (
            float(np.asarray(equilibrium_time, dtype=float).reshape(-1)[equilibrium_index])
            if equilibrium_time is not None else None
        ),
        "core_profiles_time_s": profiles["time"],
        "profile_coordinate": "sqrt(psi_N) (GENRAY indexrho=4)",
        "profile_data_span": profiles["data_span_sqrt_psi_n"],
        "zeff_source": profiles["zeff_source"],
        "density_source": profiles["density_source"],
        "temperature_floor_ev": profiles["temperature_floor_ev"],
        "temperature_points_floored": profiles["temperature_points_floored"],
        "launcher": {k: v for k, v in launcher.items()},
        "mode": config.mode,
        "harmonic": int(config.harmonic),
        "namelist_overrides": {str(g): dict(e) for g, e in config.namelist_overrides.items()},
    }
    record = directory / PROVENANCE_NAME
    record.write_text(json.dumps(provenance, indent=2, default=float) + "\n", encoding="utf-8")
    return GENRAYInputs(
        workdir=directory,
        genray_in=genray_in,
        eqdsk=Path(eqdsk),
        provenance=provenance,
        files=(genray_in, Path(eqdsk), record),
    )


__all__ = [
    "EQDSK_NAME",
    "GENRAY_INPUT_NAME",
    "PROVENANCE_NAME",
    "eccone_angles",
    "genray_namelists",
    "prepare_genray_inputs",
    "render_namelists",
]

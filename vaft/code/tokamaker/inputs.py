"""Build TokaMaker forward-solve inputs from an ODS.

``prepare_tokamaker_inputs`` reads machine geometry (wall limiter, ``pf_active``
coil rectangles), the prescribed coil currents at the case time, the vacuum
toroidal field (``F0 = R0*B0``), and the global targets from an ODS, then
resolves the mesh-cache location. It never imports OpenFUSIONToolkit and never
builds a mesh — that is deferred to ``build_tokamaker_mesh``/``run_tokamaker``
so input preparation works in OFT-free environments (tests, CI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dataclasses import replace

from .config import TokaMakerConfig, TokaMakerEvolutionInputs, TokaMakerInputs
from .geometry import (
    _coil_name,
    geometry_signature,
    split_coil_names,
    tokamaker_geometry_from_ods,
)


# ----------------------------------------------------------------------------- #
#  ODS readers
# ----------------------------------------------------------------------------- #
def _infer_shot(ods: Any, config: TokaMakerConfig) -> int:
    if config.shot is not None:
        return int(config.shot)
    if ods is not None:
        for path in (
            "dataset_description.data_entry.pulse",
            "summary.global_quantities.pulse",
        ):
            try:
                return int(ods[path])
            except Exception:
                pass
    raise ValueError(
        "TokaMaker shot number is required in TokaMakerConfig.shot or ODS metadata"
    )


def _resolve_time(ods: Any, config: TokaMakerConfig) -> float:
    """Resolve the case time [s] from an explicit slice index or ``time``.

    When ``time_index`` is given it selects a slice of the chosen constraint
    source (the ``equilibrium`` time array, or the ``magnetics`` Ip time array)
    and the corresponding time is returned; otherwise ``time`` is used directly.
    """
    if config.time_index is not None:
        idx = int(config.time_index)
        if config.constraint_source == "magnetics":
            t = np.asarray(ods["magnetics.ip.0.time"], dtype=float)[idx]
        else:
            t = np.asarray(ods["equilibrium.time"], dtype=float)[idx]
        return float(t)
    if config.time is not None:
        return float(config.time)
    raise ValueError("TokaMakerConfig.time (seconds) or TokaMakerConfig.time_index is required")


def _ip_from_magnetics(ods: Any, time: float) -> float:
    mg = ods["magnetics"]
    return float(np.interp(time, mg["ip.0.time"], mg["ip.0.data"]))


def _resolve_targets(ods: Any, config: TokaMakerConfig, time: float) -> dict[str, float]:
    """Resolve the ``set_targets`` kwargs honouring ``constraint_source``.

    - ``"equilibrium"``: Ip comes from the equilibrium slice nearest ``time``
      (explicit ``time_index`` wins), falling back to magnetics.
    - ``"magnetics"``: Ip comes from magnetics only; the equilibrium IDS is
      never read.

    ``ip``/``pax``/``ip_ratio``/``r0_target``/``v0_target`` set explicitly on
    the config always take precedence; optional targets are only forwarded when
    set, so unset targets stay disabled in TokaMaker.
    """
    source = config.constraint_source
    if source not in ("equilibrium", "magnetics"):
        raise ValueError(
            "TokaMakerConfig.constraint_source must be 'equilibrium' or "
            f"'magnetics', got {source!r}"
        )

    if config.ip is not None:
        ip = float(config.ip)
    elif source == "magnetics":
        ip = _ip_from_magnetics(ods, time)
    else:
        # The whole equilibrium read is fallback-guarded: a raw shot with
        # magnetics but no reconstruction has an empty/absent equilibrium IDS,
        # and the time-array lookup must fall back just like a missing slice.
        try:
            eq = ods["equilibrium"]
            idx = config.time_index
            if idx is None:
                eqtime = np.asarray(eq["time"], dtype=float)
                idx = int(np.argmin(np.abs(eqtime - time)))
            ip = float(eq[f"time_slice.{int(idx)}.global_quantities.ip"])
        except Exception:
            ip = _ip_from_magnetics(ods, time)

    if not np.isfinite(ip) or ip <= 0.0:
        raise ValueError(
            f"TokaMaker Ip target must be a positive current in amperes, got {ip!r}. "
            "TokaMaker rejects Ip <= 0; for a reversed-current convention flip the "
            "signs explicitly (Ip, F0, coil currents) instead of relying on the adapter."
        )

    targets: dict[str, float] = {"Ip": ip}
    if config.pax is not None:
        targets["pax"] = float(config.pax)
    if config.ip_ratio is not None:
        targets["Ip_ratio"] = float(config.ip_ratio)
    if config.r0_target is not None:
        targets["R0"] = float(config.r0_target)
    if config.v0_target is not None:
        targets["V0"] = float(config.v0_target)
    return targets


def _f0_from_ods(ods: Any, config: TokaMakerConfig, time: float) -> float:
    """Resolve F0 = R0*B0 [T·m]: explicit ``f0`` > ``bt0 * major_r`` > tf IDS.

    ``tf.b_field_tor_vacuum_r`` *is* R0*B0 (no division by ``tf.r0`` — that was
    the TES Bt0 convention, not F0).
    """
    if config.f0 is not None:
        return float(config.f0)
    if config.bt0 is not None:
        return float(config.bt0) * float(config.major_r)
    tf = ods["tf"]
    f0 = float(np.interp(time, tf["time"], np.asarray(tf["b_field_tor_vacuum_r.data"], dtype=float)))
    if f0 == 0.0:
        raise ValueError(
            "Vacuum toroidal field resolved to F0 = 0; supply TokaMakerConfig.f0 "
            "(= R0*B0 [T·m]) or bt0 explicitly."
        )
    return f0


def _coil_currents_from_ods(
    ods: Any, config: TokaMakerConfig, geometry: dict, time: float
) -> dict[str, float]:
    """Per-coil-set currents [A] at ``time`` for TokaMaker.set_coil_currents.

    Currents are amps per turn (pf_active terminal current); TokaMaker applies
    the ``nturns`` scaling internally. An explicit ``config.coil_currents``
    mapping replaces the ODS interpolation entirely (missing coils default to
    0 A inside TokaMaker).
    """
    coil_sets = {entry["coil_set"] for entry in geometry["coils"].values()}
    if config.coil_currents is not None:
        explicit = {str(name).upper(): float(value) for name, value in config.coil_currents.items()}
        unknown = sorted(set(explicit) - coil_sets)
        if unknown:
            raise ValueError(
                "coil_currents names not present in the coil sets: "
                + ", ".join(unknown)
                + f". Valid sets: {', '.join(sorted(coil_sets))}."
            )
        # coils omitted from the mapping default to 0 A inside TokaMaker
        return explicit

    pf = ods["pf_active"]
    pf_time = np.asarray(pf["time"], dtype=float)
    by_coil: dict[str, float] = {}
    ncoil = len(pf["coil"])
    for i in range(ncoil):
        name = _coil_name(ods, i)
        data = np.asarray(pf[f"coil.{i}.current.data"], dtype=float)
        by_coil[name] = float(np.interp(time, pf_time, data))

    split_parents = split_coil_names(config)
    currents: dict[str, float] = {}
    missing = []
    for coil_set in coil_sets:
        parent, _, suffix = coil_set.rpartition("_")
        if coil_set in by_coil:
            currents[coil_set] = by_coil[coil_set]
        elif suffix in ("U", "L") and parent in split_parents and parent in by_coil:
            # Split halves carry the parent coil's measured current; any
            # asymmetry comes from explicit overrides (or the virtual '#VSC').
            currents[coil_set] = by_coil[parent]
        else:
            missing.append(coil_set)
    if missing:
        raise ValueError(
            "No pf_active current found for coil set(s): "
            + ", ".join(sorted(missing))
            + ". Supply TokaMakerConfig.coil_currents explicitly."
        )
    return currents


def resolve_mesh_file(geometry: dict, config: TokaMakerConfig) -> tuple[Path, bool]:
    """Resolve the mesh-cache path (explicit ``mesh_file`` or hash-named) and existence."""
    if config.mesh_file is not None:
        path = Path(config.mesh_file).expanduser()
    else:
        signature = geometry_signature(geometry, config)
        path = Path(config.workdir).expanduser() / f"vest_gs_mesh_{signature}.h5"
    return path, path.is_file()


# ----------------------------------------------------------------------------- #
#  Public entry point
# ----------------------------------------------------------------------------- #
def prepare_tokamaker_inputs(ods: Any, config: TokaMakerConfig) -> TokaMakerInputs:
    """Resolve geometry, currents, targets, and the mesh cache from an ODS.

    Pure preparation: no OpenFUSIONToolkit import and no mesh build happen here.
    The geometry dict is also written to ``workdir/geometry.json`` for
    provenance and offline inspection.
    """
    workdir = Path(config.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    shot = _infer_shot(ods, config)
    time = _resolve_time(ods, config)

    geometry = tokamaker_geometry_from_ods(ods, config)
    targets = _resolve_targets(ods, config, time)
    f0 = _f0_from_ods(ods, config, time)
    coil_currents = _coil_currents_from_ods(ods, config, geometry, time)
    mesh_file, mesh_exists = resolve_mesh_file(geometry, config)

    geometry_file = workdir / "geometry.json"
    geometry_file.write_text(json.dumps(geometry, indent=2, sort_keys=True))

    return TokaMakerInputs(
        workdir=workdir,
        geometry=geometry,
        mesh_file=mesh_file,
        mesh_exists=mesh_exists,
        coil_currents=coil_currents,
        targets=targets,
        f0=f0,
        shot=shot,
        time=time,
        ods=ods,
        files=(geometry_file,),
    )


# ----------------------------------------------------------------------------- #
#  Quasi-static evolution preparation
# ----------------------------------------------------------------------------- #
def _resolve_evolution_times(config: TokaMakerConfig) -> tuple[float, ...]:
    if config.evolve_times is not None:
        times = tuple(float(t) for t in config.evolve_times)
    else:
        if config.evolve_start is None or config.evolve_end is None or config.evolve_dt is None:
            raise ValueError(
                "Evolution needs TokaMakerConfig.evolve_times, or all of "
                "evolve_start/evolve_end/evolve_dt."
            )
        times = tuple(
            float(t) for t in np.arange(config.evolve_start, config.evolve_end, config.evolve_dt)
        )
    if len(times) < 2:
        raise ValueError(f"Evolution needs at least 2 time slices, got {len(times)}")
    if any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
        raise ValueError("Evolution times must be strictly increasing")
    if not config.evolve_vacuum:
        # per-slice g-files are named g<shot>.<integer ms>; enforce uniqueness
        ms = [int(round(t * 1000)) for t in times]
        if len(set(ms)) != len(ms):
            raise ValueError(
                "Evolution times must round to distinct integer milliseconds "
                "(g-file naming and the multi-slice equilibrium merge key on "
                "them); use a grid with >= 1 ms spacing or explicit evolve_times."
            )
    return times


def _coil_waveforms_from_ods(
    ods: Any, config: TokaMakerConfig, geometry: dict, times: tuple[float, ...]
) -> dict[str, np.ndarray]:
    """Per-coil-set current waveforms sampled on the evolution grid."""
    waveforms: dict[str, list[float]] = {}
    for t in times:
        currents = _coil_currents_from_ods(ods, config, geometry, t)
        for name, value in currents.items():
            waveforms.setdefault(name, []).append(value)
    return {name: np.asarray(values, dtype=float) for name, values in waveforms.items()}


def prepare_tokamaker_evolution_inputs(
    ods: Any, config: TokaMakerConfig
) -> TokaMakerEvolutionInputs:
    """Resolve everything a quasi-static evolution needs from an ODS.

    Calls :func:`prepare_tokamaker_inputs` once at the first slice time, then
    samples the per-coil-set current waveforms and the per-slice Ip targets on
    the evolution grid. ``evolve_vacuum=True`` skips plasma targets entirely
    (``vac_solve`` mode for plasma-free windows).
    """
    if not config.include_vessel:
        raise ValueError(
            "Quasi-static evolution models wall eddy currents and requires "
            "TokaMakerConfig.include_vessel=True."
        )
    times = _resolve_evolution_times(config)

    # time_index must be cleared throughout: _resolve_time prefers it over
    # ``time``, and both the base inputs (F0, currents) and the per-step Ip
    # targets belong to the evolution grid, not a fixed slice index.
    step_config = replace(config, time_index=None)
    if config.evolve_vacuum:
        # No plasma: bypass the Ip-target resolution (which requires Ip > 0).
        base = prepare_tokamaker_inputs(ods, replace(step_config, time=times[0], ip=1.0))
        base.targets = {}
        ip_targets = np.zeros(len(times))
    else:
        base = prepare_tokamaker_inputs(ods, replace(step_config, time=times[0]))
        ip_targets = np.asarray(
            [_resolve_targets(ods, step_config, t)["Ip"] for t in times], dtype=float
        )

    coil_waveforms = _coil_waveforms_from_ods(ods, step_config, base.geometry, times)
    return TokaMakerEvolutionInputs(
        base=base,
        times=times,
        coil_waveforms=coil_waveforms,
        ip_targets=ip_targets,
        vacuum=bool(config.evolve_vacuum),
    )

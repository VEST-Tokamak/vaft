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

from .config import TokaMakerConfig, TokaMakerInputs
from .geometry import _coil_name, geometry_signature, tokamaker_geometry_from_ods


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
    currents: dict[str, float] = {}
    ncoil = len(pf["coil"])
    for i in range(ncoil):
        name = _coil_name(ods, i)
        if name not in coil_sets:
            continue
        data = np.asarray(pf[f"coil.{i}.current.data"], dtype=float)
        currents[name] = float(np.interp(time, pf_time, data))
    missing = coil_sets - currents.keys()
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

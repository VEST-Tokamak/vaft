"""OMAS-first upstream stages for the VEST production workflow."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Sequence

import numpy as np
from omas import ODS

from vaft.database import raw as raw_db
from vaft.database._local import load_ods
from vaft.data.resources import data_path
from vaft.machine_mapping.barometry import barometry
from vaft.machine_mapping.dataset_description import dataset_description
from vaft.machine_mapping.em_coupling import DEFAULT_VERSIONED_COUPLING, em_coupling
from vaft.machine_mapping.magnetics import (
    TOROIDAL_MIRNOV_REFERENCE_CHANNELS,
    vest_md_channel_definitions,
    vfit_magnetics_dynamic,
    vfit_magnetics_static,
)
from vaft.machine_mapping.pf_active import (
    PF_COIL_COUNT,
    resolve_geometry_asset,
    vfit_pf_active_dynamic,
    vfit_pf_active_static,
)
from vaft.machine_mapping.pf_passive import DEFAULT_STATIC_GEOMETRY, pf_passive
from vaft.machine_mapping.spectrometer_uv import spectrometer_uv
from vaft.machine_mapping.tf import vfit_tf_dynamic, vfit_tf_static
from vaft.machine_mapping.wall import wall
from vaft.omas import save
from vaft.omas.process_wrapper import compute_eddy_currents
from vaft.process.magnetics import VestMagneticsProcessingConfig


@dataclass(frozen=True)
class VestMachineEra:
    name: str
    first_shot: int | None
    last_shot: int | None
    pf_geometry: str
    reference_shot: int

    def contains(self, shot: int) -> bool:
        return (self.first_shot is None or shot >= self.first_shot) and (
            self.last_shot is None or shot <= self.last_shot
        )


# 43017 and 45967 are retained as legacy configuration boundaries.  The
# corrected PF6/PF7 geometry begins at shot 45958, producing a deliberate
# intermediate era that the old three-file selection could not represent.
VEST_MACHINE_ERAS = (
    VestMachineEra("vest-pre-43017-pf1906", None, 43016, "1906", 43016),
    VestMachineEra("vest-43017-45957-pf1906", 43017, 45957, "1906", 43017),
    VestMachineEra("vest-45958-45966-pf2507", 45958, 45966, "2507", 45958),
    VestMachineEra("vest-45967-plus-pf2507", 45967, None, "2507", 45967),
)


def machine_era_for_shot(shot: int) -> VestMachineEra:
    """Return the explicit VEST machine era for a shot."""
    for era in VEST_MACHINE_ERAS:
        if era.contains(int(shot)):
            return era
    raise ValueError(f"No VEST machine era is defined for shot {shot}")


def machine_era(name: str) -> VestMachineEra:
    """Resolve a configured machine-era name."""
    for era in VEST_MACHINE_ERAS:
        if era.name == name:
            return era
    choices = ", ".join(item.name for item in VEST_MACHINE_ERAS)
    raise ValueError(f"Unknown VEST machine era {name!r}; expected one of: {choices}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _disabled_pf_coils() -> list[str]:
    import scipy.io

    info = scipy.io.loadmat(resolve_geometry_asset("Coil_info.mat"))
    active = {int(value) for value in np.asarray(info["CoilNumber"]).reshape(-1)}
    return [f"PF{index}" for index in range(1, PF_COIL_COUNT + 1) if index not in active]


def _read_raw_payload(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _archived_field_codes(path: Path) -> set[int]:
    return {int(code) for code in _read_raw_payload(path).get("fields", {})}


def _archived_pulse_datetime(path: Path) -> datetime | None:
    """The raw dump's own SQL-sourced pulse_datetime, if it carries one.

    Populated by `vaft.database.raw.dump_all_raw_signals_for_shot` from the
    authoritative `shot` table (or carried forward from an archive) -- absent
    for older dumps written before that field existed, which is not an error.
    """
    raw_value = _read_raw_payload(path).get("pulse_datetime")
    if not isinstance(raw_value, str):
        return None
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError:
        return None


def _write_raw_payload(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if path.suffix == ".gz":
        with path.open("wb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as handle:
                handle.write(serialized)
    else:
        path.write_bytes(serialized)


def build_static_ods(machine_version: str) -> tuple[ODS, dict[str, Any]]:
    """Build one finalized, versioned static VEST machine ODS."""
    era = machine_era(machine_version)
    ods = ODS(consistency_check=False)
    wall(ods)
    vfit_pf_active_static(ods, shot=era.reference_shot)
    pf_passive(ods)
    em_coupling(ods, shot=era.reference_shot)
    vfit_magnetics_static(ods)
    vfit_tf_static(ods)
    ods["wall.ids_properties.comment"] = (
        f"VEST static wall; machine era {era.name}"
    )
    ods["em_coupling.ids_properties.comment"] = (
        f"VEST electromagnetic coupling; machine era {era.name}; "
        f"PF geometry {era.pf_geometry}"
    )
    # pf_active, pf_passive, magnetics, and tf each have a dynamic counterpart
    # that legitimately sets homogeneous_time=1 once it adds a `.time` node
    # (the per-shot diagnostics stage). This product never adds one, so per
    # the DD's homogeneous_time rule it must be 2, not whatever their static
    # or asset-inherited default is.
    for ids_name in ("pf_active", "pf_passive", "magnetics", "tf"):
        ods[f"{ids_name}.ids_properties.homogeneous_time"] = 2
    pf_geometry_asset = resolve_geometry_asset(
        f"VEST_DiscretizedCoilGeometry_Full_ver_{era.pf_geometry}.mat"
    )
    manifest = {
        "schema_version": 1,
        "stage": "static",
        "status": "success",
        "machine_era": asdict(era),
        "contents": [
            "wall",
            "pf_active",
            "pf_passive",
            "em_coupling",
            "magnetics",
            "tf",
        ],
        "input": {
            "static_geometry": {
                "name": Path(DEFAULT_STATIC_GEOMETRY).name,
                "sha256": sha256_file(DEFAULT_STATIC_GEOMETRY),
            },
            "coupling": {
                "name": Path(DEFAULT_VERSIONED_COUPLING).name,
                "sha256": sha256_file(DEFAULT_VERSIONED_COUPLING),
            },
            "pf_geometry": {
                "name": pf_geometry_asset.name,
                "sha256": sha256_file(pf_geometry_asset),
            },
            "magnetics_geometry": {
                "name": "VEST_MagneticsGeometry_Full_ver_2302.yaml",
                "source_version": "2409",
                "sha256": sha256_file(
                    data_path("geometry/VEST_MagneticsGeometry_Full_ver_2302.yaml")
                ),
            },
            "magnetics_calibration": {
                "name": "MD.yaml",
                "source_version": "2409",
                "sha256": sha256_file(data_path("geometry/MD.yaml")),
            },
        },
        "channel_status": {
            "pf_active": {
                "status": "success",
                "disabled_channels": _disabled_pf_coils(),
            }
        },
        "quality_summary": {
            "missing": [],
            "repaired": [],
            "disabled": _disabled_pf_coils(),
            "rejected": [],
            "unavailable": [],
        },
    }
    return ods, manifest


def _copy_ids(target: ODS, source: ODS, ids_names: tuple[str, ...]) -> None:
    for ids_name in ids_names:
        if ids_name in source:
            target[ids_name] = copy.deepcopy(source[ids_name])


def build_diagnostics_ods(
    *,
    shot: int,
    raw_source: str | Path,
    static_ods: str | Path,
    tstart: float = 0.26,
    tend: float = 0.36,
    dt: float = 4e-5,
    run: int = 1,
    vest_magnetics_processing: dict[str, Any] | None = None,
) -> tuple[ODS, dict[str, Any]]:
    """Build independent diagnostic IDSs without losing valid siblings."""
    raw_path = Path(raw_source)
    static_path = Path(static_ods)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dump not found: {raw_path}")
    if not static_path.exists():
        raise FileNotFoundError(f"Static ODS not found: {static_path}")

    shot = int(shot)
    era = machine_era_for_shot(shot)
    static, _ = load_ods(static_path)
    ods = ODS(consistency_check=False)
    dataset_description(
        ods,
        shot,
        {
            "source_type": "shot",
            "run": run,
            "machine": "VEST",
            "user": "vaft",
            "description": f"VEST diagnostics; machine era {era.name}",
            "pulse_datetime": _archived_pulse_datetime(raw_path),
        },
    )
    statuses: dict[str, Any] = {}
    archived_fields = _archived_field_codes(raw_path)

    def run_component(
        name: str,
        ids_names: tuple[str, ...],
        mapper: Callable[[ODS], None],
        component_status: str = "success",
        **details: Any,
    ) -> None:
        component = ODS(consistency_check=False)
        try:
            mapper(component)
        except (raw_db.RawSignalUnavailableError, FileNotFoundError) as error:
            statuses[name] = {"status": "unavailable", "reason": str(error), **details}
            return
        _copy_ids(ods, component, ids_names)
        for ids_name in ids_names:
            # A component built by copying static geometry (homogeneous_time=2,
            # no `.time` node) and then adding dynamic data now has a `.time`
            # node, so homogeneous_time must become 1 to match it.
            if f"{ids_name}.time" in component:
                ods[f"{ids_name}.ids_properties.homogeneous_time"] = 1
        statuses[name] = {"status": component_status, **details}

    run_component(
        "pf_active",
        ("pf_active",),
        lambda component: (
            _copy_ids(component, static, ("pf_active",)),
            vfit_pf_active_dynamic(
                component, shot, tstart, tend, dt, raw_source=raw_path
            ),
        ),
        disabled_channels=_disabled_pf_coils(),
    )
    run_component(
        "spectrometer_uv",
        ("spectrometer_uv",),
        lambda component: spectrometer_uv(
            component, shot, tstart, tend, dt, raw_source=raw_path
        ),
    )
    run_component(
        "barometry",
        ("barometry",),
        lambda component: barometry(
            component, shot, tstart, tend, dt, raw_source=raw_path
        ),
    )
    run_component(
        "tf",
        ("tf",),
        lambda component: (
            _copy_ids(component, static, ("tf",)),
            vfit_tf_dynamic(
                component, shot, tstart, tend, dt, raw_source=raw_path
            ),
        ),
    )
    processing = (
        VestMagneticsProcessingConfig(**vest_magnetics_processing)
        if vest_magnetics_processing
        else None
    )
    magnetics_channels = [
        int(channel["field_code"]) for channel in vest_md_channel_definitions()
    ] + [int(channel["field_code"]) for channel in TOROIDAL_MIRNOV_REFERENCE_CHANNELS]
    missing_magnetics_channels = sorted(
        field for field in magnetics_channels if field not in archived_fields
    )
    run_component(
        "magnetics",
        ("magnetics",),
        lambda component: (
            _copy_ids(component, static, ("magnetics",)),
            vfit_magnetics_dynamic(
                component,
                shot,
                tstart,
                tend,
                dt,
                processing_config=processing,
                raw_source=raw_path,
            ),
        ),
        processing="VestMagneticsProcessingConfig",
        component_status="partial" if missing_magnetics_channels else "success",
        missing_channels=missing_magnetics_channels,
    )

    successes = sum(value["status"] == "success" for value in statuses.values())
    unavailable = sorted(
        name for name, value in statuses.items() if value["status"] == "unavailable"
    )
    missing_channels = sorted(
        f"{name}:field-{field}"
        for name, value in statuses.items()
        for field in value.get("missing_channels", [])
    )
    manifest = {
        "schema_version": 1,
        "stage": "diagnostics",
        "shot": shot,
        "machine_version": era.name,
        "status": "success" if successes == len(statuses) else "partial",
        "input": {
            "raw_sha256": sha256_file(raw_path),
            "static_sha256": sha256_file(static_path),
        },
        "configuration": {
            "tstart": float(tstart),
            "tend": float(tend),
            "dt": float(dt),
            "run": int(run),
            "vest_magnetics_processing": vest_magnetics_processing or {},
        },
        "channel_status": statuses,
        "quality_summary": {
            "missing": sorted(unavailable + missing_channels),
            "repaired": [],
            "disabled": _disabled_pf_coils(),
            "rejected": [],
            "unavailable": unavailable,
        },
    }
    return ods, manifest


def build_eddy_ods(
    *,
    shot: int,
    diagnostics_ods: str | Path,
    static_ods: str | Path,
    filament_r: list[float],
    filament_z: list[float],
    filament_fraction: list[float],
    dt_sub: float = 5e-5,
) -> tuple[ODS, dict[str, Any]]:
    """Compute target-shot passive currents from finalized input ODSs."""
    if not (len(filament_r) == len(filament_z) == len(filament_fraction)):
        raise ValueError("Filament r, z, and fraction lists must have the same length")
    diagnostics_path = Path(diagnostics_ods)
    static_path = Path(static_ods)
    diagnostics, _ = load_ods(diagnostics_path)
    static, _ = load_ods(static_path)
    missing = [
        path
        for path in ("pf_active.time", "magnetics.ip.0.time", "magnetics.ip.0.data")
        if path not in diagnostics
    ]
    if missing:
        raise raw_db.RawSignalUnavailableError(
            shot,
            "eddy-input",
            "diagnostics ODS is missing " + ", ".join(missing),
            signal_name="eddy-current constraints",
        )

    ods = diagnostics
    _copy_ids(ods, static, ("wall", "pf_passive", "em_coupling"))
    pf_time = np.asarray(ods["pf_active.time"], dtype=float)
    ip_time = np.asarray(ods["magnetics.ip.0.time"], dtype=float)
    ip_data = np.asarray(ods["magnetics.ip.0.data"], dtype=float)
    ip_on_pf_time = np.interp(pf_time, ip_time, ip_data)
    plasma = list(zip(filament_r, filament_z))
    plasma_currents = [ip_on_pf_time * fraction for fraction in filament_fraction]
    compute_eddy_currents(ods, plasma, plasma_currents, dt_sub=dt_sub)
    # pf_passive was copied from static (homogeneous_time=2, no `.time` node);
    # compute_eddy_currents() just added one, so this must become 1 to match.
    if "pf_passive.time" in ods:
        ods["pf_passive.ids_properties.homogeneous_time"] = 1
    manifest = {
        "schema_version": 1,
        "stage": "eddy",
        "shot": int(shot),
        "machine_version": machine_era_for_shot(int(shot)).name,
        "status": "success",
        "input": {
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "static_sha256": sha256_file(static_path),
        },
        "filaments": [
            {"r": r, "z": z, "current_fraction": fraction}
            for r, z, fraction in zip(filament_r, filament_z, filament_fraction)
        ],
        "dt_sub": float(dt_sub),
    }
    return ods, manifest


def build_mhd_linear_ods(
    *,
    shot: int,
    time_values: Sequence[int | str],
    workdir: str | Path,
    modules: Sequence[str] = ("dcon", "rdcon", "stride"),
    modes: Sequence[int] = (1, 2),
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Build the ``mhd_linear`` IDS from a completed GPEC-suite run directory.

    ``time_values`` is the shot's set of refined-gfile time labels (whatever
    ``GPECCaseInputs.time_ms`` was for each ``run_gpec_suite_case`` call, in
    milliseconds) -- one ``mhd_linear.time_slice`` entry is built per value,
    in order. Reads DCON/RDCON/STRIDE ``.nc`` output already written by
    :func:`vaft.code.gpec.run_gpec_suite_case` under
    ``{workdir}/{time_label}/{module}/nn={mode}/`` -- computed via the same
    helper the adapter itself uses (:mod:`vaft.code.gpec._runtime`) so this
    never re-derives, and risks diverging from, that directory grammar.

    A missing or unparseable ``(time, module, mode)`` cell is recorded in the
    manifest's ``modules_modes`` breakdown rather than aborting the whole
    build -- one failed RDCON case should not prevent DCON's results from
    being captured.
    """
    from vaft.code.gpec import _runtime as gpec_runtime
    from vaft.machine_mapping.mhd_linear import mhd_linear as mhd_linear_mapper

    shot = int(shot)
    era = machine_era_for_shot(shot)

    ods = ODS(consistency_check=False)
    dataset_description(
        ods,
        shot,
        {
            "source_type": "shot",
            "run": run,
            "machine": "VEST",
            "user": "vaft",
            "description": f"VEST linear MHD stability (DCON/RDCON/STRIDE); machine era {era.name}",
        },
    )

    ods["mhd_linear"]["ids_properties"]["homogeneous_time"] = 1
    ods["mhd_linear"]["time"] = [float(t) / 1000.0 for t in time_values]

    modules_modes: dict[str, Any] = {}
    inputs_hashes: dict[str, str] = {}
    workdir_path = Path(workdir)
    for time_slice, time_ms in enumerate(time_values):
        for module in modules:
            for mode in modes:
                key = f"t={time_ms}/{module}/n={mode}"
                run_dir = gpec_runtime.module_dir(workdir_path, time_ms, module, mode)
                if not run_dir.is_dir():
                    modules_modes[key] = {"status": "missing", "reason": f"run directory not found: {run_dir}"}
                    continue
                try:
                    extras = mhd_linear_mapper(ods, str(run_dir), {"time_slice": time_slice, "module": module})
                except Exception as exc:
                    modules_modes[key] = {"status": "failed", "reason": str(exc)}
                    continue
                if not extras:
                    modules_modes[key] = {"status": "no_output", "reason": "no matching .nc output found"}
                    continue
                modules_modes[key] = {"status": "success", "modes": extras}
                for nc_path in sorted(run_dir.glob("*.nc")):
                    inputs_hashes[f"{key}/{nc_path.name}"] = sha256_file(nc_path)

    status = "success" if any(cell["status"] == "success" for cell in modules_modes.values()) else "empty"
    manifest = {
        "schema_version": 1,
        "stage": "mhd_linear",
        "shot": shot,
        "time_values": list(time_values),
        "machine_version": era.name,
        "status": status,
        "input": inputs_hashes,
        "modules_modes": modules_modes,
    }
    return ods, manifest


def write_stage_product(
    ods: ODS,
    manifest: dict[str, Any],
    *,
    output: str | Path,
    metadata: str | Path,
) -> None:
    """Write deterministic ODS and manifest products."""
    output_path = save(ods, output)
    final_manifest = dict(manifest)
    final_manifest["output"] = {
        "name": output_path.name,
        "sha256": sha256_file(output_path),
    }
    metadata_path = Path(metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def archive_raw_source(
    *, shot: int, output: str | Path, source: str | Path | None = None
) -> dict[str, Any]:
    """Copy an explicit archive or export one shot from live SQL."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        source_path = Path(str(source).format(shot=int(shot))).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Archived raw source not found: {source_path}")
        opener = gzip.open if source_path.suffix == ".gz" else open
        with opener(source_path, "rt", encoding="utf-8") as handle:
            source_payload = json.load(handle)
        source_shot = source_payload.get("shot")
        try:
            source_shot_number = int(source_shot)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Raw archive has no valid shot number: {source_path}") from error
        if source_shot_number != int(shot):
            raise ValueError(
                f"Raw archive shot mismatch: requested {shot}, "
                f"but {source_path} contains shot {source_shot}"
            )
        if not isinstance(source_payload.get("fields"), dict):
            raise ValueError(f"Raw archive has no fields mapping: {source_path}")
        if source_path.resolve() != output_path.resolve():
            if (source_path.suffix == ".gz") == (output_path.suffix == ".gz"):
                shutil.copyfile(source_path, output_path)
            else:
                _write_raw_payload(output_path, source_payload)
        source_kind = "archive"
        source_name = source_path.name
    else:
        temporary_dump: Path | None = None
        dump_path = output_path
        if output_path.suffix != ".gz":
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".json.gz",
                delete=False,
            ) as handle:
                temporary_dump = Path(handle.name)
            dump_path = temporary_dump
        try:
            if not raw_db.dump_all_raw_signals_for_shot(int(shot), str(dump_path)):
                raise RuntimeError(f"Failed to export VEST raw data for shot {shot}")
            with gzip.open(dump_path, "rt", encoding="utf-8") as handle:
                source_payload = json.load(handle)
            if temporary_dump is not None:
                _write_raw_payload(output_path, source_payload)
        finally:
            if temporary_dump is not None:
                temporary_dump.unlink(missing_ok=True)
        source_kind = "vest-sql"
        source_name = None
    field_codes = sorted(int(code) for code in source_payload["fields"])
    return {
        "schema_version": 1,
        "stage": "raw",
        "shot": int(shot),
        "status": "success",
        "source": {"kind": source_kind, "name": source_name},
        "inventory": {"field_count": len(field_codes), "field_codes": field_codes},
        "output": {"name": output_path.name, "sha256": sha256_file(output_path)},
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "VEST_MACHINE_ERAS",
    "VestMachineEra",
    "archive_raw_source",
    "build_diagnostics_ods",
    "build_eddy_ods",
    "build_static_ods",
    "machine_era",
    "machine_era_for_shot",
    "write_manifest",
    "write_stage_product",
]

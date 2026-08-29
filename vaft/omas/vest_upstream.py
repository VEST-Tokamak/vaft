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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from omas import ODS

from vaft.database import raw as raw_db
from vaft.database._local import load_ods
from vaft.data.resources import data_path
from vaft.machine_mapping.barometry import barometry
from vaft.machine_mapping.dataset_description import dataset_description
from vaft.machine_mapping.em_coupling import DEFAULT_VERSIONED_COUPLING, em_coupling
from vaft.machine_mapping.impa import impa as impa_mapper, impa_expected_fields
from vaft.machine_mapping.langmuir_probes import langmuir_probes
from vaft.machine_mapping.magnetics import (
    LIMITER_SHUNT_CHANNELS,
    TOROIDAL_MIRNOV_REFERENCE_CHANNELS,
    FLUCTUATION_MIRNOV_FIRST_SHOT,
    LIMITER_SHUNT_CHANNELS,
    TOROIDAL_MIRNOV_REFERENCE_CHANNELS,
    fluctuation_mirnov_channel_definitions,
    vest_equilibrium_magnetics_channel_definitions,
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
from vaft.machine_mapping.utils import get_path, path_exists
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


def _canonical_diagnostics_time(tstart: float, tend: float, dt: float) -> np.ndarray:
    """Build the one processed grid used by the diagnostics product.

    Diagnostics windows are always half-open: ``tstart <= t < tend``.  Native
    acquisition timebases are handled explicitly by their mapper and are not
    selected by overloading a non-positive ``dt`` value.
    """
    tstart, tend, dt = float(tstart), float(tend), float(dt)
    if not all(np.isfinite(value) for value in (tstart, tend, dt)):
        raise ValueError("Diagnostics tstart, tend, and dt must be finite")
    if tend <= tstart:
        raise ValueError("Diagnostics tend must be greater than tstart")
    if dt <= 0.0:
        raise ValueError("Diagnostics dt must be positive; native time is an explicit mapper mode")
    time = np.arange(tstart, tend, dt, dtype=float)
    if time.size == 0:
        raise ValueError("Diagnostics window produces an empty processed time grid")
    return time


def _validate_time_data_pair(ods: ODS, time_path: str, data_path: str) -> None:
    """Check the basic invariant shared by all mapped waveform nodes."""
    if not path_exists(ods, time_path) or not path_exists(ods, data_path):
        return
    time = np.asarray(get_path(ods, time_path), dtype=float).reshape(-1)
    data = np.asarray(get_path(ods, data_path), dtype=float).reshape(-1)
    if time.size != data.size:
        raise ValueError(f"{data_path} has {data.size} samples but {time_path} has {time.size}")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{time_path} must be strictly monotonic")


def _validate_data_on_grid(ods: ODS, data_path: str, grid: np.ndarray) -> None:
    if not path_exists(ods, data_path):
        return
    data = np.asarray(get_path(ods, data_path), dtype=float).reshape(-1)
    if data.size == 0:
        return
    if data.size != grid.size:
        raise ValueError(
            f"{data_path} has {data.size} samples but the canonical grid has {grid.size}"
        )


def _time_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.allclose(left, right, rtol=0.0, atol=1e-12)


def _validate_diagnostics_time_coordinates(
    ods: ODS,
    processed_time: np.ndarray,
    *,
    tstart: float,
    tend: float,
    dt: float,
) -> dict[str, Any]:
    """Validate realized diagnostic coordinates and return manifest metadata."""
    processed_time = np.asarray(processed_time, dtype=float)
    present_ids = set(ods.keys())
    if processed_time.size > 1 and np.any(np.diff(processed_time) <= 0.0):
        raise ValueError("Canonical diagnostics time must be strictly monotonic")

    canonical_paths = (
        "pf_active.time",
        "spectrometer_uv.time",
        "tf.time",
        "magnetics.time",
        "barometry.gauge.0.pressure.time",
    )
    for path in canonical_paths:
        if path.split(".", 1)[0] not in present_ids:
            continue
        if path_exists(ods, path):
            actual = np.asarray(get_path(ods, path), dtype=float).reshape(-1)
            if not _time_equal(actual, processed_time):
                raise ValueError(f"{path} does not use the canonical diagnostics grid")

    for time_path, signal_data_path in (
        ("pf_active.coil.0.current.time", "pf_active.coil.0.current.data"),
        ("tf.b_field_tor_vacuum_r.time", "tf.b_field_tor_vacuum_r.data"),
        ("tf.coil.0.current.time", "tf.coil.0.current.data"),
        ("magnetics.ip.0.time", "magnetics.ip.0.data"),
        ("magnetics.diamagnetic_flux.0.time", "magnetics.diamagnetic_flux.0.data"),
        ("barometry.gauge.0.pressure.time", "barometry.gauge.0.pressure.data"),
    ):
        if time_path.split(".", 1)[0] not in present_ids:
            continue
        _validate_time_data_pair(ods, time_path, signal_data_path)

    if "pf_active" in present_ids:
        for index in range(len(ods["pf_active.coil"])):
            _validate_time_data_pair(
                ods,
                f"pf_active.coil.{index}.current.time",
                f"pf_active.coil.{index}.current.data",
            )
    if "magnetics" in present_ids:
        for index in range(len(ods["magnetics.flux_loop"])):
            base = f"magnetics.flux_loop.{index}.flux"
            _validate_time_data_pair(ods, f"{base}.time", f"{base}.data")
            _validate_data_on_grid(ods, f"{base}.data", processed_time)
            if (
                path_exists(ods, f"{base}.data")
                and np.asarray(get_path(ods, f"{base}.data")).size > 0
                and path_exists(ods, f"{base}.time")
                and not _time_equal(
                np.asarray(get_path(ods, f"{base}.time"), dtype=float).reshape(-1),
                processed_time,
                )
            ):
                raise ValueError(f"{base}.time does not use the canonical diagnostics grid")
        for index in range(len(ods["magnetics.b_field_pol_probe"])):
            base = f"magnetics.b_field_pol_probe.{index}.field"
            _validate_time_data_pair(ods, f"{base}.time", f"{base}.data")
            _validate_data_on_grid(ods, f"{base}.data", processed_time)
            if (
                path_exists(ods, f"{base}.data")
                and np.asarray(get_path(ods, f"{base}.data")).size > 0
                and path_exists(ods, f"{base}.time")
                and not _time_equal(
                np.asarray(get_path(ods, f"{base}.time"), dtype=float).reshape(-1),
                processed_time,
                )
            ):
                raise ValueError(f"{base}.time does not use the canonical diagnostics grid")

    native_paths: list[str] = []
    native_time_metadata: list[dict[str, Any]] = []
    native_flux_loop_metadata: list[dict[str, Any]] = []
    if "magnetics" in present_ids:
        root_time = np.asarray(get_path(ods, "magnetics.time"), dtype=float).reshape(-1)
        for index in range(len(ods["magnetics.b_field_pol_probe"])):
            base = f"magnetics.b_field_pol_probe.{index}.voltage"
            time_path, data_path = f"{base}.time", f"{base}.data"
            _validate_time_data_pair(ods, time_path, data_path)
            if not path_exists(ods, time_path):
                continue
            voltage_time = np.asarray(get_path(ods, time_path), dtype=float).reshape(-1)
            if voltage_time.size == 0:
                continue
            if np.any(voltage_time < tstart) or np.any(voltage_time >= tend):
                raise ValueError(f"{time_path} is outside the diagnostics analysis window")
            if not _time_equal(voltage_time, root_time):
                native_paths.append(time_path)
                native_dt = (
                    float(np.median(np.diff(voltage_time)))
                    if voltage_time.size > 1
                    else None
                )
                native_time_metadata.append(
                    {
                        "path": time_path,
                        "sample_count": int(voltage_time.size),
                        "dt": native_dt,
                        "sampling_rate": 1.0 / native_dt if native_dt else None,
                    }
                )
        # Flux-loop terminal voltage (issue #209) is stored at the native
        # acquisition rate, cropped to the analysis window, and is therefore
        # validated as a native coordinate rather than against the canonical
        # processed grid used by flux_loop[*].flux above.
        for index in range(len(ods["magnetics.flux_loop"])):
            base = f"magnetics.flux_loop.{index}.voltage"
            time_path, data_path = f"{base}.time", f"{base}.data"
            _validate_time_data_pair(ods, time_path, data_path)
            if not path_exists(ods, time_path):
                continue
            voltage_time = np.asarray(get_path(ods, time_path), dtype=float).reshape(-1)
            if voltage_time.size == 0:
                continue
            if np.any(voltage_time < tstart) or np.any(voltage_time >= tend):
                raise ValueError(f"{time_path} is outside the diagnostics analysis window")
            if not _time_equal(voltage_time, root_time):
                native_paths.append(time_path)
                native_dt = (
                    float(np.median(np.diff(voltage_time)))
                    if voltage_time.size > 1
                    else None
                )
                native_flux_loop_metadata.append(
                    {
                        "path": time_path,
                        "sample_count": int(voltage_time.size),
                        "dt": native_dt,
                        "sampling_rate": 1.0 / native_dt if native_dt else None,
                    }
                )
        for index in range(len(ods["magnetics.shunt"])):
            base = f"magnetics.shunt.{index}.voltage"
            time_path, data_path = f"{base}.time", f"{base}.data"
            _validate_time_data_pair(ods, time_path, data_path)
            if path_exists(ods, time_path):
                shunt_time = np.asarray(get_path(ods, time_path), dtype=float).reshape(-1)
                if shunt_time.size and not _time_equal(shunt_time, root_time):
                    native_paths.append(time_path)
        # IMAS defines homogeneous mode as one root-level coordinate for every
        # dynamic quantity. Native Mirnov coordinates therefore require mode 0.
        ods["magnetics.ids_properties.homogeneous_time"] = 0 if native_paths else 1

    realized_dt = float(np.median(np.diff(processed_time))) if processed_time.size > 1 else None
    if realized_dt is not None and not np.isclose(realized_dt, dt, rtol=0.0, atol=1e-12):
        raise ValueError("Canonical diagnostics time does not realize the configured dt")
    return {
        "requested_start": float(tstart),
        "requested_end": float(tend),
        "requested_dt": float(dt),
        "processed_start": float(processed_time[0]),
        "processed_end_exclusive": float(processed_time[0] + processed_time.size * (realized_dt or 0.0)),
        "processed_sample_count": int(processed_time.size),
        "realized_dt": realized_dt,
        "processed_time_clipped": False,
        "source_clipping": False,
        "magnetics_homogeneous_time": int(ods["magnetics.ids_properties.homogeneous_time"])
        if "magnetics" in present_ids
        else None,
        "native_time_paths": native_paths,
        # Quantity-specific: `native_mirnov` stays probe-only so consumers that
        # index it keep reading Mirnov sampling, while flux-loop terminal
        # voltage (issue #209) reports under its own key.
        "native_mirnov": native_time_metadata,
        "native_flux_loop_voltage": native_flux_loop_metadata,
    }


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
    processed_time = _canonical_diagnostics_time(tstart, tend, dt)
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
                component, shot, tstart, tend, dt, raw_source=raw_path,
                target_time=processed_time,
            ),
        ),
        disabled_channels=_disabled_pf_coils(),
    )
    run_component(
        "spectrometer_uv",
        ("spectrometer_uv",),
        lambda component: spectrometer_uv(
            component, shot, tstart, tend, dt, raw_source=raw_path,
            target_time=processed_time,
        ),
    )
    run_component(
        "barometry",
        ("barometry",),
        lambda component: barometry(
            component, shot, tstart, tend, dt, raw_source=raw_path,
            target_time=processed_time,
        ),
    )
    run_component(
        "langmuir_probes",
        ("langmuir_probes",),
        lambda component: langmuir_probes(
            component, shot, tstart, tend, dt, raw_source=raw_path
        ),
    )
    run_component(
        "tf",
        ("tf",),
        lambda component: (
            _copy_ids(component, static, ("tf",)),
            vfit_tf_dynamic(
                component, shot, tstart, tend, dt, raw_source=raw_path,
                target_time=processed_time,
            ),
        ),
    )
    processing = (
        VestMagneticsProcessingConfig(**vest_magnetics_processing)
        if vest_magnetics_processing
        else None
    )
    magnetics_channels = [
        int(channel["field_code"]) for channel in vest_equilibrium_magnetics_channel_definitions()
    ] + [int(channel["field_code"]) for channel in TOROIDAL_MIRNOV_REFERENCE_CHANNELS] + [
        int(channel["field_code"]) for channel in LIMITER_SHUNT_CHANNELS
    ]
    if int(shot) >= FLUCTUATION_MIRNOV_FIRST_SHOT:
        magnetics_channels += [
            int(channel["field"]) for channel in fluctuation_mirnov_channel_definitions()
        ]
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
                target_time=processed_time,
            ),
        ),
        processing="VestMagneticsProcessingConfig",
        component_status="partial" if missing_magnetics_channels else "success",
        missing_channels=missing_magnetics_channels,
    )

    # IMPA extends magnetics.b_field_pol_probe, so the component is seeded from
    # the magnetics IDS built above and copied back.  A failed IMPA calibration
    # must never discard valid magnetics data.
    impa_status: dict[str, Any] = {}

    def _map_impa(component: ODS) -> None:
        _copy_ids(component, ods, ("magnetics",))
        impa_status.update(
            impa_mapper(component, shot, tstart, tend, dt, raw_source=raw_path)
        )

    # This shot's own era may not wire every channel (the 2022-04-23 block
    # runs seven, not eight); use the same effective per-shot field list the
    # mapper reads, not the unfiltered default, so an intentionally-absent
    # channel is not reported as missing.
    impa_fields = sorted(impa_expected_fields(shot))
    missing_impa_channels = sorted(
        field for field in impa_fields if field not in archived_fields
    )
    if "magnetics" in ods:
        run_component(
            "impa",
            ("magnetics",),
            _map_impa,
            component_status="partial" if missing_impa_channels else "success",
            missing_channels=missing_impa_channels,
        )
        if impa_status:
            statuses["impa"].update(
                {
                    "calibration_status": impa_status.get("status"),
                    "checks": impa_status.get("checks", {}),
                    "reasons": impa_status.get("reasons", []),
                    "geometry_method": impa_status.get("geometry_method"),
                    "r0": impa_status.get("r0"),
                    "geometry_nrmse": impa_status.get("geometry_nrmse"),
                    "calibration_window": impa_status.get("provenance", {}).get(
                        "calibration_window"
                    ),
                }
            )
            # A rejected self-calibration is a real quality outcome, not a
            # successful mapping.
            if impa_status.get("status") == "invalid":
                statuses["impa"]["status"] = "partial"
    else:
        statuses["impa"] = {
            "status": "unavailable",
            "reason": "magnetics IDS is unavailable, so IMPA channels cannot be appended",
            "missing_channels": missing_impa_channels,
        }

    time_grid = _validate_diagnostics_time_coordinates(
        ods, processed_time, tstart=tstart, tend=tend, dt=dt
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
        "time_grid": time_grid,
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
    workdir: str | Path | None = None,
    module_workdirs: Mapping[tuple[str, int], str | Path] | None = None,
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

    times_seconds = [float(t) / 1000.0 for t in time_values]
    ods["mhd_linear"]["ids_properties"]["homogeneous_time"] = 1
    ods["mhd_linear"]["time"] = times_seconds

    # Lay the whole (time, n_tor) grid out before any solver runs, so the IDS
    # is dense on both axes regardless of which cells succeed: every requested
    # time slice exists, every requested mode holds the same array position in
    # each of them, and each entry states its own `n_tor`. Cells no solver
    # fills keep only that `n_tor` -- never a fabricated payload -- and the
    # slice's negative `code.output_flag` says the result is not usable.
    from vaft.machine_mapping.mhd_linear import (
        ensure_toroidal_mode_grid,
        initialize_output_flags,
    )

    mode_grid = [int(mode) for mode in modes]
    for time_slice in range(len(time_values)):
        ensure_toroidal_mode_grid(ods, time_slice, mode_grid)
    initialize_output_flags(ods, "mhd_linear", len(time_values))

    modules_modes: dict[str, Any] = {}
    inputs_hashes: dict[str, str] = {}
    if workdir is None and not module_workdirs:
        raise ValueError("workdir or module_workdirs is required")
    workdir_path = Path(workdir) if workdir is not None else None
    for time_slice, time_ms in enumerate(time_values):
        for module in modules:
            for mode in modes:
                key = f"t={time_ms}/{module}/n={mode}"
                cell_root = (
                    Path(module_workdirs[(module, mode)])
                    if module_workdirs and (module, mode) in module_workdirs
                    else workdir_path
                )
                if cell_root is None:
                    modules_modes[key] = {"status": "missing", "reason": "no work directory registered"}
                    continue
                run_dir = gpec_runtime.module_dir(cell_root, time_ms, module, mode)
                if not run_dir.is_dir():
                    modules_modes[key] = {"status": "missing", "reason": f"run directory not found: {run_dir}"}
                    continue
                try:
                    extras = mhd_linear_mapper(
                        ods,
                        str(run_dir),
                        {"time_slice": time_slice, "module": module, "modes": mode_grid},
                    )
                except Exception as exc:
                    modules_modes[key] = {"status": "failed", "reason": str(exc)}
                    continue
                if not extras:
                    modules_modes[key] = {"status": "no_output", "reason": "no matching .nc output found"}
                    continue
                modules_modes[key] = {"status": "success", "modes": extras}
                for nc_path in sorted(run_dir.glob("*.nc")):
                    inputs_hashes[f"{key}/{nc_path.name}"] = sha256_file(nc_path)

    # `ntms` carries RDCON/STRIDE's classical Delta-prime (mhd_linear has no
    # field for it -- see vaft.machine_mapping.mhd_linear). Only give it a time
    # base if some cell actually populated it: a DCON-only run would otherwise
    # be left with an `ntms.time` vector and no `time_slice` entries at all,
    # which is a length mismatch under homogeneous_time=1. When it *is*
    # populated, the AOS is padded out to the full time base so every declared
    # time has a slice, empty or not.
    # `ntms`'s time axis is made dense the same way `mhd_linear`'s is. Its mode
    # axis deliberately is not: an `ntms.mode` entry is one *rational surface*
    # (an (m, n) pair the solver locates in the equilibrium), not a requested
    # toroidal mode, so there is no caller-supplied grid to pad it against --
    # how many surfaces exist is itself a result. Slices with no surfaces stay
    # empty and are marked unusable by the negative output flag.
    if "ntms.time_slice" in ods and len(ods["ntms.time_slice"]):
        for index in range(len(ods["ntms.time_slice"]), len(times_seconds)):
            ods["ntms"]["time_slice"][index]
        ods["ntms"]["ids_properties"]["homogeneous_time"] = 1
        ods["ntms"]["time"] = times_seconds
        initialize_output_flags(ods, "ntms", len(times_seconds))

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
    *,
    shot: int,
    output: str | Path,
    source: str | Path | None = None,
    max_retries: int = 3,
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
            retry_options = (
                {} if int(max_retries) == 3 else {"max_retries": int(max_retries)}
            )
            if not raw_db.dump_all_raw_signals_for_shot(
                int(shot), str(dump_path), **retry_options
            ):
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

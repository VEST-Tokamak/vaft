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
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from omas import ODS

from vaft.database import raw as raw_db
from vaft.process.signal_processing import SignalRepairError
from vaft.database._local import load_ods
from vaft.data.resources import data_path
from vaft.machine_mapping.barometry import barometry
from vaft.machine_mapping.dataset_description import dataset_description
from vaft.machine_mapping.em_coupling import DEFAULT_VERSIONED_COUPLING, em_coupling
from vaft.machine_mapping.impa import (
    impa as impa_mapper,
    impa_expected_fields,
    impa_probe_indices,
    resolve_impa_config,
)
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
from vaft.machine_mapping.utils import (
    DiagnosticsTimePolicy,
    DiagnosticsTimePolicyTable,
    get_path,
    path_exists,
    resolve_diagnostics_time_policies,
)
from vaft.omas import save
from vaft.validation.imas import is_condemned_channel
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
    # Prefix the era; keep the mapper's reciprocity/provenance clause so a
    # product says which coupling asset it carries (issues #347/#373).
    mapper_comment = str(ods["em_coupling.ids_properties.comment"])
    _, _, reciprocity = mapper_comment.partition("; mutual_passive_passive ")
    ods["em_coupling.ids_properties.comment"] = (
        f"VEST electromagnetic coupling; machine era {era.name}; "
        f"PF geometry {era.pf_geometry}"
        + (f"; mutual_passive_passive {reciprocity}" if reciprocity else "")
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
    """Build one processed grid for the diagnostics product.

    Diagnostics windows are always half-open: ``tstart <= t < tend``.  Native
    acquisition timebases are handled explicitly by their mapper and are not
    selected by overloading a non-positive ``dt`` value.

    Since issue #244 the product carries more than one such grid -- the short
    plasma-analysis window and the full-discharge window -- so this builds *a*
    grid, not *the* grid.
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


def _policy_time(policy: DiagnosticsTimePolicy) -> np.ndarray:
    """Build the processed grid for one time policy (issue #244).

    Every policy uses the same half-open convention, so the analysis window and
    the full-discharge window differ only in their numbers.
    """
    return _canonical_diagnostics_time(policy.tstart, policy.tend, policy.dt)


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


def _component_grid(
    grids: Mapping[str, np.ndarray], component: str
) -> np.ndarray | None:
    grid = grids.get(component)
    if grid is None:
        return None
    return np.asarray(grid, dtype=float).reshape(-1)


def _realized_window(grid: np.ndarray) -> dict[str, Any]:
    """Describe one realized coordinate: span, cadence, and sample count."""
    grid = np.asarray(grid, dtype=float).reshape(-1)
    realized_dt = float(np.median(np.diff(grid))) if grid.size > 1 else None
    return {
        "realized_start": float(grid[0]) if grid.size else None,
        "realized_end_exclusive": (
            float(grid[0] + grid.size * realized_dt) if grid.size and realized_dt else None
        ),
        "realized_dt": realized_dt,
        "sample_count": int(grid.size),
    }


def _validate_realized_axis(
    component: str, grid: np.ndarray, policy: DiagnosticsTimePolicy
) -> None:
    """Check a realized coordinate against the policy that asked for it.

    Components given an explicit target axis satisfy this trivially.  It earns
    its keep for the full-window components, whose axis is built inside the
    mapper and read back out of the product: without this the only comparison
    left would be that axis against itself, and a mapper emitting the wrong
    cadence -- or running past the window -- would validate cleanly.
    """
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if grid.size == 0:
        return
    if grid.size > 1:
        spacing = np.diff(grid)
        if not np.allclose(spacing, policy.dt, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"{component} time is not uniform at the {policy.name} cadence "
                f"{policy.dt}: realized spacing spans "
                f"[{float(spacing.min())}, {float(spacing.max())}]"
            )
    # Clipping may narrow the span but must never widen it: the axis has to
    # stay inside the half-open window the policy asked for.
    if grid[0] < policy.tstart - 0.5 * policy.dt:
        raise ValueError(
            f"{component} time starts at {float(grid[0])}, before the "
            f"{policy.name} window start {policy.tstart}"
        )
    if grid[-1] >= policy.tend:
        raise ValueError(
            f"{component} time reaches {float(grid[-1])}, at or past the "
            f"{policy.name} window's exclusive end {policy.tend}"
        )


def _validate_diagnostics_time_coordinates(
    ods: ODS,
    grids: Mapping[str, np.ndarray],
    *,
    policies: DiagnosticsTimePolicyTable,
) -> dict[str, Any]:
    """Validate realized diagnostic coordinates and return manifest metadata.

    Each component is checked against *its own* time policy's grid.  A single
    ODS-wide time axis is not required and must not be imposed: TF and
    barometry carry the full discharge history while equilibrium magnetics
    carries the short analysis window, and both are valid at once (issue #244).
    """
    present_ids = set(ods.keys())
    magnetics_policy = policies.get("magnetics") or policies.default
    magnetics_grid = _component_grid(grids, "magnetics")

    for component, grid in grids.items():
        grid = np.asarray(grid, dtype=float).reshape(-1)
        if grid.size > 1 and np.any(np.diff(grid) <= 0.0):
            raise ValueError(f"{component} diagnostics time must be strictly monotonic")
        _validate_realized_axis(component, grid, policies[component])

    # Each canonical coordinate belongs to the component that produced it, so
    # it is compared against that component's grid rather than a shared one.
    canonical_paths = (
        ("pf_active.time", "pf_active"),
        ("spectrometer_uv.time", "spectrometer_uv"),
        ("tf.time", "tf"),
        ("magnetics.time", "magnetics"),
        ("barometry.gauge.0.pressure.time", "barometry"),
    )
    for path, component in canonical_paths:
        if path.split(".", 1)[0] not in present_ids:
            continue
        grid = _component_grid(grids, component)
        if grid is None or not path_exists(ods, path):
            continue
        actual = np.asarray(get_path(ods, path), dtype=float).reshape(-1)
        if not _time_equal(actual, grid):
            raise ValueError(
                f"{path} does not use the {policies[component].name} diagnostics grid"
            )

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
    if "magnetics" in present_ids and magnetics_grid is not None:
        for index in range(len(ods["magnetics.flux_loop"])):
            base = f"magnetics.flux_loop.{index}.flux"
            _validate_time_data_pair(ods, f"{base}.time", f"{base}.data")
            _validate_data_on_grid(ods, f"{base}.data", magnetics_grid)
            if (
                path_exists(ods, f"{base}.data")
                and np.asarray(get_path(ods, f"{base}.data")).size > 0
                and path_exists(ods, f"{base}.time")
                and not _time_equal(
                np.asarray(get_path(ods, f"{base}.time"), dtype=float).reshape(-1),
                magnetics_grid,
                )
            ):
                raise ValueError(f"{base}.time does not use the magnetics diagnostics grid")
        for index in range(len(ods["magnetics.b_field_pol_probe"])):
            base = f"magnetics.b_field_pol_probe.{index}.field"
            _validate_time_data_pair(ods, f"{base}.time", f"{base}.data")
            _validate_data_on_grid(ods, f"{base}.data", magnetics_grid)
            if (
                path_exists(ods, f"{base}.data")
                and np.asarray(get_path(ods, f"{base}.data")).size > 0
                and path_exists(ods, f"{base}.time")
                and not _time_equal(
                np.asarray(get_path(ods, f"{base}.time"), dtype=float).reshape(-1),
                magnetics_grid,
                )
            ):
                raise ValueError(f"{base}.time does not use the magnetics diagnostics grid")

    native_paths: list[str] = []
    native_time_metadata: list[dict[str, Any]] = []
    native_flux_loop_metadata: list[dict[str, Any]] = []
    if "magnetics" in present_ids:
        tstart, tend = magnetics_policy.tstart, magnetics_policy.tend
        root_time = np.asarray(get_path(ods, "magnetics.time"), dtype=float).reshape(-1)
        for index in range(len(ods["magnetics.b_field_pol_probe"])):
            base = f"magnetics.b_field_pol_probe.{index}.voltage"
            time_path, data_path = f"{base}.time", f"{base}.data"
            _validate_time_data_pair(ods, time_path, data_path)
            if not path_exists(ods, time_path) or not path_exists(ods, data_path):
                continue
            # Optional late-shot fluctuation-Mirnov channels are represented
            # by empty waveforms when their DAQ field is unavailable.  IMAS
            # can expose the corresponding unset time leaf as its scalar
            # default, so decide whether a native coordinate exists from the
            # waveform first rather than validating that placeholder.
            voltage_data = np.asarray(get_path(ods, data_path)).reshape(-1)
            if is_condemned_channel(ods, base) or voltage_data.size == 0:
                continue
            voltage_time = np.asarray(get_path(ods, time_path), dtype=float).reshape(-1)
            if voltage_time.size == 0:
                continue
            if np.any(voltage_time < tstart) or np.any(voltage_time >= tend):
                raise ValueError(f"{time_path} is outside the magnetics analysis window")
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
                raise ValueError(f"{time_path} is outside the magnetics analysis window")
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

    # Per-component provenance: what each policy asked for, what the source
    # actually supported, and whether the difference cost any coverage.
    component_metadata: dict[str, Any] = {}
    for component, grid in sorted(grids.items()):
        policy = policies[component]
        realized = _realized_window(np.asarray(grid, dtype=float).reshape(-1))
        requested = _policy_time(policy)
        # Counted in samples rather than compared as floats: a record that ends
        # one sample short of the nominal window is the DAQ's own half-open
        # convention (25 000 slow samples span 0 to 0.99996 s), not coverage
        # the source failed to provide.
        missing = int(requested.size) - int(realized["sample_count"])
        component_metadata[component] = {
            "policy": policy.name,
            "requested_start": policy.tstart,
            "requested_end": policy.tend,
            "requested_dt": policy.dt,
            "requested_sample_count": int(requested.size),
            **realized,
            "missing_samples": max(missing, 0),
            "source_clipping": missing > 1,
        }

    default_policy = policies.default
    default_time = _policy_time(default_policy)
    realized_dt = (
        float(np.median(np.diff(default_time))) if default_time.size > 1 else None
    )
    if realized_dt is not None and not np.isclose(
        realized_dt, default_policy.dt, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Canonical diagnostics time does not realize the configured dt")
    return {
        # The unprefixed keys describe the default (plasma-analysis) window,
        # which is what this manifest section meant before issue #244 added
        # per-component policies below.
        "requested_start": float(default_policy.tstart),
        "requested_end": float(default_policy.tend),
        "requested_dt": float(default_policy.dt),
        "processed_start": float(default_time[0]),
        "processed_end_exclusive": float(
            default_time[0] + default_time.size * (realized_dt or 0.0)
        ),
        "processed_sample_count": int(default_time.size),
        "realized_dt": realized_dt,
        "processed_time_clipped": False,
        "source_clipping": any(
            entry["source_clipping"] for entry in component_metadata.values()
        ),
        "magnetics_homogeneous_time": int(ods["magnetics.ids_properties.homogeneous_time"])
        if "magnetics" in present_ids
        else None,
        "native_time_paths": native_paths,
        # Quantity-specific: `native_mirnov` stays probe-only so consumers that
        # index it keep reading Mirnov sampling, while flux-loop terminal
        # voltage (issue #209) reports under its own key.
        "native_mirnov": native_time_metadata,
        "native_flux_loop_voltage": native_flux_loop_metadata,
        "default_policy": default_policy.name,
        "policies": {
            name: {"tstart": window.tstart, "tend": window.tend, "dt": window.dt}
            for name, window in sorted(policies.windows.items())
        },
        "components": component_metadata,
    }


def build_diagnostics_ods(
    *,
    shot: int,
    raw_source: str | Path,
    static_ods: str | Path,
    tstart: float | None = None,
    tend: float | None = None,
    dt: float | None = None,
    run: int = 1,
    vest_magnetics_processing: dict[str, Any] | None = None,
    time_policies: Mapping[str, Any] | None = None,
) -> tuple[ODS, dict[str, Any]]:
    """Build independent diagnostic IDSs without losing valid siblings.

    ``tstart``/``tend``/``dt`` retune the plasma-analysis window, which is what
    the equilibrium magnetics need; left unset, that window comes from the
    configured policy document so `vest.yaml` stays the single source of truth
    for it.  Components whose physics spans the whole discharge -- TF,
    barometry, and EC power once #165 lands -- follow their own configured
    window instead, so the product carries more than one temporal coverage
    (issue #244).  ``time_policies`` overrides the whole policy document.
    """
    raw_path = Path(raw_source)
    static_path = Path(static_ods)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dump not found: {raw_path}")
    if not static_path.exists():
        raise FileNotFoundError(f"Static ODS not found: {static_path}")

    shot = int(shot)
    policies = resolve_diagnostics_time_policies(
        analysis_override={"tstart": tstart, "tend": tend, "dt": dt},
        overrides=time_policies,
    )
    # `tstart`/`tend`/`dt` may be unset; report what was actually applied.
    analysis = policies.default
    # One grid per distinct window, built once and shared by every component
    # that uses it.  Components whose window may exceed their source coverage
    # realize their own axis inside the mapper and are filled in below.
    policy_grids: dict[str, np.ndarray] = {}
    grids: dict[str, np.ndarray] = {}
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

    def policy_grid(component: str) -> np.ndarray:
        """The exact grid for a component whose window is inside its source."""
        policy = policies[component]
        if policy.name not in policy_grids:
            policy_grids[policy.name] = _policy_time(policy)
        return policy_grids[policy.name]

    def record_realized_grid(component: str, ids_name: str, path: str) -> None:
        """Record the axis a full-window component actually realized.

        These mappers clip their window to real source coverage rather than
        extrapolating, so the realized axis -- not the requested one -- is what
        validation and the manifest must describe.

        The `ods.keys()` pre-guard this used to need is gone: `path_exists` no
        longer materializes what it probes, so asking about a path under an
        absent IDS is safe on its own (issue #118).
        """
        if statuses.get(component, {}).get("status", "unavailable") == "unavailable":
            return
        if not path_exists(ods, path):
            return
        grids[component] = np.asarray(get_path(ods, path), dtype=float).reshape(-1)

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
        except (
            raw_db.RawSignalUnavailableError,
            FileNotFoundError,
            SignalRepairError,
        ) as error:
            # A fully saturated waveform is a property of the shot, not a fault
            # in the run: refusing to reconstruct one is correct, but it makes
            # that component unavailable rather than the whole stage. Letting it
            # escape cost 88 shots their entire diagnostics product for one
            # clipped diamagnetic-flux signal.
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

    pf_active_policy = policies["pf_active"]
    run_component(
        "pf_active",
        ("pf_active",),
        lambda component: (
            _copy_ids(component, static, ("pf_active",)),
            vfit_pf_active_dynamic(
                component,
                shot,
                pf_active_policy.tstart,
                pf_active_policy.tend,
                pf_active_policy.dt,
                raw_source=raw_path,
                target_time=policy_grid("pf_active"),
            ),
        ),
        disabled_channels=_disabled_pf_coils(),
    )
    grids["pf_active"] = policy_grid("pf_active")
    uv_policy = policies["spectrometer_uv"]
    run_component(
        "spectrometer_uv",
        ("spectrometer_uv",),
        lambda component: spectrometer_uv(
            component,
            shot,
            uv_policy.tstart,
            uv_policy.tend,
            uv_policy.dt,
            raw_source=raw_path,
            target_time=policy_grid("spectrometer_uv"),
        ),
    )
    grids["spectrometer_uv"] = policy_grid("spectrometer_uv")
    # barometry and tf follow the full-discharge policy: the prefill history
    # before breakdown and the TF ramp-up/ramp-down are physical information
    # that the short analysis window would crop away.  They are given no
    # explicit target axis so the mapper clips their window to whatever the
    # source actually recorded instead of extrapolating past it.
    barometry_policy = policies["barometry"]
    run_component(
        "barometry",
        ("barometry",),
        lambda component: barometry(
            component,
            shot,
            barometry_policy.tstart,
            barometry_policy.tend,
            barometry_policy.dt,
            raw_source=raw_path,
        ),
    )
    record_realized_grid("barometry", "barometry", "barometry.gauge.0.pressure.time")
    langmuir_policy = policies["langmuir_probes"]
    run_component(
        "langmuir_probes",
        ("langmuir_probes",),
        lambda component: langmuir_probes(
            component,
            shot,
            langmuir_policy.tstart,
            langmuir_policy.tend,
            langmuir_policy.dt,
            raw_source=raw_path,
        ),
    )
    grids["langmuir_probes"] = policy_grid("langmuir_probes")
    tf_policy = policies["tf"]
    run_component(
        "tf",
        ("tf",),
        lambda component: (
            _copy_ids(component, static, ("tf",)),
            vfit_tf_dynamic(
                component,
                shot,
                tf_policy.tstart,
                tf_policy.tend,
                tf_policy.dt,
                raw_source=raw_path,
            ),
        ),
    )
    record_realized_grid("tf", "tf", "tf.time")
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
    magnetics_policy = policies["magnetics"]
    run_component(
        "magnetics",
        ("magnetics",),
        lambda component: (
            _copy_ids(component, static, ("magnetics",)),
            vfit_magnetics_dynamic(
                component,
                shot,
                magnetics_policy.tstart,
                magnetics_policy.tend,
                magnetics_policy.dt,
                processing_config=processing,
                raw_source=raw_path,
                target_time=policy_grid("magnetics"),
            ),
        ),
        processing="VestMagneticsProcessingConfig",
        component_status="partial" if missing_magnetics_channels else "success",
        missing_channels=missing_magnetics_channels,
    )
    grids["magnetics"] = policy_grid("magnetics")

    # A component whose raw signals were unavailable produced no coordinate,
    # so it must not appear in the realized-coverage record either.
    grids = {
        name: grid
        for name, grid in grids.items()
        if statuses.get(name, {}).get("status", "unavailable") != "unavailable"
    }
    time_grid = _validate_diagnostics_time_coordinates(ods, grids, policies=policies)

    # Magnetics signal quality (issue #189), projected into the native IDS
    # validity nodes so every downstream stage reads one answer instead of
    # rediscovering sensor health.  It runs here rather than in the mapper
    # because mapping is a transformation layer and this is an assessment of
    # what it produced (#253 §2).  Report-only: nothing is dropped or
    # rewritten, and a channel with no waveform is left untouched rather than
    # asserted invalid.
    magnetics_quality: dict[str, Any] = {}
    if "magnetics" in ods:
        from vaft.validation.magnetics import (
            magnetics_quality_metrics,
            project_validity,
            validate_magnetics_signals,
        )

        quality_report = validate_magnetics_signals(ods)
        project_validity(ods, quality_report)
        magnetics_quality = magnetics_quality_metrics(ods, quality_report)

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
            # These stay the plasma-analysis window; per-component coverage is
            # reported under `time_grid` (issue #244).
            "tstart": float(analysis.tstart),
            "tend": float(analysis.tend),
            "dt": float(analysis.dt),
            "run": int(run),
            "vest_magnetics_processing": vest_magnetics_processing or {},
            "time_policies": time_policies or {},
        },
        "time_grid": time_grid,
        "magnetics_quality": magnetics_quality,
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


def _impa_realized_grid(ods: ODS) -> np.ndarray | None:
    """Return the axis the IMPA channels actually realized, if any.

    The mapper clips its window to real source coverage rather than
    extrapolating, so the realized axis -- not the requested one -- is what the
    product's coordinate and manifest must describe.
    """
    for node in ("magnetics.b_field_tor_probe", "magnetics.b_field_pol_probe"):
        # Guarded: asking `impa_probe_indices` about an absent node would
        # materialize it, and this ODS is the product that gets written.
        if not path_exists(ods, node):
            continue
        for index in impa_probe_indices(ods, node):
            path = f"{node}.{index}.field.time"
            if path_exists(ods, path):
                return np.asarray(get_path(ods, path), dtype=float).reshape(-1)
    return None


def _impa_configuration_sha256(config: Mapping[str, Any]) -> str:
    """Hash the shot's resolved IMPA machine description.

    The IMPA product is published into its own source with no static product
    beside it, so the wiring and geometry priors it was built from have to be
    identifiable from the product itself (issue #305).
    """
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def build_impa_ods(
    *,
    shot: int,
    raw_source: str | Path,
    tstart: float | None = None,
    tend: float | None = None,
    dt: float | None = None,
    run: int = 1,
    time_policies: Mapping[str, Any] | None = None,
) -> tuple[ODS, dict[str, Any]]:
    """Build the standalone IMPA product for one shot (issue #305).

    IMPA is an insertable, campaign-dependent diagnostic: raw fields may exist
    while the array is withdrawn, its geometry is self-calibrated per shot, and
    its Bz sensors are still being qualified (#154, #304).  Those are poor
    invariants for the baseline `magnetics` product, so the array gets its own
    stage and its own HSDS source instead of being appended to the diagnostics
    one.

    The product is self-contained rather than a copy of the baseline shot: the
    mapper appends after whatever probes are already present, so on a fresh ODS
    the Hall channels land at ``b_field_tor_probe.0..n`` and the vertical-field
    sensors at ``b_field_pol_probe.0..n``, and nothing here reads the
    diagnostics product.  Composing the two for analysis is explicit and lives
    in :mod:`vaft.database.composition`.
    """
    raw_path = Path(raw_source)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dump not found: {raw_path}")

    shot = int(shot)
    policies = resolve_diagnostics_time_policies(
        analysis_override={"tstart": tstart, "tend": tend, "dt": dt},
        overrides=time_policies,
    )
    policy = policies["impa"]
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
            "description": f"VEST IMPA; machine era {era.name}",
            "pulse_datetime": _archived_pulse_datetime(raw_path),
        },
    )

    config = resolve_impa_config(shot)
    # This shot's own era may not wire every channel (the 2022-04-23 block runs
    # seven, not eight), so an intentionally-absent channel is not missing.
    expected_fields = sorted(impa_expected_fields(shot, config))
    archived_fields = _archived_field_codes(raw_path)
    missing_channels = sorted(
        field for field in expected_fields if field not in archived_fields
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "impa",
        "shot": shot,
        "machine_version": era.name,
        "status": "unavailable",
        "input": {
            "raw_sha256": sha256_file(raw_path),
            "impa_configuration_sha256": _impa_configuration_sha256(config),
        },
        "configuration": {
            "tstart": float(policy.tstart),
            "tend": float(policy.tend),
            "dt": float(policy.dt),
            "run": int(run),
            "time_policies": time_policies or {},
            "expected_fields": expected_fields,
        },
        "time_grid": {},
        "calibration": {},
        "channel_status": {},
        "quality_summary": {
            "missing": [f"impa:field-{field}" for field in missing_channels],
            "repaired": [],
            "disabled": [],
            "rejected": [],
            "unavailable": [],
        },
    }

    # No wired channel was archived at all: the array was not recording for this
    # shot, which is a normal outcome for an insertable diagnostic and not a
    # fault in the run.
    if not any(field in archived_fields for field in expected_fields):
        manifest["quality_summary"]["unavailable"] = ["impa"]
        manifest["reason"] = "no wired IMPA channel is archived for this shot"
        return ods, manifest

    try:
        status = impa_mapper(
            ods, shot, policy.tstart, policy.tend, policy.dt, raw_source=raw_path
        )
    except (
        raw_db.RawSignalUnavailableError,
        FileNotFoundError,
        SignalRepairError,
    ) as error:
        manifest["quality_summary"]["unavailable"] = ["impa"]
        manifest["reason"] = str(error)
        return ods, manifest

    grid = _impa_realized_grid(ods)
    if grid is not None:
        if grid.size > 1 and np.any(np.diff(grid) <= 0.0):
            raise ValueError("impa diagnostics time must be strictly monotonic")
        _validate_realized_axis("impa", grid, policy)
        ods["magnetics.time"] = grid
        ods["magnetics.ids_properties.homogeneous_time"] = 1
        manifest["time_grid"] = {"impa": _realized_window(grid)}
    else:
        # No channel produced a waveform, so the product carries geometry and
        # validity only. Per the DD's homogeneous_time rule an IDS with no
        # `.time` node is 2, not unset -- the same statement `build_static_ods`
        # makes for the static product.
        ods["magnetics.ids_properties.homogeneous_time"] = 2

    calibration_status = status.get("status")
    manifest["calibration"] = {
        "status": calibration_status,
        "checks": status.get("checks", {}),
        "reasons": status.get("reasons", []),
        "orientation": status.get("orientation"),
        "ids_node": status.get("ids_node"),
        "geometry_method": status.get("geometry_method"),
        "r0": status.get("r0"),
        "geometry_nrmse": status.get("geometry_nrmse"),
        "fitted_pitch": status.get("fitted_pitch"),
        "incident_angle_deg": status.get("incident_angle_deg"),
        "calibration_window": status.get("provenance", {}).get("calibration_window"),
        "provenance": status.get("provenance", {}),
    }
    manifest["channel_status"] = {
        "channels": status.get("channels", {}),
        "bz_channels": status.get("bz_channels", {}),
        "missing_channels": missing_channels,
    }

    # A rejected self-calibration is a real quality outcome, and the product it
    # produced is kept locally with the verdict on it -- but it is not published
    # (issue #305): `rejected` is not a replicable status, so the IMPA source
    # never presents an unqualified geometry as a measurement.
    if calibration_status == "invalid":
        manifest["status"] = "rejected"
        manifest["quality_summary"]["rejected"] = ["impa"]
    elif calibration_status == "warning" or missing_channels:
        manifest["status"] = "partial"
    else:
        manifest["status"] = "success"
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


def build_gpec_ideal_ods(
    *,
    shot: int,
    time_values: Sequence[int | str],
    workdir: str | Path | None = None,
    mode_workdirs: Mapping[int, str | Path] | None = None,
    modes: Sequence[int] = (1,),
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Build ``mhd_linear`` + ``coils_non_axisymmetric`` from ideal-GPEC runs.

    Mirrors :func:`build_mhd_linear_ods` for the ``gpec`` module: one
    ``mhd_linear.time_slice`` per entry of ``time_values`` (milliseconds), a
    dense ``toroidal_mode`` grid over ``modes``, run directories resolved via
    the adapter's own directory grammar
    (``{workdir}/{time_label}/gpec/nn={mode}/``).  The shot/time identity is
    written from these arguments -- GPEC's own netCDF ``shot``/``time``
    attributes are 0 for VEST runs and are never trusted.

    The canonical static 3D coil geometry is always written; each cell's
    ``coil.in`` is read back so the run's excitation travels with the field
    it produced (currents matched by stable identifier, zero-current sets
    untouched).
    """
    from vaft.code.gpec import _runtime as gpec_runtime
    from vaft.code.gpec import read_coil_in
    from vaft.machine_mapping.coils_non_axisymmetric_geometry import (
        VEST_3D_COIL_SETS,
        CoilExcitation,
    )
    from vaft.machine_mapping.coils_non_axisymmetric import (
        apply_coil_excitation,
        coils_non_axisymmetric,
    )
    from vaft.machine_mapping.gpec_ideal import gpec_ideal as gpec_ideal_mapper
    from vaft.machine_mapping.mhd_linear import (
        ensure_toroidal_mode_grid,
        initialize_output_flags,
    )

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
            "description": f"VEST ideal-GPEC 3D response; machine era {era.name}",
        },
    )

    times_seconds = [float(t) / 1000.0 for t in time_values]
    ods["mhd_linear"]["ids_properties"]["homogeneous_time"] = 1
    ods["mhd_linear"]["time"] = times_seconds

    mode_grid = [int(mode) for mode in modes]
    for time_slice in range(len(time_values)):
        ensure_toroidal_mode_grid(ods, time_slice, mode_grid)
    initialize_output_flags(ods, "mhd_linear", len(time_values))

    coils_non_axisymmetric(ods)

    if workdir is None and not mode_workdirs:
        raise ValueError("workdir or mode_workdirs is required")
    workdir_path = Path(workdir) if workdir is not None else None
    cells: dict[str, Any] = {}
    inputs_hashes: dict[str, str] = {}
    excitations: dict[str, CoilExcitation] = {}
    for time_slice, time_ms in enumerate(time_values):
        for mode in mode_grid:
            key = f"t={time_ms}/gpec/n={mode}"
            cell_root = (
                Path(mode_workdirs[mode])
                if mode_workdirs and mode in mode_workdirs
                else workdir_path
            )
            if cell_root is None:
                cells[key] = {"status": "missing", "reason": "no work directory registered"}
                continue
            run_dir = gpec_runtime.module_dir(cell_root, time_ms, "gpec", mode)
            if not run_dir.is_dir():
                cells[key] = {"status": "missing", "reason": f"run directory not found: {run_dir}"}
                continue
            try:
                extras = gpec_ideal_mapper(
                    ods,
                    str(run_dir),
                    {
                        "time_slice": time_slice,
                        "mode": mode,
                        "modes": mode_grid,
                        "time_s": times_seconds[time_slice],
                    },
                )
            except Exception as exc:
                cells[key] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                continue
            cells[key] = {"status": "success", "modes": extras}
            for nc_path in sorted(run_dir.glob("*.nc")):
                inputs_hashes[f"{key}/{nc_path.name}"] = sha256_file(nc_path)
            coil_in = run_dir / "coil.in"
            if coil_in.is_file():
                inputs_hashes[f"{key}/coil.in"] = sha256_file(coil_in)
                for spec in read_coil_in(coil_in):
                    if spec.name in VEST_3D_COIL_SETS and spec.currents_a:
                        excitations[spec.name] = CoilExcitation(
                            coil_set=spec.name, currents_a=spec.currents_a
                        )

    if excitations:
        apply_coil_excitation(
            ods,
            list(excitations.values()),
            time_s=times_seconds[0] if times_seconds else 0.0,
        )

    status = "success" if any(cell["status"] == "success" for cell in cells.values()) else "empty"
    manifest = {
        "schema_version": 1,
        "stage": "gpec_ideal",
        "shot": shot,
        "time_values": list(time_values),
        "machine_version": era.name,
        "status": status,
        "input": inputs_hashes,
        "modules_modes": cells,
    }
    return ods, manifest


def _external_profile_manifest(stage: str, shot: int, run: int, era_name: str) -> dict[str, Any]:
    """The manifest shape shared by the externally-uploaded profile stages."""
    return {
        "schema_version": 1,
        "stage": stage,
        "shot": int(shot),
        "machine_version": era_name,
        # Absence is the default outcome. Most VEST shots carry no external
        # profile upload at all, and that is a normal result rather than a
        # failure, so the stage starts unavailable and is promoted only once a
        # mapper has actually written channels.
        "status": "unavailable",
        "input": {},
        "configuration": {"run": int(run)},
        "quality_summary": {
            "missing": [],
            "repaired": [],
            "disabled": [],
            "rejected": [],
            "unavailable": [stage],
        },
    }


def build_thomson_ods(
    *,
    shot: int,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Build the standalone Thomson scattering product for one shot.

    The mapper resolves its own `.mat` file across the layouts VEST has used
    over the years, so an explicit ``mat_file`` is an override rather than a
    requirement.  A shot with no upload yields a provenance-only product whose
    manifest says ``unavailable``: the stage is optional, so that is recorded
    and replication skips it rather than failing the run.
    """
    from vaft.machine_mapping.thomson_scattering import thomson_scattering

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
            "description": f"VEST Thomson scattering; machine era {era.name}",
        },
    )
    manifest = _external_profile_manifest("thomson", shot, run, era.name)
    if mat_file is not None:
        manifest["input"]["mat_file"] = str(mat_file)

    try:
        thomson_scattering(ods, shot, data_root=data_root, mat_file=mat_file)
    except FileNotFoundError as error:
        # No upload for this shot. Distinguished from a malformed one: the
        # mapper raises FileNotFoundError only after exhausting every layout.
        manifest["error"] = f"{type(error).__name__}: {error}"
        return ods, manifest

    channels = len(ods["thomson_scattering.channel"]) if "thomson_scattering.channel" in ods else 0
    if channels:
        manifest["status"] = "success"
        manifest["quality_summary"]["unavailable"] = []
        manifest["channels"] = channels
    return ods, manifest


def build_ces_ods(
    *,
    shot: int,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
    options: str = "ces",
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Build the standalone charge-exchange product for one shot.

    ``options`` selects which external layout is read (``ces`` for
    ``CES_{shot}.mat``, ``ids`` for ``IDS_{shot}.mat``); both land in the same
    ``charge_exchange`` IDS.  Presence of this product is what later routes a
    shot's kinetic reconstruction to `kinetic-efit` rather than `electron-efit`,
    so an absent upload must be recorded rather than inferred.
    """
    from vaft.machine_mapping.charge_exchange import charge_exchange

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
            "description": f"VEST charge exchange; machine era {era.name}",
        },
    )
    manifest = _external_profile_manifest("ces", shot, run, era.name)
    manifest["configuration"]["options"] = options
    if mat_file is not None:
        manifest["input"]["mat_file"] = str(mat_file)

    try:
        charge_exchange(ods, shot, options=options, data_root=data_root, mat_file=mat_file)
    except FileNotFoundError as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        return ods, manifest

    channels = len(ods["charge_exchange.channel"]) if "charge_exchange.channel" in ods else 0
    if channels:
        manifest["status"] = "success"
        manifest["quality_summary"]["unavailable"] = []
        manifest["channels"] = channels
    return ods, manifest


#: How far a profile time may sit from the reconstructed slice it is mapped
#: against.  Tighter than the 1 ms the profile layer resolves its own samples
#: within (`vaft.process.profile`), because the failure modes are not
#: symmetric: a profile with no equilibrium is counted and visible, while one
#: mapped onto the wrong equilibrium is a number nobody can tell is wrong.
#: Refusing a 1 ms match costs a slice; accepting a misaligned one costs the
#: result's meaning.
EQUILIBRIUM_TIME_TOLERANCE_MS = 0.5

#: How CHEASE names the equilibrium it refined, per shot and millisecond.
#: EFIT's convention is a five-digit, zero-padded time, so the padding has to be
#: in the format spec rather than typed out as literal zeros -- a discharge at
#: or past 1000 ms would otherwise build a six-digit suffix that matches no file
#: and be indistinguishable from having no equilibrium at all.
GEQDSK_NAME_TEMPLATE = "g0{shot}.{time_ms:05d}"


def build_core_profiles_ods(
    *,
    shot: int,
    thomson_product: str | Path,
    ces_product: str | Path | None = None,
    efit_product: str | Path | None = None,
    equilibrium_tolerance_ms: float = EQUILIBRIUM_TIME_TOLERANCE_MS,
    geqdsk_dir: str | Path | None = None,
    geqdsk_template: str = GEQDSK_NAME_TEMPLATE,
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Map Thomson (and CES, when present) onto an equilibrium grid.

    One slice is built per Thomson time that has an equilibrium to map against.
    A time the ion diagnostic actually covers becomes a kinetic slice; every
    other time stays electron-only and is stripped of slice-total pressure,
    because a total computed with a phantom Ti=Te would read as measured to
    anything that later loads this product.

    The manifest records how many of each were built.  That count is what routes
    the reconstruction afterwards -- a product with no kinetic slice can only
    support the `electron-efit` lineage -- so it is written down rather than
    re-derived by each consumer.
    """
    import omas as _omas

    from vaft.code.efit import build_kinetic_core_profiles
    from vaft.data import read_geqdsk
    from vaft.data.eqdsk import from_equilibrium
    from vaft.process import profile as _profile
    from vaft.process.equilibrium import as_equilibrium

    shot = int(shot)
    era = machine_era_for_shot(shot)
    ods, _ = load_ods(Path(thomson_product))
    if ces_product is not None and Path(ces_product).exists():
        ion, _ = load_ods(Path(ces_product))
        if "charge_exchange" in ion:
            ods["charge_exchange"] = ion["charge_exchange"]

    manifest = _external_profile_manifest("core_profiles", shot, run, era.name)
    manifest["input"] = {
        "thomson_product": str(thomson_product),
        "ces_product": str(ces_product) if ces_product is not None else None,
        "efit_product": str(efit_product) if efit_product is not None else None,
        "geqdsk_dir": str(geqdsk_dir) if geqdsk_dir is not None else None,
    }
    manifest["configuration"]["equilibrium_tolerance_ms"] = float(equilibrium_tolerance_ms)

    if "thomson_scattering.time" not in ods:
        manifest["error"] = "the Thomson product carries no thomson_scattering.time"
        return ods, manifest

    # Test the full path, not the parent: reading `charge_exchange.channel`
    # off an ODS that has the IDS but no channels would materialize the node,
    # and `flat()` does not show an empty one, so the damage is silent.
    has_ion = (
        "charge_exchange.channel" in ods
        and len(ods["charge_exchange.channel"]) > 0
    )
    times_ms = np.asarray(ods["thomson_scattering.time"], dtype=float) * 1e3

    # A re-run must not leave a stale slice behind: the mix of kinetic and
    # electron-only slices depends on which CES product was available, and that
    # can change between runs.
    if "core_profiles" in ods:
        del ods["core_profiles"]

    # Preferred source of the psi map: the EFIT stage's own product, mapped per
    # Thomson time to its nearest reconstructed slice. `geqdsk_dir` remains for
    # a CHEASE directory that predates the stage, but it ties this stage to a
    # filesystem layout it does not own, and the kinetic stages already read the
    # equilibrium out of the product instead.
    solution = None
    eq_times_ms = None
    if efit_product is not None:
        # A path that does not resolve is a caller error, not an absent
        # equilibrium. Falling through to `geqdsk_dir` would report "no
        # equilibrium" for every time and never mention the typo.
        if not Path(efit_product).exists():
            manifest["error"] = f"the efit product is missing: {efit_product}"
            return ods, manifest
        solution, _ = load_ods(Path(efit_product))
        eq_times_ms = _time_array_ms(solution, "equilibrium.time")
        if eq_times_ms is None:
            # Present but unusable is not the same as absent, and it is the
            # common case: a shot whose EFIT stage is `no_output` -- 48224 --
            # produces exactly this. Reporting it as "no equilibrium at every
            # time" would blame the time bases for a product that was never
            # reconstructed.
            manifest["error"] = (
                f"the efit product carries no equilibrium.time: {efit_product}"
            )
            return ods, manifest

    n_kinetic = 0
    n_electron = 0
    missing_equilibrium: list[float] = []
    for time_ms in times_ms:
        geq = None
        if solution is not None:
            index = int(np.argmin(np.abs(eq_times_ms - float(time_ms))))
            # Nearest is not the same as close. On 48226 the Thomson window
            # (298-307 ms) and the reconstructed window (308-311 ms) do not
            # overlap at all, and an unguarded argmin mapped every profile onto
            # an equilibrium 7-10 ms away while reporting nothing missing.
            if abs(float(eq_times_ms[index]) - float(time_ms)) > equilibrium_tolerance_ms:
                geq = None
            else:
                try:
                    geq = from_equilibrium(as_equilibrium(solution, time_index=index))
                except Exception as error:  # noqa: BLE001 - one bad slice is a missing one
                    # Recorded once. "An unusable slice is a missing one" holds
                    # for one slice; when every slice fails for the same
                    # structural reason the count alone reads as a time-base
                    # mismatch rather than an unreadable equilibrium.
                    manifest.setdefault(
                        "equilibrium_error", f"{type(error).__name__}: {error}"
                    )
                    geq = None
        elif geqdsk_dir is not None:
            path = Path(geqdsk_dir) / geqdsk_template.format(
                shot=shot, time_ms=int(time_ms)
            )
            if path.exists():
                try:
                    geq = read_geqdsk(path)
                except Exception:  # noqa: BLE001 - a bad g-file is a missing one here
                    geq = None
        if geq is None:
            missing_equilibrium.append(round(float(time_ms), 3))
            continue

        built_kinetic = False
        if has_ion:
            try:
                # require_ion: only where the ion diagnostic actually covers this
                # time. Without it a time with no ion data would silently take
                # the Ti=Te fallback and be indistinguishable from a measured one.
                build_kinetic_core_profiles(
                    ods,
                    geq,
                    float(time_ms),
                    ion_index=0,
                    require_thomson=True,
                    require_ion=True,
                    ti_te_fallback=False,
                )
                built_kinetic = True
                n_kinetic += 1
            except Exception:  # noqa: BLE001 - no ion coverage here; fall back
                built_kinetic = False

        if not built_kinetic:
            try:
                mapped = _profile.equilibrium_mapping_thomson_scattering(ods, geq)
                n_e_fn, t_e_fn, *_ = _profile.profile_fitting_thomson_scattering(
                    ods,
                    float(time_ms),
                    mapped,
                    Te_order=2,
                    Ne_order=2,
                    fitting_function_te="polynomial",
                    fitting_function_ne="exponential",
                )
                _profile.core_profiles(
                    ods,
                    float(time_ms),
                    mapped,
                    n_e_fn,
                    t_e_fn,
                    geq=geq,
                    ti_te_fallback=False,
                )
                n_electron += 1
            except Exception:  # noqa: BLE001 - one failed fit is not a failed stage
                continue

    # Once, not per slice: the call takes the whole IDS, so running it inside
    # the loop reprocessed every slice built so far on each pass.
    try:
        _omas.omas_physics.core_profiles_pressures(ods, update=True)
    except Exception:  # noqa: BLE001 - pressures are derived, not measured
        pass

    _profile.strip_electron_only_pressure(ods)

    manifest["slices"] = {
        "kinetic": n_kinetic,
        "electron_only": n_electron,
        "no_equilibrium": len(missing_equilibrium),
    }
    # Named so the reconstruction stages can read their own eligibility off the
    # manifest instead of reloading and re-inspecting the product.
    manifest["supports_kinetic_efit"] = n_kinetic > 0
    if n_kinetic or n_electron:
        manifest["status"] = "success"
        manifest["quality_summary"]["unavailable"] = []
    else:
        manifest["quality_summary"]["missing"] = [
            f"core_profiles:t={t}ms" for t in missing_equilibrium[:20]
        ]
    return ods, manifest


def _time_array_ms(ods: ODS, path: str) -> np.ndarray | None:
    """A time array in milliseconds, or ``None`` when there is not one.

    Total where the obvious form is not: the path may be absent, may resolve
    through a scalar parent, or may itself be a bare float that ``len`` refuses
    -- and every caller here is an optional stage that must record rather than
    raise.
    """
    if not _path_present(ods, path):
        return None
    try:
        values = np.asarray(ods[path], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return values * 1e3 if values.size else None


def _path_present(ods: ODS, path: str) -> bool:
    """Whether ``path`` exists, answering False where ``in`` would raise.

    OMAS resolves a membership test segment by segment, so asking about a deep
    path whose parent turned out to be a scalar raises instead of returning
    False.  A reloaded product is exactly where that happens -- the shape it
    was saved with is not guaranteed to be the shape the caller assumes.
    """
    try:
        return path in ods
    except (AttributeError, TypeError, KeyError, IndexError, ValueError):
        return False


def _has_kinetic_slice(ods: ODS) -> bool:
    """Whether any core_profiles slice carries a measured ion temperature.

    The same predicate :func:`vaft.process.profile.strip_electron_only_pressure`
    uses to decide which slices may keep a total pressure, so the two cannot
    disagree about what counts as kinetic.
    """
    # `in` is not total over a reloaded product. OMAS walks the path segment by
    # segment, so a membership test whose parent resolves to a scalar raises
    # `AttributeError: 'float' object has no attribute 'omas_data'` rather than
    # answering False -- and `core_profiles` being present says nothing about
    # whether `profiles_1d` is a populated AoS.
    if not _path_present(ods, "core_profiles.profiles_1d"):
        return False
    try:
        count = len(ods["core_profiles.profiles_1d"])
    except TypeError:  # an AoS that is not a list is an AoS with no slices
        return False
    return any(
        _path_present(ods, f"core_profiles.profiles_1d.{index}.ion.0.temperature")
        for index in range(count)
    )


def build_kinetic_efit_ods(
    *,
    shot: int,
    stage: str,
    core_profiles_product: str | Path,
    constraints_product: str | Path,
    efit_product: str | Path,
    time_ms: float | None = None,
    equilibrium_tolerance_ms: float = EQUILIBRIUM_TIME_TOLERANCE_MS,
    workdir: str | Path | None = None,
    executable: str | None = None,
    encoding: str = "raw6",
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Reconstruct a kinetic-pressure equilibrium from products already on disk.

    Nothing is read from outside the FileDB.  The base magnetic kfile is built
    by :func:`vaft.code.efit.generate_kfile` -- the default path through
    ``prepare_kinetic_efit_inputs`` -- from the EFIT stage's own *constraints*
    product, and the R -> psi_N map comes from the magnetic EFIT *solution*
    through :func:`vaft.data.eqdsk.from_equilibrium`, so no g-file has to be
    located on a filesystem the stage does not own.

    Those are two different products and both are `equilibrium`, which is why
    they are not merged: the constraints product carries
    ``code.parameters`` (the ``IN1`` namelist ``generate_kfile`` reads) and
    ``time_slice.*.constraints``, while the solution carries the flux map.  The
    psi map travels as ``geq``, which ``run_kinetic_chain`` already takes
    separately from the ODS, so neither has to overwrite the other.

    ``stage`` is ``electron_efit`` or ``kinetic_efit``, and it must agree with
    what the core-profile product actually measured: the ion diagnostic is what
    separates a measured Ti from one assumed through a Ti/Te ratio, and a run
    that quietly produced the other kind would be published into the wrong
    lineage.  A mismatch is recorded, not raised, because both stages are asked
    for and only one of them can apply to a given shot.
    """
    from vaft.code.efit.kinetic import KineticEFITConfig, run_kinetic_chain
    from vaft.data.eqdsk import from_equilibrium, to_omas
    from vaft.process.equilibrium import as_equilibrium

    if stage not in ("electron_efit", "kinetic_efit"):
        raise ValueError(f"stage must be electron_efit or kinetic_efit; got {stage!r}")

    shot = int(shot)
    era = machine_era_for_shot(shot)
    manifest = _external_profile_manifest(stage, shot, run, era.name)
    manifest["input"] = {
        "core_profiles_product": str(core_profiles_product),
        "constraints_product": str(constraints_product),
        "efit_product": str(efit_product),
    }
    manifest["configuration"]["encoding"] = encoding

    out = ODS(consistency_check=False)
    dataset_description(
        out,
        shot,
        {
            "source_type": "shot",
            "run": run,
            "machine": "VEST",
            "user": "vaft",
            "description": f"VEST {stage.replace('_', ' ')}; machine era {era.name}",
        },
    )

    def unavailable(reason: str) -> tuple[ODS, dict[str, Any]]:
        manifest["error"] = reason
        return out, manifest

    for label, path in (
        ("core_profiles", core_profiles_product),
        ("constraints", constraints_product),
        ("efit", efit_product),
    ):
        if not Path(path).exists():
            return unavailable(f"the {label} product is missing: {path}")

    ods, _ = load_ods(Path(core_profiles_product))
    # What decides the lineage is whether a slice was actually built with a
    # measured ion temperature, not whether the product happens to carry a
    # `charge_exchange` IDS. On 48226 it carries one and has no kinetic slice
    # at all -- the ion diagnostic covers 298-307 ms and the equilibrium does
    # not -- so keying off the IDS refused `electron_efit` for a product only
    # `electron_efit` can serve, while `kinetic_efit` could not serve it
    # either. Both lineages declined the same shot.
    has_kinetic = _has_kinetic_slice(ods)
    wants_kinetic = stage == "kinetic_efit"
    if has_kinetic != wants_kinetic:
        measured = "with" if has_kinetic else "without"
        return unavailable(
            f"this shot's profiles were built {measured} a measured ion "
            f"temperature, so it belongs to the other lineage, not {stage}"
        )
    if "thomson_scattering" not in ods:
        return unavailable("the core-profile product carries no thomson_scattering")

    # What `generate_kfile` actually reads is the EFIT stage's *constraints*
    # product: `code.parameters` holds the IN1 namelist (TABLE_DIR/INPUT_DIR)
    # and `time_slice.*.constraints` the measurements already folded in from
    # magnetics. The efit *output* product cannot stand in for it -- its
    # `code.parameters` records collection provenance instead, and on products
    # written before #642 it is a flat string rather than a tree.
    constraints, _ = load_ods(Path(constraints_product))
    if "equilibrium" not in constraints:
        return unavailable("the constraints product carries no equilibrium")
    ods["equilibrium"] = constraints["equilibrium"]

    # The solution stays out of the ODS on purpose: it is `equilibrium` too, so
    # merging it would overwrite the constraints the kfile is built from. It
    # only has to reach `geq`, which is a separate argument.
    solution, _ = load_ods(Path(efit_product))
    if (
        "equilibrium.time_slice" not in solution
        or not len(solution["equilibrium.time_slice"])
    ):
        return unavailable("the efit product carries no equilibrium time slice")

    # `time_slice` existing does not guarantee `time` does, and argmin over an
    # empty array raises. Every other bad-input path here records and returns,
    # so this one must too -- an optional stage that raises fails the whole run
    # instead of being skipped.
    if "equilibrium.time" not in solution:
        return unavailable("the efit product carries no equilibrium.time")
    eq_times = np.asarray(solution["equilibrium.time"], dtype=float).reshape(-1) * 1e3
    if eq_times.size == 0:
        return unavailable("the efit product's equilibrium.time is empty")

    if time_ms is None:
        # Same two forms this file just stopped trusting: `in` is not total
        # over a reloaded product, and a single time saved as a scalar makes
        # `len` raise out of a stage contracted never to raise.
        profile_times = _time_array_ms(ods, "core_profiles.time")
        if profile_times is None:
            return unavailable("no core_profiles slice to reconstruct at")
        # The best-aligned slice, not the first one. Which profile time happens
        # to come first says nothing about which has an equilibrium to be
        # reconstructed against, and with a sub-millisecond tolerance an
        # arbitrary choice refuses shots a better-aligned slice would serve.
        offsets = np.min(np.abs(profile_times[:, None] - eq_times[None, :]), axis=1)
        time_ms = float(profile_times[int(np.argmin(offsets))])

    time_index = int(np.argmin(np.abs(eq_times - float(time_ms))))
    manifest["configuration"]["time_ms"] = float(time_ms)
    manifest["configuration"]["equilibrium_time_ms"] = float(eq_times[time_index])
    manifest["configuration"]["equilibrium_tolerance_ms"] = float(equilibrium_tolerance_ms)
    # Reconstructing a 298 ms profile against a 308 ms equilibrium is not a
    # near miss in a discharge this short -- it is a different plasma. Refuse
    # rather than produce a number nobody can tell is meaningless.
    offset = abs(float(eq_times[time_index]) - float(time_ms))
    if offset > equilibrium_tolerance_ms:
        return unavailable(
            f"the nearest reconstructed equilibrium is {offset:.1f} ms from the "
            f"profile time ({time_ms:.1f} ms), beyond the "
            f"{equilibrium_tolerance_ms:.1f} ms tolerance"
        )

    # The psi this carries is the ODS's, in weber (#278/#281), where a g-file's
    # is weber per radian -- measured on 48224, SIMAG and SIBRY come out exactly
    # 2*pi larger. It does not matter *here*, because every consumer normalizes:
    # psi_N = (psi - SIMAG)/(SIBRY - SIMAG) in `_psin_of_r`, and psi_norm /
    # rho_pol_norm / rho_tor_norm in the profile mapper, so a global factor -- or
    # a COCOS sign flip -- cancels in numerator and denominator alike. Verified
    # against the packaged g-file: psi_N agrees to 2e-16. Anything that later
    # wants *absolute* psi off this object has to convert first.
    try:
        geq = from_equilibrium(as_equilibrium(solution, time_index=time_index))
    except Exception as error:  # noqa: BLE001 - an unusable equilibrium is not a crash
        return unavailable(f"cannot build a psi map from the equilibrium: {error}")

    # A caller-supplied workdir is the caller's to keep. One we invent is ours to
    # remove: EFIT leaves a kfile and its outputs per PLASMA scale, and this stage
    # is registered for every shot, so leaking one directory per shot would fill
    # the filesystem the products themselves live on.
    ephemeral = workdir is None
    work = Path(tempfile.mkdtemp(prefix=f"{stage}-")) if ephemeral else Path(workdir)
    try:
        config = KineticEFITConfig(
            executable=executable,
            workdir=work,
            shot=shot,
            time_ms=float(time_ms),
            encoding=encoding,
            # 'auto' is the VEST policy ratio, and it only applies when the ODS
            # has no ion data -- which is exactly the electron_efit case.
            ti_te_ratio="auto",
        )
        try:
            chain = run_kinetic_chain(ods, geq, float(time_ms), efit_config=config)
        except Exception as error:  # noqa: BLE001 - recorded, one shot cannot fail a run
            manifest["status"] = "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
            manifest["traceback"] = traceback.format_exc()
            return out, manifest

        result = chain["kinetic_efit"]
        manifest["reconstruction"] = {
            "status": result.status,
            "converged": bool(result.converged),
            "scale": result.scale,
            "chi2": result.chi2,
            "reason": result.reason,
        }
        if result.status == "skipped":
            return unavailable(result.reason or "EFIT executable is not configured")
        if not result.converged or result.gfile is None:
            manifest["status"] = "no_output"
            manifest["error"] = result.reason or "no PLASMA scale converged"
            return out, manifest

        # Read the g-file before the workdir goes away.
        to_omas(read_geqdsk_file(result.gfile), ods=out)
        manifest["status"] = "success"
        manifest["quality_summary"]["unavailable"] = []
        # Recording a path under a directory about to be deleted would name a
        # file nobody can open; the name is what identifies the reconstruction.
        manifest["output_gfile"] = (
            Path(result.gfile).name if ephemeral else str(result.gfile)
        )
        return out, manifest
    finally:
        if ephemeral:
            shutil.rmtree(work, ignore_errors=True)


def read_geqdsk_file(path: str | Path):
    """Read a g-file. Thin indirection so the builder stays import-light."""
    from vaft.data import read_geqdsk

    return read_geqdsk(path)


def write_stage_product(
    ods: ODS,
    manifest: dict[str, Any],
    *,
    output: str | Path,
    metadata: str | Path,
    compression: str | None = None,
) -> None:
    """Write deterministic ODS and manifest products.

    ``compression`` names an HDF5 filter (``"gzip"``) for ``.h5``/``.hdf5``
    outputs. It changes the container encoding, not the data, so the recorded
    ``output.sha256`` legitimately differs from an uncompressed product built
    from the same ODS -- that hash identifies the file, not the physics.
    """
    output_path = save(ods, output, compression=compression)
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
    "build_gpec_ideal_ods",
    "build_static_ods",
    "machine_era",
    "machine_era_for_shot",
    "write_manifest",
    "write_stage_product",
]

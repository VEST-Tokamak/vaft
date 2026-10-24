"""OMAS-first upstream stages for the VEST production workflow."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from omas import ODS

from vaft.database import raw as raw_db
from vaft.database.sources import STAGE_REPLICATION
from vaft.process.signal_processing import SignalRepairError
from vaft.database._local import load_ods
from vaft.data.resources import data_path
from vaft.machine_mapping.barometry import barometry
from vaft.machine_mapping.ec_launchers import ec_launchers
from vaft.machine_mapping.dataset_description import dataset_description
from vaft.machine_mapping.em_coupling import DEFAULT_VERSIONED_COUPLING, em_coupling
from vaft.machine_mapping.effective_config import vest_configuration_record
from vaft.machine_mapping.impa import (
    impa as impa_mapper,
    impa_expected_fields,
    impa_probe_indices,
    resolve_impa_config,
)
from vaft.machine_mapping.langmuir_probes import LangmuirProbeEraGapError, langmuir_probes
from vaft.machine_mapping.magnetics import (
    FLUCTUATION_MIRNOV_FIRST_SHOT,
    LIMITER_SHUNT_CHANNELS,
    equilibrium_magnetics_processing_config,
    fluctuation_mirnov_channel_definitions,
    known_magnetics_faults,
    toroidal_mirnov_reference_channels,
    vest_equilibrium_magnetics_channel_definitions,
    vfit_magnetics_dynamic,
    vfit_magnetics_static,
)
from vaft.machine_mapping.pf_active import (
    PF_COIL_COUNT,
    require_acquired_pf_currents,
    resolve_geometry_asset,
    unacquired_pf_coils,
    vfit_pf_active_dynamic,
    vfit_pf_active_static,
)
from vaft.machine_mapping.pf_passive import (
    DEFAULT_STATIC_GEOMETRY,
    DEFAULT_WALL_2409_ADDITIONS,
    pf_passive,
    wall_geometry_version_for_shot,
)
from vaft.machine_mapping.spectrometer_uv import spectrometer_uv
from vaft.machine_mapping.tf import vfit_tf_dynamic, vfit_tf_static
from vaft.machine_mapping.wall import wall
from vaft.machine_mapping.utils import (
    DiagnosticsTimePolicy,
    DiagnosticsTimePolicyTable,
    VestConfigurationError,
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
    wall_geometry: str = "1512"

    def contains(self, shot: int) -> bool:
        return (self.first_shot is None or shot >= self.first_shot) and (
            self.last_shot is None or shot <= self.last_shot
        )


# One era per machine access that changed a conductor the static product
# carries (issue #956). Each boundary is the first shot after a multi-day gap
# in the raw `pulse_datetime`, which is when the vessel could be opened:
#   43017  passive wall 1512 -> 2409 (+15 SUS316LN elements at Z = -1.164 m);
#          43016 is 2024-07-05, 43017 is 2024-07-15.
#   45968  PF6/PF7 rebuilt, coil geometry 1906 -> 2507; 45950-45967 run every
#          ~12 minutes on 2025-06-30 and 45968 follows on 2025-07-07.
# The PF boundary was 45958 before #956, with no recorded source, which split
# off a ten-shot 45958-45966 era and a 45967 legacy boundary; both are gone.
# Code-owned on purpose (#2159, audit gap G14): an era names the static
# product's geometry assets and product identity (``machine_version``), which
# stage products and HSDS sources are keyed by, so it changes with a code
# release and its tests, not with a vest.yaml edit. The shot-era *processing*
# policy stays in vest.yaml.
VEST_MACHINE_ERAS = (
    VestMachineEra("vest-pre-43017-pf1906", None, 43016, "1906", 43016, "1512"),
    VestMachineEra("vest-43017-45967-pf1906", 43017, 45967, "1906", 43017, "2409"),
    VestMachineEra("vest-45968-plus-pf2507", 45968, None, "2507", 45968, "2409"),
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


def _disabled_pf_coils(shot: int) -> list[str]:
    """Circuits with no acquisition channel at all, as of this shot's era.

    Read from `vest.yaml`'s `pf_active.processing.coil_field_codes`, which is
    where the channel map lives since #708. It used to come from the
    five-entry `Coil_info.mat`, which reported PF2 as permanently disabled --
    when in fact PF2 is acquired on field 4, and was simply unused by the
    shots anyone had looked at.

    "Disabled" here means *no channel is known*, which is a statement about
    the machine. It is not the same as *this discharge left the coil cold*,
    which is a statement about the discharge and is carried by the current
    waveform itself.
    """
    from vaft.machine_mapping.pf_active import coil_field_code_by_index

    acquirable = coil_field_code_by_index(shot)
    return [
        f"PF{index}"
        for index in range(1, PF_COIL_COUNT + 1)
        if (index - 1) not in acquirable
    ]


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
    pf_passive(ods, shot=era.reference_shot)
    em_coupling(ods, shot=era.reference_shot)
    if wall_geometry_version_for_shot(era.reference_shot) != era.wall_geometry:
        raise ValueError(
            f"machine era {era.name} declares wall {era.wall_geometry}, but its "
            f"reference shot {era.reference_shot} selects wall "
            f"{wall_geometry_version_for_shot(era.reference_shot)}"
        )
    # Deliberately un-gated (shot=0, the inventory). A static product describes
    # a machine *era*, and the phase-reference Mirnov boundary (shot 35520)
    # falls inside the first era, whose reference_shot is 43016 -- so no single
    # shot represents that era's magnetics truthfully. Per-shot availability is
    # applied in `build_shot_components`, which rebuilds this IDS with the shot
    # rather than copying it.
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
        f"PF geometry {era.pf_geometry}; wall {era.wall_geometry}"
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
            **(
                {
                    "wall_additions": {
                        "name": Path(DEFAULT_WALL_2409_ADDITIONS).name,
                        "wall_geometry": era.wall_geometry,
                        "sha256": sha256_file(DEFAULT_WALL_2409_ADDITIONS),
                    }
                }
                if era.wall_geometry != "1512"
                else {}
            ),
            # The static product carries probe positions only. Which raw field
            # feeds each position is per shot (vest.yaml `wiring`, issue #956)
            # and recorded in the diagnostics provenance.
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
                "disabled_channels": _disabled_pf_coils(era.reference_shot),
            }
        },
        "quality_summary": {
            "missing": [],
            "repaired": [],
            "disabled": _disabled_pf_coils(era.reference_shot),
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
        ("ec_launchers.beam.0.power_launched.time", "ec_power"),
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
        (
            "ec_launchers.beam.0.power_launched.time",
            "ec_launchers.beam.0.power_launched.data",
        ),
        ("ec_launchers.beam.0.frequency.time", "ec_launchers.beam.0.frequency.data"),
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


#: ``VestMagneticsProcessingConfig`` fields that state the acquisition (#2159):
#: which DAQ chain recorded the shot and at what rate. An override may restate
#: the era's value but never change it.
MAGNETICS_MACHINE_FACT_FIELDS = ("daq_mode", "fast_sample_rate")
#: Fields that change the calibration of every channel; changing one needs
#: ``calibration_override_reason`` in the override block, which is recorded.
MAGNETICS_CALIBRATION_FIELDS = ("calibration_mode", "flux_output_per_radian")
#: Reserved key of an override block carrying the reason for a calibration override.
CALIBRATION_OVERRIDE_REASON_KEY = "calibration_override_reason"
#: Fields the vest.yaml era (``equilibrium_magnetics.processing.window``) sets.
_MAGNETICS_ERA_FIELDS = (
    "window_override", "flux_baseline_window", "flux_baseline_samples", "daq_mode",
    "allow_zero_fallback",
)


@dataclass(frozen=True)
class MagneticsProcessingResolution:
    """The equilibrium-magnetics processing a shot actually runs with (#2159).

    ``config`` is what the mapper receives (the era policy when ``overridden``
    is false); ``effective`` its JSON form, ``fingerprint`` the canonical
    SHA-256 of ``effective`` (see :mod:`vaft.machine_mapping.effective_config`),
    and ``provenance`` where each field came from.
    """

    config: VestMagneticsProcessingConfig
    overridden: bool
    effective: Mapping[str, Any]
    fingerprint: str
    provenance: Mapping[str, Any]


def resolve_magnetics_processing(
    shot: int, overrides: Mapping[str, Any] | None = None
) -> MagneticsProcessingResolution:
    """Resolve and validate the equilibrium-magnetics processing for *shot*.

    The era policy comes from vest.yaml; ``overrides`` (the workflow's
    ``vest.magnetics.processing`` block) changes only the keys it names, and
    is validated first:

    * an unknown key is a :class:`VestConfigurationError` naming the valid ones;
    * a machine fact (``daq_mode``, ``fast_sample_rate``) may be restated but
      not changed;
    * a calibration field (``calibration_mode``, ``flux_output_per_radian``)
      changes only with a non-empty ``calibration_override_reason``;
    * a key the era decides (the legacy window ladder, the per-sample flux
      baselines) set off its default would change nothing and is refused;
    * the output-grid keys may not move the grid the era window indexes
      unless ``window_override`` is named with them (#1541);
    * the result must pass ``VestMagneticsProcessingConfig`` validation
      (positive rates and taps, cutoff below Nyquist).
    """
    from vaft.machine_mapping.effective_config import (
        configuration_fingerprint,
        resolve_vest_processing_config,
    )

    shot = int(shot)
    era = equilibrium_magnetics_processing_config(shot)
    window_revision = next(
        (
            dict(record)
            for record in resolve_vest_processing_config(shot, "equilibrium_magnetics").provenance[
                "revisions"
            ]
            if record["path"] == "processing.window"
        ),
        None,
    )
    if overrides is not None and not isinstance(overrides, Mapping):
        raise VestConfigurationError(
            f"shot {shot}: the magnetics processing override must be a mapping, not {type(overrides).__name__}"
        )
    overrides = dict(overrides or {})
    reason = overrides.pop(CALIBRATION_OVERRIDE_REASON_KEY, None)
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise VestConfigurationError(
            f"shot {shot}: {CALIBRATION_OVERRIDE_REASON_KEY} must be a non-empty string"
        )
    known = {f.name for f in fields(VestMagneticsProcessingConfig)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise VestConfigurationError(
            f"shot {shot}: unknown magnetics processing key(s) {unknown}; valid keys are "
            f"{sorted(known)} (plus {CALIBRATION_OVERRIDE_REASON_KEY!r})"
        )
    changed = sorted(
        key for key in overrides if _as_tuple(overrides[key]) != _as_tuple(getattr(era, key))
    )
    facts = [key for key in changed if key in MAGNETICS_MACHINE_FACT_FIELDS]
    if facts:
        raise VestConfigurationError(
            f"shot {shot}: {facts} state the shot's acquisition (vest.yaml era "
            f"{dict((key, getattr(era, key)) for key in facts)}) and cannot be overridden"
        )
    calibrations = [key for key in changed if key in MAGNETICS_CALIBRATION_FIELDS]
    if calibrations and reason is None:
        raise VestConfigurationError(
            f"shot {shot}: {calibrations} change the calibration of every channel; add "
            f"{CALIBRATION_OVERRIDE_REASON_KEY!r} to the override block to accept that"
        )
    shadowed = set(_WINDOW_RULE_KEYS) if era.window_override is not None else set()
    if era.flux_baseline_window is not None or era.flux_baseline_samples is not None:
        shadowed |= set(_FLUX_BASELINE_RULE_KEYS)
    defaults = {f.name: f.default for f in fields(VestMagneticsProcessingConfig)}
    # A shadowed key left at its default is inert (the routine block restates
    # them all); one set to anything else is a change that would be dropped.
    dropped = sorted(
        key for key in shadowed & set(overrides)
        if _as_tuple(overrides[key]) != _as_tuple(defaults[key])
    )
    if dropped:
        raise VestConfigurationError(
            f"shot {shot}: the vest.yaml era policy decides {dropped}, so this override would "
            "change nothing; set window_override / flux_baseline_window / flux_baseline_samples instead"
        )
    # The era window is a triple of INDICES into the output grid
    # linspace(time_start, time_end, sample_count). An override that moves the
    # grid while the era keeps the indices moves the analysis window and the
    # probe baseline with it, silently: `sample_count: 20000` puts the
    # native-DAQ window at 0.325-0.45 s instead of 0.26-0.36 s. The grid keys
    # are therefore refused unless the override also names `window_override`,
    # so a scan that moves the grid owns the indices into it as well.
    if era.window_override is not None and "window_override" not in overrides:
        moved = sorted(
            key for key in _OUTPUT_GRID_KEYS
            if key in overrides and _as_tuple(overrides[key]) != _as_tuple(getattr(era, key))
        )
        if moved:
            raise VestConfigurationError(
                f"shot {shot}: {moved} would move the output grid that the vest.yaml era window "
                f"{tuple(era.window_override)} indexes, shifting the analysis window and the probe "
                "baseline; name window_override alongside them or leave them at the era's values"
            )
    try:
        config = replace(era, **overrides) if overrides else era
    except ValueError as error:
        raise VestConfigurationError(f"shot {shot}: invalid magnetics processing override: {error}") from error

    effective = json.loads(json.dumps(asdict(config)))
    keys = {}
    for name in sorted(known):
        if name in changed:
            origin = "override"
        elif name in _MAGNETICS_ERA_FIELDS:
            origin = "vest.yaml"
        else:
            origin = "default"
        keys[name] = {
            "origin": origin,
            "overridden": name in changed,
            "revision": window_revision if origin == "vest.yaml" else None,
        }
    provenance = {
        "era_window_revision": window_revision,
        "overrides": {
            "requested": json.loads(json.dumps(overrides)),
            "changed": changed,
            CALIBRATION_OVERRIDE_REASON_KEY: reason if calibrations else None,
        },
        "keys": keys,
    }
    return MagneticsProcessingResolution(
        config=config,
        overridden=bool(overrides),
        effective=effective,
        fingerprint=configuration_fingerprint(effective),
        provenance=provenance,
    )


def _time_policy_effective(policies: DiagnosticsTimePolicyTable) -> dict[str, Any]:
    """The applied time policies as plain data: every window, the default, each component's."""
    return {
        "default": policies.default.as_dict() if policies.default is not None else None,
        "windows": {name: policy.as_dict() for name, policy in sorted(policies.windows.items())},
        "components": {name: policy.name for name, policy in sorted(policies.items())},
    }


def magnetics_processing_for_shot(
    shot: int, overrides: Mapping[str, Any] | None
) -> VestMagneticsProcessingConfig | None:
    """The equilibrium-magnetics processing for *shot*: its era policy, with *overrides* on top.

    ``None`` without overrides, so the mapper resolves the era itself. An
    override changes only the keys it names. It used to *replace* the era
    policy: the routine workflow's ``vest.magnetics.processing`` block, which
    restates the dataclass defaults, then ran every native-DAQ shot (46404 on)
    through the slow-DAQ path, with the wrong fields and baselines, as soon as
    the workflow configuration was actually loaded (#1541). The acquisition era
    (``daq_mode``, the baseline rules, the output window) comes from
    ``vest.yaml``; :func:`resolve_magnetics_processing` lists every guard.
    """
    if not overrides:
        return None
    return resolve_magnetics_processing(shot, overrides).config


#: Keys only the legacy hardcoded window ladder reads; an era ``window_override`` shadows them.
_WINDOW_RULE_KEYS = (
    "default_index_start", "default_index_end", "default_probe_baseline_end", "late_shot_min",
    "transient_shot_min", "transient_shot_max", "late_index_start", "late_index_end", "late_probe_baseline_end",
)
#: The output grid the era's ``window_override`` indices index into.
_OUTPUT_GRID_KEYS = ("time_start", "time_end", "sample_count")
#: Keys only the legacy per-sample flux baselines read; an era flux-baseline rule shadows them.
_FLUX_BASELINE_RULE_KEYS = (
    "flux_baseline_first_start", "flux_baseline_first_end", "flux_baseline_second_start",
    "flux_baseline_second_end", "flux_baseline_late_start", "flux_baseline_late_end",
    "flux_baseline_late_loop_numbers",
)


def _as_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, (list, tuple)) else value


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
    barometry, and EC power (``ec_launchers``, issue #165) -- follow their own
    configured window instead, so the product carries more than one temporal
    coverage (issue #244).  ``time_policies`` overrides the whole policy document.
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
    # Grid key -> the `statuses` key of the component that realized it.
    grid_status_names: dict[str, str] = {}
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

        ``component`` names the time policy and ``ids_name`` the status entry
        `run_component` wrote; they differ for EC power (``ec_power`` policy,
        ``ec_launchers`` IDS).
        """
        grid_status_names[component] = ids_name
        if statuses.get(ids_name, {}).get("status", "unavailable") == "unavailable":
            return
        if not path_exists(ods, path):
            return
        grids[component] = np.asarray(get_path(ods, path), dtype=float).reshape(-1)

    def run_component(
        name: str,
        ids_names: tuple[str, ...],
        mapper: Callable[[ODS], None],
        component_status: str = "success",
        optional: bool = False,
        **details: Any,
    ) -> None:
        component = ODS(consistency_check=False)
        try:
            mapper(component)
        except (
            raw_db.RawSignalUnavailableError,
            FileNotFoundError,
            SignalRepairError,
            # A shot between two documented Langmuir bias/tip eras has no
            # known setting (#152); that is the Langmuir component's gap,
            # not a reason to lose every other diagnostic (#989).
            LangmuirProbeEraGapError,
        ) as error:
            # A fully saturated waveform is a property of the shot, not a fault
            # in the run: refusing to reconstruct one is correct, but it makes
            # that component unavailable rather than the whole stage. Letting it
            # escape cost 88 shots their entire diagnostics product for one
            # clipped diamagnetic-flux signal.
            statuses[name] = {"status": "unavailable", "reason": str(error), **details}
            if optional:
                # Hardware a shot may simply not have run (EC power before
                # ~29500 or on a no-ECH shot): recorded, never "partial".
                statuses[name]["optional"] = True
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
        disabled_channels=_disabled_pf_coils(shot),
    )
    unacquired_pf = unacquired_pf_coils(ods)
    if unacquired_pf and statuses["pf_active"]["status"] != "unavailable":
        # Published without the circuits whose channel was empty (#1568):
        # they carry NaN, and eddy/EFIT refuse the shot.
        statuses["pf_active"]["status"] = "partial"
        statuses["pf_active"]["unacquired_channels"] = unacquired_pf
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
    # EC power (issue #165) is full-discharge for the same reason: the ECH
    # pulse precedes breakdown and outlasts the analysis window.  The component
    # is `ec_power` in the policy table and `ec_launchers` as an IDS.
    ec_power_policy = policies["ec_power"]
    run_component(
        "ec_launchers",
        ("ec_launchers",),
        lambda component: ec_launchers(
            component,
            shot,
            ec_power_policy.tstart,
            ec_power_policy.tend,
            ec_power_policy.dt,
            raw_source=raw_path,
        ),
        optional=True,
    )
    record_realized_grid(
        "ec_power", "ec_launchers", "ec_launchers.beam.0.power_launched.time"
    )
    langmuir_policy = policies["langmuir_probes"]
    # Filled by the mapper; statuses keeps this same dict, so the counts reach
    # the manifest without opening the ODS (#915).
    langmuir_unfiltered: dict[str, Any] = {}
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
            report=langmuir_unfiltered,
        ),
        anti_alias={
            "policy": (
                "finite runs of the triple-probe solve shorter than filtfilt "
                "accepts are decimated unfiltered; every target sample they "
                "contribute to has validity_timed = -1 (#915)"
            ),
            "unfiltered": langmuir_unfiltered,
        },
    )
    grids["langmuir_probes"] = policy_grid("langmuir_probes")
    tf_policy = policies["tf"]
    # Filled by the mapper: the TF-current acquisition excursions it replaced
    # (#1543), so the manifest's `repaired` says what `tf.coil.0.current` no
    # longer measures rather than a literal empty list.
    tf_report: dict[str, Any] = {}
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
                report=tf_report,
            ),
        ),
    )
    record_realized_grid("tf", "tf", "tf.time")
    tf_repaired = [{"component": "tf", **entry} for entry in tf_report.get("repaired", [])]
    if statuses["tf"]["status"] != "unavailable":
        statuses["tf"]["repaired"] = tf_repaired
    magnetics_resolution = resolve_magnetics_processing(shot, vest_magnetics_processing)
    # Without an override the mapper resolves the era itself, as it always did;
    # the resolution is the same config and is recorded in the manifest.
    processing = magnetics_resolution.config if magnetics_resolution.overridden else None
    magnetics_channels = [
        int(channel["field_code"])
        for channel in vest_equilibrium_magnetics_channel_definitions(int(shot))
    ] + [
        # Gated, like the fluctuation list below: a channel the mapper declines
        # to map past its last operational shot is not a channel the archive is
        # missing, and reporting it as one pins component_status at "partial"
        # for every modern shot.
        int(channel["field_code"]) for channel in toroidal_mirnov_reference_channels(int(shot))
    ] + [
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
    magnetics_left_out: dict[str, str] = {}
    run_component(
        "magnetics",
        ("magnetics",),
        lambda component: (
            # Rebuilt with the shot rather than copied from the era static:
            # the static carries the un-gated inventory, and copying it would
            # publish phase-reference channels the machine no longer had while
            # `vfit_mirnov_raw_dynamic` gated them out -- leaving the array at
            # different indices depending on whether a product came from this
            # pipeline or from a fresh mapping.
            vfit_magnetics_static(component, shot),
            magnetics_left_out.update(vfit_magnetics_dynamic(
                component,
                shot,
                magnetics_policy.tstart,
                magnetics_policy.tend,
                magnetics_policy.dt,
                processing_config=processing,
                raw_source=raw_path,
                target_time=policy_grid("magnetics"),
            ) or {}),
        ),
        processing="VestMagneticsProcessingConfig",
        component_status="partial" if missing_magnetics_channels else "success",
        missing_channels=missing_magnetics_channels,
    )
    grids["magnetics"] = policy_grid("magnetics")
    if magnetics_left_out and statuses["magnetics"]["status"] != "unavailable":
        # The IDS was built without these signals (#993): a partial record,
        # with the reason each one is missing, not a silent success.
        statuses["magnetics"]["status"] = "partial"
        statuses["magnetics"]["signals_left_out"] = dict(magnetics_left_out)

    # A component whose raw signals were unavailable produced no coordinate,
    # so it must not appear in the realized-coverage record either.
    grids = {
        name: grid
        for name, grid in grids.items()
        if statuses.get(grid_status_names.get(name, name), {}).get("status", "unavailable")
        != "unavailable"
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

        # Channels recorded as broken for this shot (vest.yaml
        # `known_faults`, issue #956) are condemned by the same pass, so
        # EFIT and every other reader see them through validity alone.
        quality_report = validate_magnetics_signals(
            ods, known_faults=known_magnetics_faults(int(shot))
        )
        project_validity(ods, quality_report)
        magnetics_quality = magnetics_quality_metrics(ods, quality_report)

    successes = sum(
        value["status"] == "success" or bool(value.get("optional"))
        for value in statuses.values()
    )
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
            # What the processing actually used: the era policy with any override
            # on top. Recorded on every shot, not only overridden ones (#2159).
            "vest_magnetics_processing_effective": dict(magnetics_resolution.effective),
            "vest_magnetics_processing_fingerprint": magnetics_resolution.fingerprint,
            "vest_magnetics_processing_provenance": json.loads(
                json.dumps(magnetics_resolution.provenance, default=str)
            ),
            "time_policies": time_policies or {},
            # Effective vest.yaml configuration per diagnostic, its fingerprint,
            # the vest.yaml sha256 and the vaft version (#2159, audit gap G12).
            "vest_configuration": vest_configuration_record(
                shot,
                extra={
                    "magnetics_processing": magnetics_resolution.effective,
                    "time_policies": _time_policy_effective(policies),
                },
            ),
        },
        "time_grid": time_grid,
        "magnetics_quality": magnetics_quality,
        "channel_status": statuses,
        "quality_summary": {
            "missing": sorted(
                unavailable
                + missing_channels
                + [
                    f"magnetics:{signal}"
                    for signal in statuses.get("magnetics", {}).get("signals_left_out", {})
                ]
                + [
                    f"pf_active:{coil}"
                    for coil in statuses.get("pf_active", {}).get("unacquired_channels", [])
                ]
            ),
            "repaired": tf_repaired,
            "disabled": _disabled_pf_coils(shot),
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
    in :mod:`vaft.database.composition`.  :func:`build_eddy_ods` follows the same
    rule: it solves against the diagnostics product but stores only the
    ``pf_passive`` it owns.
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
            # The effective IMPA configuration (era-validated), its fingerprint,
            # the vest.yaml sha256 and the vaft version (#2159).
            "vest_configuration": vest_configuration_record(
                shot,
                diagnostics=("impa",),
                extra={"time_policies": _time_policy_effective(policies)},
            ),
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


#: ``eddy_status`` prefix of an eddy stage whose inputs do not exist in the data.
#: Replication reads it as a result, not a fault (vaft.database.replication).
EDDY_INPUT_UNAVAILABLE = "skipped: required input unavailable"


def eddy_no_output_product(
    *, shot: int, diagnostics_ods: str | Path, static_ods: str | Path, reason: str
) -> tuple[ODS, dict[str, Any]]:
    """The eddy product for a shot with an unrecorded PF circuit (#1568).

    :func:`build_eddy_ods` refuses such a shot, and that stays an error for a
    direct caller. The routine stage records it instead: a product with no
    ``pf_passive`` and a manifest saying ``no_output`` and why, so the shot is
    finished with nothing to compute rather than a failure retried forever.
    Only an unrecorded circuit takes this path; a missing diagnostics
    component still fails the stage, because it can be a configuration fault.
    """
    diagnostics_path, static_path = Path(diagnostics_ods), Path(static_ods)
    product = ODS(consistency_check=False)
    product["dataset_description.data_entry.pulse"] = int(shot)
    product["dataset_description.ids_properties.homogeneous_time"] = 2
    product["dataset_description.ids_properties.comment"] = f"eddy produced no output: {reason}"
    manifest = {
        "schema_version": 1,
        "stage": "eddy",
        "shot": int(shot),
        "machine_version": machine_era_for_shot(int(shot)).name,
        "status": "no_output",
        "eddy_status": f"{EDDY_INPUT_UNAVAILABLE}: {reason}",
        "input": {
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "static_sha256": sha256_file(static_path),
        },
    }
    return product, manifest


#: The eddy solve's inputs: the diagnostics component that provides each path.
_EDDY_INPUT_COMPONENTS = {
    "pf_active.time": "pf_active",
    "magnetics.ip.0.time": "magnetics",
    "magnetics.ip.0.data": "magnetics",
}


#: The EFIT constraints' inputs that only the diagnostics stage provides:
#: BTOR comes from the TF coil current, and nothing else measures it.
_EFIT_CONSTRAINT_INPUT_COMPONENTS = {
    "tf.b_field_tor_vacuum_r.data": "tf",
}


def eddy_inputs_absent_from_data(diagnostics_ods: str | Path, diagnostics_manifest: str | Path) -> str | None:
    """Why the eddy solve has no inputs, when the diagnostics stage recorded the raw data as absent.

    ``None`` unless every component the solve needs and the diagnostics product
    lacks (``pf_active``, ``magnetics``) is recorded in the diagnostics
    manifest's ``channel_status`` as ``unavailable`` because the raw store
    returned no waveform for it: a truncated raw
    dump, a shot without PF acquisition (#1799). That is a property of the
    shot, recorded as the stage's result. Anything else -- a component missing
    without such a record, or unavailable for another reason -- stays an error,
    because it can be a configuration fault.
    """
    return _inputs_absent_from_data(diagnostics_ods, diagnostics_manifest, _EDDY_INPUT_COMPONENTS)


def efit_constraint_inputs_absent_from_data(
    diagnostics_ods: str | Path, diagnostics_manifest: str | Path
) -> str | None:
    """Why EFIT constraints cannot be built, when the raw record lacks a required signal.

    The same rule as :func:`eddy_inputs_absent_from_data`, for the inputs only
    the diagnostics product supplies to the constraints: the TF coil current
    (BTOR). A shot whose raw dump has no TF waveform (42675, #1799) gets a
    ``not applicable`` result; a TF missing for any other reason stays an error.
    """
    return _inputs_absent_from_data(diagnostics_ods, diagnostics_manifest, _EFIT_CONSTRAINT_INPUT_COMPONENTS)


def _inputs_absent_from_data(
    diagnostics_ods: str | Path, diagnostics_manifest: str | Path, required: dict[str, str]
) -> str | None:
    diagnostics, _ = load_ods(Path(diagnostics_ods))
    missing = sorted({component for path, component in required.items() if path not in diagnostics})
    if not missing:
        return None
    try:
        status = json.loads(Path(diagnostics_manifest).read_text(encoding="utf-8")).get("channel_status") or {}
    except (OSError, ValueError):
        return None
    reasons = []
    for component in missing:
        entry = status.get(component) or {}
        reason = str(entry.get("reason") or "")
        # "returned no waveform" is the raw store's absence verdict; the shared
        # "Required VEST raw signal is unavailable" prefix also heads unusable-data
        # errors (mismatched lengths, extra channels), which are not absences.
        if entry.get("status") != "unavailable" or "returned no waveform" not in reason.lower():
            return None
        reasons.append(f"{component}: {reason}")
    return "; ".join(reasons)


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
    """Compute target-shot passive currents from finalized input ODSs.

    The returned product holds only what this stage owns --
    ``STAGE_REPLICATION["eddy"].ids`` plus ``dataset_description`` -- so the
    local product and the projection :func:`vaft.database.replication._project`
    publishes are the same set rather than two that can drift.  The solve itself
    runs on the full diagnostics ODS, because
    :func:`~vaft.omas.process_wrapper.compute_eddy_currents` needs
    ``pf_active``, ``pf_passive`` and ``em_coupling`` together; the projection
    happens after it.

    Two things survive that look like the thing being removed:

    ``dataset_description`` is copied from the diagnostics ODS rather than
    rebuilt.  It identifies the data entry, and the two products are the same
    entry -- minting a second one here would need a ``run`` this builder is
    never given, and a ``run`` that disagreed with the diagnostics replica of
    the same shot would be a silent pairing error.

    ``pf_passive.loop.*`` geometry still comes from the static product and still
    travels.  The rule is that a product carries no top-level IDS it does not
    own, not that it carries no static-derived bytes: ``static`` has no per-shot
    destination, so geometry stripped from inside an owned IDS would reach no
    reader at all.

    Analysis that needs ``magnetics`` or ``pf_active`` alongside these currents
    composes the two products explicitly, in
    :func:`vaft.database.composition.compose_stage_products`.
    """
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

    # A coil kept as not-acquired (#1568) would drive the vessel with NaN.
    require_acquired_pf_currents(diagnostics, int(shot), "the eddy-current solve")

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

    # `ods` is the diagnostics ODS the solve ran on; the product is not.  The
    # owned set is read from the replication registry rather than restated, so
    # the two cannot disagree about what this stage publishes.
    product = ODS(consistency_check=False)
    _copy_ids(product, ods, ("dataset_description",) + STAGE_REPLICATION["eddy"].ids)
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
    return product, manifest


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


def _append_record_fields(
    ods: ODS, path: str, fields: Mapping[str, Any], *, when_absent: str = ""
) -> None:
    """Add ``key=value`` fields to a provenance record, keeping what it holds.

    A record is read by key, so an appended field never disturbs a parser of the
    grammar already there; a key the record carries is left as written.

    ``when_absent`` starts a record that does not exist yet.  It matters for an
    ion temperature: an absent record is read as *measured*, while a record with
    only these fields is read as *unknown* and a measured ``T_i`` would be
    refused downstream -- so the caller states what the absence meant.
    """
    from vaft.machine_mapping.core_profiles import ti_record_fields
    from vaft.ods_access import path_value

    current = path_value(ods, path, None)
    text = "" if current is None else str(current).strip().rstrip(";").strip()
    text = text or when_absent
    present = ti_record_fields(text)
    added = "; ".join(f"{key}={value}" for key, value in fields.items() if key not in present)
    if added:
        ods[path] = f"{text}; {added}" if text else added


def _volume_weights_on_grid(solution: ODS, index: int, rho: np.ndarray) -> np.ndarray | None:
    """``dV`` per ``rho_tor_norm`` point from the equilibrium's own ``V(rho)`` at ``index``.

    The composition normalisation is an ``n_e dV`` mean, so it needs the volume
    element of the surface each point sits on.  Points beyond the equilibrium's
    ``rho`` range are clamped by ``np.interp`` and so get ``dV = 0``.
    """
    from vaft.ods_access import path_value

    base = f"equilibrium.time_slice.{index}.profiles_1d"
    rho_eq = path_value(solution, f"{base}.rho_tor_norm", None)
    volume = path_value(solution, f"{base}.volume", None)
    if rho_eq is None or volume is None or rho.size < 2:
        return None
    rho_eq, volume = np.asarray(rho_eq, dtype=float), np.asarray(volume, dtype=float)
    ok = np.isfinite(rho_eq) & np.isfinite(volume)
    if not ok.any():
        return None
    order = np.argsort(rho_eq[ok])
    edges = np.concatenate(([0.0], 0.5 * (rho[1:] + rho[:-1]), [rho[-1]]))
    return np.clip(np.diff(np.interp(edges, rho_eq[ok][order], volume[ok][order])), 0.0, None)


def _plasma_onset(diagnostics_product: str | Path | None) -> tuple[float | None, str]:
    """``(onset [s], source)`` of the discharge, or ``(None, why)`` when there is none."""
    if diagnostics_product is None or not Path(diagnostics_product).exists():
        return None, "no diagnostics product"
    try:
        from vaft.omas.plasma_timing import plasma_timing

        diagnostics, _ = load_ods(Path(diagnostics_product))
        timing = plasma_timing(diagnostics)
    except Exception as error:  # noqa: BLE001 - recorded; the stage reports it per slice
        return None, f"{type(error).__name__}: {error}"[:120]
    return (float(timing.onset), str(timing.source)) if timing.found else (None, "no plasma found")


COMPOSITION_IONIZATIONS = ("transient", "coronal")

#: Recorded beside the main-ion density so a reader can tell a pressure that
#: counts the diluted ions from the legacy ``e n_e (T_e + T_i)`` (n_i = n_e).
PRESSURE_CLOSURE_RECORD = {"pressure_closure": "infer_kinetic_closure", "dilution": "species"}


def _write_species_pressures(ods: ODS, base: str) -> None:
    """``ion[k].pressure`` = e n_k T_i for every ion of a slice that has an ion temperature.

    Only the total ``pressure_ion_total`` was stored, so a reader drawing the
    impurity pressure had to rebuild it; this states it per species.  A slice with
    no ion temperature gets none (its total was not written either).
    """
    from vaft.ods_access import path_value

    t_i = path_value(ods, f"{base}.ion.0.temperature", None)
    if t_i is None:
        return
    t_i = np.asarray(t_i, dtype=float)
    for k in range(len(ods[f"{base}.ion"])):
        n_k = path_value(ods, f"{base}.ion.{k}.density", None)
        if n_k is None:
            continue
        p_k = 1.602176634e-19 * np.asarray(n_k, dtype=float) * t_i
        ods[f"{base}.ion.{k}.pressure"] = p_k
        ods[f"{base}.ion.{k}.pressure_thermal"] = p_k


def _write_closure_pressures(ods: ODS, index: int, time_s: float) -> str:
    """Rewrite one slice's pressures from its (now diluted) composition; say what was done.

    The measured product stored ``pressure_thermal = e n_e (T_e + T_i)``, which
    counts one ion per electron.  Once the main ion is diluted that value is no
    longer the pressure of the ions the slice carries, so it is recomputed with
    :func:`vaft.process.kinetic_closure.infer_kinetic_closure` (``species``):

    * a **measured** ion temperature gives ``electrons.pressure``,
      ``electrons.pressure_thermal``, ``pressure_ion_total`` and
      ``pressure_thermal = p_e + p_i``;
    * **no** ion temperature gives the electron pressure only -- the slice stays
      free of any assumed-``T_i`` total, as it was;
    * an ion temperature the product itself assumed or inferred is not a
      measurement, so its own pressure is left as it was written.

    Returns ``"ion"``, ``"electron_only"`` or ``"unchanged"``.
    """
    from vaft.formula.equilibrium import electron_pressure
    from vaft.ods_access import path_value
    from vaft.process.kinetic_closure import infer_kinetic_closure

    base = f"core_profiles.profiles_1d.{index}"
    try:
        closure = infer_kinetic_closure(ods, time=float(time_s), dilution="species")
    except ValueError as error:
        if "stores no ion temperature" not in str(error):
            raise
        closure = None
    if closure is None:
        ne = path_value(ods, f"{base}.electrons.density_thermal", None)
        ne = np.asarray(ods[f"{base}.electrons.density"] if ne is None else ne, dtype=float)
        te = np.asarray(ods[f"{base}.electrons.temperature"], dtype=float)
        p_e = np.where(np.isfinite(ne) & np.isfinite(te),
                       np.asarray(electron_pressure(np.nan_to_num(ne), np.nan_to_num(te)), dtype=float),
                       np.nan)
        ods[f"{base}.electrons.pressure"] = p_e
        ods[f"{base}.electrons.pressure_thermal"] = p_e
        return "electron_only"
    if closure.provenance["ti"]["status"] != "measured":
        return "unchanged"
    pressure = closure.pressure
    ods[f"{base}.electrons.pressure"] = pressure.p_e
    ods[f"{base}.electrons.pressure_thermal"] = pressure.p_e
    ods[f"{base}.pressure_ion_total"] = pressure.p_i_thermal
    ods[f"{base}.pressure_thermal"] = pressure.p_thermal
    _append_record_fields(ods, f"{base}.ion.0.density_fit.parameters", PRESSURE_CLOSURE_RECORD)
    return "ion"


def build_composition_ods(
    *,
    shot: int,
    core_profiles_product: str | Path,
    efit_product: str | Path | None = None,
    diagnostics_product: str | Path | None = None,
    ionization: str = "transient",
    normalization: str = "ne_weighted_mean",
    equilibrium_tolerance_ms: float = EQUILIBRIUM_TIME_TOLERANCE_MS,
    cache_dir: str | None = None,
    tables: Mapping[str, Any] | None = None,
    assume_ti_eq_te: bool = True,
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Add the VEST impurity composition, ``n_i`` and plasma ``Z_eff(rho)`` to measured core profiles.

    Each ``core_profiles`` slice of the measured product is given the charge-state
    resolved C/O composition of ``machine_mapping/vest.yaml`` ``impurity_model``:
    the elemental ratio and the ``n_e dV``-mean ``Z_eff`` target are the model's,
    the radial charge states come from ADF11 data at the slice's ``T_e(rho)`` and
    ``n_e(rho)``, and the main ion is diluted accordingly.  Everything written is
    an *assumption* (``origin=assumed``/``derived``); a measured ``zeff`` is never
    overwritten.

    ``transient`` ionization needs the plasma age, ``t - onset``, with the onset
    from ``plasma_timing`` on the diagnostics product.  A slice with no positive
    age is skipped and named in the manifest -- it is never silently computed as
    coronal, which overstates the charge on VEST's tens-of-ms discharges.

    The slice's pressures are recomputed from the diluted ions with the
    ``species`` kinetic closure, so ``pressure_thermal`` no longer counts one
    ion per electron; the closure is named in ``ion.0.density_fit.parameters``
    (``pressure_closure=infer_kinetic_closure``).

    Each composed slice also records which equilibrium it was matched to
    (``equilibrium_lineage``, ``equilibrium_occurrence``, ``equilibrium_time_s``),
    in the electron temperature fit record and, when the slice has an ion
    temperature, in its record (an ion temperature with no record is stated as
    ``origin=measured`` first, which is what its absence meant): the pipeline-1
    EFIT is the magnetic lineage.
    The product is the measured one plus these additions; the replicated IDS is
    ``core_profiles``, which this product replaces in ``main`` only when it exists.

    A slice with **no measured** ion temperature is given ``T_i = T_e`` when
    ``assume_ti_eq_te`` (the default): written as an *assumption* in the ratio
    grammar (``ti_te_ratio=1; status=assumed``), with ``t_i_average`` and the
    total pressure following from it.  A measured ``T_i`` is never replaced.

    ``cache_dir`` is where the OPEN-ADAS ADF11 files are kept (they are fetched on
    first use, so a server without network needs the cache warmed); ``tables``
    overrides them with ``{element: (acd, scd)}`` paths and exists for tests.
    """
    from vaft.machine_mapping.core_profiles import vest_impurity_model
    from vaft.ods_access import path_value
    from vaft.process.impurity import populate_radial_impurity_profiles, resolve_radial_composition
    from vaft.process.kinetic_closure import infer_kinetic_closure

    if ionization not in COMPOSITION_IONIZATIONS:
        raise ValueError(f"ionization must be one of {COMPOSITION_IONIZATIONS}, got {ionization!r}")
    shot = int(shot)
    era = machine_era_for_shot(shot)
    manifest = _external_profile_manifest("composition", shot, run, era.name)
    manifest["input"] = {
        "core_profiles_product": str(core_profiles_product),
        "efit_product": str(efit_product) if efit_product is not None else None,
        "diagnostics_product": str(diagnostics_product) if diagnostics_product is not None else None,
    }
    manifest["configuration"].update(
        ionization=ionization, normalization=normalization,
        equilibrium_tolerance_ms=float(equilibrium_tolerance_ms),
        assume_ti_eq_te=bool(assume_ti_eq_te),
    )

    if not Path(core_profiles_product).exists():
        # A path that does not resolve is a caller error, not an absent product.
        raise FileNotFoundError(f"the core_profiles product is missing: {core_profiles_product}")
    ods, _ = load_ods(Path(core_profiles_product))
    times = path_value(ods, "core_profiles.time", None)
    if times is None or np.asarray(times).size == 0:
        manifest["error"] = "the core_profiles product carries no slices"
        return ods, manifest
    times = np.atleast_1d(np.asarray(times, dtype=float))

    model = vest_impurity_model(shot)
    if model is None:
        manifest["error"] = f"vest.yaml defines no impurity_model for shot {shot}"
        return ods, manifest
    weights = {item["element"]: float(item["relative_density"]) for item in model["species"]}
    target_zeff = float(model["target_zeff"])
    manifest["composition"] = {
        "model_status": model["status"], "elemental_weights": weights,
        "target_zeff": target_zeff, "main_ion": model["main_ion"],
    }

    solution = eq_times_ms = None
    if efit_product is not None and Path(efit_product).exists():
        solution, _ = load_ods(Path(efit_product))
        eq_times_ms = _time_array_ms(solution, "equilibrium.time")
    onset, onset_source = _plasma_onset(diagnostics_product)
    manifest["composition"].update(onset_s=onset, onset_source=onset_source)

    composed, assumed_ti, skipped = 0, 0, []
    # The machine policy's ratio sigma travels with the assumption, so a reader that
    # weights the pressure (the spline EFIT encoding, #1414) sees its uncertainty.
    from vaft.machine_mapping.core_profiles import vest_core_profiles_policy

    assumed_ti_sigma = float(vest_core_profiles_policy(shot).ti_te_ratio_sigma)
    assumed_ti_record = (
        f"ti_te_ratio=1; sigma={assumed_ti_sigma:g}; status=assumed; source=assumption; assume_ti_eq_te=1"
    )
    pressures = {"ion": 0, "electron_only": 0, "unchanged": 0}
    for index, time_s in enumerate(times):
        base = f"core_profiles.profiles_1d.{index}"
        rho = path_value(ods, f"{base}.grid.rho_tor_norm", None)
        te = path_value(ods, f"{base}.electrons.temperature", None)
        ne = path_value(ods, f"{base}.electrons.density_thermal", None)
        if ne is None:
            ne = path_value(ods, f"{base}.electrons.density", None)
        if rho is None or te is None or ne is None:
            skipped.append({"time_s": float(time_s), "reason": "slice lacks rho, T_e or n_e"})
            continue
        rho, te, ne = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (rho, te, ne))

        equilibrium_index = None
        if solution is not None and eq_times_ms is not None:
            nearest = int(np.argmin(np.abs(eq_times_ms - time_s * 1e3)))
            if abs(float(eq_times_ms[nearest]) - time_s * 1e3) <= equilibrium_tolerance_ms:
                equilibrium_index = nearest
        dv = (
            _volume_weights_on_grid(solution, equilibrium_index, rho)
            if equilibrium_index is not None else None
        )

        age = None if onset is None else float(time_s) - onset
        positive_age = age if age is not None and age > 0.0 else None
        if ionization == "transient" and positive_age is None:
            skipped.append({"time_s": float(time_s), "reason": f"no positive plasma age ({onset_source})"})
            continue
        try:
            radial = resolve_radial_composition(
                te, ne, rho, weights, normalization=normalization, target_zeff=target_zeff,
                ionization=ionization, plasma_age_s=positive_age, volume_weights=dv,
                cache_dir=cache_dir, tables=tables, time=float(time_s),
            )
            ods = populate_radial_impurity_profiles(ods, radial, time=float(time_s))
        except Exception as error:  # noqa: BLE001 - one slice that cannot be composed is a skipped one
            skipped.append({"time_s": float(time_s), "reason": f"{type(error).__name__}: {error}"[:200]})
            continue
        composed += 1
        if assume_ti_eq_te and path_value(ods, f"{base}.ion.0.temperature", None) is None:
            for k in range(len(ods[f"{base}.ion"])):
                ods[f"{base}.ion.{k}.temperature"] = te.copy()
                ods[f"{base}.ion.{k}.temperature_fit.parameters"] = assumed_ti_record
            ods[f"{base}.t_i_average"] = te.copy()
            ods[f"{base}.t_i_average_fit.parameters"] = assumed_ti_record
            assumed_ti += 1
            closure = infer_kinetic_closure(ods, time=float(time_s), dilution="species", ti_te_ratio=1.0)
            pressure = closure.pressure
            ods[f"{base}.electrons.pressure"] = pressure.p_e
            ods[f"{base}.electrons.pressure_thermal"] = pressure.p_e
            ods[f"{base}.pressure_ion_total"] = pressure.p_i_thermal
            ods[f"{base}.pressure_thermal"] = pressure.p_thermal
            pressures["ion"] += 1
        else:
            pressures[_write_closure_pressures(ods, index, float(time_s))] += 1

        _append_record_fields(ods, f"{base}.zeff_fit.parameters", {"target_zeff": target_zeff})
        # the record that states the normalization also states what it normalised to
        for k in range(len(ods[f"{base}.ion"])):
            _append_record_fields(ods, f"{base}.ion.{k}.density_fit.parameters", {"target_zeff": target_zeff})
        _write_species_pressures(ods, base)
        if equilibrium_index is not None:
            lineage = {
                "equilibrium_lineage": "magnetics", "equilibrium_occurrence": 0,
                "equilibrium_time_s": round(float(eq_times_ms[equilibrium_index]) * 1e-3, 6),
            }
            _append_record_fields(ods, f"{base}.electrons.temperature_fit.parameters", lineage)
            if path_value(ods, f"{base}.ion.0.temperature", None) is not None:
                # No record on a stored T_i means "measured" to every reader
                # (classify_ti_record); say so before adding to it.
                _append_record_fields(
                    ods, f"{base}.ion.0.temperature_fit.parameters", lineage,
                    when_absent="origin=measured",
                )

    manifest["slices"] = {
        "composed": composed, "skipped": len(skipped), "total": int(times.size),
    }
    manifest["ion_temperature"] = {"assumed_ti_eq_te": assumed_ti}
    manifest["pressure"] = {"closure": "infer_kinetic_closure", "dilution": "species", **pressures}
    manifest["skipped_slices"] = skipped[:20]
    if composed:
        manifest["status"] = "success"
        manifest["quality_summary"]["unavailable"] = []
        if skipped:
            manifest["quality_summary"]["missing"] = [
                f"composition:t={item['time_s']:.4f}s" for item in skipped[:20]
            ]
    else:
        manifest["error"] = skipped[0]["reason"] if skipped else "no slice could be composed"
    return ods, manifest


#: Relative one-sigma model-form uncertainty of the EFIT pressure (#874), and the
#: window of neighbouring slices whose spread can exceed it (#1426).
TI_INFERENCE_FLOOR = 0.17
TI_INFERENCE_WINDOW_S = 1e-3


class _SliceSkipped(Exception):
    """A slice the T_i inference cannot be made for; the message is the reason recorded."""


def _magnetic_efit_lineage(efit_product: str | Path) -> str | None:
    """``"magnetics"`` when the product lives under the FileDB ``efit/magnetic`` stage, else ``None``.

    The pressure-partition guard trusts the lineage it is given, and nothing in
    an equilibrium product records one, so it is read off *where the product
    was loaded from* -- never asserted by the caller.
    """
    parts = Path(str(efit_product)).parts
    return "magnetics" if any(pair == ("efit", "magnetic") for pair in zip(parts, parts[1:])) else None


def _thomson_relative_errors(ods: ODS, time_s: float) -> tuple[float, float]:
    """Median relative one-sigma of ``n_e`` and ``T_e`` over the valid channels nearest ``time_s``.

    The fitted ``core_profiles`` carry no uncertainty of their own, so the
    channels' median relative error is transferred to the fit (``NaN`` when the
    product has no usable channel, which the kernel records as a missing sigma).
    """
    from vaft.ods_access import path_value

    times = path_value(ods, "thomson_scattering.time", None)
    if times is None or not _path_present(ods, "thomson_scattering.channel"):
        return math.nan, math.nan
    times = np.atleast_1d(np.asarray(times, dtype=float))
    it = int(np.argmin(np.abs(times - time_s)))
    rel_n, rel_t = [], []
    for i in range(len(ods["thomson_scattering.channel"])):
        base = f"thomson_scattering.channel.{i}"
        for leaf, store in (("n_e", rel_n), ("t_e", rel_t)):
            data = path_value(ods, f"{base}.{leaf}.data", None)
            error = path_value(ods, f"{base}.{leaf}.data_error_upper", None)
            if data is None or error is None:
                continue
            data, error = np.atleast_1d(np.asarray(data, float)), np.atleast_1d(np.asarray(error, float))
            if it < data.size and it < error.size and data[it] > 0 and np.isfinite(error[it]) and error[it] > 0:
                store.append(float(error[it] / data[it]))
    return (
        float(np.median(rel_n)) if rel_n else math.nan,
        float(np.median(rel_t)) if rel_t else math.nan,
    )


def _equilibrium_pressure_on(solution: ODS, index: int, psi_norm: np.ndarray) -> np.ndarray | None:
    """The reconstructed pressure of slice ``index`` interpolated onto ``psi_norm`` points."""
    from vaft.ods_access import path_value
    from vaft.validation import kinetic_state as ks

    grid = ks.rho_tor_norm_of(solution, index).get("psi_norm")
    pressure = path_value(solution, f"equilibrium.time_slice.{index}.profiles_1d.pressure", None)
    if grid is None or pressure is None:
        return None
    grid, pressure = np.asarray(grid, dtype=float), np.asarray(pressure, dtype=float)
    if grid.size != pressure.size:
        return None
    order = np.argsort(grid)
    return np.interp(psi_norm, grid[order], pressure[order])


def build_ti_inferred_ods(
    *,
    shot: int,
    core_profiles_product: str | Path,
    efit_product: str | Path,
    floor: float = TI_INFERENCE_FLOOR,
    window_s: float = TI_INFERENCE_WINDOW_S,
    equilibrium_tolerance_ms: float = EQUILIBRIUM_TIME_TOLERANCE_MS,
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Give the electron-only slices an ion temperature inferred from the magnetic pressure partition.

    On a slice with no measured ``T_i``, ``T_i = p_eq / (e f n_e) - T_e / f`` with
    ``p_eq`` the pipeline-1 magnetic EFIT pressure at the slice's own flux
    surfaces and ``f = n_i,tot / n_e`` the composed product's diluted ion
    density (:func:`vaft.process.kinetic_closure.infer_ti_pressure_partition`).
    The result is **inferred**, never measured: it is written with
    ``origin=inferred`` in the ion ``temperature_fit.parameters`` record (the
    spelling ``classify_ti_record`` reads back), points where ``p_i`` is not
    positive and significant stay NaN, and the slice's pressures are recomputed
    from it, so ``p_thermal`` equals the equilibrium pressure there by
    construction -- a closure identity, not an agreement test.

    Only the ``efit/magnetic`` lineage is accepted, read from where the product
    lives; the electron-kinetic EFIT was fitted with an assumed ``T_i/T_e`` and
    would return that assumption.  Slices that already carry a measured ``T_i``
    are left alone.  A slice with no composition is skipped rather than inferred
    with ``n_i = n_e``: the ion density is half of the formula.  The stage reads
    the *composed* product and must never be read by the kinetic EFIT stages.
    """
    from vaft.machine_mapping.core_profiles import classify_ti_record, inferred_ti_text, ti_record_fields
    from vaft.ods_access import path_value
    from vaft.process.kinetic_closure import (
        TI_FLAGS,
        assemble_pressure,
        infer_kinetic_closure,
        infer_ti_pressure_partition,
    )
    from vaft.validation import kinetic_state as ks

    shot = int(shot)
    era = machine_era_for_shot(shot)
    manifest = _external_profile_manifest("ti_inferred", shot, run, era.name)
    manifest["input"] = {
        "core_profiles_product": str(core_profiles_product),
        "efit_product": str(efit_product),
    }
    manifest["configuration"].update(
        floor=float(floor), window_s=float(window_s),
        equilibrium_tolerance_ms=float(equilibrium_tolerance_ms),
    )
    if not Path(core_profiles_product).exists():
        raise FileNotFoundError(f"the core_profiles product is missing: {core_profiles_product}")
    ods, _ = load_ods(Path(core_profiles_product))
    times = path_value(ods, "core_profiles.time", None)
    if times is None or np.asarray(times).size == 0:
        manifest["error"] = "the core_profiles product carries no slices"
        return ods, manifest
    times = np.atleast_1d(np.asarray(times, dtype=float))

    lineage = _magnetic_efit_lineage(efit_product)
    manifest["configuration"]["equilibrium_lineage"] = lineage
    if lineage is None:
        manifest["error"] = (
            f"the efit product {efit_product} is not under omas/efit/magnetic, so its lineage is "
            "not established; the inference is refused rather than guessed"
        )
        return ods, manifest
    if not Path(efit_product).exists():
        manifest["error"] = f"the magnetic efit product is missing: {efit_product}"
        return ods, manifest
    solution, _ = load_ods(Path(efit_product))
    eq_times_ms = _time_array_ms(solution, "equilibrium.time")
    if eq_times_ms is None:
        manifest["error"] = "the efit product carries no equilibrium.time"
        return ods, manifest
    usable_ms = usable_equilibrium_times_ms(solution)

    counts = {"inferred": 0, "measured": 0, "unavailable": 0}
    skipped: list[dict[str, Any]] = []
    flag_totals: dict[str, int] = {}
    ratios: list[float] = []

    def infer_slice(index: int, time_s: float) -> None:
        base = f"core_profiles.profiles_1d.{index}"
        nearest = int(np.argmin(np.abs(eq_times_ms - time_s * 1e3)))
        if abs(float(eq_times_ms[nearest]) - time_s * 1e3) > equilibrium_tolerance_ms:
            raise _SliceSkipped(
                f"no magnetic equilibrium within {equilibrium_tolerance_ms:g} ms of the slice"
            )
        origin = ti_record_fields(path_value(ods, f"{base}.zeff_fit.parameters", None)).get("origin")
        if origin not in ("assumed", "derived"):
            raise _SliceSkipped(f"the slice carries no composition (zeff record origin={origin})")
        # n_i,tot/n_e from the stored ions; the ratio only lets the closure run
        # on a slice with no ion temperature -- no T_i enters the densities.
        closure = infer_kinetic_closure(ods, time=time_s, dilution="species", ti_te_ratio=1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.sum(list(closure.ion_densities.values()), axis=0) / closure.n_e

        electron = ks.core_profiles_electron_pressure(ods, time_s=time_s, tolerance_s=5e-4)
        if not electron["available"] or electron["index"] != index:
            raise _SliceSkipped(electron.get("reason", "the profile slice cannot be matched by time"))
        placed = ks.core_profiles_grid_on_slice(ks.rho_tor_norm_of(solution, nearest), electron)
        if not placed["available"]:
            raise _SliceSkipped(placed["reason"])
        psi = np.asarray(placed["psi_norm"], dtype=float)
        inside = psi <= 1.0
        if inside.sum() < 3:
            raise _SliceSkipped("fewer than three profile points inside the last closed flux surface")
        p_eq = _equilibrium_pressure_on(solution, nearest, psi[inside])
        if p_eq is None:
            raise _SliceSkipped("the equilibrium pressure cannot be sampled on the profile grid")
        others = []
        for k, eq_ms in enumerate(eq_times_ms):
            if k == nearest or abs(eq_ms * 1e-3 - time_s) > window_s + 1e-9:
                continue
            if usable_ms is not None and not np.any(np.abs(usable_ms - eq_ms) < 1e-6):
                continue
            pressure = _equilibrium_pressure_on(solution, k, psi[inside])
            if pressure is not None:
                others.append(pressure)
        sigma_p, basis = ks.pressure_sigma(p_eq, others, floor=floor)
        rel_n, rel_t = _thomson_relative_errors(ods, time_s)
        n_e, t_e = np.asarray(electron["n_e"], float)[inside], np.asarray(electron["t_e"], float)[inside]
        result = infer_ti_pressure_partition(
            p_eq, n_e, t_e,
            composition={"ion_density_per_electron": fraction[inside],
                         "composition_source": f"composition stage ions (zeff record origin={origin})"},
            sigma_p_eq=sigma_p, sigma_n_e=rel_n * n_e, sigma_t_e=rel_t * t_e,
            equilibrium_lineage=lineage,
        )
        if not result["eligible"]:
            raise _SliceSkipped(result["reason"])
        valid = np.isfinite(result["t_i"])
        if not valid.any():
            seen = sorted({f for flags in result["flags"] for f in flags})
            raise _SliceSkipped(f"no point has a positive, significant p_i (flags: {', '.join(seen) or 'none'})")

        rho = np.asarray(electron["rho_tor_norm"], dtype=float)
        t_i = np.full(rho.shape, np.nan)
        t_i[inside] = result["t_i"]
        record = "; ".join([
            inferred_ti_text("equilibrium_pressure_partition"),
            f"equilibrium_lineage={lineage}", "equilibrium_occurrence=0",
            f"equilibrium_time_s={float(eq_times_ms[nearest]) * 1e-3:.6f}",
            f"sigma_p_eq_basis={basis}", f"valid_points={int(valid.sum())}/{valid.size}",
            f"rho_valid={float(rho[inside][valid].min()):.4g}:{float(rho[inside][valid].max()):.4g}",
            "not measured",
        ])
        for k in range(len(ods[f"{base}.ion"])):
            ods[f"{base}.ion.{k}.temperature"] = t_i
            ods[f"{base}.ion.{k}.temperature_fit.parameters"] = record
        ods[f"{base}.t_i_average"] = t_i
        pressure = assemble_pressure(
            closure.n_e, closure.T_e, closure.ion_densities,
            {label: t_i for label in closure.ion_densities},
        )
        ods[f"{base}.pressure_ion_total"] = pressure.p_i_thermal
        ods[f"{base}.pressure_thermal"] = pressure.p_thermal
        ods[f"{base}.t_i_average_fit.parameters"] = record
        _write_species_pressures(ods, base)  # the composed ones were built on the assumed T_i
        _append_record_fields(ods, f"{base}.ion.0.density_fit.parameters", PRESSURE_CLOSURE_RECORD)
        for flags in result["flags"]:
            for flag in flags:
                flag_totals[flag] = flag_totals.get(flag, 0) + 1
        ratios.extend(float(x) for x in result["ti_te"][valid])

    for index, time_s in enumerate(times):
        base_i = f"core_profiles.profiles_1d.{index}.ion.0"
        has_ti = path_value(ods, f"{base_i}.temperature", None) is not None
        # A T_i the composition stage *assumed* (T_i = T_e) is not a measurement: it is
        # the electron-only state this stage exists to improve on.
        kind = classify_ti_record(path_value(ods, f"{base_i}.temperature_fit.parameters", None)) if has_ti else None
        if kind is not None and kind != "assumed":
            # measured, or a T_i some earlier writer inferred / left unclassifiable: kept, and named as such
            key = "measured" if kind == "measured" else f"kept_{kind}"
            counts[key] = counts.get(key, 0) + 1
            continue
        try:
            infer_slice(index, float(time_s))
        except _SliceSkipped as skip:
            counts["unavailable"] += 1
            skipped.append({"time_s": float(time_s), "reason": str(skip)[:240]})
            continue
        counts["inferred"] += 1

    manifest["slices"] = {**counts, "total": int(times.size)}
    manifest["skipped_slices"] = skipped[:20]
    manifest["inference"] = {
        "method": "equilibrium_pressure_partition", "equilibrium_lineage": lineage,
        "flags": {f: flag_totals.get(f, 0) for f in (*TI_FLAGS, "sigma_unavailable")},
        "ti_te_median": float(np.median(ratios)) if ratios else None,
    }
    if counts["inferred"]:
        manifest["status"] = "success"
        manifest["quality_summary"]["unavailable"] = []
        if skipped:
            manifest["quality_summary"]["missing"] = [
                f"ti_inferred:t={item['time_s']:.4f}s" for item in skipped[:20]
            ]
    else:
        manifest["error"] = (
            skipped[0]["reason"] if skipped
            else "every slice already carries an ion temperature that is not assumed: "
            + ", ".join(f"{k}={v}" for k, v in counts.items() if k != "inferred" and v)
        )
    return ods, manifest


#: The conductivity models Lane Z's estimator offers (#1214); ``redl`` is the neoclassical one.
RESISTIVE_MODELS = ("redl", "sauter", "spitzer_nrl")


def _composition_equivalent_scalar(
    states: Sequence[Any], solution: ODS, ods: ODS, *, model: str, ln_lambda: Any, bounds: tuple[float, float]
) -> dict[str, Any]:
    """The resistive scalar the stored composition ``Z_eff(rho)`` would give, for the cross-check.

    Each state's surfaces are put on the composition's ``rho_tor_norm`` through
    the equilibrium's own ``rho_tor(psi_N)``, and Lane L's projection
    (:func:`vaft.process.zeff_projection.project_window_to_resistive_scalar`)
    turns the profiles into the scalar Lane Z's estimator would report if they
    were the truth -- the same objective, so the two numbers are comparable.
    """
    from vaft.machine_mapping.core_profiles import ti_record_fields
    from vaft.ods_access import path_value
    from vaft.process.zeff_projection import project_window_to_resistive_scalar
    from vaft.validation import kinetic_state as ks

    profiles, used, skipped = [], [], []
    for state in states:
        base = f"core_profiles.profiles_1d.{state.source['cp_index']}"
        origin = ti_record_fields(path_value(ods, f"{base}.zeff_fit.parameters", None)).get("origin")
        zeff = path_value(ods, f"{base}.zeff", None)
        rho_cp = path_value(ods, f"{base}.grid.rho_tor_norm", None)
        if origin not in ("assumed", "derived") or zeff is None or rho_cp is None:
            skipped.append(f"t={state.time:.4f}: no composition Z_eff")
            continue
        coordinate = ks.rho_tor_norm_of(solution, int(state.source["eq_index"]))
        if coordinate.get("coordinate") != "rho_tor_norm":
            skipped.append(f"t={state.time:.4f}: {coordinate.get('reason')}")
            continue
        order = np.argsort(coordinate["psi_norm"])
        rho_state = np.interp(state.psi_norm, coordinate["psi_norm"][order], coordinate["rho_tor_norm"][order])
        rho_cp, zeff = np.asarray(rho_cp, dtype=float), np.asarray(zeff, dtype=float)
        good = np.isfinite(zeff) & np.isfinite(rho_cp)
        if good.sum() < 2:
            skipped.append(f"t={state.time:.4f}: composition Z_eff undefined")
            continue
        sort = np.argsort(rho_cp[good])
        # Surfaces beyond the composition's support take its end values (np.interp holds them).
        profiles.append(np.interp(rho_state, rho_cp[good][sort], zeff[good][sort]))
        used.append(state)
    if not used:
        return {"available": False, "reason": "; ".join(skipped) or "no state"}
    projection = project_window_to_resistive_scalar(
        used, profiles, model=model, ln_lambda=ln_lambda, bounds=bounds,
        weights="uniform", profile_source="composition stage",
    )
    return {"available": True, "zeff_equivalent": float(projection.zeff_equivalent),
            "status": str(projection.convergence["status"]), "states": len(used), "skipped": skipped}


def _romero_window(
    work: ODS,
    *,
    model: str,
    ln_lambda: Any,
    bounds: tuple[float, float],
    i_ni_a: float,
    smoothing: Any,
    time_range: tuple[float, float] | None,
    max_excluded_current: float,
    inductive_max: float,
    time_tolerance_s: float,
) -> dict[str, Any]:
    """Lane Z's Romero window estimate on an ODS with an equilibrium and ``core_profiles`` slices.

    The orchestration the resistive stage and the staged kinetic-state inference
    share: Romero's boundary-flux balance and the flux-surface states from
    :func:`vaft.omas.resistive_zeff.compute_resistive_zeff_ods`, the two state
    filters of Lane Z's atlas, then the window estimate and the per-slice
    diagnostic.  Nothing is stored and nothing is judged: the result carries
    ``error`` (a message) when no estimate could be made, and ``inference`` is
    ``None`` then.

    Keys: ``result`` (Lane Z's dict), ``window``, ``observed``, ``states``,
    ``skipped``, ``usable``, ``inference``, ``diagnostic``, ``error``.
    """
    from vaft.omas.resistive_zeff import compute_resistive_zeff_ods
    from vaft.process.resistive_zeff import infer_resistive_zeff, per_slice_resistive_zeff

    out: dict[str, Any] = {
        "result": None, "window": None, "observed": None, "states": [], "skipped": [], "usable": [],
        "inference": None, "diagnostic": None, "error": None,
    }
    try:
        result = compute_resistive_zeff_ods(
            work, time_range=time_range, model=model, ln_lambda=ln_lambda, bounds=bounds,
            I_ni=float(i_ni_a), smoothing=smoothing, time_tolerance_s=time_tolerance_s,
        )
    except (ValueError, KeyError, IndexError) as error:
        # Lane Z refuses on purpose (an undetected flux unit, no current window, ...):
        # that is an absence for this shot, with the reason, not a failed stage.
        out["error"] = f"{type(error).__name__}: {error}"[:300]
        return out
    observed, states = result["observed"], result["states"]
    out.update(result=result, window=result["window"], observed=observed, states=states,
               skipped=result["skipped"])

    # The model leaves out the surfaces beyond the profiles' support -- the cold, most
    # resistive edge -- while the observed R_p includes them, so a state missing much of
    # the current would bias Z_eff high; and where V_I dwarfs V_B, V_R is a small
    # difference of large numbers. Lane Z's atlas applies the same two filters.
    def inductive_at(state):
        return float(observed.inductive_fraction[int(np.argmin(np.abs(observed.time - state.time)))])

    usable = [
        s for s in states
        if (s.source.get("excluded_current_fraction") or 0.0) <= max_excluded_current
        and inductive_at(s) < inductive_max
    ]
    out["usable"] = usable
    if not usable:
        out["error"] = (
            f"no kinetic state is usable: {len(states)} built, each either beyond "
            f"{max_excluded_current:g} of j_tor outside the profiles or with |V_I|/|V_B| >= {inductive_max:g}"
            if states else "no core_profiles slice in the window has electron profiles at an equilibrium time"
        )
        return out
    kwargs = dict(model=model, ln_lambda=ln_lambda, bounds=bounds, time_tolerance_s=time_tolerance_s)
    out["inference"] = infer_resistive_zeff(observed, usable, weights="uniform", **kwargs)
    out["diagnostic"] = per_slice_resistive_zeff(observed, usable, **kwargs)
    return out


def build_resistive_zeff_ods(
    *,
    shot: int,
    core_profiles_product: str | Path,
    efit_product: str | Path,
    model: str = "redl",
    ln_lambda: Any = "sauter",
    bounds: tuple[float, float] = (1.0, 8.0),
    i_ni_a: float = 0.0,
    smoothing_window_s: float | None = None,
    smoothing_order: int = 2,
    time_range: tuple[float, float] | None = None,
    max_excluded_current: float = 0.05,
    inductive_max: float = 1.0,
    time_tolerance_s: float = 5e-5,
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Add the Romero resistive ``Z_eff`` to ``core_profiles.global_quantities.z_eff_resistive``.

    The pipeline-1 magnetic EFIT supplies the boundary flux, ``I_p`` and
    ``l_i`` over the current-carrying window (Romero's transformer balance,
    :func:`vaft.omas.resistive_zeff.compute_resistive_zeff_ods`) and the
    Grad-Shafranov geometry at each kinetic slice; the profiles supply
    ``T_e`` and ``n_e``.  Lane Z's estimator then finds the one ``Z_eff`` that
    makes the model resistance match the observed one.  Nothing of that physics
    is re-implemented here.

    What is stored is the **window estimate**: one scalar for the whole window
    (#1214 Sec. 9), written at every usable kinetic slice and NaN elsewhere.
    The per-slice solutions are a diagnostic -- each carries the full noise of a
    differentiated ``V_R`` -- and go to the manifest, labelled so, not into the
    IDS.  A fit that sits on a bound (``bound_hit``) or is not identifiable is
    recorded and not stored.  The value is a model-inferred estimate, kept apart
    from the composition ``Z_eff`` (``profiles_1d.zeff``), which stays as it was.

    Provenance goes in the slice's ``zeff_fit.parameters`` as ``resistive_*``
    fields (never ``origin``/``method``, which belong to the composition):
    model, conductivity, window, status, uncertainty, the equilibrium lineage
    and occurrence, ``I_ni`` and the smoothing -- stated, because Lane Z's
    estimator has no defaults for them.

    The manifest cross-checks the result against the composition: the stored
    ``Z_eff(rho)`` is turned into the scalar the same estimator would report
    were it the truth (Lane L's projection) and the two are compared.

    Only the ``efit/magnetic`` lineage is accepted, read from where the product
    lives: the geometry and ``<J.B>`` must not come from a pressure fitted with
    an assumed ``T_i``.  The product is read from the inferred-``T_i`` stage's,
    so the overlay that replaces ``core_profiles`` in ``main`` keeps what that
    stage and the composition wrote.
    """
    import copy

    from vaft.ods_access import path_value
    from vaft.process.resistive_zeff import (
        Smoothing,
    )

    if model not in RESISTIVE_MODELS:
        raise ValueError(f"model must be one of {RESISTIVE_MODELS}, got {model!r}")
    shot = int(shot)
    bounds = (float(bounds[0]), float(bounds[1]))
    smoothing = (
        Smoothing("none") if smoothing_window_s is None
        else Smoothing("local_polynomial", window_s=float(smoothing_window_s), order=int(smoothing_order))
    )
    era = machine_era_for_shot(shot)
    manifest = _external_profile_manifest("resistive_zeff", shot, run, era.name)
    manifest["input"] = {
        "core_profiles_product": str(core_profiles_product), "efit_product": str(efit_product),
    }
    lineage = _magnetic_efit_lineage(efit_product)
    manifest["configuration"].update(
        model=model, ln_lambda=str(ln_lambda), bounds=list(bounds), i_ni_a=float(i_ni_a),
        smoothing=smoothing.describe(), max_excluded_current=float(max_excluded_current),
        inductive_max=float(inductive_max), equilibrium_lineage=lineage,
        time_range=None if time_range is None else [float(time_range[0]), float(time_range[1])],
    )
    if not Path(core_profiles_product).exists():
        raise FileNotFoundError(f"the core_profiles product is missing: {core_profiles_product}")
    ods, _ = load_ods(Path(core_profiles_product))
    times = path_value(ods, "core_profiles.time", None)
    if times is None or np.asarray(times).size == 0:
        manifest["error"] = "the core_profiles product carries no slices"
        return ods, manifest
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if lineage is None:
        manifest["error"] = (
            f"the efit product {efit_product} is not under omas/efit/magnetic, so its lineage is "
            "not established; the geometry is refused rather than guessed"
        )
        return ods, manifest
    if not Path(efit_product).exists():
        manifest["error"] = f"the magnetic efit product is missing: {efit_product}"
        return ods, manifest
    solution, _ = load_ods(Path(efit_product))
    if not _path_present(solution, "equilibrium.time_slice") or len(solution["equilibrium.time_slice"]) < 3:
        manifest["error"] = "the efit product has fewer than three equilibrium slices; the balance needs a window"
        return ods, manifest

    work = ODS(consistency_check=False)
    work["core_profiles"] = copy.deepcopy(ods["core_profiles"])
    work["equilibrium"] = solution["equilibrium"]
    window = _romero_window(
        work, model=model, ln_lambda=ln_lambda, bounds=bounds, i_ni_a=i_ni_a, smoothing=smoothing,
        time_range=time_range, max_excluded_current=max_excluded_current,
        inductive_max=inductive_max, time_tolerance_s=time_tolerance_s,
    )
    if window["result"] is None:
        manifest["error"] = window["error"]
        return ods, manifest
    observed, states, usable = window["observed"], window["states"], window["usable"]
    manifest["window"] = {"t_start_s": float(window["window"][0]), "t_end_s": float(window["window"][1])}
    manifest["states"] = {
        "built": len(states), "skipped": [{"time_s": t, "reason": r[:200]} for t, r in window["skipped"][:20]],
        "usable": len(usable),
    }
    if window["inference"] is None:
        manifest["error"] = window["error"]
        return ods, manifest
    inference, diagnostic = window["inference"], window["diagnostic"]
    result = {"window": window["window"]}
    estimate = inference.estimate
    manifest["inference"] = {
        "status": inference.status, "zeff": inference.zeff, "uncertainty": estimate.get("uncertainty"),
        "reason": inference.reason, "median_inductive_fraction": float(np.nanmedian(observed.inductive_fraction)),
    }
    manifest["per_slice_diagnostic"] = {
        "note": "each solved alone: the full noise of a differentiated V_R; a diagnostic, not the estimate",
        "time_s": [float(t) for t in diagnostic["time"]],
        "zeff": [None if not np.isfinite(z) else float(z) for z in diagnostic["zeff"]],
        "status": list(diagnostic["status"]),
    }
    manifest["crosscheck"] = _composition_equivalent_scalar(
        usable, solution, ods, model=model, ln_lambda=ln_lambda, bounds=bounds)
    if manifest["crosscheck"]["available"] and inference.zeff is not None:
        check = manifest["crosscheck"]
        check["resistive_zeff"] = float(inference.zeff)
        check["difference"] = check["zeff_equivalent"] - float(inference.zeff)
        check["relative_difference"] = check["difference"] / float(inference.zeff)

    if inference.status != "ok":
        manifest["error"] = (
            f"the window estimate is {inference.status}"
            + (f" ({inference.reason})" if inference.reason else "")
            + "; it is recorded here and not stored as z_eff_resistive"
        )
        return ods, manifest

    stored = np.full(times.shape, np.nan)
    fields = {
        "resistive_origin": "inferred", "resistive_model": "romero_transformer",
        "resistive_conductivity": model, "resistive_scope": "window",
        "resistive_window": f"{result['window'][0]:.4f}:{result['window'][1]:.4f}",
        "resistive_status": inference.status, "resistive_n_states": len(usable),
        "resistive_equilibrium_lineage": lineage, "resistive_equilibrium_occurrence": 0,
        "resistive_i_ni_a": float(i_ni_a), "resistive_smoothing": smoothing.describe().replace("=", ":"),
        "resistive_ln_lambda": str(ln_lambda),
    }
    if estimate.get("uncertainty") is not None and np.isfinite(estimate["uncertainty"]):
        fields["resistive_sigma"] = f"{float(estimate['uncertainty']):.4g}"
    for state in usable:
        index = int(state.source["cp_index"])
        stored[index] = float(inference.zeff)
        _append_record_fields(ods, f"core_profiles.profiles_1d.{index}.zeff_fit.parameters", fields)
    ods["core_profiles.global_quantities.z_eff_resistive"] = stored
    manifest["status"] = "success"
    manifest["quality_summary"]["unavailable"] = []
    manifest["stored"] = {"slices": int(np.isfinite(stored).sum()), "total": int(times.size)}
    return ods, manifest


def _has_kinetic_slice(ods: ODS) -> bool:
    """Whether any core_profiles slice carries a measured ion temperature.

    Measured is read from the slice's ion temperature record
    (:func:`vaft.machine_mapping.core_profiles.classify_ti_record`), not from the
    presence of ``ion.0.temperature``.
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
    # A T_i the composition stage assumed (T_i = T_e) or ti_inferred inferred does
    # not make a slice kinetic: its record decides (no record means measured).
    # strip_electron_only_pressure keys on presence, which agrees on the measured
    # core_profiles product it runs on -- no assumed or inferred T_i exists there.
    from vaft.machine_mapping.core_profiles import classify_ti_record

    def measured(index: int) -> bool:
        base = f"core_profiles.profiles_1d.{index}.ion.0"
        if not _path_present(ods, f"{base}.temperature"):
            return False
        record = ods[f"{base}.temperature_fit.parameters"] if _path_present(
            ods, f"{base}.temperature_fit.parameters") else None
        return classify_ti_record(record) == "measured"

    return any(measured(index) for index in range(count))


KINETIC_DILUTIONS = ("species", "none")


def _ion_density_ratio_function(
    ods: ODS, time_ms: float, *, tolerance_s: float = 5e-4
) -> tuple[Callable[[np.ndarray], np.ndarray] | None, dict[str, Any]]:
    """``f(psi_N) = n_i,tot / n_e`` of the composed slice at ``time_ms``, and the record of it.

    Taken from the ions the composition stage stored, through the species
    kinetic closure, so the EFIT pressure points count the same ions the
    product's ``pressure_thermal`` does.  ``(None, {"applied": False, ...})``
    names why when the slice carries no composition -- the pressure is then the
    legacy ``n_i = n_e``, and the record says so rather than leaving it to be
    inferred from a number.  ``T_i`` plays no part: a placeholder ratio only
    lets the closure run on a slice with no ion temperature.
    """
    from vaft.machine_mapping.core_profiles import ti_record_fields
    from vaft.ods_access import path_value
    from vaft.process.kinetic_closure import infer_kinetic_closure

    record: dict[str, Any] = {"mode": "species", "applied": False}
    times = path_value(ods, "core_profiles.time", None)
    if times is None:
        record["reason"] = "the product carries no core_profiles slice"
        return None, record
    times = np.atleast_1d(np.asarray(times, dtype=float))
    index = int(np.argmin(np.abs(times - time_ms * 1e-3)))
    if abs(float(times[index]) - time_ms * 1e-3) > tolerance_s:
        record["reason"] = "no core_profiles slice at the reconstruction time"
        return None, record
    base = f"core_profiles.profiles_1d.{index}"
    origin = ti_record_fields(path_value(ods, f"{base}.zeff_fit.parameters", None)).get("origin")
    if origin not in ("assumed", "derived"):
        record["reason"] = f"the slice carries no composition (zeff record origin={origin})"
        return None, record
    try:
        closure = infer_kinetic_closure(
            ods, time=float(times[index]), dilution="species", ti_te_ratio=1.0
        )
    except ValueError as error:
        record["reason"] = f"{type(error).__name__}: {error}"[:200]
        return None, record
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.sum(list(closure.ion_densities.values()), axis=0) / closure.n_e
    psi = path_value(ods, f"{base}.grid.psi", None)
    if psi is not None:
        psi = np.asarray(psi, dtype=float)
        psi_n = (psi - psi[0]) / (psi[-1] - psi[0])
    else:
        rho_pol = path_value(ods, f"{base}.grid.rho_pol_norm", None)
        if rho_pol is None:
            record["reason"] = "the slice has neither grid.psi nor grid.rho_pol_norm"
            return None, record
        psi_n = np.asarray(rho_pol, dtype=float) ** 2
    ok = np.isfinite(ratio) & np.isfinite(psi_n) & (ratio > 0.0)
    if ok.sum() < 2:
        record["reason"] = "fewer than two usable n_i/n_e points"
        return None, record
    order = np.argsort(psi_n[ok])
    x, y = psi_n[ok][order], ratio[ok][order]

    def ion_density_ratio(psi_normalised: np.ndarray) -> np.ndarray:
        # np.interp holds the end values, so a point in the scrape-off layer
        # (psi_N > 1) takes the last confined ratio rather than extrapolating.
        return np.interp(np.asarray(psi_normalised, dtype=float), x, y)

    record.update(
        applied=True, closure="infer_kinetic_closure", slice_time_s=float(times[index]),
        n_i_over_n_e_axis=float(y[0]), n_i_over_n_e_edge=float(y[-1]),
        composition_source=str(closure.provenance["composition"]["source"]),
    )
    return ion_density_ratio, record


def kinetic_efit_parameters(
    stage: str, efit_preset: str | None, dilution: Mapping[str, Any] | None = None
) -> str:
    """The ``equilibrium.code.parameters`` payload of a kinetic EFIT product.

    Names the preset that built the base k-file and, as the top-level
    ``uncertainty_model`` that :func:`vaft.omas.efit_quality.constraint_uncertainty_model`
    reads, the uncertainty model its constraints were fitted under.  No preset
    named is the default (``DEFAULT_PRESET``), which is what
    :func:`vaft.code.efit.kinetic.prepare_kinetic_efit_inputs` builds the
    k-file with then, so it is recorded by that name.

    ``dilution`` is the record of the ion-density correction the pressure
    constraint was built with (:func:`_ion_density_ratio_function`), so a
    reader can tell ``n_i = n_e`` from a diluted composition.
    """
    from vaft.code.efit.presets import DEFAULT_PRESET
    from vaft.code.efit.presets import efit_preset as _efit_preset

    preset = _efit_preset(efit_preset or DEFAULT_PRESET)
    record = preset.record()
    scientific = preset.scientific
    return json.dumps(
        {
            "kinetic_efit": {
                "stage": stage, "efit_preset": record,
                **({"dilution": dict(dilution)} if dilution is not None else {}),
            },
            "uncertainty_model": scientific.constraints.uncertainty_mode,
        },
        sort_keys=True,
    )


def usable_equilibrium_times_ms(solution: Any) -> np.ndarray | None:
    """Times [ms] of the slices the EFIT collection judged ``usable``.

    ``None`` when the product records no per-slice verdict (written before the
    collection stored one, or not an EFIT product), so a caller can tell "no
    verdicts" from "no usable slice".
    """
    try:
        payload = json.loads(str(solution["equilibrium.code.parameters"]))
        statuses = payload["efit_collection"]["slice_statuses"]
    except Exception:  # noqa: BLE001 - absent or older payloads carry no verdicts
        return None
    return np.asarray(
        [1e3 * float(entry["time"]) for entry in statuses if entry.get("overall_status") == "usable"],
        dtype=float,
    )


#: How many aligned profile times the kinetic lineages try before giving up.
KINETIC_MAX_ATTEMPTS = 3


def profile_time_candidates(
    profile_times_ms: np.ndarray,
    eq_times_ms: np.ndarray,
    usable_eq_times_ms: np.ndarray | None,
    *,
    tolerance_ms: float = EQUILIBRIUM_TIME_TOLERANCE_MS,
) -> tuple[list[float], str]:
    """Profile times to try, best first, and why: :func:`choose_profile_time`'s
    choice first, then every other aligned, usable one by offset and time."""
    first, why = choose_profile_time(profile_times_ms, eq_times_ms, usable_eq_times_ms, tolerance_ms=tolerance_ms)
    profile = np.asarray(profile_times_ms, dtype=float).reshape(-1)
    eq = np.asarray(eq_times_ms, dtype=float).reshape(-1)
    distance = np.abs(profile[:, None] - eq[None, :])
    offsets = distance.min(axis=1)
    nearest = eq[distance.argmin(axis=1)]
    aligned = offsets <= tolerance_ms
    if why == "aligned and usable":
        usable = np.round(np.asarray(usable_eq_times_ms, dtype=float).reshape(-1), 3)
        good = np.flatnonzero(aligned & np.isin(np.round(nearest, 3), usable))
        order = sorted(good, key=lambda i: (offsets[i], profile[i]))
        return [float(profile[i]) for i in order], why
    usable = np.asarray(usable_eq_times_ms if usable_eq_times_ms is not None else [], dtype=float).reshape(-1)
    if why == "aligned (no aligned slice is usable)" and usable.size and aligned.any():
        # Thomson fired outside the magnetically usable window (the 42xxx
        # series: profiles at 322-331 ms, usable EFIT from 333 ms). The kinetic
        # fit is likeliest to converge next to that window, so try the aligned
        # times closest to it first rather than the earliest.
        gap = np.min(np.abs(profile[:, None] - usable[None, :]), axis=1)
        order = sorted(np.flatnonzero(aligned), key=lambda i: (gap[i], offsets[i], profile[i]))
        return [float(profile[i]) for i in order], "aligned, nearest the usable window"
    return [first], why


def choose_profile_time(
    profile_times_ms: np.ndarray,
    eq_times_ms: np.ndarray,
    usable_eq_times_ms: np.ndarray | None,
    *,
    tolerance_ms: float = EQUILIBRIUM_TIME_TOLERANCE_MS,
) -> tuple[float, str]:
    """The profile time to reconstruct at, and why it was chosen.

    The best-aligned slice, not the first one: which profile time comes first
    says nothing about which has an equilibrium to reconstruct against.  Among
    the aligned ones, a slice whose magnetic equilibrium the collection judged
    ``usable`` wins -- when Thomson and EFIT share a millisecond grid every
    profile time is aligned, and the first of them was 39915 @ 312 ms, a
    negative-pressure slice the kinetic fit could not converge from (#1331).
    Falls back to the best-aligned slice when no aligned one is usable or the
    product records no verdicts; the returned reason says which happened.
    """
    profile = np.asarray(profile_times_ms, dtype=float).reshape(-1)
    eq = np.asarray(eq_times_ms, dtype=float).reshape(-1)
    distance = np.abs(profile[:, None] - eq[None, :])
    offsets = distance.min(axis=1)
    best = float(profile[int(np.argmin(offsets))])
    if usable_eq_times_ms is None:
        return best, "aligned (no per-slice verdicts)"
    usable = np.asarray(usable_eq_times_ms, dtype=float).reshape(-1)
    nearest = eq[distance.argmin(axis=1)]
    good = (offsets <= tolerance_ms) & np.isin(np.round(nearest, 3), np.round(usable, 3))
    if not good.any():
        return best, "aligned (no aligned slice is usable)"
    candidates = np.flatnonzero(good)
    return float(profile[candidates[int(np.argmin(offsets[candidates]))]]), "aligned and usable"


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
    efit_preset: str | None = None,
    dilution: str = "species",
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

    ``dilution="species"`` (default) builds the raw pressure points with the ion
    density the composition stage stored, ``p = e n_e (T_e + f T_i)`` with
    ``f = n_i,tot / n_e`` at each Thomson radius; a slice that carries no
    composition falls back to ``n_i = n_e`` and the manifest says why.
    ``"none"`` is the legacy pressure.  Read the composed product
    (``composition``) to get the dilution; the measured one has none to apply.
    """
    from vaft.code.efit.kinetic import KineticEFITConfig, run_kinetic_chain
    from vaft.data.eqdsk import from_equilibrium, to_omas
    from vaft.process.equilibrium import as_equilibrium

    if stage not in ("electron_efit", "kinetic_efit"):
        raise ValueError(f"stage must be electron_efit or kinetic_efit; got {stage!r}")
    if dilution not in KINETIC_DILUTIONS:
        raise ValueError(f"dilution must be one of {KINETIC_DILUTIONS}, got {dilution!r}")

    shot = int(shot)
    era = machine_era_for_shot(shot)
    manifest = _external_profile_manifest(stage, shot, run, era.name)
    manifest["input"] = {
        "core_profiles_product": str(core_profiles_product),
        "constraints_product": str(constraints_product),
        "efit_product": str(efit_product),
    }
    manifest["configuration"]["encoding"] = encoding
    manifest["configuration"]["dilution"] = dilution
    # Always: a product names the configuration that built it, the default
    # included (DEFAULT_PRESET when none was named).
    from vaft.code.efit.presets import DEFAULT_PRESET
    from vaft.code.efit.presets import efit_preset as _efit_preset

    manifest["configuration"]["efit_preset"] = _efit_preset(efit_preset or DEFAULT_PRESET).record()

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
        candidates, choice = profile_time_candidates(
            profile_times, eq_times, usable_equilibrium_times_ms(solution),
            tolerance_ms=equilibrium_tolerance_ms,
        )
        candidates = candidates[:KINETIC_MAX_ATTEMPTS]
        manifest["configuration"]["time_choice"] = choice
    else:
        candidates = [float(time_ms)]
        manifest["configuration"]["time_choice"] = "caller"
    manifest["configuration"]["equilibrium_tolerance_ms"] = float(equilibrium_tolerance_ms)

    # A caller-supplied workdir is the caller's to keep. One we invent is ours to
    # remove: EFIT leaves a kfile and its outputs per PLASMA scale, and this stage
    # is registered for every shot, so leaking one directory per shot would fill
    # the filesystem the products themselves live on.
    ephemeral = workdir is None
    work = Path(tempfile.mkdtemp(prefix=f"{stage}-")) if ephemeral else Path(workdir)
    attempts: list[dict[str, Any]] = []
    retry = len(candidates) > 1

    def record(time_ms: float, **fields: Any) -> None:
        if retry:
            attempts.append({"time_ms": float(time_ms), **fields})
            manifest["attempts"] = attempts

    try:
        # Candidates in order until one converges (#1331): a slice whose
        # magnetic fit is usable can still lose its boundary once the pressure
        # constraint is added (39915 @ 315 ms), while its neighbour converges.
        for time_ms in candidates:
            # Nothing from a previous attempt may describe this one.
            for key in ("reconstruction", "traceback", "error"):
                manifest.pop(key, None)
            time_index = int(np.argmin(np.abs(eq_times - float(time_ms))))
            manifest["configuration"]["time_ms"] = float(time_ms)
            manifest["configuration"]["equilibrium_time_ms"] = float(eq_times[time_index])
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

            # The ODS psi is weber (#278/#281); from_equilibrium writes it per radian,
            # as a g-file stores it (#1292), so SIMAG/SIBRY are the solution's flux over
            # 2*pi and `geq.to_omas()` gives back the solution's own weber. That matters:
            # `core_profiles` takes `grid.psi` from `geq.to_omas()`, which before #1292
            # came out 2*pi too large. psi_N in `_psin_of_r` and the profile mapper is
            # normalized and was never affected.
            try:
                geq = from_equilibrium(as_equilibrium(solution, time_index=time_index))
            except Exception as error:  # noqa: BLE001 - an unusable equilibrium is not a crash
                message = f"cannot build a psi map from the equilibrium: {error}"
                if not retry:
                    return unavailable(message)
                manifest["status"], manifest["error"] = "unavailable", message
                record(time_ms, status="unavailable", reason=message)
                continue

            if dilution == "species":
                ion_density_ratio, dilution_record = _ion_density_ratio_function(ods, float(time_ms))
            else:
                ion_density_ratio, dilution_record = None, {"mode": "none", "applied": False}
            manifest["dilution"] = dilution_record
            config = KineticEFITConfig(
                executable=executable,
                workdir=work / f"t{time_ms:.3f}" if retry else work,
                shot=shot,
                time_ms=float(time_ms),
                encoding=encoding,
                # 'auto' is the VEST policy ratio, and it only applies when the ODS
                # has no ion data -- which is exactly the electron_efit case.
                ti_te_ratio="auto",
                efit_preset=efit_preset or None,
                ion_density_ratio=ion_density_ratio,
            )
            try:
                chain = run_kinetic_chain(ods, geq, float(time_ms), efit_config=config)
            except Exception as error:  # noqa: BLE001 - recorded, one shot cannot fail a run
                manifest["status"] = "failed"
                manifest["error"] = f"{type(error).__name__}: {error}"
                manifest["traceback"] = traceback.format_exc()
                record(time_ms, status="failed", reason=manifest["error"])
                continue  # a failure is the slice's, not the stage's: try the next

            result = chain["kinetic_efit"]
            manifest["reconstruction"] = {
                "status": result.status,
                "converged": bool(result.converged),
                "scale": result.scale,
                "chi2": result.chi2,
                "reason": result.reason,
            }
            # The attempt log carries the stage's verdict on the attempt, not
            # the chain's "EFIT ran" status: an attempt that ran and did not
            # converge is the stage's ``no_output``, and logging the chain's
            # ``ok`` beside ``converged: False`` overstated it (cold review
            # 0.8.0 delta-absorb-2 F3).
            if result.status == "skipped":
                verdict = "unavailable"
            elif not result.converged or result.gfile is None:
                verdict = "no_output"
            else:
                verdict = "success"
            record(time_ms, **{**manifest["reconstruction"], "status": verdict})
            if result.status == "skipped":
                return unavailable(result.reason or "EFIT executable is not configured")
            if not result.converged or result.gfile is None:
                manifest["status"] = "no_output"
                manifest["error"] = result.reason or "no PLASMA scale converged"
                continue

            # Read the g-file before the workdir goes away.
            to_omas(read_geqdsk_file(result.gfile), ods=out)
            # The g-file carries no record of how the constraints were weighted;
            # the preset (or its absence) does.  Serialized, as `code.parameters`
            # is a STR_0D that a nested tree does not survive (#380/#642).
            out["equilibrium.code.parameters"] = kinetic_efit_parameters(
                stage, efit_preset, manifest.get("dilution")
            )
            manifest["status"] = "success"
            manifest["quality_summary"]["unavailable"] = []
            # Recording a path under a directory about to be deleted would name a
            # file nobody can open; the name is what identifies the reconstruction.
            manifest["output_gfile"] = (
                Path(result.gfile).name if ephemeral else str(result.gfile)
            )
            return out, manifest
        return out, manifest
    finally:
        if ephemeral:
            shutil.rmtree(work, ignore_errors=True)


def read_geqdsk_file(path: str | Path):
    """Read a g-file. Thin indirection so the builder stays import-light."""
    from vaft.data import read_geqdsk

    return read_geqdsk(path)


def build_neoclassical_ods(
    *,
    shot: int,
    state: str | Path,
    run_directory: str | Path,
    time: float | None = None,
    time_index: int = 0,
    z_eff: float | None = None,
    ion_index: int = 0,
    rho_max: float | None = 0.95,
    lineage: str = "kinetic",
    run: int = 1,
) -> tuple[ODS, dict[str, Any]]:
    """Build the ``neoclassical`` stage product from one NEO run (#550 phase 8).

    ``state`` is the qualified kinetic state the run was made from and
    ``run_directory`` the directory NEO wrote into. The product holds what NEO
    produced -- ``core_profiles.j_bootstrap`` and the ``core_transport`` fluxes --
    plus the identity of the state it came from, not a copy of that state: the
    arrangement the per-product stability sources use, and what keeps this product
    from re-publishing profiles the kinetic stage already owns.

    The state is qualified before anything is mapped. A state that does not qualify
    raises, because a product built from an input the rule rejects would be exactly
    the unqualified promotion #550 defers: the rule has to gate the write, not merely
    describe it afterwards.

    ``lineage`` is the caller's statement of which reconstruction the state came from,
    and it is an argument rather than something read back because no ODS records it:
    an electron-only fit with an assumed Ti/Te ratio carries an ion temperature that
    looks exactly like a measured one. Only the kinetic lineage has a destination
    today -- ``kinetic-efit/neoclassical`` in
    :data:`vaft.database.sources.STAGE_REPLICATION` -- so anything else is refused
    here rather than published into a source that would misdescribe it.

    Returns the product and its manifest. The manifest records the readiness result
    in full, so the assumptions a number rests on -- an assumed Z_eff above all --
    travel with the number rather than living in whoever ran it.
    """
    from vaft.code.gacode.neo.outputs import collect_neo_outputs
    from vaft.machine_mapping.neoclassical import (
        core_profiles_from_neo,
        core_transport_from_neo,
    )
    from vaft.validation.neoclassical import input_readiness

    shot = int(shot)
    if lineage != "kinetic":
        raise ValueError(
            f"lineage {lineage!r} has no neoclassical destination; only 'kinetic' is "
            "wired, because the ion temperature NEO needs is measured only on that "
            "lineage. Add a source to vaft.database.sources.STAGE_REPLICATION before "
            "promoting another."
        )
    state_path = Path(state)
    source_state, _ = load_ods(state_path)

    readiness = input_readiness(
        source_state,
        time=time,
        time_index=time_index if time is None else None,
        rho_max=rho_max,
        z_eff=z_eff,
        ion_index=ion_index,
    )
    if not readiness["qualified"]:
        reasons = "; ".join(
            f"{check['name']}: {check['detail']}"
            for check in readiness["requirements"]
            if not check["satisfied"]
        )
        raise ValueError(
            f"shot {shot} does not qualify for neoclassical promotion ({reasons}). "
            "vaft.validation.neoclassical.input_readiness is the rule; a state that "
            "fails it is not promoted with a caveat, it is not promoted."
        )

    native = collect_neo_outputs(run_directory)
    if native is None or not native.solved:
        missing = "no NEO output in that directory" if native is None else (
            "the run did not solve: " + ", ".join(native.errors or native.missing())
        )
        raise ValueError(f"shot {shot} has no usable NEO result ({missing})")

    resolved = readiness["provenance"].get("time", {})
    slice_time = float(
        resolved.get("equilibrium_time", time if time is not None else 0.0)
    )
    b0 = _vacuum_field_at(source_state, resolved.get("equilibrium_index", 0))

    ods = ODS(consistency_check=False)
    dataset_description(
        ods,
        shot,
        {
            "source_type": "shot",
            "run": run,
            "machine": "VEST",
            "user": "vaft",
            "description": (
                f"VEST neoclassical transport from NEO; machine era "
                f"{machine_era_for_shot(shot).name}"
            ),
        },
    )
    profiles = core_profiles_from_neo(
        ods, native, time=slice_time, time_index=0, b0=b0
    )
    # The product is a fresh ODS without core_profiles ions, so each ion's nuclear
    # charge is read from the slice NEO's input was paired from -- by index, since
    # slice_time is the equilibrium's and the profile slice may sit off it.
    cp_index = resolved.get("core_profiles_index")
    transport = core_transport_from_neo(
        ods, native, time=slice_time, time_index=0,
        species_source=source_state,
        species_index=None if cp_index is None else int(cp_index),
    )

    written = tuple(profiles["written"]) + tuple(transport["written"])
    # The bootstrap current is what this product is for. A run that mapped only the
    # fluxes -- because no vacuum field was available to normalise by, say -- is a
    # partial result, and calling it "success" would publish a product the summary
    # then produces no row for, indistinguishable from a shot never run.
    if "j_bootstrap" in written:
        status = "success"
    elif written:
        status = "partial"
    else:
        status = "empty"
    manifest = {
        "schema_version": 1,
        "stage": "neoclassical",
        "shot": shot,
        "machine_version": machine_era_for_shot(shot).name,
        "status": status,
        "input": {
            "state": state_path.name,
            "state_sha256": sha256_file(state_path),
            # The run directory's *name* only: an absolute path is meaningless on
            # every machine but the one that ran it, and this manifest is published.
            "run_name": Path(run_directory).name,
        },
        "solver": {
            "code": "neo",
            **{key: str(value) for key, value in (native.version or {}).items()},
            "surfaces": int(native.grid.n_radial) if native.grid is not None else 0,
        },
        "lineage": lineage,
        "time_s": slice_time,
        "vacuum_b0_t": None if b0 is None else float(b0),
        "written": list(written),
        "skipped": list(profiles["skipped"]) + list(transport["skipped"]),
        "readiness": _jsonable(readiness),
    }
    return ods, manifest


def _vacuum_field_at(ods: ODS, index: Any) -> float | None:
    """The vacuum field of one equilibrium slice, by index rather than by position."""
    try:
        field = np.ravel(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], dtype=float))
    except Exception:
        return None
    if field.size == 0:
        return None
    # A single stored value applies to every slice; otherwise index it. Never
    # position 0 for a later slice -- that error has been made twice already
    # in this chain (PR #696).
    if field.size == 1:
        return float(field[0])
    position = int(index or 0)
    return float(field[position]) if position < field.size else None


def _jsonable(value: Any) -> Any:
    """Plain types only: a manifest is written as JSON and read back by a pipeline."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    return value


def write_neoclassical_product(
    ods: ODS,
    manifest: Mapping[str, Any],
    native: Any,
    *,
    output: str | Path,
    metadata: str | Path,
    compression: str | None = None,
) -> Path:
    """Write the stage product, its manifest, and NEO's own result beside them.

    #527's principle: the solver-native result is preserved alongside the queryable
    one. The standardized product is what the database and every query read; the
    native JSON is there for the question the mapping did not anticipate -- NEO's own
    normalisation, its Sauter and Redl columns, the surfaces it actually solved --
    and is never the query contract, which is why it sits beside the product rather
    than inside it.

    Returns the path the native result was written to.
    """
    write_stage_product(
        ods, dict(manifest), output=output, metadata=metadata, compression=compression
    )
    native_path = Path(output).with_suffix("")
    native_path = native_path.with_name(native_path.name + ".neo.json")
    return native.write_json(native_path)


# --------------------------------------------------------------------------- #
# turbulent_transport: one TGLF configuration on one state (#1900)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _AtlasSurface:
    """The part of a ``TGLFInput`` the ``core_transport`` mapper reads.

    The routine driver does not keep the ``TGLFInput`` object, but it keeps
    everything the mapper needs from it: the species names and the gyro-Bohm
    normalisation in ``state.json``, and the charges and masses in the native
    ``input.tglf`` the run read. Rebuilding this view rather than restating the
    mapping keeps one statement of how TGLF's fluxes reach IMAS.
    """

    rho: float
    names: tuple[str, ...]
    zs: np.ndarray
    mass: np.ndarray
    vexb_shear: float | None
    normalisation: Any

    def missing(self) -> tuple[str, ...]:
        # The mapper's momentum guard reads `vexb_shear` *and* `missing()`. An
        # empty `missing()` is safe today only because the routine driver never
        # gives TGLF a rotation: `vexb_shear` is always None in its state.json,
        # so the value check alone refuses the toroidal stress. If the driver
        # starts recording rotation, this must report the provenance the
        # summary does not carry, or a momentum flux could be published from a
        # rotation whose source nobody recorded.
        return ()


@dataclass(frozen=True)
class _SurfaceBridge:
    """``r/a`` against ``rho_tor_norm`` at the surfaces the atlas mapped.

    The mapper interpolates ``rho`` against ``rmin/rmin[-1]``. The axis and the
    boundary are added because both coordinates are 0 and 1 there by
    definition; at every solved surface the interpolation then returns exactly
    the ``rho_tor_norm`` the driver's own projection computed from the full
    profile.
    """

    rmin: np.ndarray
    rho: np.ndarray


def _tglf_field_model(parameters: Mapping[str, Any]) -> str | None:
    bper, bpar = bool(parameters.get("use_bper")), bool(parameters.get("use_bpar"))
    if bper and bpar:
        return "em-bper-bpar"
    if bper:
        return "em-bper"
    return None if bpar else "es"


def _normalisation_from_summary(record: Mapping[str, Any]) -> Any:
    """Rebuild ``TGLFNormalisation`` from the driver's JSON summary of it.

    The summary keeps ``a``, ``B_unit``, ``rho_s``, ``c_s``, ``Q_GB`` and
    ``Gamma_GB``; density and temperature follow from the last two by the
    definitions ``Gamma_GB = n_e c_s (rho_s/a)^2`` and ``Q_GB = Gamma_GB T_e``.
    """
    from vaft.code.gacode.tglf.inputs import TGLFNormalisation

    a = float(record["a_m"])
    rho_s = float(record["rho_s_m"])
    c_s = float(record["c_s_m_s"])
    gamma_gb = float(record["gamma_gb_m2_s"])
    q_gb = float(record["q_gb_W_m2"])
    if not all(np.isfinite([a, rho_s, c_s, gamma_gb, q_gb])) or 0.0 in (a, rho_s, c_s, gamma_gb):
        raise ValueError(f"normalisation record {dict(record)!r} cannot be inverted")
    return TGLFNormalisation(
        electron_density=gamma_gb / (c_s * (rho_s / a) ** 2),
        electron_temperature=q_gb / gamma_gb,
        sound_speed=c_s,
        gyroradius=rho_s,
        minor_radius=a,
        b_unit=float(record["b_unit_T"]),
    )


def _read_tglf_state(directory: Path) -> dict[str, Any]:
    """Load and qualify one routine-TGLF state tree, or raise saying why."""
    from vaft.code.gacode.tglf.outputs import TglfOutputs
    from vaft.code.mitim.compare import read_input_tglf
    from vaft.database.sources import (
        family_for_atlas_lineage,
        parse_tglf_configuration,
    )
    from vaft.process.transport_state import EFIT_QUALITIES

    path = directory / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    label = state.get("tglf_config")
    sat_rule, field_model = parse_tglf_configuration(label)
    parameters = dict(state.get("tglf_parameters") or {})
    named = (parameters.get("sat_rule"), _tglf_field_model(parameters))
    if named != (sat_rule, field_model):
        raise ValueError(
            f"{path}: tglf_config {label!r} does not name its settings "
            f"(tglf_parameters give SAT_RULE={named[0]!r}, field model {named[1]!r}); "
            "a product filed under the wrong configuration would overwrite another's."
        )
    quality = state.get("efit_quality")
    if quality not in EFIT_QUALITIES:
        raise ValueError(
            f"{path}: efit_quality {quality!r} is not one of {EFIT_QUALITIES}; a state "
            "the #1331 labels do not admit is not promoted."
        )
    family = family_for_atlas_lineage(state.get("efit_lineage"))
    if state.get("status") not in ("solved", "partial"):
        raise ValueError(f"{path}: state status {state.get('status')!r} has no TGLF result")

    projected = {
        round(float(row["r_over_a"]), 6): row
        for row in ((state.get("core_transport") or {}).get("surfaces") or [])
    }
    surfaces = []
    unsolved = []
    for record in state.get("surfaces") or []:
        radius = float(record["r_over_a"])
        if record.get("status") != "solved":
            unsolved.append({"r_over_a": radius, "status": record.get("status")})
            continue
        workdir = directory / f"r{radius:.2f}"
        native = TglfOutputs.read_json(workdir / "outputs.json")
        if not native.solved:
            raise ValueError(f"{workdir}: recorded solved, but outputs.json is not a solved run")
        settings = read_input_tglf(workdir / "input.tglf")
        ran = (
            int(settings.get("SAT_RULE", 0)),
            _tglf_field_model({"use_bper": settings.get("USE_BPER", False),
                               "use_bpar": settings.get("USE_BPAR", False)}),
        )
        if ran != (sat_rule, field_model):
            raise ValueError(
                f"{workdir}: input.tglf ran SAT_RULE={ran[0]} with field model {ran[1]!r}, "
                f"but the tree is labelled {label!r}."
            )
        local = record.get("local") or {}
        names = tuple(str(name) for name in local.get("species") or ())
        count = int(settings.get("NS", 0))
        if count != len(names) or count == 0:
            raise ValueError(
                f"{workdir}: input.tglf declares NS={count} but state.json names "
                f"{len(names)} species"
            )
        row = projected.get(round(radius, 6))
        if row is None:
            raise ValueError(
                f"{path}: surface r/a={radius} solved but the driver projected no "
                "rho_tor_norm for it, so it cannot be placed on core_transport's grid"
            )
        if local.get("normalisation") is None:
            raise ValueError(f"{path}: surface r/a={radius} carries no normalisation")
        surfaces.append({
            "local": _AtlasSurface(
                rho=radius,
                names=names,
                zs=np.asarray([float(settings[f"ZS_{i}"]) for i in range(1, count + 1)]),
                mass=np.asarray([float(settings[f"MASS_{i}"]) for i in range(1, count + 1)]),
                vexb_shear=local.get("vexb_shear"),
                normalisation=_normalisation_from_summary(local["normalisation"]),
            ),
            "native": native,
            "projected": row,
            "run_identity": record.get("run_identity"),
            "input_sha256": sha256_file(workdir / "input.tglf"),
        })
    if not surfaces:
        raise ValueError(f"{path}: no solved surface to promote")
    return {
        "path": path,
        "state": state,
        "configuration": label,
        "family": family,
        "surfaces": sorted(surfaces, key=lambda item: item["local"].rho),
        "unsolved": unsolved,
    }


def _check_against_projection(
    ods: ODS, base: str, surfaces: Sequence[Mapping[str, Any]], rtol: float, where: Path
) -> None:
    """Refuse a rebuild that disagrees with the driver's own projection.

    Electron and per-ion energy and particle fluxes are compared. A rebuild that
    could be compared with nothing is refused too: a check that passes by
    comparing no values is no evidence that the tree is consistent.
    """

    def stored(column: str, label: str | None = None) -> np.ndarray:
        values = []
        for item in surfaces:
            value = item["projected"].get(column)
            if label is not None:
                value = value.get(label) if isinstance(value, Mapping) else None
            values.append(np.nan if value is None else value)
        return np.asarray(values, dtype=float)

    checks = [
        ("electrons.energy.flux", stored("electron_energy_flux_W_m2")),
        ("electrons.particles.flux", stored("electron_particle_flux_m2_s")),
    ]
    index = 0
    while f"{base}.ion.{index}.label" in ods:
        label = str(ods[f"{base}.ion.{index}.label"])
        checks.append((f"ion.{index}.energy.flux", stored("ion_energy_flux_W_m2", label)))
        checks.append((f"ion.{index}.particles.flux", stored("ion_particle_flux_m2_s", label)))
        index += 1

    compared = 0
    for leaf, expected in checks:
        if f"{base}.{leaf}" not in ods:
            continue
        rebuilt = np.asarray(ods[f"{base}.{leaf}"], dtype=float)
        known = np.isfinite(expected)
        compared += int(known.sum())
        if not np.allclose(rebuilt[known], expected[known], rtol=rtol, atol=0.0):
            raise ValueError(
                f"{where}: {leaf} rebuilt from the native outputs ({rebuilt.tolist()}) "
                f"disagrees with the driver's projection ({expected.tolist()}) beyond "
                f"rtol={rtol}; the tree is not internally consistent."
            )
    if compared == 0:
        raise ValueError(
            f"{where}: no rebuilt flux could be compared with the driver's projection, "
            "so the rebuild is unverified and is not promoted."
        )


def build_turbulent_transport_ods(
    *,
    state_directories: str | Path | Sequence[str | Path],
    run: int = 1,
    rtol: float = 1e-6,
) -> tuple[ODS, dict[str, Any], dict[str, Any]]:
    """Build the ``turbulent_transport`` stage product from routine TGLF trees (#1900).

    ``state_directories`` are ``<config>/<shot>/<lineage>/<time_ms>/`` trees written
    by ``workflow/transport_atlas/run_tglf.py``: one ``state.json`` and one
    ``r<r/a>/`` native directory (``outputs.json`` and ``input.tglf``) per surface.
    One directory is the usual case. Several are accepted only when they are
    distinct times of one shot, lineage and configuration, because the product is
    per shot and a second build for the same shot would replace the first.

    Each state is qualified before anything is mapped. A state is refused when its
    ``efit_quality`` is not good/admissible, when its ``tglf_config`` label does not
    match its ``tglf_parameters``, or when the native ``input.tglf`` ran another
    configuration than the label says. Only surfaces whose status is ``solved`` are
    mapped, through :func:`vaft.machine_mapping.turbulence.core_transport_from_tglf`,
    and the rebuilt fluxes are checked against the driver's own projection.

    The product owns ``core_transport`` only. Every IDS it owns carries
    ``ids_properties.name`` set to
    :func:`vaft.database.sources.turbulent_transport_occurrence_name`, the name the
    occurrence will carry under #1128/#1129, so the payload is self-describing now
    and its migration out of the transitional source renames nothing. The identity
    of the upstream state -- ``state_identity``, the per-surface run identities, the
    TGLF parameters and the GACODE revision -- is recorded in
    ``core_transport.model[:].code.parameters`` (JSON) and in the manifest.

    Several states in one product must also agree on ``ti_lineage``,
    ``tglf_parameters`` and ``gacode_revision``, which the product states once.

    Coexistence holds across configurations and lineages. Alternative Ti
    assumptions of one state share the product's path and source, so they are
    refused at write time (:func:`write_turbulent_transport_product`), not stored
    side by side, until #1128/#1129.

    Returns ``(ods, manifest, native)``; ``native`` is TGLF's own per-surface
    result for :func:`write_turbulent_transport_product` to keep beside the
    product.
    """
    from vaft.database.sources import (
        FAMILY_LINEAGE_NAMES,
        TRANSITIONAL_UNTIL,
        turbulent_transport_occurrence_name,
        turbulent_transport_source,
    )
    from vaft.machine_mapping.turbulence import core_transport_from_tglf

    if isinstance(state_directories, (str, Path)):
        state_directories = [state_directories]
    states = [_read_tglf_state(Path(directory)) for directory in state_directories]
    if not states:
        raise ValueError("no state directory given")

    first = states[0]
    shot = int(first["state"]["shot"])
    for item in states[1:]:
        key = (int(item["state"]["shot"]), item["family"], item["configuration"])
        if key != (shot, first["family"], first["configuration"]):
            raise ValueError(
                f"{item['path']}: one product holds one shot, family and configuration; "
                f"got {key} beside {(shot, first['family'], first['configuration'])}"
            )
    # The top-level code.parameters and manifest describe every time at once, so
    # the times must agree on everything stated there once: a Ti assumption, a
    # TGLF setting or a GACODE revision that differed between times would be
    # misdescribed for some of them.
    for item in states[1:]:
        for key in ("ti_lineage", "tglf_parameters", "gacode_revision"):
            if item["state"].get(key) != first["state"].get(key):
                raise ValueError(
                    f"{item['path']}: {key} {item['state'].get(key)!r} differs from "
                    f"{first['state'].get(key)!r} in {first['path']}; one product "
                    "states it once for all its times, so they cannot share one."
                )
    states.sort(key=lambda item: float(item["state"]["time_efit_s"]))
    times = [float(item["state"]["time_efit_s"]) for item in states]
    if len(set(times)) != len(times):
        raise ValueError(
            f"shot {shot}: two states at one time {times}; the second would replace the first"
        )

    family = first["family"]
    configuration = first["configuration"]
    name = turbulent_transport_occurrence_name(family, configuration)
    source = turbulent_transport_source(family, configuration)

    ods = ODS(consistency_check=False)
    dataset_description(
        ods,
        shot,
        {
            "source_type": "shot",
            "run": run,
            "machine": "VEST",
            "user": "vaft",
            "description": (
                f"VEST turbulent transport from TGLF ({configuration}, "
                f"{FAMILY_LINEAGE_NAMES[family]}); machine era "
                f"{machine_era_for_shot(shot).name}"
            ),
        },
    )

    written: list[str] = []
    skipped: list[str] = []
    provenance_states = []
    native_states = []
    model = 0
    for index, item in enumerate(states):
        state = item["state"]
        bridge = _SurfaceBridge(
            rmin=np.asarray([0.0, *(s["local"].rho for s in item["surfaces"]), 1.0]),
            rho=np.asarray(
                [0.0, *(float(s["projected"]["rho_tor_norm"]) for s in item["surfaces"]), 1.0]
            ),
        )
        report = core_transport_from_tglf(
            ods,
            [(s["local"], s["native"]) for s in item["surfaces"]],
            bridge,
            time=float(state["time_efit_s"]),
            time_index=index,
            state_identity=state.get("state_identity"),
            configuration={"collection_label": configuration},
        )
        model = report["model"]
        base = f"core_transport.model.{model}.profiles_1d.{index}"
        if not report["written"]:
            raise ValueError(
                f"{item['path']}: the mapper wrote nothing ({'; '.join(report['skipped'])})"
            )
        _check_against_projection(ods, base, item["surfaces"], rtol, item["path"])
        written.extend(f"t={state['time_efit_s']}: {leaf}" for leaf in report["written"])
        skipped.extend(f"t={state['time_efit_s']}: {leaf}" for leaf in report["skipped"])
        provenance_states.append({
            "time_s": float(state["time_efit_s"]),
            "state_identity": state.get("state_identity"),
            "efit_lineage": state.get("efit_lineage"),
            "efit_quality": state.get("efit_quality"),
            "ti_lineage": state.get("ti_lineage"),
            "gacode_revision": state.get("gacode_revision"),
            "surfaces": [
                {"r_over_a": s["local"].rho, "run_identity": s["run_identity"],
                 "input_tglf_sha256": s["input_sha256"]}
                for s in item["surfaces"]
            ],
            "unsolved_surfaces": item["unsolved"],
        })
        native_states.append({
            "time_s": float(state["time_efit_s"]),
            "state_identity": state.get("state_identity"),
            "surfaces": [
                {"r_over_a": s["local"].rho, "run_identity": s["run_identity"],
                 "outputs": s["native"].to_dict()}
                for s in item["surfaces"]
            ],
        })

    parameters = {
        "schema_version": 1,
        "occurrence_name": name,
        "transitional_source": source,
        "transitional_until": TRANSITIONAL_UNTIL,
        "tglf_config": configuration,
        "tglf_parameters": _jsonable(first["state"].get("tglf_parameters") or {}),
        "family": family,
        "states": provenance_states,
    }
    # One JSON string: it survives the stage product and the Access Layer verbatim,
    # where a nested block would not (#561).
    ods[f"core_transport.model.{model}.code.parameters"] = json.dumps(
        _jsonable(parameters), sort_keys=True
    )
    ods["core_transport.ids_properties.name"] = name
    ods["core_transport.ids_properties.comment"] = (
        f"TGLF {configuration} on the {FAMILY_LINEAGE_NAMES[family]} lineage; "
        f"stored in the transitional source {source!r} until {TRANSITIONAL_UNTIL}"
    )
    ods["core_transport.ids_properties.provenance.node.0.path"] = f"model.{model}"
    ods["core_transport.ids_properties.provenance.node.0.sources"] = [
        f"equilibrium/{FAMILY_LINEAGE_NAMES[family]}"
    ]

    partial = any(item["unsolved"] for item in states)
    manifest = {
        "schema_version": 1,
        "stage": "turbulent_transport",
        "shot": shot,
        "machine_version": machine_era_for_shot(shot).name,
        "status": "partial" if partial else "success",
        "family": family,
        "configuration": configuration,
        "occurrence_name": name,
        "transitional_source": source,
        "transitional_until": TRANSITIONAL_UNTIL,
        "input": [
            {"state": "/".join(item["path"].parts[-5:]),
             "state_sha256": sha256_file(item["path"])}
            for item in states
        ],
        "solver": {"code": "tglf", "parameters": parameters["tglf_parameters"]},
        "states": provenance_states,
        "written": written,
        "skipped": skipped,
    }
    native = {
        "schema": "vaft.turbulent_transport.native",
        "schema_version": 1,
        "shot": shot,
        "configuration": configuration,
        "family": family,
        "states": native_states,
    }
    return ods, manifest, native


def _state_identities(manifest: Mapping[str, Any]) -> dict[float, Any]:
    return {
        round(float(state["time_s"]), 9): state.get("state_identity")
        for state in manifest.get("states") or []
    }


def _refuse_a_different_state(manifest: Mapping[str, Any], metadata: Path, output: Path) -> None:
    """Raise when ``output`` already holds a product of other states."""
    if not metadata.is_file():
        if output.exists():
            raise ValueError(
                f"{output} exists without its manifest {metadata}; its states cannot "
                "be compared with this one's, so it is not overwritten."
            )
        return
    existing = json.loads(metadata.read_text(encoding="utf-8"))
    before, after = _state_identities(existing), _state_identities(manifest)
    if before != after:
        raise ValueError(
            f"{output} already holds the states {before}; this product's are {after}. "
            "The path does not name the Ti assumption or the state identity, so "
            "replacing it would silently drop the stored result. Alternative states "
            "of one shot and configuration are refused, not stored side by side, "
            "until the #1128/#1129 occurrence model; remove the old product "
            "deliberately if it is superseded."
        )


def write_turbulent_transport_product(
    ods: ODS,
    manifest: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    output: str | Path,
    metadata: str | Path,
    compression: str | None = None,
) -> Path:
    """Write the stage product, its manifest, and TGLF's own results beside them.

    As for NEO (#527): the standardized ``core_transport`` is the query contract,
    and the native per-surface outputs -- spectra, saturation parameters -- sit
    beside it for the question the mapping did not anticipate.

    The path ``{family}/{configuration}/{shot}`` -- and its transitional source --
    does not name the Ti assumption or the rest of the state's identity. A
    product already there whose states differ from this one's (by time and
    ``state_identity``) is therefore refused rather than replaced: coexistence
    holds across configurations and lineages, but alternative Ti assumptions of
    one state are refused, not stored side by side, until the #1128/#1129
    occurrence model can name them. Rewriting the same states is allowed.

    Returns the path the native result was written to.
    """
    _refuse_a_different_state(manifest, Path(metadata), Path(output))
    write_stage_product(
        ods, dict(manifest), output=output, metadata=metadata, compression=compression
    )
    native_path = Path(output).with_suffix("")
    native_path = native_path.with_name(native_path.name + ".tglf.json")
    native_path.write_text(
        json.dumps(native, indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    return native_path


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

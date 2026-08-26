"""VEST interferometer mapping for the OMAS ``interferometer`` IDS.

Two independent diagnostic systems are supported:

- a 5-chord 94 GHz horizontal system, each chord reflected off an inboard
  mirror (three-point line of sight);
- a 1-chord 282 GHz vertical system (two-point line of sight).

Both systems deliver already-postprocessed line-integrated electron density
(``line_den``) -- this module maps it directly to ``n_e_line.data`` and must
not apply a second phase-to-density conversion. See
https://github.com/VEST-Tokamak/vaft/issues/153 for the full specification
this module implements, including the geometry, wavelength constants, and the
validity/fringe-jump policy (both left unset in this first implementation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from vaft.process.signal_processing import line_average_density as _line_average_density

from .utils import load_yaml, resolve_data_root, set_path, _resolve_info_file_path

CODE_NAME = "vaft.machine_mapping.interferometer"
CODE_VERSION = "1"


def _interferometer_config(info_file: str | None = None) -> dict[str, Any]:
    content = load_yaml(_resolve_info_file_path(info_file))
    defaults = content.get("0") or content.get(0) or {}
    config = defaults.get("interferometer") if isinstance(defaults, dict) else None
    if not isinstance(config, dict):
        raise KeyError("No 'interferometer' configuration block found in vest.yaml")
    return config


def _candidate_interferometer_paths(
    shot: int, filename_pattern_hint: str, data_root: Path
) -> list[Path]:
    del filename_pattern_hint  # candidates are filename-shape-specific, not pattern-driven
    return [
        data_root / "legacy" / f"{int(shot)}_ALL_LID.mat",
        data_root / "interferometry" / f"{int(shot)}_ALL_LID.mat",
        data_root / f"{int(shot)}_ALL_LID.mat",
    ]


def _resolve_interferometer_mat_file(
    shot: int,
    system: str,
    *,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
) -> Path:
    """Resolve the MAT file for one interferometer ``system`` ("94ghz"|"282ghz")."""
    if mat_file is not None:
        path = Path(mat_file).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Interferometer MAT file not found: {path}")

    root = resolve_data_root(data_root)
    shot = int(shot)
    if system == "94ghz":
        candidates = [
            root / "legacy" / f"{shot}_056789_LID.mat",
            root / "interferometry" / f"{shot}_056789_LID.mat",
            root / f"{shot}_056789_LID.mat",
        ]
    elif system == "282ghz":
        candidates = [
            root / "legacy" / f"{shot}_ALL_LID.mat",
            root / "interferometry" / f"{shot}_ALL_LID.mat",
            root / f"{shot}_ALL_LID.mat",
        ]
    else:
        raise ValueError(f"Unknown interferometer system {system!r}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Cannot find interferometer LID MAT file for shot {shot}, system "
        f"{system!r}. Packaged samples are not included in the PyPI "
        "distribution; provide mat_file/data_root or clone the VAFT GitHub "
        f"repository. Searched: {searched}"
    )


def _load_lid_mat(path: Path, shot: int, expected_columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Load, validate, and return ``(time, line_den)`` from one LID MAT file.

    ``trigger_time`` is intentionally ignored -- it is legacy metadata that
    does not affect the mapped time (see issue #153).
    """
    mat_data = loadmat(str(path))

    if "shotNum" not in mat_data:
        raise KeyError(f"{path}: missing 'shotNum'")
    file_shot = int(np.asarray(mat_data["shotNum"]).reshape(-1)[0])
    if file_shot != int(shot):
        raise ValueError(f"{path}: shotNum {file_shot} does not match requested shot {shot}")

    if "time" not in mat_data:
        raise KeyError(f"{path}: missing 'time'")
    time = np.asarray(mat_data["time"], dtype=float)
    if time.ndim != 2 or time.shape[1] != 1:
        raise ValueError(f"{path}: 'time' must be a column vector, got shape {time.shape}")
    time = time.reshape(-1)
    if not np.all(np.isfinite(time)):
        raise ValueError(f"{path}: 'time' contains non-finite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: 'time' is not strictly monotonic increasing")

    if "line_den" not in mat_data:
        raise KeyError(f"{path}: missing 'line_den'")
    line_den = np.asarray(mat_data["line_den"], dtype=float)
    if line_den.ndim != 2:
        raise ValueError(f"{path}: 'line_den' must be 2D, got shape {line_den.shape}")
    if line_den.shape[0] != time.shape[0]:
        raise ValueError(
            f"{path}: line_den.shape[0] ({line_den.shape[0]}) != len(time) ({time.shape[0]}); "
            "transposed or ambiguous input is not silently corrected"
        )
    if line_den.shape[1] != expected_columns:
        raise ValueError(
            f"{path}: expected exactly {expected_columns} line_den column(s), "
            f"got {line_den.shape[1]}"
        )

    return time, line_den


def vfit_interferometer_94ghz_static(ods: Any, *, info_file: str | None = None) -> None:
    config = _interferometer_config(info_file)["horizontal_94ghz"]
    physical_channels = config["physical_channels"]
    z_values = config["z_m"]
    phi = float(config["phi_rad"])
    launch_r = float(config["launch_r_m"])
    mirror_r = float(config["mirror_r_m"])
    identifiers = config["identifiers"]
    wavelength = float(config["wavelength_m"])
    phase_to_n_e_line = float(config["phase_to_n_e_line"])

    if launch_r < 0 or mirror_r < 0:
        raise ValueError("94 GHz LOS radii must be non-negative")

    set_path(ods, "interferometer.ids_properties.comment", "VEST 94 GHz horizontal interferometer")
    set_path(ods, "interferometer.code.name", CODE_NAME)
    set_path(ods, "interferometer.code.version", CODE_VERSION)

    for index, (physical_channel, z_m, identifier) in enumerate(
        zip(physical_channels, z_values, identifiers)
    ):
        prefix = f"interferometer.channel.{index}"
        name = f"94 GHz horizontal chord {physical_channel}"
        set_path(ods, f"{prefix}.name", name)
        set_path(ods, f"{prefix}.identifier", identifier)

        for point_name, r_m in (("first_point", launch_r), ("second_point", mirror_r), ("third_point", launch_r)):
            set_path(ods, f"{prefix}.line_of_sight.{point_name}.r", r_m)
            set_path(ods, f"{prefix}.line_of_sight.{point_name}.phi", phi)
            set_path(ods, f"{prefix}.line_of_sight.{point_name}.z", float(z_m))

        set_path(ods, f"{prefix}.wavelength.0.value", wavelength)
        set_path(ods, f"{prefix}.wavelength.0.phase_to_n_e_line", phase_to_n_e_line)


def vfit_interferometer_94ghz_dynamic(
    ods: Any,
    shot: int,
    *,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
    compute_line_average: bool = False,
    info_file: str | None = None,
) -> None:
    config = _interferometer_config(info_file)["horizontal_94ghz"]
    physical_channels = config["physical_channels"]
    path_length_m = float(config["path_length_m"])

    source_path = _resolve_interferometer_mat_file(
        shot, "94ghz", data_root=data_root, mat_file=mat_file
    )
    time, line_den = _load_lid_mat(source_path, shot, expected_columns=len(physical_channels))

    set_path(ods, "interferometer.ids_properties.homogeneous_time", 1)
    set_path(ods, "interferometer.time", time)
    set_path(
        ods,
        "interferometer.ids_properties.provenance.node.0.path",
        "channel(:)/n_e_line",
    )
    set_path(
        ods,
        "interferometer.ids_properties.provenance.node.0.sources",
        [
            f"{source_path.name}; shot={int(shot)}; system=94GHz_horizontal; "
            f"variable=line_den; physical_channels={list(physical_channels)}; "
            "already postprocessed to line-integrated density; no automatic "
            "validity or fringe-jump correction applied"
        ],
    )

    for index in range(len(physical_channels)):
        prefix = f"interferometer.channel.{index}"
        set_path(ods, f"{prefix}.n_e_line.data", line_den[:, index])
        if compute_line_average:
            set_path(
                ods,
                f"{prefix}.n_e_line_average.data",
                _line_average_density(line_den[:, index], path_length_m),
            )


def interferometer_94ghz(
    ods: Any,
    shot: int,
    *,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
    compute_line_average: bool = False,
    info_file: str | None = None,
) -> None:
    """Populate ``ods`` with the 94 GHz horizontal interferometer (own occurrence)."""
    vfit_interferometer_94ghz_static(ods, info_file=info_file)
    vfit_interferometer_94ghz_dynamic(
        ods,
        shot,
        data_root=data_root,
        mat_file=mat_file,
        compute_line_average=compute_line_average,
        info_file=info_file,
    )


def vfit_interferometer_282ghz_static(ods: Any, *, info_file: str | None = None) -> None:
    config = _interferometer_config(info_file)["vertical_282ghz"]
    r_m = float(config["r_m"])
    phi = float(config["phi_rad"])
    z_bottom = float(config["z_bottom_m"])
    z_top = float(config["z_top_m"])
    identifier = config["identifier"]
    wavelength = float(config["wavelength_m"])
    phase_to_n_e_line = float(config["phase_to_n_e_line"])

    if r_m < 0:
        raise ValueError("282 GHz LOS radius must be non-negative")

    set_path(ods, "interferometer.ids_properties.comment", "VEST 282 GHz vertical interferometer")
    set_path(ods, "interferometer.code.name", CODE_NAME)
    set_path(ods, "interferometer.code.version", CODE_VERSION)

    prefix = "interferometer.channel.0"
    set_path(ods, f"{prefix}.name", "282 GHz vertical chord")
    set_path(ods, f"{prefix}.identifier", identifier)
    set_path(ods, f"{prefix}.line_of_sight.first_point.r", r_m)
    set_path(ods, f"{prefix}.line_of_sight.first_point.phi", phi)
    set_path(ods, f"{prefix}.line_of_sight.first_point.z", z_bottom)
    set_path(ods, f"{prefix}.line_of_sight.second_point.r", r_m)
    set_path(ods, f"{prefix}.line_of_sight.second_point.phi", phi)
    set_path(ods, f"{prefix}.line_of_sight.second_point.z", z_top)
    set_path(ods, f"{prefix}.wavelength.0.value", wavelength)
    set_path(ods, f"{prefix}.wavelength.0.phase_to_n_e_line", phase_to_n_e_line)


def vfit_interferometer_282ghz_dynamic(
    ods: Any,
    shot: int,
    *,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
    compute_line_average: bool = False,
    info_file: str | None = None,
) -> None:
    config = _interferometer_config(info_file)["vertical_282ghz"]
    path_length_m = float(config["path_length_m"])

    source_path = _resolve_interferometer_mat_file(
        shot, "282ghz", data_root=data_root, mat_file=mat_file
    )
    time, line_den = _load_lid_mat(source_path, shot, expected_columns=1)

    set_path(ods, "interferometer.ids_properties.homogeneous_time", 1)
    set_path(ods, "interferometer.time", time)
    set_path(
        ods,
        "interferometer.ids_properties.provenance.node.0.path",
        "channel(0)/n_e_line",
    )
    set_path(
        ods,
        "interferometer.ids_properties.provenance.node.0.sources",
        [
            f"{source_path.name}; shot={int(shot)}; system=282GHz_vertical; "
            "variable=line_den; already postprocessed to line-integrated "
            "density; no automatic validity or fringe-jump correction applied"
        ],
    )

    prefix = "interferometer.channel.0"
    set_path(ods, f"{prefix}.n_e_line.data", line_den[:, 0])
    if compute_line_average:
        set_path(
            ods,
            f"{prefix}.n_e_line_average.data",
            _line_average_density(line_den[:, 0], path_length_m),
        )


def interferometer_282ghz(
    ods: Any,
    shot: int,
    *,
    data_root: str | Path | None = None,
    mat_file: str | Path | None = None,
    compute_line_average: bool = False,
    info_file: str | None = None,
) -> None:
    """Populate ``ods`` with the 282 GHz vertical interferometer (own occurrence)."""
    vfit_interferometer_282ghz_static(ods, info_file=info_file)
    vfit_interferometer_282ghz_dynamic(
        ods,
        shot,
        data_root=data_root,
        mat_file=mat_file,
        compute_line_average=compute_line_average,
        info_file=info_file,
    )


def interferometer(
    ods: Any,
    shot: int,
    *,
    data_root: str | Path | None = None,
    mat_file_94ghz: str | Path | None = None,
    mat_file_282ghz: str | Path | None = None,
    compute_line_average: bool = False,
    info_file: str | None = None,
) -> None:
    """Populate ``ods`` with both interferometer systems in one heterogeneous-time IDS.

    This is the compatibility fallback described in issue #153: the current
    VAFT/OMAS storage interface has no clean way to represent the two systems
    as separate IDS occurrences within one in-memory ODS, so both are packed
    as channels of a single ``interferometer`` IDS with
    ``homogeneous_time = 0`` -- the 94 GHz horizontal chords as channels 0-4
    and the 282 GHz vertical chord as channel 5, each channel keeping its own
    ``n_e_line.time``. Prefer :func:`interferometer_94ghz` and
    :func:`interferometer_282ghz` (writing into separate ODS objects) when the
    caller can save them as distinct IMAS occurrences instead.
    """
    horizontal = type(ods)() if not isinstance(ods, dict) else {}
    vertical = type(ods)() if not isinstance(ods, dict) else {}

    interferometer_94ghz(
        horizontal,
        shot,
        data_root=data_root,
        mat_file=mat_file_94ghz,
        compute_line_average=compute_line_average,
        info_file=info_file,
    )
    interferometer_282ghz(
        vertical,
        shot,
        data_root=data_root,
        mat_file=mat_file_282ghz,
        compute_line_average=compute_line_average,
        info_file=info_file,
    )

    set_path(ods, "interferometer.ids_properties.homogeneous_time", 0)
    set_path(ods, "interferometer.ids_properties.comment", "VEST interferometer (94 GHz horizontal + 282 GHz vertical)")
    set_path(ods, "interferometer.code.name", CODE_NAME)
    set_path(ods, "interferometer.code.version", CODE_VERSION)

    n_horizontal = len(horizontal["interferometer"]["channel"])
    horizontal_time = horizontal["interferometer"]["time"]
    for index in range(n_horizontal):
        src = horizontal["interferometer"]["channel"][index]
        prefix = f"interferometer.channel.{index}"
        set_path(ods, f"{prefix}.name", src["name"])
        set_path(ods, f"{prefix}.identifier", src["identifier"])
        for point_name in ("first_point", "second_point", "third_point"):
            for coord in ("r", "phi", "z"):
                set_path(
                    ods,
                    f"{prefix}.line_of_sight.{point_name}.{coord}",
                    src["line_of_sight"][point_name][coord],
                )
        set_path(ods, f"{prefix}.wavelength.0.value", src["wavelength"][0]["value"])
        set_path(
            ods,
            f"{prefix}.wavelength.0.phase_to_n_e_line",
            src["wavelength"][0]["phase_to_n_e_line"],
        )
        set_path(ods, f"{prefix}.n_e_line.time", horizontal_time)
        set_path(ods, f"{prefix}.n_e_line.data", src["n_e_line"]["data"])
        if "n_e_line_average" in src:
            set_path(ods, f"{prefix}.n_e_line_average.time", horizontal_time)
            set_path(ods, f"{prefix}.n_e_line_average.data", src["n_e_line_average"]["data"])

    vertical_time = vertical["interferometer"]["time"]
    vertical_src = vertical["interferometer"]["channel"][0]
    prefix = f"interferometer.channel.{n_horizontal}"
    set_path(ods, f"{prefix}.name", vertical_src["name"])
    set_path(ods, f"{prefix}.identifier", vertical_src["identifier"])
    for point_name in ("first_point", "second_point"):
        for coord in ("r", "phi", "z"):
            set_path(
                ods,
                f"{prefix}.line_of_sight.{point_name}.{coord}",
                vertical_src["line_of_sight"][point_name][coord],
            )
    set_path(ods, f"{prefix}.wavelength.0.value", vertical_src["wavelength"][0]["value"])
    set_path(
        ods,
        f"{prefix}.wavelength.0.phase_to_n_e_line",
        vertical_src["wavelength"][0]["phase_to_n_e_line"],
    )
    set_path(ods, f"{prefix}.n_e_line.time", vertical_time)
    set_path(ods, f"{prefix}.n_e_line.data", vertical_src["n_e_line"]["data"])
    if "n_e_line_average" in vertical_src:
        set_path(ods, f"{prefix}.n_e_line_average.time", vertical_time)
        set_path(ods, f"{prefix}.n_e_line_average.data", vertical_src["n_e_line_average"]["data"])


__all__ = [
    "interferometer",
    "interferometer_94ghz",
    "interferometer_282ghz",
    "vfit_interferometer_94ghz_static",
    "vfit_interferometer_94ghz_dynamic",
    "vfit_interferometer_282ghz_static",
    "vfit_interferometer_282ghz_dynamic",
]

"""Machine-mapping helpers for assembling equilibrium IDS data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
from typing import Any, Optional

import numpy as np

from omas import ODS

from vaft.data.eqdsk import GEQDSK, read_geqdsk


_GFILE_NAME = re.compile(r"[A-Za-z]0?(?P<shot>\d+)\.(?P<time>\d+)")


def _source_metadata(path: Path) -> tuple[int, float | None]:
    match = _GFILE_NAME.match(path.name)
    if match is None:
        return 0, None
    return int(match.group("shot")), int(match.group("time")) / 1000.0


def _existing_slice_count(ods: ODS) -> int:
    try:
        return len(ods["equilibrium.time_slice"])
    except (KeyError, TypeError):
        return 0


def _existing_times(ods: ODS) -> np.ndarray:
    count = _existing_slice_count(ods)
    if count == 0:
        return np.array([], dtype=float)
    return np.asarray(
        [float(ods[f"equilibrium.time_slice.{index}.time"]) for index in range(count)],
        dtype=float,
    )


def _normalise_sources(source: Sequence[str | Path]) -> list[Path]:
    if isinstance(source, (str, bytes, Path)) or not isinstance(source, Sequence):
        raise TypeError("source must be a non-empty sequence of g-file paths")
    paths = [Path(item).expanduser().resolve() for item in source]
    if not paths:
        raise ValueError("source must contain at least one g-file path")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"g-file paths do not exist or are not files: {missing}")
    return paths


def _resolve_times(paths: list[Path], options: dict[str, Any]) -> tuple[list[int], list[float]]:
    metadata = [_source_metadata(path) for path in paths]
    shots = [shot for shot, _ in metadata]
    inferred_times = [time for _, time in metadata]

    explicit_times = options.get("times")
    if explicit_times is not None:
        times = [float(value) for value in explicit_times]
        if len(times) != len(paths):
            raise ValueError("options['times'] must have the same length as source")
    else:
        if any(value is None for value in inferred_times):
            unresolved = [str(path) for path, value in zip(paths, inferred_times) if value is None]
            raise ValueError(
                "Could not infer time from g-file name; provide options['times'] for: "
                f"{unresolved}"
            )
        times = [float(value) for value in inferred_times if value is not None]

    nonzero_shots = {shot for shot in shots if shot != 0}
    if len(nonzero_shots) > 1:
        raise ValueError(f"All g-files must describe the same shot, got {sorted(nonzero_shots)}")
    if len(set(times)) != len(times):
        raise ValueError("Duplicate equilibrium times are not allowed")
    return shots, times


def equilibrium(
    ods: ODS,
    source: Sequence[str | Path],
    options: Optional[dict[str, Any]] = None,
) -> None:
    """Merge a sequence of GEQDSK files into ``ods.equilibrium``.

    Times are inferred from conventional names such as ``g040330.00320`` or
    supplied in seconds through ``options['times']``. Existing time slices are
    appended to unless ``options['replace']`` is true.
    """
    options = dict(options or {})
    paths = _normalise_sources(source)
    shots, times = _resolve_times(paths, options)
    entries: list[tuple[float, Path, GEQDSK, int]] = []
    for path, shot, time in zip(paths, shots, times):
        entries.append((time, path, read_geqdsk(path), shot))
    entries.sort(key=lambda item: item[0])

    replace = bool(options.get("replace", False))
    if replace and "equilibrium" in ods:
        del ods["equilibrium"]
    start_index = _existing_slice_count(ods)
    existing_times = _existing_times(ods)
    new_times = np.asarray([entry[0] for entry in entries], dtype=float)
    if existing_times.size and np.intersect1d(existing_times, new_times).size:
        raise ValueError("New equilibrium times overlap existing time slices")

    allow_derived_data = bool(options.get("allow_derived_data", True))
    for offset, (time, _path, geqdsk, _shot) in enumerate(entries):
        index = start_index + offset
        geqdsk.to_omas(
            ods=ods,
            time_index=index,
            profile_index=0,
            allow_derived_data=allow_derived_data,
        )
        ods[f"equilibrium.time_slice.{index}.time"] = time
        try:
            ods.set_time_array("equilibrium.time", index, time)
            ods.set_time_array("wall.time", index, time)
        except Exception:
            ods[f"equilibrium.time.{index}"] = time
            ods[f"wall.time.{index}"] = time

    all_times = np.concatenate((existing_times, new_times))
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    ods["equilibrium.time"] = all_times
    if "wall" in ods:
        ods["wall.time"] = all_times

    shot_values = {shot for shot in shots if shot != 0}
    if shot_values:
        ods["dataset_description.data_entry.pulse"] = shot_values.pop()


__all__ = ["equilibrium"]

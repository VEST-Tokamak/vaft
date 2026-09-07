"""VEST FAST-camera mapping for the OMAS ``camera_visible`` IDS.

The FAST visible-light camera's raw output is a per-shot sequence of
``{shot}_{frame:08d}.bmp`` grayscale frames plus a sidecar ``{shot}_bmp.txt``
header (see the VEST_Fast Camera_Diagnostics repository's ``bmp_arranger.py``).
This module reuses that donor tool's header-parsing convention and frame-to-time
formula, but replaces its Ip/H-alpha SQL-based valid-interval gate with a purely
image-content-based one (near-black frame rejection), and writes only raw,
uncalibrated frames into ``camera_visible`` -- no radiometric calibration or
geometry is available, so ``frame[:].radiance`` and calibration/geometry nodes
are never populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Iterator, Sequence

import numpy as np

from .utils import resolve_data_root, set_path

DEFAULT_NEAR_BLACK_THRESHOLD = 35
DEFAULT_NEAR_BLACK_PERCENTAGE = 0.98
DEFAULT_BUFFER_FRAMES = 2

# Fixed 0-based line indices in `{shot}_bmp.txt`, verified stable across all
# sample shots in the VEST_Fast Camera_Diagnostics repository. The "Top Frame"/
# "Bottom Frame" lines at 74/75 belong to the header's *File Information* block
# (the exported BMP sub-range), not the earlier *Recording Information* block.
# `ShutterSpeed:` sits at 24, not 25 -- 25 is `CustomShutterValueNumerator`.
# The parser returns None rather than raising on a line mismatch, so the
# off-by-one silently dropped exposure_time from every camera IDS ever built.
_FRAMES_LINE_INDEX = 15
_TOP_FRAME_LINE_INDEX = 74
_BOTTOM_FRAME_LINE_INDEX = 75
_SHUTTER_SPEED_LINE_INDEX = 24

_TOTAL_FRAMES_PATTERN = re.compile(r"Frames:\s*(\d+)")
# The relative time carries a sign, and it means something: a centre-triggered
# acquisition (`TriggerSelect: CENTER`) starts before the trigger, so its top
# frame sits at a negative time. Requiring a leading "+" both rejected those
# headers outright and would have placed a pre-trigger frame after the trigger
# had it matched.
_TOP_FRAME_PATTERN = re.compile(r"^Top Frame,.+,([+-]?\d+\.\d+)")
_BOTTOM_FRAME_PATTERN = re.compile(r"^Bottom Frame,.+,([+-]?\d+\.\d+)")
# The frame-rate prefix in front of the parenthesised exposure is not always a
# number: a camera run with no frame-rate-derived shutter records
# `ShutterSpeed: OPEN(19.1us)`. Requiring `<number>k(` skipped the exposure of
# every such acquisition -- 102 routine shots and every fluctuation shot in the
# archive -- so the prefix is now anything up to the parenthesis.
_SHUTTER_SPEED_PATTERN = re.compile(r"ShutterSpeed:\s*[^(\n]*\(([\d.]+)us\)")


class CameraFrameSelectionError(RuntimeError):
    """Raised when no non-dark frame can be found in a shot's frame sequence."""


@dataclass(frozen=True)
class CameraHeaderInfo:
    """Parsed contents of a `{shot}_bmp.txt` camera header."""

    start_time_ms: float
    end_time_ms: float
    total_frames: int
    exposure_time_s: float | None


#: VEST's two camera-viewing ports, as the visible-camera calibration uses
#: them: a rectangular aperture on the vessel's outer wall, given by its major
#: radius, its chord width, and the top and bottom of its opening. The two
#: ports sit 120 degrees apart, the first centred on 30 degrees. Machine
#: geometry, kept here rather than in the notebooks that project it.
PORT_MAJOR_RADIUS_M = 0.803
PORT_CHORD_WIDTH_M = 0.24
PORT_TOP_M = 0.57 - 0.2355
PORT_HEIGHT_M = 0.692
PORT_FIRST_CENTRE_RAD = np.deg2rad(30.0)
PORT_SEPARATION_RAD = np.deg2rad(120.0)


def vest_port_corner_points() -> np.ndarray:
    """The camera ports' corner and edge-midpoint markers, in world centimeters.

    Eight markers per port -- four corners, then the mid-height points of the
    two vertical edges and the mid-width points of the top and bottom edges --
    for both ports, in the ``(X, Y, Z)`` centimeter frame
    :func:`vaft.process.camera_geometry.project_points` expects, matching
    :func:`vaft.process.camera_geometry.sweep_toroidal`.

    The aperture's angular half-width follows from its chord across the port
    circle, ``cos(w) = 1 - c^2 / (2 R^2)``.
    """
    z_top = PORT_TOP_M
    z_bottom = z_top - PORT_HEIGHT_M
    z_middle = 0.5 * (z_top + z_bottom)
    radius = PORT_MAJOR_RADIUS_M
    width_rad = np.arccos(1.0 - PORT_CHORD_WIDTH_M ** 2 / (2.0 * radius ** 2))

    def corners(base_angle: float) -> np.ndarray:
        left, right = base_angle, base_angle + width_rad
        centre = base_angle + width_rad / 2.0
        x_left, y_left = radius * np.cos(left), radius * np.sin(left)
        x_right, y_right = radius * np.cos(right), radius * np.sin(right)
        x_centre, y_centre = radius * np.cos(centre), radius * np.sin(centre)
        return np.array([
            [x_right, y_right, z_top],
            [x_right, y_right, z_bottom],
            [x_left, y_left, z_bottom],
            [x_left, y_left, z_top],
            [x_right, y_right, z_middle],
            [x_centre, y_centre, z_bottom],
            [x_left, y_left, z_middle],
            [x_centre, y_centre, z_top],
        ]) * 100.0

    first = PORT_FIRST_CENTRE_RAD - width_rad / 2.0
    return np.vstack([corners(first), corners(first + PORT_SEPARATION_RAD)])


def _parse_bmp_header(path: str | Path) -> CameraHeaderInfo:
    """Parse a FAST-camera `{shot}_bmp.txt` header.

    Reuses the fixed-line-index convention of the donor `bmp_arranger.py`
    ``extract_data`` function rather than searching by content, since that
    convention was verified stable across every sample shot header available.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    required_index = max(_FRAMES_LINE_INDEX, _TOP_FRAME_LINE_INDEX, _BOTTOM_FRAME_LINE_INDEX)
    if len(lines) <= required_index:
        raise ValueError(
            f"Camera header {path} has only {len(lines)} lines; expected at least "
            f"{required_index + 1} to read Frames/Top Frame/Bottom Frame."
        )

    frames_match = _TOTAL_FRAMES_PATTERN.search(lines[_FRAMES_LINE_INDEX])
    if frames_match is None:
        raise ValueError(
            f"Camera header {path} line {_FRAMES_LINE_INDEX + 1} does not match "
            f"'Frames: N': {lines[_FRAMES_LINE_INDEX]!r}"
        )
    total_frames = int(frames_match.group(1))

    top_match = _TOP_FRAME_PATTERN.search(lines[_TOP_FRAME_LINE_INDEX])
    if top_match is None:
        raise ValueError(
            f"Camera header {path} line {_TOP_FRAME_LINE_INDEX + 1} does not match "
            f"'Top Frame' data: {lines[_TOP_FRAME_LINE_INDEX]!r}"
        )
    start_time_ms = float(top_match.group(1)) * 1000.0

    bottom_match = _BOTTOM_FRAME_PATTERN.search(lines[_BOTTOM_FRAME_LINE_INDEX])
    if bottom_match is None:
        raise ValueError(
            f"Camera header {path} line {_BOTTOM_FRAME_LINE_INDEX + 1} does not match "
            f"'Bottom Frame' data: {lines[_BOTTOM_FRAME_LINE_INDEX]!r}"
        )
    end_time_ms = float(bottom_match.group(1)) * 1000.0

    exposure_time_s: float | None = None
    if len(lines) > _SHUTTER_SPEED_LINE_INDEX:
        shutter_match = _SHUTTER_SPEED_PATTERN.search(lines[_SHUTTER_SPEED_LINE_INDEX])
        if shutter_match is not None:
            exposure_time_s = float(shutter_match.group(1)) * 1e-6

    return CameraHeaderInfo(
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        total_frames=total_frames,
        exposure_time_s=exposure_time_s,
    )


def frame_time_ms(frame_index: int, total_frames: int, start_time_ms: float, end_time_ms: float) -> float:
    """Linear-interpolate a frame's time, reusing the donor `bmp_arranger.py` formula.

    ``time_ms = start + (end - start) * frame / (total_frames - 1)``. The
    formula is index/total_frames-based, so it stays correct for any single
    frame regardless of which other frames are later discarded.
    """
    if total_frames <= 1:
        raise ValueError("total_frames must be greater than 1 to interpolate a frame time.")
    fraction = frame_index / (total_frames - 1)
    return start_time_ms + (end_time_ms - start_time_ms) * fraction


def is_near_black(
    image: np.ndarray,
    *,
    threshold: int = DEFAULT_NEAR_BLACK_THRESHOLD,
    percentage: float = DEFAULT_NEAR_BLACK_PERCENTAGE,
) -> bool:
    """Return whether ``image`` is near-black, reusing the donor `is_near_black` formula."""
    array = np.asarray(image)
    black_pixels = np.sum(array < threshold)
    total_pixels = array.size
    return (black_pixels / total_pixels) > percentage


def find_valid_frame_interval(
    frames: Sequence[np.ndarray | None],
    *,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    threshold: int = DEFAULT_NEAR_BLACK_THRESHOLD,
    percentage: float = DEFAULT_NEAR_BLACK_PERCENTAGE,
) -> tuple[int, int]:
    """Find the ``[onset, end]`` interval of non-dark frames, with padding.

    Mirrors the donor `find_valid_frame_range`: scans for the first/last
    non-dark, non-missing frame, then pads by ``buffer_frames`` on each side,
    clamped to the available index range.
    """
    total_frames = len(frames)
    first_valid: int | None = None
    last_valid: int | None = None

    for index, frame in enumerate(frames):
        if frame is None:
            continue
        if not is_near_black(frame, threshold=threshold, percentage=percentage):
            if first_valid is None:
                first_valid = index
            last_valid = index

    if first_valid is None or last_valid is None:
        raise CameraFrameSelectionError(
            "No valid (non-dark) frames found; every available frame is near-black "
            "or missing."
        )

    onset = max(0, first_valid - buffer_frames)
    end = min(total_frames - 1, last_valid + buffer_frames)
    return onset, end


def _load_raw_frame(shot_dir: Path, shot: int, frame_index: int) -> np.ndarray | None:
    """Load `{shot}_{frame_index:08d}.bmp` as grayscale, or None if missing."""
    import cv2

    filepath = shot_dir / f"{shot}_{frame_index:08d}.bmp"
    if not filepath.exists():
        return None
    image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image


def _resolve_shot_paths(
    shot: int,
    *,
    data_root: str | Path | None = None,
    frame_dir: str | Path | None = None,
    header_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve ``(shot_dir, header_path)`` for a shot's raw camera frames.

    Raw FAST-camera frames are external, per-shot data that is never packaged
    with vaft, so callers normally pass ``frame_dir``/``header_path``
    explicitly; ``data_root`` is a convenience for a shared local mirror laid
    out as ``data_root/{shot}/{shot}_bmp.txt``.
    """
    if frame_dir is not None:
        shot_dir = Path(frame_dir).expanduser()
        if not shot_dir.is_dir():
            raise FileNotFoundError(f"Camera frame directory not found: {shot_dir}")
    else:
        root = resolve_data_root(data_root)
        shot_dir = root / str(int(shot))
        if not shot_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot find camera frame directory for shot {shot}. Raw FAST-camera "
                "frames are not packaged with vaft; provide frame_dir or data_root. "
                f"Searched: {shot_dir}"
            )

    if header_path is not None:
        resolved_header = Path(header_path).expanduser()
    else:
        resolved_header = shot_dir / f"{int(shot)}_bmp.txt"
    if not resolved_header.exists():
        raise FileNotFoundError(f"Camera header file not found: {resolved_header}")

    return shot_dir, resolved_header


def vfit_camera_visible_static(
    ods: Any,
    *,
    lines_n: int,
    columns_n: int,
    exposure_time_s: float | None = None,
    channel_name: str = "Fast Camera",
    source: str | None = None,
    comment_extra: str = "",
) -> None:
    """Fill static `camera_visible` IDS metadata: channel/detector geometry-free facts."""
    comment = (
        "VEST FAST-camera raw grayscale frames; no radiometric calibration available "
        "(frame_raw only, radiance/counts_to_radiance intentionally unset); no aperture, "
        "optical_element, fibre_bundle, viewing_angle, or pixel_to_alpha/beta geometry "
        "available (intentionally unset)."
    )
    if comment_extra:
        comment = f"{comment} {comment_extra}"

    set_path(ods, "camera_visible.name", channel_name)
    set_path(ods, "camera_visible.ids_properties.homogeneous_time", 1)
    set_path(ods, "camera_visible.ids_properties.name", channel_name)
    set_path(ods, "camera_visible.ids_properties.comment", comment)
    set_path(ods, "camera_visible.ids_properties.creation_date", datetime.now(timezone.utc).isoformat())
    if source is not None:
        set_path(ods, "camera_visible.ids_properties.source", str(source))

    set_path(ods, "camera_visible.channel.0.name", channel_name)
    set_path(ods, "camera_visible.channel.0.detector.0.lines_n", int(lines_n))
    set_path(ods, "camera_visible.channel.0.detector.0.columns_n", int(columns_n))
    if exposure_time_s is not None:
        set_path(ods, "camera_visible.channel.0.detector.0.exposure_time", float(exposure_time_s))


def vfit_camera_visible_dynamic(
    ods: Any,
    *,
    images: Sequence[np.ndarray],
    times_s: Sequence[float],
) -> None:
    """Fill dynamic `camera_visible` frame data: reindexed ``frame[:]`` array."""
    if len(images) != len(times_s):
        raise ValueError("images and times_s must have the same length.")
    if len(images) == 0:
        raise ValueError("images must contain at least one frame.")

    shape = images[0].shape
    for image in images:
        if image.shape != shape:
            raise ValueError(f"All frames must share the same shape; got {image.shape} and {shape}.")

    time_values = np.asarray(times_s, dtype=float)
    set_path(ods, "camera_visible.time", time_values)
    for index, (image, time_value) in enumerate(zip(images, times_s)):
        prefix = f"camera_visible.channel.0.detector.0.frame.{index}"
        # No cast here: `image_raw` is an INT_2D node, and OMAS applies
        # `value.astype(int)` itself on every assignment to one.
        set_path(ods, f"{prefix}.image_raw", np.asarray(image))
        set_path(ods, f"{prefix}.time", float(time_value))


def _image_raw_paths(ods: Any) -> Iterator[str]:
    """Yield every populated ``image_raw`` path in a ``camera_visible`` ODS.

    The mapping only ever writes channel 0 / detector 0, but this is a public
    entry point and a caller may hand it an ODS assembled elsewhere. Walking
    what is actually there beats hardcoding the one shape this module happens
    to produce, which would silently leave other channels at their original
    width while the product claims to be narrowed throughout.
    """
    channels = ods["camera_visible"].get("channel", {})
    for channel_index in sorted(channels):
        detectors = channels[channel_index].get("detector", {})
        for detector_index in sorted(detectors):
            frames = detectors[detector_index].get("frame", {})
            for frame_index in sorted(frames):
                if "image_raw" in frames[frame_index]:
                    yield (
                        f"camera_visible.channel.{channel_index}"
                        f".detector.{detector_index}.frame.{frame_index}.image_raw"
                    )


def narrow_image_storage(ods: Any) -> None:
    """Re-state every ``camera_visible`` ``image_raw`` frame as ``int32``.

    This cannot live in :func:`vfit_camera_visible_dynamic`, and not for want
    of trying: OMAS upcasts on assignment. Every write to an ``INT_2D`` node
    runs ``value = value.astype(int)`` (``omas/omas_core.py``), which is
    platform ``int`` -- ``int64`` here -- so a frame assigned as ``uint8``,
    ``int32`` or ``float32`` comes back out as ``int64`` regardless. The width
    can therefore only be set immediately before storage, after the mapping
    has finished building and validating the ODS and with the consistency
    check suspended so the re-assignment reaches the node untouched.

    Halving the stored width costs nothing in fidelity: VEST FAST-camera
    frames are 8-bit grayscale, so every value fits ``int32`` with three
    orders of magnitude to spare, and the re-cast is exact.

    The check is *not* switched back on afterwards, and that is the same
    upcast again rather than an oversight: re-enabling it re-runs OMAS's
    ``consistency_checker`` over every leaf and re-applies ``astype(int)``,
    putting ``image_raw`` straight back to ``int64``. Validation before the
    narrowing is what the mapping already did -- every assignment in
    :func:`vfit_camera_visible_dynamic` ran under the check -- and validation
    afterwards belongs to the stored product, which reloads through the
    ordinary loader reporting ``consistency_check == True``. Nothing but
    storage should follow this call.

    Call this on a finished ODS, immediately before ``vaft.omas.save``. It is
    deliberately not folded into ``save``: applying it to every ODS would
    silently re-type unrelated integer nodes in other IDSs.
    """
    # Membership, not indexing. Reading a missing path through an ODS creates
    # it, so a `try: ods[path] except KeyError` guard would never fire here --
    # it would quietly graft an empty camera_visible onto an ODS that has none
    # and then disable its consistency check for nothing.
    if "camera_visible" not in ods:
        return

    paths = list(_image_raw_paths(ods))
    if not paths:
        return

    ods.consistency_check = False
    for path in paths:
        image = np.asarray(ods[path])
        if image.dtype == np.int32:
            continue
        ods[path] = image.astype(np.int32)


def camera_visible(
    ods: Any,
    shot: int,
    *,
    data_root: str | Path | None = None,
    frame_dir: str | Path | None = None,
    header_path: str | Path | None = None,
    near_black_threshold: int = DEFAULT_NEAR_BLACK_THRESHOLD,
    near_black_percentage: float = DEFAULT_NEAR_BLACK_PERCENTAGE,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    channel_name: str = "Fast Camera",
) -> None:
    """Populate ``ods`` with VEST FAST-camera raw frames for ``shot``.

    Valid frames are selected from image content: frames outside the
    near-black-rejection interval (padded by ``buffer_frames`` on each side)
    are discarded, and the retained frames are reindexed from 0. This is the
    same reindex-from-zero, image-content-based selection as the donor
    ``bmp_arranger.py`` tool, not the Ip/H-alpha SQL-based gate used by its
    ``bmp_arranger_batch.py`` variant.
    """
    shot_dir, resolved_header = _resolve_shot_paths(
        shot, data_root=data_root, frame_dir=frame_dir, header_path=header_path
    )
    header = _parse_bmp_header(resolved_header)

    raw_frames: list[np.ndarray | None] = [
        _load_raw_frame(shot_dir, int(shot), index) for index in range(header.total_frames)
    ]

    onset, end = find_valid_frame_interval(
        raw_frames,
        buffer_frames=buffer_frames,
        threshold=near_black_threshold,
        percentage=near_black_percentage,
    )

    retained_indices = [
        index for index in range(onset, end + 1) if raw_frames[index] is not None
    ]
    if not retained_indices:
        raise CameraFrameSelectionError(
            f"All frames in the valid interval [{onset}, {end}] for shot {shot} are missing."
        )

    images = [raw_frames[index] for index in retained_indices]
    times_s = [
        frame_time_ms(index, header.total_frames, header.start_time_ms, header.end_time_ms) / 1000.0
        for index in retained_indices
    ]

    lines_n, columns_n = images[0].shape

    exposure_note = (
        f"exposure_time from header ShutterSpeed line {_SHUTTER_SPEED_LINE_INDEX + 1}."
        if header.exposure_time_s is not None
        else "exposure_time unavailable: no ShutterSpeed line found in header."
    )
    selection_note = (
        f"Frames selected by image-content dark-frame rejection "
        f"(threshold={near_black_threshold}, percentage={near_black_percentage}, "
        f"buffer_frames={buffer_frames}); retained original frame indices "
        f"[{retained_indices[0]}, {retained_indices[-1]}] out of {header.total_frames}, "
        f"reindexed from 0. {exposure_note}"
    )

    vfit_camera_visible_static(
        ods,
        lines_n=lines_n,
        columns_n=columns_n,
        exposure_time_s=header.exposure_time_s,
        channel_name=channel_name,
        source=str(resolved_header),
        comment_extra=selection_note,
    )
    vfit_camera_visible_dynamic(ods, images=images, times_s=times_s)


def camera_visible_from_frame_dir(
    shot: int,
    *,
    consistency_check: bool = True,
    **kwargs: Any,
):
    """Create and return an OMAS ODS filled with one VEST FAST-camera shot."""
    from omas import ODS

    ods = ODS(consistency_check=consistency_check)
    camera_visible(ods, shot, **kwargs)
    return ods


def save_camera_visible_ods(
    output_path: str | Path,
    shot: int,
    **kwargs: Any,
):
    """Build and save a camera_visible ODS. The file extension selects OMAS format."""
    output = Path(output_path).expanduser()
    consistency_check = kwargs.pop("consistency_check", True)
    ods = camera_visible_from_frame_dir(
        shot,
        consistency_check=consistency_check,
        **kwargs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ods.save(str(output))
    return ods


camera_visible_from_raw_database = camera_visible

__all__ = [
    "CameraFrameSelectionError",
    "CameraHeaderInfo",
    "DEFAULT_BUFFER_FRAMES",
    "DEFAULT_NEAR_BLACK_PERCENTAGE",
    "DEFAULT_NEAR_BLACK_THRESHOLD",
    "camera_visible",
    "camera_visible_from_frame_dir",
    "camera_visible_from_raw_database",
    "find_valid_frame_interval",
    "frame_time_ms",
    "is_near_black",
    "narrow_image_storage",
    "save_camera_visible_ods",
    "vfit_camera_visible_dynamic",
    "vfit_camera_visible_static",
]

"""VEST FAST-camera mapping for the OMAS ``camera_visible`` IDS.

The FAST visible-light camera's raw output is a per-shot sequence of
``{shot}_{frame:08d}.bmp`` grayscale frames plus a sidecar ``{shot}_bmp.txt``
header (see the VEST_Fast Camera_Diagnostics repository's ``bmp_arranger.py``).
This module reuses that donor tool's header-parsing convention and frame-to-time
formula, but replaces its Ip/H-alpha SQL-based valid-interval gate with a purely
image-content-based one (near-black frame rejection against each shot's own
dark level, see :func:`select_valid_frames`), and writes only raw,
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

from .registry import port_phi
from .utils import resolve_data_root, set_path

#: The donor's fixed near-black level. It is now used only when a caller passes
#: it explicitly: frame selection derives its threshold from each shot's own
#: dark level (see :func:`select_valid_frames`).
DEFAULT_NEAR_BLACK_THRESHOLD = 35
DEFAULT_NEAR_BLACK_PERCENTAGE = 0.98
DEFAULT_BUFFER_FRAMES = 2
#: A frame is near-black when more than ``percentage`` of its pixels sit below
#: ``dark_level + DEFAULT_DARK_MARGIN``. The fixed 35 only fitted the original
#: exports, whose sensor floor is ~25 DN. The BatchConv2 re-exports (GX-8
#: headers) use another gain/LUT and put an empty vessel at ~48 DN, so no frame
#: of e.g. shot 48909 counted as dark and every frame was kept. Calibrated
#: against the I_p window on 55 plasma shots of both exports: +20 trims
#: re-exports to the I_p window within ~1-2 ms and moves the cut on the
#: original exports by at most a frame or two.
DEFAULT_DARK_MARGIN = 20.0
#: The dark level is the lower quartile, over the shot's frames, of each
#: frame's 1st-percentile pixel. Plasma frames still have dark corners outside
#: the port, so this tracks the sensor floor; the quartile (not the minimum)
#: ignores a stray blank frame and still holds when up to three quarters of
#: the recording is bright.
DARK_LEVEL_PERCENTILE = 1.0
DARK_LEVEL_FRAME_QUANTILE = 0.25
#: Measured floors run 15-52 DN across both exports. A recording lit almost
#: throughout (diffuse light, saturation) would otherwise estimate its "dark"
#: level at the lit level and reject every frame; the cap keeps such frames.
DARK_LEVEL_CEILING = 60.0
#: Names the selection rule in the IDS comment and the stage manifest, so a
#: product can be audited for which rule it was built with.
FRAME_SELECTION_RULE = "relative-dark-v2"
FIXED_FRAME_SELECTION_RULE = "fixed-v1"

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

# A second header layout, written by BatchConv2 (the MEMRECAM HXLink batch
# converter) when an archived `.mcf` is re-exported long after the shot. It
# starts with `Type: GX-8` and carries the exported sub-range as `TopFrame:` /
# `BottomFrame:` keys, the rate as `Frame_Rate:` and the shutter as a bare
# `Shutter_SRC_Speed: 200k`, but it has neither the `Frames:` line nor the
# `Top Frame,...,<relative time>` rows the fixed-index parser needs. Its
# `[CONV_PARAM]` block also has `BEGIN_TIME=`/`END_TIME=`, and those must NOT
# be used: they are copied from the converter's parameter template and read
# +0.280000/+0.340000 whatever range was exported (2,990 of 5,312 re-exported
# shots disagree with their own frame numbers).
_GX8_TYPE_PATTERN = re.compile(r"^Type:\s*GX-8\b")
_GX8_KEY_PATTERN = re.compile(r"^([A-Za-z_]+)\s*:\s*(.*?)\s*$")
# `200k` -> 200000 frames-per-second-equivalent shutter, i.e. 1/200000 s.
# `OPEN` and `CUSTOM` carry no exposure in this layout and leave it unset.
_GX8_SHUTTER_PATTERN = re.compile(r"^([\d.]+)\s*([kK]?)$")


class CameraFrameSelectionError(RuntimeError):
    """Raised when no non-dark frame can be found in a shot's frame sequence."""


@dataclass(frozen=True)
class CameraHeaderInfo:
    """Parsed contents of a `{shot}_bmp.txt` camera header."""

    start_time_ms: float
    end_time_ms: float
    total_frames: int
    exposure_time_s: float | None
    #: Where ``exposure_time_s`` came from, for the IDS comment.
    exposure_source: str = f"header ShutterSpeed line {_SHUTTER_SPEED_LINE_INDEX + 1}"


#: VEST's two camera-viewing ports, as the visible-camera calibration uses
#: them: a rectangular aperture on the vessel's outer wall, given by its major
#: radius, its chord width, and the top and bottom of its opening. The two
#: ports sit 120 degrees apart, the first centred on 30 degrees. Machine
#: geometry, kept here rather than in the notebooks that project it.
#:
#: PROVISIONAL / UNRECONCILED with the port table (issue #746). The rectangular
#: main-chamber
#: ports are 2MR, 6MR and 10MR, which *are* 120 degrees apart, and the camera
#: itself looks through 6MR -- but 2MR and 10MR are at 300 and 60 degrees of
#: IMAS phi (60 and 300 of VEST clock angle), and neither pair is the 30/150
#: below. So these two angles are in some third frame, plausibly one centred on
#: the camera. They are left untouched because they feed a projection in
#: vaft.process.camera_geometry that is calibrated against real images;
#: changing them to match the port table without redoing that calibration would
#: break a working result to satisfy a naming convention.
PORT_MAJOR_RADIUS_M = 0.803
PORT_CHORD_WIDTH_M = 0.24
PORT_TOP_M = 0.57 - 0.2355
PORT_HEIGHT_M = 0.692
PORT_FIRST_CENTRE_RAD = np.deg2rad(30.0)
PORT_SEPARATION_RAD = np.deg2rad(120.0)

#: The port the fast camera itself looks through.  The port-status document
#: states this outright -- ``6MR : Entrance``, with the fast camera, the
#: H-alpha / O I filterscope and the hard X-ray detector all listed there -- so
#: unlike the two landmark angles above this is not an inference.
CAMERA_PORT = "6MR"

# ---------------------------------------------------------------------------
# What the camera actually sees (VEST optical-diagnostics slide, "Fast camera")
# ---------------------------------------------------------------------------
#
# The view is TANGENTIAL, not a radial look through the port. The slide's top
# view draws a fan from the camera optics that grazes the machine at an inner
# tangency of 0.13-0.22 m and reaches the outboard side at 0.65-0.75 m, and it
# labels the 50 kHz frame a "tangential view".
#
# That matters beyond documentation: it is why PORT_FIRST_CENTRE_RAD and
# PORT_SEPARATION_RAD above sit in a frame that matches neither the VEST clock
# angles nor IMAS phi (issue #746). A tangential camera does not see the
# rectangular ports at their port-table angles -- it sees them projected along
# its own sightline -- so a camera-centred frame is the expected shape of that
# discrepancy rather than evidence of a mistake. Reconciling it means redoing
# the projection with the tangency geometry below, not renumbering two angles.
#
# Recorded, not yet written into the IDS: turning these radii into
# `viewing_angle_alpha_bounds` needs the camera's own position along its
# sightline, which the slide does not give.

TANGENTIAL_VIEW = True
VIEWING_TANGENCY_RANGE_M = (0.13, 0.22)
"""Inner tangency radius the viewing fan grazes (``R_in`` on the slide)."""

VIEWING_OUTBOARD_RANGE_M = (0.65, 0.75)
"""How far out the fan reaches on the far side (``R_out`` on the slide)."""

VESSEL_OUTBOARD_RADIUS_M = 0.88
"""The vessel outboard radius the slide marks (``R_outboard``).

Distinct from :data:`PORT_MAJOR_RADIUS_M` (0.803, the port flange this module
projects from) and from the packaged limiter outline, which reaches 0.760 m.
Three different surfaces; none is a substitute for another.
"""

VIEWING_RADIUS_RANGE_M = (0.1, 0.7)
"""``R_viewing`` on the slide: the radial span the 208x208 frame covers."""

PIXEL_SCALE_AT_TANGENCY_M = (0.0025, 0.0028)
"""What one pixel subtends at the point of tangency, in metres."""


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


def _parse_gx8_header(path: Path, lines: Sequence[str]) -> CameraHeaderInfo:
    """Parse the BatchConv2 ``Type: GX-8`` header layout by key.

    Frame numbers are counted from the trigger frame, exactly as in the
    original layout, whose ``Top Frame`` row reads ``+700 ... +0.280000`` at
    2500 fps. So the top and bottom times are ``frame / Frame_Rate``, and the
    exported frame count is ``BottomFrame - TopFrame + 1``. Verified against
    the three shots (26151, 26153, 26155) that exist in both layouts: same
    start, end and exposure. The frame numbers are indices into the camera's
    source recording, whose rate is ``Frame_SRC_Rate``; ``Frame_Rate`` is the
    export rate, read first here with ``Frame_SRC_Rate`` as the fallback.
    Every header seen so far carries the same value in both, so a header
    where they differ would need the source rate for the times and has not
    been met.
    """
    values: dict[str, str] = {}
    for line in lines:
        match = _GX8_KEY_PATTERN.match(line)
        if match is not None:
            values.setdefault(match.group(1), match.group(2))

    def required_int(key: str) -> int:
        try:
            return int(values[key])
        except (KeyError, ValueError):
            raise ValueError(
                f"GX-8 camera header {path} has no integer {key!r}: {values.get(key)!r}"
            ) from None

    top_frame = required_int("TopFrame")
    bottom_frame = required_int("BottomFrame")
    rate_text = values.get("Frame_Rate") or values.get("Frame_SRC_Rate")
    try:
        frame_rate = float(rate_text) if rate_text is not None else float("nan")
    except ValueError:
        frame_rate = float("nan")
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"GX-8 camera header {path} has no positive Frame_Rate: {rate_text!r}")
    if bottom_frame <= top_frame:
        raise ValueError(
            f"GX-8 camera header {path} has BottomFrame {bottom_frame} <= TopFrame {top_frame}."
        )

    exposure_time_s: float | None = None
    shutter_key = "Shutter_SRC_Speed" if "Shutter_SRC_Speed" in values else "Shutter_Speed"
    shutter_text = values.get(shutter_key)
    if shutter_text is not None:
        shutter_match = _GX8_SHUTTER_PATTERN.match(shutter_text)
        if shutter_match is not None:
            try:
                speed = float(shutter_match.group(1)) * (1e3 if shutter_match.group(2) else 1.0)
            except ValueError:
                speed = 0.0
            if speed > 0:
                exposure_time_s = 1.0 / speed

    return CameraHeaderInfo(
        start_time_ms=top_frame / frame_rate * 1000.0,
        end_time_ms=bottom_frame / frame_rate * 1000.0,
        total_frames=bottom_frame - top_frame + 1,
        exposure_time_s=exposure_time_s,
        exposure_source=f"GX-8 header {shutter_key} {shutter_text!r} as 1/speed",
    )


def _parse_bmp_header(path: str | Path) -> CameraHeaderInfo:
    """Parse a FAST-camera `{shot}_bmp.txt` header.

    Reuses the fixed-line-index convention of the donor `bmp_arranger.py`
    ``extract_data`` function rather than searching by content, since that
    convention was verified stable across every sample shot header available.
    A BatchConv2 re-export (``Type: GX-8`` on line 2) is parsed by key instead;
    see :func:`_parse_gx8_header`.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    if len(lines) > 1 and _GX8_TYPE_PATTERN.match(lines[1]):
        return _parse_gx8_header(path, lines)

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
    threshold: float = DEFAULT_NEAR_BLACK_THRESHOLD,
    percentage: float = DEFAULT_NEAR_BLACK_PERCENTAGE,
) -> bool:
    """Return whether ``image`` is near-black, reusing the donor `is_near_black` formula."""
    array = np.asarray(image)
    black_pixels = np.sum(array < threshold)
    total_pixels = array.size
    return (black_pixels / total_pixels) > percentage


def estimate_dark_level(
    frames: Sequence[np.ndarray | None],
    *,
    percentile: float = DARK_LEVEL_PERCENTILE,
) -> float:
    """Return a shot's sensor dark level (see ``DARK_LEVEL_FRAME_QUANTILE``)."""
    levels = [float(np.percentile(frame, percentile)) for frame in frames if frame is not None]
    if not levels:
        raise CameraFrameSelectionError("No frames available to estimate the dark level.")
    return float(np.quantile(levels, DARK_LEVEL_FRAME_QUANTILE))


@dataclass(frozen=True)
class FrameSelection:
    """The outcome of dark-frame rejection on one shot's frame sequence."""

    onset: int
    end: int
    total_frames: int
    threshold: float
    percentage: float
    buffer_frames: int
    rule: str
    #: First and last frame inside ``[onset, end]`` that exists: the original
    #: indices of the first and last frame actually stored. They differ from
    #: ``onset``/``end`` only when padding lands on missing frames.
    first_retained: int
    last_retained: int
    #: ``None`` when the caller fixed ``threshold`` (rule ``fixed-v1``).
    dark_level: float | None = None

    def as_record(self) -> dict[str, Any]:
        """JSON-ready form, as stored in the stage manifest."""
        return {
            "rule": self.rule,
            "first_retained": self.first_retained,
            "last_retained": self.last_retained,
            "total_frames": self.total_frames,
            "threshold": self.threshold,
            "percentage": self.percentage,
            "buffer_frames": self.buffer_frames,
            "dark_level": self.dark_level,
        }


def select_valid_frames(
    frames: Sequence[np.ndarray | None],
    *,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    threshold: float | None = None,
    percentage: float = DEFAULT_NEAR_BLACK_PERCENTAGE,
    dark_margin: float = DEFAULT_DARK_MARGIN,
) -> FrameSelection:
    """Find the ``[onset, end]`` interval of non-dark frames, with padding.

    Mirrors the donor `find_valid_frame_range`: scans for the first/last
    non-dark, non-missing frame, then pads by ``buffer_frames`` on each side,
    clamped to the available index range. Dark frames *inside* the interval
    are kept.

    With ``threshold=None`` (the default) the near-black level is
    ``estimate_dark_level(frames) + dark_margin``, so the rule holds across
    exports whose dark floor differs. Passing a number restores the donor's
    fixed level (``DEFAULT_NEAR_BLACK_THRESHOLD`` reproduces it).
    """
    if threshold is None:
        dark_level: float | None = min(estimate_dark_level(frames), DARK_LEVEL_CEILING)
        resolved = dark_level + float(dark_margin)
        rule = FRAME_SELECTION_RULE
    else:
        dark_level = None
        resolved = float(threshold)
        rule = FIXED_FRAME_SELECTION_RULE

    total_frames = len(frames)
    first_valid: int | None = None
    last_valid: int | None = None

    for index, frame in enumerate(frames):
        if frame is None:
            continue
        if not is_near_black(frame, threshold=resolved, percentage=percentage):
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
    present = [index for index in range(onset, end + 1) if frames[index] is not None]
    return FrameSelection(
        onset=onset,
        end=end,
        total_frames=total_frames,
        first_retained=present[0],
        last_retained=present[-1],
        threshold=resolved,
        percentage=float(percentage),
        buffer_frames=int(buffer_frames),
        rule=rule,
        dark_level=dark_level,
    )


def find_valid_frame_interval(
    frames: Sequence[np.ndarray | None],
    *,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    threshold: float | None = None,
    percentage: float = DEFAULT_NEAR_BLACK_PERCENTAGE,
    dark_margin: float = DEFAULT_DARK_MARGIN,
) -> tuple[int, int]:
    """Return only ``(onset, end)`` of :func:`select_valid_frames`."""
    selection = select_valid_frames(
        frames,
        buffer_frames=buffer_frames,
        threshold=threshold,
        percentage=percentage,
        dark_margin=dark_margin,
    )
    return selection.onset, selection.end


_SELECTION_RANGE_PATTERN = re.compile(
    r"retained original frame indices \[(\d+), (\d+)\] out of (\d+)"
)
_SELECTION_PARAMETER_PATTERN = re.compile(r"\b(rule|threshold|percentage|buffer_frames|dark_level)=([^,;)\s]+)")


def parse_frame_selection(comment: str) -> dict[str, Any] | None:
    """Read the frame selection back out of a ``camera_visible`` IDS comment.

    Returns the :meth:`FrameSelection.as_record` fields, or ``None`` when the
    comment carries no selection. The comment names the original indices of
    the first and last *stored* frame, so those are what come back. Products written before the rule was named
    have no ``rule=`` and are reported as ``fixed-v1``.
    """
    match = _SELECTION_RANGE_PATTERN.search(comment or "")
    if match is None:
        return None
    parameters = dict(_SELECTION_PARAMETER_PATTERN.findall(comment))
    rule = parameters.get("rule", FIXED_FRAME_SELECTION_RULE)

    def number(key: str) -> float | None:
        value = parameters.get(key)
        return None if value is None else float(value)

    buffer_frames = number("buffer_frames")
    return {
        "rule": rule,
        "first_retained": int(match.group(1)),
        "last_retained": int(match.group(2)),
        "total_frames": int(match.group(3)),
        "threshold": number("threshold"),
        "percentage": number("percentage"),
        "buffer_frames": None if buffer_frames is None else int(buffer_frames),
        "dark_level": number("dark_level"),
    }


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
    # Where the camera views from. The two landmark angles above are a separate,
    # still-unreconciled frame (issue #746); this one comes straight from the
    # port document.
    set_path(ods, "camera_visible.channel.0.aperture.0.centre.r", PORT_MAJOR_RADIUS_M)
    set_path(ods, "camera_visible.channel.0.aperture.0.centre.phi", port_phi(CAMERA_PORT))
    set_path(
        ods,
        "camera_visible.channel.0.aperture.0.centre.z",
        PORT_TOP_M - 0.5 * PORT_HEIGHT_M,
    )
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
    near_black_threshold: float | None = None,
    near_black_percentage: float = DEFAULT_NEAR_BLACK_PERCENTAGE,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    dark_margin: float = DEFAULT_DARK_MARGIN,
    channel_name: str = "Fast Camera",
) -> None:
    """Populate ``ods`` with VEST FAST-camera raw frames for ``shot``.

    Valid frames are selected from image content: frames outside the
    near-black-rejection interval (padded by ``buffer_frames`` on each side)
    are discarded, and the retained frames are reindexed from 0. The
    near-black level is the shot's own dark level plus ``dark_margin`` unless
    ``near_black_threshold`` fixes it (:func:`select_valid_frames`); the
    outcome is written into ``ids_properties.comment`` in the form
    :func:`parse_frame_selection` reads back. This is the
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

    selection = select_valid_frames(
        raw_frames,
        buffer_frames=buffer_frames,
        threshold=near_black_threshold,
        percentage=near_black_percentage,
        dark_margin=dark_margin,
    )
    onset, end = selection.onset, selection.end

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
        f"exposure_time from {header.exposure_source}."
        if header.exposure_time_s is not None
        else "exposure_time unavailable: no ShutterSpeed line found in header."
    )
    dark_level_note = (
        f", dark_level={selection.dark_level:g}" if selection.dark_level is not None else ""
    )
    selection_note = (
        f"Frames selected by image-content dark-frame rejection "
        f"(rule={selection.rule}, threshold={selection.threshold:g}{dark_level_note}, "
        f"percentage={selection.percentage:g}, buffer_frames={selection.buffer_frames}); "
        f"retained original frame indices "
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
    "DARK_LEVEL_CEILING",
    "DARK_LEVEL_FRAME_QUANTILE",
    "DARK_LEVEL_PERCENTILE",
    "DEFAULT_BUFFER_FRAMES",
    "DEFAULT_DARK_MARGIN",
    "DEFAULT_NEAR_BLACK_PERCENTAGE",
    "DEFAULT_NEAR_BLACK_THRESHOLD",
    "FIXED_FRAME_SELECTION_RULE",
    "FRAME_SELECTION_RULE",
    "FrameSelection",
    "camera_visible",
    "camera_visible_from_frame_dir",
    "camera_visible_from_raw_database",
    "estimate_dark_level",
    "find_valid_frame_interval",
    "frame_time_ms",
    "is_near_black",
    "narrow_image_storage",
    "parse_frame_selection",
    "save_camera_visible_ods",
    "select_valid_frames",
    "vfit_camera_visible_dynamic",
    "vfit_camera_visible_static",
]

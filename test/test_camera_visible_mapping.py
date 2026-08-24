from pathlib import Path

import cv2
import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.camera_visible import (
    CameraFrameSelectionError,
    camera_visible,
    camera_visible_from_frame_dir,
    find_valid_frame_interval,
    frame_time_ms,
    is_near_black,
)

SHOT = 99999
IMAGE_SHAPE = (6, 8)  # (lines_n, columns_n) -- tiny synthetic frame, not real 1280x1024.


def _write_header(
    shot_dir: Path,
    shot: int,
    *,
    total_frames: int,
    start_time_ms: float = 280.0,
    end_time_ms: float = 320.0,
    shutter_speed_line: str | None = "ShutterSpeed: 333.3k(3.0us)\n",
) -> Path:
    """Build a synthetic `{shot}_bmp.txt` using the real fixed-line-index layout."""
    lines = ["\n"] * 76
    lines[15] = f"Frames: {total_frames}\n"
    if shutter_speed_line is not None:
        lines[25] = shutter_speed_line
    lines[74] = f"Top Frame,+700,20/06/18 21:41:01.708809,+{start_time_ms / 1000:012.6f}\n"
    lines[75] = f"Bottom Frame,+850,20/06/18 21:41:01.768809,+{end_time_ms / 1000:012.6f}\n"

    header_path = shot_dir / f"{shot}_bmp.txt"
    header_path.write_text("".join(lines), encoding="utf-8")
    return header_path


def _write_frame(shot_dir: Path, shot: int, index: int, *, bright: bool) -> None:
    value = 200 if bright else 5
    image = np.full(IMAGE_SHAPE, value, dtype=np.uint8)
    cv2.imwrite(str(shot_dir / f"{shot}_{index:08d}.bmp"), image)


def _write_shot(
    tmp_path: Path,
    *,
    total_frames: int,
    bright_indices: set[int],
    missing_indices: set[int] = frozenset(),
    shutter_speed_line: str | None = "ShutterSpeed: 333.3k(3.0us)\n",
    shot: int = SHOT,
) -> Path:
    shot_dir = tmp_path / str(shot)
    shot_dir.mkdir()
    _write_header(shot_dir, shot, total_frames=total_frames, shutter_speed_line=shutter_speed_line)
    for index in range(total_frames):
        if index in missing_indices:
            continue
        _write_frame(shot_dir, shot, index, bright=index in bright_indices)
    return shot_dir


def test_frame_time_ms_matches_donor_linear_formula():
    assert frame_time_ms(0, 11, 280.0, 320.0) == pytest.approx(280.0)
    assert frame_time_ms(10, 11, 280.0, 320.0) == pytest.approx(320.0)
    assert frame_time_ms(5, 11, 280.0, 320.0) == pytest.approx(300.0)


def test_frame_time_ms_rejects_degenerate_total_frames():
    with pytest.raises(ValueError):
        frame_time_ms(0, 1, 280.0, 320.0)


def test_is_near_black_matches_donor_threshold_and_percentage():
    dark = np.full((4, 4), 5, dtype=np.uint8)
    bright = np.full((4, 4), 200, dtype=np.uint8)
    assert bool(is_near_black(dark)) is True
    assert bool(is_near_black(bright)) is False


def test_find_valid_frame_interval_pads_two_frames_each_side():
    # frames: dark dark dark bright bright bright bright dark dark
    frames = [
        np.full((2, 2), 5, dtype=np.uint8) if i not in (3, 4, 5, 6) else np.full((2, 2), 200, dtype=np.uint8)
        for i in range(9)
    ]
    onset, end = find_valid_frame_interval(frames, buffer_frames=2)
    assert (onset, end) == (1, 8)


def test_find_valid_frame_interval_clamps_at_boundaries():
    # bright frame at index 0 and at the last index -- padding must clamp, not go out of range.
    frames = [np.full((2, 2), 5, dtype=np.uint8) for _ in range(5)]
    frames[0] = np.full((2, 2), 200, dtype=np.uint8)
    frames[4] = np.full((2, 2), 200, dtype=np.uint8)
    onset, end = find_valid_frame_interval(frames, buffer_frames=2)
    assert (onset, end) == (0, 4)


def test_find_valid_frame_interval_raises_when_all_dark():
    frames = [np.full((2, 2), 5, dtype=np.uint8) for _ in range(5)]
    with pytest.raises(CameraFrameSelectionError):
        find_valid_frame_interval(frames, buffer_frames=2)


def test_camera_visible_frame_ordering_and_padding(tmp_path):
    total_frames = 10
    bright_indices = {4, 5, 6}
    shot_dir = _write_shot(tmp_path, total_frames=total_frames, bright_indices=bright_indices)

    ods = ODS()
    camera_visible(ods, SHOT, frame_dir=shot_dir)

    # Expected retained original indices: onset=max(0,4-2)=2, end=min(9,6+2)=8 -> 2..8 (7 frames).
    n_frames = len(ods["camera_visible.channel.0.detector.0.frame"])
    assert n_frames == 7

    expected_original_indices = list(range(2, 9))
    expected_times_s = [
        frame_time_ms(idx, total_frames, 280.0, 320.0) / 1000.0 for idx in expected_original_indices
    ]
    actual_times_s = [
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.time"] for i in range(n_frames)
    ]
    np.testing.assert_allclose(actual_times_s, expected_times_s)
    # ascending order
    assert actual_times_s == sorted(actual_times_s)


def test_camera_visible_image_dimensions_and_round_trip(tmp_path):
    shot_dir = _write_shot(tmp_path, total_frames=6, bright_indices={2, 3})
    ods = ODS()
    camera_visible(ods, SHOT, frame_dir=shot_dir)

    lines_n = ods["camera_visible.channel.0.detector.0.lines_n"]
    columns_n = ods["camera_visible.channel.0.detector.0.columns_n"]
    assert (lines_n, columns_n) == IMAGE_SHAPE

    image = ods["camera_visible.channel.0.detector.0.frame.0.image_raw"]
    assert image.shape == IMAGE_SHAPE


def test_camera_visible_interior_missing_frame_is_skipped(tmp_path):
    total_frames = 8
    bright_indices = {2, 3, 4, 5}
    shot_dir = _write_shot(
        tmp_path,
        total_frames=total_frames,
        bright_indices=bright_indices,
        missing_indices={4},
    )

    ods = ODS()
    camera_visible(ods, SHOT, frame_dir=shot_dir)

    # onset=max(0,2-2)=0, end=min(7,5+2)=7 -> indices 0..7 minus missing index 4 -> 7 frames.
    n_frames = len(ods["camera_visible.channel.0.detector.0.frame"])
    assert n_frames == 7

    expected_original_indices = [i for i in range(0, 8) if i != 4]
    expected_times_s = [
        frame_time_ms(idx, total_frames, 280.0, 320.0) / 1000.0 for idx in expected_original_indices
    ]
    actual_times_s = [
        ods[f"camera_visible.channel.0.detector.0.frame.{i}.time"] for i in range(n_frames)
    ]
    np.testing.assert_allclose(actual_times_s, expected_times_s)


def test_camera_visible_exposure_time_from_shutter_speed(tmp_path):
    shot_dir = _write_shot(tmp_path, total_frames=6, bright_indices={2, 3})
    ods = ODS()
    camera_visible(ods, SHOT, frame_dir=shot_dir)
    assert ods["camera_visible.channel.0.detector.0.exposure_time"] == pytest.approx(3.0e-6)


def test_camera_visible_missing_shutter_speed_leaves_exposure_time_unset(tmp_path):
    shot_dir = _write_shot(
        tmp_path, total_frames=6, bright_indices={2, 3}, shutter_speed_line=None
    )
    ods = ODS()
    camera_visible(ods, SHOT, frame_dir=shot_dir)
    assert "exposure_time" not in ods["camera_visible.channel.0.detector.0"].keys()


def test_camera_visible_all_dark_shot_raises(tmp_path):
    shot_dir = _write_shot(tmp_path, total_frames=5, bright_indices=set())
    ods = ODS()
    with pytest.raises(CameraFrameSelectionError):
        camera_visible(ods, SHOT, frame_dir=shot_dir)


def test_camera_visible_missing_frame_dir_raises(tmp_path):
    ods = ODS()
    with pytest.raises(FileNotFoundError):
        camera_visible(ods, SHOT, frame_dir=tmp_path / "does_not_exist")


def test_camera_visible_from_frame_dir_schema_conformance(tmp_path):
    shot_dir = _write_shot(tmp_path, total_frames=6, bright_indices={2, 3})
    ods = camera_visible_from_frame_dir(SHOT, frame_dir=shot_dir, consistency_check=True)

    assert ods["camera_visible.ids_properties.homogeneous_time"] == 1
    image = ods["camera_visible.channel.0.detector.0.frame.0.image_raw"]
    assert np.issubdtype(image.dtype, np.integer)
    assert isinstance(ods["camera_visible.channel.0.detector.0.lines_n"], (int, np.integer))
    assert isinstance(ods["camera_visible.channel.0.detector.0.columns_n"], (int, np.integer))

    # No radiometric calibration data must ever be populated.
    assert "radiance" not in ods["camera_visible.channel.0.detector.0.frame.0"].keys()
    assert "counts_to_radiance" not in ods["camera_visible.channel.0.detector.0"].keys()

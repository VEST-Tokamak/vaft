from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
from omas import ODS

import vaft.omas
from vaft.machine_mapping.camera_visible import (
    CameraFrameSelectionError,
    _parse_bmp_header,
    camera_visible,
    camera_visible_from_frame_dir,
    find_valid_frame_interval,
    frame_time_ms,
    is_near_black,
    narrow_image_storage,
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
        # 24, not 25. Every real header puts `ShutterSpeed:` here and
        # `CustomShutterValueNumerator` at 25; this fixture used to plant it at
        # 25 and so agreed with the parser's off-by-one instead of catching it.
        lines[24] = shutter_speed_line
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


def test_centre_triggered_header_keeps_its_negative_start_time(tmp_path):
    """A centre-triggered acquisition begins before the trigger.

    Real VEST headers (shots 38635 and 38769) record the top frame's relative
    time as `-00000.433200`. Requiring a leading "+" rejected those headers
    outright; dropping the sign instead would have been worse, placing a
    pre-trigger frame after the trigger.
    """
    shot_dir = tmp_path / "38635"
    shot_dir.mkdir()
    header = _write_header(shot_dir, 38635, total_frames=3)
    lines = header.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[74] = "Top Frame,-1083,23/02/27 16:34:59.513349,-00000.433200\n"
    lines[75] = "Bottom Frame,+1083,23/02/27 16:35:00.379749,+00000.433200\n"
    header.write_text("".join(lines), encoding="utf-8")

    info = _parse_bmp_header(header)
    assert info.start_time_ms == pytest.approx(-433.2)
    assert info.end_time_ms == pytest.approx(433.2)
    assert frame_time_ms(0, 3, info.start_time_ms, info.end_time_ms) == pytest.approx(-433.2)
    assert frame_time_ms(1, 3, info.start_time_ms, info.end_time_ms) == pytest.approx(0.0)


def test_exposure_parses_from_the_line_the_real_headers_use(tmp_path):
    """`ShutterSpeed:` is at index 24; index 25 is CustomShutterValueNumerator.

    The parser read 25 and, because a mismatch yields None rather than an
    error, dropped exposure_time from every camera IDS ever produced without
    saying so. The fixture agreed with the bug, so nothing caught it.
    """
    shot_dir = tmp_path / str(SHOT)
    shot_dir.mkdir()
    header = _write_header(shot_dir, SHOT, total_frames=3, shutter_speed_line=None)
    lines = header.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[24] = "ShutterSpeed: 50k(20.0us)\n"
    lines[25] = "CustomShutterValueNumerator: 397486\n"
    header.write_text("".join(lines), encoding="utf-8")

    assert _parse_bmp_header(header).exposure_time_s == pytest.approx(20.0e-6)


@pytest.mark.parametrize(
    ("shutter_line", "expected_s"),
    [
        # Routine acquisitions state a frame rate before the exposure...
        ("ShutterSpeed: 50k(20.0us)\n", 20.0e-6),
        # ...but a shutter left open states OPEN, and the old pattern's
        # required `<number>k(` prefix silently skipped all of those: 102
        # routine shots and every fluctuation shot in the archive.
        ("ShutterSpeed: OPEN(19.1us)\n", 19.1e-6),
    ],
)
def test_camera_visible_exposure_time_forms(tmp_path, shutter_line, expected_s):
    shot_dir = _write_shot(
        tmp_path, total_frames=6, bright_indices={2, 3}, shutter_speed_line=shutter_line
    )
    ods = ODS()
    camera_visible(ods, SHOT, frame_dir=shot_dir)
    assert ods["camera_visible.channel.0.detector.0.exposure_time"] == pytest.approx(expected_s)


def test_narrow_image_storage_halves_the_width_without_moving_a_pixel(tmp_path):
    shot_dir = _write_shot(tmp_path, total_frames=6, bright_indices={2, 3})
    ods = camera_visible_from_frame_dir(SHOT, frame_dir=shot_dir, consistency_check=True)

    prefix = "camera_visible.channel.0.detector.0.frame"
    n_frames = len(ods[prefix])
    before = [np.asarray(ods[f"{prefix}.{i}.image_raw"]).copy() for i in range(n_frames)]
    # OMAS upcasts INT_2D on assignment, so the mapping cannot leave int32 behind.
    assert all(image.dtype == np.int64 for image in before)

    narrow_image_storage(ods)

    for index, original in enumerate(before):
        narrowed = np.asarray(ods[f"{prefix}.{index}.image_raw"])
        assert narrowed.dtype == np.int32
        assert np.array_equal(narrowed, original)


def test_narrowed_product_stores_int32_and_round_trips_identically(tmp_path):
    shot_dir = _write_shot(tmp_path, total_frames=6, bright_indices={2, 3})
    ods = camera_visible_from_frame_dir(SHOT, frame_dir=shot_dir, consistency_check=True)
    prefix = "camera_visible.channel.0.detector.0.frame"
    n_frames = len(ods[prefix])
    expected = [np.asarray(ods[f"{prefix}.{i}.image_raw"]).copy() for i in range(n_frames)]
    expected_time = np.asarray(ods["camera_visible.time"], dtype=float).copy()

    narrow_image_storage(ods)
    target = tmp_path / f"{SHOT}_camera_visible.h5"
    vaft.omas.save(ods, target, compression="gzip")

    with h5py.File(target, "r") as handle:
        dataset = handle["camera_visible/channel/0/detector/0/frame/0/image_raw"]
        assert dataset.dtype == np.int32
        assert dataset.compression == "gzip"

    reloaded = vaft.omas.load(target)
    assert len(reloaded[prefix]) == n_frames
    for index, original in enumerate(expected):
        assert np.array_equal(np.asarray(reloaded[f"{prefix}.{index}.image_raw"]), original)
    np.testing.assert_array_equal(
        np.asarray(reloaded["camera_visible.time"], dtype=float), expected_time
    )


def test_compressed_product_still_satisfies_the_imas_schema(tmp_path):
    """Compression is a container detail, so a gzip product must still validate."""
    from omas.omas_h5 import load_omas_h5

    shot_dir = _write_shot(tmp_path, total_frames=6, bright_indices={2, 3})
    ods = camera_visible_from_frame_dir(SHOT, frame_dir=shot_dir, consistency_check=True)
    narrow_image_storage(ods)
    target = tmp_path / f"{SHOT}_camera_visible.h5"
    vaft.omas.save(ods, target, compression="gzip")

    reloaded = load_omas_h5(str(target), consistency_check=True)
    assert reloaded.consistency_check is True
    assert len(reloaded["camera_visible.channel.0.detector.0.frame"]) == len(
        ods["camera_visible.channel.0.detector.0.frame"]
    )


def test_save_rejects_compression_for_a_json_target(tmp_path):
    shot_dir = _write_shot(tmp_path, total_frames=6, bright_indices={2, 3})
    ods = camera_visible_from_frame_dir(SHOT, frame_dir=shot_dir, consistency_check=True)
    with pytest.raises(ValueError, match="compression"):
        vaft.omas.save(ods, tmp_path / f"{SHOT}.json", compression="gzip")

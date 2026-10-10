from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
from omas import ODS

import vaft.omas
from vaft.machine_mapping.camera_visible import (
    DEFAULT_NEAR_BLACK_THRESHOLD,
    FIXED_FRAME_SELECTION_RULE,
    FRAME_SELECTION_RULE,
    CameraFrameSelectionError,
    _parse_bmp_header,
    camera_visible,
    camera_visible_from_frame_dir,
    find_valid_frame_interval,
    frame_time_ms,
    is_near_black,
    narrow_image_storage,
    parse_frame_selection,
    select_valid_frames,
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


def _noisy_frames(floor: int, bright_indices: set[int], total: int, *, seed: int = 0) -> list[np.ndarray]:
    """Frames with a sensor floor and +-6 DN noise; bright ones carry a lit region."""
    rng = np.random.default_rng(seed)
    frames = []
    for index in range(total):
        image = floor + rng.integers(-6, 7, size=(40, 40))
        if index in bright_indices:
            image[5:35, 5:35] += 80
        frames.append(np.clip(image, 0, 255).astype(np.uint8))
    return frames


def test_select_valid_frames_follows_a_raised_dark_floor():
    # A BatchConv2 re-export puts the empty vessel at ~48 DN, above the fixed
    # 35: the old rule kept every frame (shot 48909 kept all 101).
    frames = _noisy_frames(48, {10, 11, 12, 13}, 25)
    fixed = select_valid_frames(frames, threshold=DEFAULT_NEAR_BLACK_THRESHOLD)
    assert (fixed.onset, fixed.end) == (0, 24)
    assert fixed.rule == FIXED_FRAME_SELECTION_RULE

    relative = select_valid_frames(frames)
    assert (relative.onset, relative.end) == (8, 15)
    assert relative.rule == FRAME_SELECTION_RULE
    assert 40 <= relative.dark_level <= 48
    assert relative.threshold == pytest.approx(relative.dark_level + 20)


def test_select_valid_frames_matches_fixed_rule_on_original_exports():
    # Original exports sit at ~25 DN, where the fixed 35 already worked.
    frames = _noisy_frames(25, {10, 11, 12, 13}, 25)
    fixed = select_valid_frames(frames, threshold=DEFAULT_NEAR_BLACK_THRESHOLD)
    relative = select_valid_frames(frames)
    assert (relative.onset, relative.end) == (fixed.onset, fixed.end) == (8, 15)


def test_dark_level_survives_a_mostly_bright_recording():
    # Two thirds of the frames lit: the floor still comes from the dark third.
    bright = set(range(4, 22))
    frames = _noisy_frames(48, bright, 27)
    selection = select_valid_frames(frames)
    assert (selection.onset, selection.end) == (2, 23)


def test_uniformly_lit_recording_is_kept_not_rejected():
    # Every frame lit everywhere: without the ceiling the "dark" level would be
    # the lit level and every frame would count as dark.
    frames = [np.full((10, 10), 200, dtype=np.uint8) for _ in range(10)]
    selection = select_valid_frames(frames)
    assert (selection.onset, selection.end) == (0, 9)
    assert selection.dark_level == pytest.approx(60.0)


def test_retained_bounds_skip_missing_padding_frames():
    frames = _noisy_frames(25, {4, 5, 6}, 10)
    frames[2] = None
    selection = select_valid_frames(frames)
    assert (selection.onset, selection.end) == (2, 8)
    assert (selection.first_retained, selection.last_retained) == (3, 8)


def test_select_valid_frames_raises_when_all_dark_at_any_floor():
    with pytest.raises(CameraFrameSelectionError):
        select_valid_frames(_noisy_frames(48, set(), 10))


def test_parse_frame_selection_reads_both_comment_generations():
    legacy = (
        "Frames selected by image-content dark-frame rejection "
        "(threshold=35, percentage=0.98, buffer_frames=2); retained original frame "
        "indices [0, 100] out of 101, reindexed from 0."
    )
    record = parse_frame_selection(legacy)
    assert record["rule"] == FIXED_FRAME_SELECTION_RULE
    assert (record["first_retained"], record["last_retained"], record["total_frames"]) == (0, 100, 101)
    assert record["threshold"] == 35 and record["dark_level"] is None
    assert parse_frame_selection("no selection here") is None


def test_camera_visible_comment_round_trips_the_selection(tmp_path):
    shot_dir = _write_shot(tmp_path, total_frames=10, bright_indices={4, 5, 6})
    ods = ODS()
    camera_visible(ods, SHOT, frame_dir=shot_dir)
    record = parse_frame_selection(ods["camera_visible.ids_properties.comment"])
    assert record["rule"] == FRAME_SELECTION_RULE
    assert (record["first_retained"], record["last_retained"], record["total_frames"]) == (2, 8, 10)
    assert record["dark_level"] == pytest.approx(5.0)
    assert record["threshold"] == pytest.approx(25.0)
    assert record["buffer_frames"] == 2 and record["percentage"] == pytest.approx(0.98)


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


def test_narrow_image_storage_leaves_a_camera_less_ods_alone():
    """Reading a missing ODS path creates it, so the guard must not index.

    A `try: ods[path] except KeyError` guard never fired: the read grafted an
    empty `camera_visible` onto an unrelated ODS and then disabled its
    consistency check for a diagnostic it does not carry. `ods.flat()` does
    not show the grafted key, so nothing made it visible.
    """
    ods = ODS(consistency_check=True)
    ods["magnetics.ids_properties.homogeneous_time"] = 1

    narrow_image_storage(ods)

    assert "camera_visible" not in ods
    assert sorted(ods.keys()) == ["magnetics"]
    assert ods.consistency_check is True


def test_narrow_image_storage_covers_every_channel(tmp_path):
    """It used to narrow channel 0 only, while the manifest claimed the lot."""
    ods = ODS(consistency_check=True)
    ods["camera_visible.ids_properties.homogeneous_time"] = 1
    ods["camera_visible.time"] = np.array([0.3, 0.4])
    image = np.full((4, 4), 7, dtype=np.uint8)
    for channel in (0, 1):
        ods[f"camera_visible.channel.{channel}.detector.0.frame.0.image_raw"] = image

    narrow_image_storage(ods)

    for channel in (0, 1):
        stored = np.asarray(ods[f"camera_visible.channel.{channel}.detector.0.frame.0.image_raw"])
        assert stored.dtype == np.int32
        assert np.array_equal(stored, image)


def _write_gx8_header(
    shot_dir: Path,
    shot: int,
    *,
    top_frame: int = 740,
    bottom_frame: int = 742,
    frame_rate: int = 2500,
    shutter: str = "200k",
) -> Path:
    """A BatchConv2 re-export header, abridged from real shot 22360's.

    ``BEGIN_TIME``/``END_TIME`` are left at the converter template's values on
    purpose: they disagree with the frame numbers, as they do in the archive.
    """
    text = (
        "ID: 5\n"
        "Type: GX-8\n"
        "CID: 1955\n"
        f"TopFrame: {top_frame}\n"
        f"BottomFrame: {bottom_frame}\n"
        "Rec_Time: 19/05/07 21:52:26.567279\n"
        "TriggerSelect: CENTER\n"
        "TriggerValue: 0\n"
        f"Frame_SRC_Rate: {frame_rate}\n"
        "Frame_SRC_Size: 1024x1280\n"
        f"Shutter_SRC_Speed: {shutter}\n"
        "ShutterValueNumerator: 397486\n"
        "ShutterValueDenominator: 1000000000\n"
        "\n"
        f"Frame_Rate: {frame_rate}\n"
        f"Shutter_Speed: {shutter}\n"
        "File Type : bmp\n"
        f"Conversion Range : {top_frame}  -> {bottom_frame}\n"
        "\n"
        "[CONV_PARAM]\n"
        f"BEGIN_FRAME={top_frame}\n"
        f"END_FRAME={bottom_frame}\n"
        "BEGIN_TIME=+0.280000\n"
        "END_TIME=+0.340000\n"
        f"FRAME_RATE={frame_rate}\n"
    )
    header_path = shot_dir / f"{shot}_bmp.txt"
    header_path.write_text(text, encoding="utf-8")
    return header_path


def test_gx8_parser_documents_which_rate_the_frame_numbers_belong_to():
    # Frame numbers index the source recording; the parser reads Frame_Rate
    # first and Frame_SRC_Rate as a fallback, which is only safe while every
    # header carries the same value in both. The docstring must say so
    # (cold review 0.8.0 delta-absorb-16 diagram-docs F4).
    from vaft.machine_mapping.camera_visible import _parse_gx8_header

    doc = _parse_gx8_header.__doc__
    assert "Frame_SRC_Rate" in doc and "source recording" in doc


def test_gx8_header_times_come_from_frame_numbers_not_the_template(tmp_path):
    # The CONV_PARAM BEGIN_TIME/END_TIME (+0.28/+0.34 s) are template values;
    # frames 740..742 at 2500 fps are 0.296..0.2968 s.
    header = _parse_bmp_header(_write_gx8_header(tmp_path, SHOT))

    assert header.total_frames == 3
    assert header.start_time_ms == pytest.approx(296.0)
    assert header.end_time_ms == pytest.approx(296.8)
    assert header.exposure_time_s == pytest.approx(5e-6)


@pytest.mark.parametrize(
    "shutter, expected_s",
    [("333k", 1 / 333e3), ("5k", 2e-4), ("100", 1e-2), ("OPEN", None), ("CUSTOM", None)],
)
def test_gx8_header_exposure_forms(tmp_path, shutter, expected_s):
    header = _parse_bmp_header(_write_gx8_header(tmp_path, SHOT, shutter=shutter))

    if expected_s is None:
        assert header.exposure_time_s is None
    else:
        assert header.exposure_time_s == pytest.approx(expected_s)


def test_gx8_header_agrees_with_the_original_layout_for_the_same_export(tmp_path):
    # Shots 26151/26153/26155 exist in both layouts: TopFrame 700..850 at
    # 2500 fps, ShutterSpeed 200k(5.0us).
    original_dir = tmp_path / "original"
    original_dir.mkdir()
    reexport_dir = tmp_path / "reexport"
    reexport_dir.mkdir()
    original = _parse_bmp_header(
        _write_header(
            original_dir, SHOT, total_frames=151, start_time_ms=280.0, end_time_ms=340.0,
            shutter_speed_line="ShutterSpeed: 200k(5.0us)\n",
        )
    )
    reexport = _parse_bmp_header(
        _write_gx8_header(reexport_dir, SHOT, top_frame=700, bottom_frame=850, shutter="200k")
    )

    assert reexport.total_frames == original.total_frames
    assert reexport.start_time_ms == pytest.approx(original.start_time_ms)
    assert reexport.end_time_ms == pytest.approx(original.end_time_ms)
    assert reexport.exposure_time_s == pytest.approx(original.exposure_time_s)


def test_gx8_header_without_frame_numbers_is_rejected(tmp_path):
    header_path = _write_gx8_header(tmp_path, SHOT)
    header_path.write_text(
        header_path.read_text().replace("TopFrame: 740\n", ""), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="TopFrame"):
        _parse_bmp_header(header_path)


def test_camera_visible_reads_a_gx8_reexport_end_to_end(tmp_path):
    shot_dir = tmp_path / str(SHOT)
    shot_dir.mkdir()
    _write_gx8_header(shot_dir, SHOT, top_frame=740, bottom_frame=744)
    for index in range(5):
        _write_frame(shot_dir, SHOT, index, bright=index == 2)

    ods = ODS()
    camera_visible(ods, SHOT, frame_dir=shot_dir)

    frames = ods["camera_visible.channel.0.detector.0.frame"]
    times = [frames[i]["time"] for i in range(len(frames))]
    assert times == pytest.approx([0.296, 0.2964, 0.2968, 0.2972, 0.2976])
    assert ods["camera_visible.channel.0.detector.0.exposure_time"] == pytest.approx(5e-6)
    assert "Shutter_SRC_Speed" in ods["camera_visible.ids_properties.comment"]

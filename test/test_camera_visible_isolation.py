"""Isolation guarantees for camera mapping and ingest.

Three properties that are easy to break and expensive to notice:

* building a shot must not disturb whatever else the caller already put in the
  ODS, and must not touch the raw archive it read from;
* two builds of one shot must agree, or a regeneration can never be proved
  equivalent to what it replaced;
* the routine and fluctuation camera sets must stay apart -- routine ingest
  must not absorb fluctuation acquisitions, and their products must not share
  a path.

The mapping and ingest code that these pin lives in
``vaft/machine_mapping/camera_visible.py`` and
``workflow/automatic_pipeline_2_corrective_data_update/ingest_external_diagnostics.py``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from vaft.machine_mapping.camera_visible import camera_visible

cv2 = pytest.importorskip("cv2")
ODS = pytest.importorskip("omas").ODS


def _load_ingest():
    script = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "automatic_pipeline_2_corrective_data_update"
        / "ingest_external_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("ingest_external_diagnostics", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ingest_external_diagnostics"] = module
    spec.loader.exec_module(module)
    return module


INGEST = _load_ingest()

SHOT = 41451
IMAGE_SHAPE = (6, 8)


def _write_shot(root: Path, shot: int, *, total_frames: int = 5, bright=(1, 2, 3)) -> Path:
    """Lay out one shot the way the consolidated archive does."""
    shot_dir = root / str(shot)
    shot_dir.mkdir(parents=True)
    lines = ["\n"] * 76
    lines[15] = f"Frames: {total_frames}\n"
    lines[24] = "ShutterSpeed: OPEN(19.1us)\n"
    lines[74] = "Top Frame,+700,20/06/18 21:41:01.708809,+00000.280000\n"
    lines[75] = "Bottom Frame,+850,20/06/18 21:41:01.768809,+00000.320000\n"
    (shot_dir / f"{shot}_bmp.txt").write_text("".join(lines), encoding="utf-8")
    for index in range(total_frames):
        value = 200 if index in bright else 5
        cv2.imwrite(
            str(shot_dir / f"{shot}_{index:08d}.bmp"),
            np.full(IMAGE_SHAPE, value, dtype=np.uint8),
        )
    return shot_dir


def _tree_snapshot(directory: Path) -> dict[str, tuple[int, bytes]]:
    return {
        p.name: (p.stat().st_size, p.read_bytes())
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


class TestBuildDoesNotMutateItsInputs:
    def test_existing_ods_content_survives(self, tmp_path):
        _write_shot(tmp_path, SHOT)
        ods = ODS(consistency_check=True)
        ods["dataset_description.data_entry.pulse"] = SHOT
        ods["magnetics.ids_properties.homogeneous_time"] = 1
        before = set(ods.flat())

        camera_visible(ods, SHOT, data_root=tmp_path)

        assert before.issubset(set(ods.flat()))
        assert ods["dataset_description.data_entry.pulse"] == SHOT
        assert {key.split(".")[0] for key in ods.flat()} == {
            "dataset_description",
            "magnetics",
            "camera_visible",
        }

    def test_raw_archive_is_left_byte_identical(self, tmp_path):
        """Reading a shot must never rewrite, reorder, or touch its frames."""
        shot_dir = _write_shot(tmp_path, SHOT)
        before = _tree_snapshot(shot_dir)

        camera_visible(ODS(consistency_check=True), SHOT, data_root=tmp_path)

        assert _tree_snapshot(shot_dir) == before

    def test_two_builds_of_one_shot_agree(self, tmp_path):
        """Without this, no regeneration can be proved equivalent."""
        _write_shot(tmp_path, SHOT)
        first, second = ODS(consistency_check=True), ODS(consistency_check=True)
        camera_visible(first, SHOT, data_root=tmp_path)
        camera_visible(second, SHOT, data_root=tmp_path)

        prefix = "camera_visible.channel.0.detector.0.frame"
        a, b = first[prefix], second[prefix]
        assert len(a) == len(b)
        for index in range(len(a)):
            assert np.array_equal(
                np.asarray(a[index]["image_raw"]), np.asarray(b[index]["image_raw"])
            )
            assert a[index]["time"] == b[index]["time"]
        assert np.array_equal(
            np.asarray(first["camera_visible.time"]), np.asarray(second["camera_visible.time"])
        )


class TestRoutineAndFluctuationStayApart:
    def test_the_fluctuation_tree_uses_the_same_mapping(self):
        """Same acquisition system, same IDS -- the split is policy, not data model."""
        assert INGEST.DIAGNOSTIC_TREES["camera_visible_fluctuation"] == "camera_visible"
        assert INGEST.DIAGNOSTIC_TREES["camera_visible"] == "camera_visible"

    def test_routine_ingest_excludes_fluctuation_by_default(self):
        default = [t for t in INGEST.DIAGNOSTIC_TREES if t not in INGEST.RESERVED_TREES]
        assert "camera_visible" in default
        assert "camera_visible_fluctuation" not in default
        assert "camera_visible_fluctuation" in INGEST.RESERVED_TREES

    def test_discovery_reads_only_the_tree_it_was_asked_for(self, tmp_path):
        routine = tmp_path / "legacy" / "camera_visible"
        fluctuation = tmp_path / "legacy" / "camera_visible_fluctuation"
        _write_shot(routine, 41451)
        _write_shot(fluctuation, 27134)

        assert INGEST.discover_shots(tmp_path, "camera_visible") == [41451]
        assert INGEST.discover_shots(tmp_path, "camera_visible_fluctuation") == [27134]

    def test_discovery_of_an_absent_tree_is_empty_not_an_error(self, tmp_path):
        assert INGEST.discover_shots(tmp_path, "camera_visible_fluctuation") == []

    def test_products_of_the_two_trees_never_share_a_path(self, tmp_path):
        """Both trees can hold the same shot number; their products must not collide."""
        shot = 27134
        routine = tmp_path / "ods" / "camera_visible" / str(shot)
        fluctuation = tmp_path / "ods" / "camera_visible_fluctuation" / str(shot)
        assert routine != fluctuation
        # The tree name is in the filename too, so even a flattened copy stays distinct.
        assert f"{shot}_camera_visible.h5" != f"{shot}_camera_visible_fluctuation.h5"

    def test_each_tree_keeps_its_own_registry(self, tmp_path):
        """One shared registry let a second run erase the first one's records."""
        routine = tmp_path / "ods" / "camera_visible" / INGEST.REGISTRY_NAME
        fluctuation = tmp_path / "ods" / "camera_visible_fluctuation" / INGEST.REGISTRY_NAME
        assert routine != fluctuation

        INGEST.save_registry(routine, {"camera_visible/41451": {"status": "success"}})
        INGEST.save_registry(
            fluctuation, {"camera_visible_fluctuation/27134": {"status": "success"}}
        )
        assert json.loads(routine.read_text()) == {"camera_visible/41451": {"status": "success"}}
        assert json.loads(fluctuation.read_text()) == {
            "camera_visible_fluctuation/27134": {"status": "success"}
        }

    def test_a_second_run_merges_rather_than_replaces(self, tmp_path):
        path = tmp_path / "ods" / "camera_visible" / INGEST.REGISTRY_NAME
        INGEST.save_registry(path, {"camera_visible/1": {"status": "success"}})
        INGEST.save_registry(path, {"camera_visible/2": {"status": "failed"}})
        assert set(json.loads(path.read_text())) == {"camera_visible/1", "camera_visible/2"}

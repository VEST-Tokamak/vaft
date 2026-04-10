"""Tests for Thomson MAT path resolution (legacy filepath as ``data_root``)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import vaft
from vaft.machine_mapping.thomson_scattering import (
    _normalize_thomson_time_to_seconds,
    _resolve_thomson_mat_file,
)


class TestThomsonMatResolve(unittest.TestCase):
    def test_explicit_mat_path_as_data_root_finds_file(self) -> None:
        data_dir = Path(vaft.__file__).resolve().parent / "data"
        mat_path = data_dir / "46051_NeTe.mat"
        resolved = _resolve_thomson_mat_file(46051, data_root=mat_path)
        self.assertEqual(resolved.resolve(), mat_path.resolve())

    def test_mat_path_not_used_as_directory_for_shot_patterns(self) -> None:
        """Regression: ``foo.mat/thomson_scattering/...`` must not be searched."""
        fake_mat = Path(tempfile.gettempdir()) / "vaft_thomson_resolve_nonexistent_999.mat"
        fake_mat.unlink(missing_ok=True)
        try:
            _resolve_thomson_mat_file(46051, data_root=fake_mat)
        except FileNotFoundError as exc:
            msg = str(exc)
            self.assertNotIn(str(fake_mat / "thomson_scattering"), msg)
        else:
            self.fail("expected FileNotFoundError for missing MAT path")

    def test_time_normalization_converts_ms_to_seconds(self) -> None:
        raw_ms = np.array([300.0, 301.0, 302.0])
        time_s = _normalize_thomson_time_to_seconds(raw_ms)
        np.testing.assert_allclose(time_s, np.array([0.300, 0.301, 0.302]))

    def test_time_normalization_always_divides_by_1e3(self) -> None:
        raw_s = np.array([0.300, 0.301, 0.302])
        time_s = _normalize_thomson_time_to_seconds(raw_s)
        np.testing.assert_allclose(time_s, raw_s / 1e3)


if __name__ == "__main__":
    unittest.main()

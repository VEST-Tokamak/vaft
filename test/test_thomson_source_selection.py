"""Which Thomson MAT wins when a shot has more than one.

Eleven VEST shots carry two Thomson files. Before this rule existed the
library resolver read a fixed candidate list that placed ``_v9`` ahead of
``_v9_rev``, so nine shots resolved to a superseded analysis whose ``T_e``
differs from the revision by up to 58%; and the corrective updater did not use
that resolver at all, picking whichever path ``os.listdir`` yielded last. The
library and the pipeline could therefore disagree about which file was
authoritative for the same shot.

These tests pin the rule itself, both entry points that apply it, and the
refusal to invent a timebase.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from vaft.machine_mapping.thomson_scattering import (
    _recover_simple_time,
    _resolve_thomson_mat_file,
    thomson_scattering,
    thomson_source_rank,
)

ODS = pytest.importorskip("omas").ODS

#: Real competing pairs from /srv/vest.diagnostic.
V9_PAIR_SHOTS = (40323, 40324, 40325, 40329, 40330, 40331)
OLD_VS_REV_SHOTS = (39915, 39916)

CHANNELS = 5
SAMPLES = 10
TAGS = ("poly1R1", "poly2R2", "poly3R3", "poly4R4", "poly5R5")


def _write_v9(path: Path, *, te: float, ne: float) -> None:
    """A v9-schema MAT: per-polychromator fits keyed on ``time_TS``."""
    payload = {"time_TS": np.linspace(300.0, 309.0, SAMPLES)}
    for tag in TAGS:
        payload[f"{tag}_Te"] = np.full(SAMPLES, te)
        payload[f"{tag}_Ne"] = np.full(SAMPLES, ne)
        payload[f"{tag}_sigmaTe"] = np.full(SAMPLES, 1.0)
        payload[f"{tag}_sigmaNe"] = np.full(SAMPLES, 1e17)
    savemat(str(path), payload)


def _write_simple(path: Path, *, te: float, ne: float, with_time: bool) -> None:
    """The oldest schema: bare Ne/Te arrays, and sometimes no time at all."""
    payload = {
        "Te": np.full((SAMPLES, CHANNELS), te),
        "Ne": np.full((SAMPLES, CHANNELS), ne),
        "sigmaTe": np.full((SAMPLES, CHANNELS), 1.0),
        "sigmaNe": np.full((SAMPLES, CHANNELS), 1e17),
    }
    if with_time:
        payload["time"] = np.linspace(300.0, 309.0, SAMPLES)
    savemat(str(path), payload)


def _channel_te(ods) -> np.ndarray:
    return np.asarray(ods["thomson_scattering.channel.0.t_e.data"], dtype=float)


class TestRankingRule:
    @pytest.mark.parametrize("shot", V9_PAIR_SHOTS)
    def test_a_revision_outranks_the_version_it_revises(self, shot: int) -> None:
        assert thomson_source_rank(f"NeTe_Shot{shot}_v9_rev.mat", shot) > thomson_source_rank(
            f"NeTe_Shot{shot}_v9.mat", shot
        )

    @pytest.mark.parametrize("shot", OLD_VS_REV_SHOTS)
    def test_a_revision_outranks_the_oldest_schema(self, shot: int) -> None:
        assert thomson_source_rank(f"NeTe_Shot{shot}_v9_rev.mat", shot) > thomson_source_rank(
            f"{shot}_NeTe.mat", shot
        )

    def test_a_later_version_outranks_an_earlier_one(self) -> None:
        """So a future _v10 needs no edit to this module to be preferred."""
        assert thomson_source_rank("NeTe_Shot40330_v10.mat", 40330) > thomson_source_rank(
            "NeTe_Shot40330_v9_rev.mat", 40330
        )

    @pytest.mark.parametrize(
        "name",
        [
            "NeTe_Shot4033_v9_rev.mat",   # a shorter shot number
            "NeTe_Shot403300_v9.mat",     # a longer one
            "NeTe_Shot40323_v9.mat",      # a different shot entirely
            "IDS_40330.mat",              # charge exchange, not Thomson
            "notes.txt",
        ],
    )
    def test_a_file_belonging_to_another_shot_is_not_ranked(self, name: str) -> None:
        assert thomson_source_rank(name, 40330) is None


class TestLibraryResolver:
    @pytest.mark.parametrize("shot", V9_PAIR_SHOTS)
    def test_the_revision_wins_over_v9(self, shot: int, tmp_path: Path) -> None:
        _write_v9(tmp_path / f"NeTe_Shot{shot}_v9.mat", te=10.0, ne=1e18)
        _write_v9(tmp_path / f"NeTe_Shot{shot}_v9_rev.mat", te=20.0, ne=2e18)
        assert _resolve_thomson_mat_file(shot, data_root=tmp_path).name == (
            f"NeTe_Shot{shot}_v9_rev.mat"
        )

    @pytest.mark.parametrize("shot", OLD_VS_REV_SHOTS)
    def test_the_revision_wins_over_the_oldest_schema(self, shot: int, tmp_path: Path) -> None:
        _write_simple(tmp_path / f"{shot}_NeTe.mat", te=10.0, ne=1e18, with_time=True)
        _write_v9(tmp_path / f"NeTe_Shot{shot}_v9_rev.mat", te=20.0, ne=2e18)
        assert _resolve_thomson_mat_file(shot, data_root=tmp_path).name == (
            f"NeTe_Shot{shot}_v9_rev.mat"
        )

    def test_the_values_read_come_from_the_revision(self, tmp_path: Path) -> None:
        """The point of the rule: the revised numbers are what reach the IDS."""
        _write_v9(tmp_path / "NeTe_Shot40330_v9.mat", te=10.0, ne=1e18)
        _write_v9(tmp_path / "NeTe_Shot40330_v9_rev.mat", te=20.0, ne=2e18)
        ods = ODS(consistency_check=False)
        thomson_scattering(ods, 40330, data_root=tmp_path)
        np.testing.assert_allclose(_channel_te(ods), np.full(SAMPLES, 20.0))

    def test_selection_does_not_depend_on_directory_order(self, tmp_path: Path) -> None:
        """Same two files, opposite creation order, same answer."""
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir(), second.mkdir()
        _write_v9(first / "NeTe_Shot40330_v9.mat", te=10.0, ne=1e18)
        _write_v9(first / "NeTe_Shot40330_v9_rev.mat", te=20.0, ne=2e18)
        _write_v9(second / "NeTe_Shot40330_v9_rev.mat", te=20.0, ne=2e18)
        _write_v9(second / "NeTe_Shot40330_v9.mat", te=10.0, ne=1e18)
        assert (
            _resolve_thomson_mat_file(40330, data_root=first).name
            == _resolve_thomson_mat_file(40330, data_root=second).name
            == "NeTe_Shot40330_v9_rev.mat"
        )

    def test_a_lone_file_is_still_found(self, tmp_path: Path) -> None:
        """Shots with one file must be unaffected by the ranking."""
        _write_simple(tmp_path / "46051_NeTe.mat", te=10.0, ne=1e18, with_time=True)
        assert _resolve_thomson_mat_file(46051, data_root=tmp_path).name == "46051_NeTe.mat"


class TestUpdaterAgreesWithTheLibrary:
    @staticmethod
    def _updater():
        script = (
            Path(__file__).resolve().parents[1]
            / "workflow"
            / "automatic_pipeline_2_corrective_data_update"
            / "update_thomson_scattering_and_core_profile.py"
        )
        spec = importlib.util.spec_from_file_location("ts_updater", script)
        module = importlib.util.module_from_spec(spec)
        sys.modules["ts_updater"] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - needs the production env
            pytest.skip(f"updater not importable here: {type(exc).__name__}: {exc}")
        return module

    def test_both_entry_points_choose_the_same_file(self, tmp_path: Path) -> None:
        """The defect was that they could differ; this is what stops it."""
        updater = self._updater()
        for shot in (39915, 40330):
            _write_v9(tmp_path / f"NeTe_Shot{shot}_v9.mat", te=10.0, ne=1e18)
            _write_v9(tmp_path / f"NeTe_Shot{shot}_v9_rev.mat", te=20.0, ne=2e18)
        _write_simple(tmp_path / "39915_NeTe.mat", te=10.0, ne=1e18, with_time=True)

        chosen = updater.select_thomson_sources(tmp_path)
        for shot in (39915, 40330):
            assert chosen[shot] == _resolve_thomson_mat_file(shot, data_root=tmp_path).name
            assert chosen[shot] == f"NeTe_Shot{shot}_v9_rev.mat"

    def test_one_file_is_selected_per_shot(self, tmp_path: Path) -> None:
        """Not one entry per file, which is what let the last one win."""
        updater = self._updater()
        _write_v9(tmp_path / "NeTe_Shot40330_v9.mat", te=10.0, ne=1e18)
        _write_v9(tmp_path / "NeTe_Shot40330_v9_rev.mat", te=20.0, ne=2e18)
        _write_simple(tmp_path / "40330_NeTe.mat", te=1.0, ne=1e17, with_time=True)
        chosen = updater.select_thomson_sources(tmp_path)
        assert list(chosen) == [40330]
        assert chosen[40330] == "NeTe_Shot40330_v9_rev.mat"

    def test_non_thomson_files_are_left_alone(self, tmp_path: Path) -> None:
        updater = self._updater()
        _write_v9(tmp_path / "NeTe_Shot40330_v9_rev.mat", te=20.0, ne=2e18)
        (tmp_path / "IDS_48224.mat").write_bytes(b"")
        (tmp_path / "CES_47514.mat").write_bytes(b"")
        assert list(updater.select_thomson_sources(tmp_path)) == [40330]


class TestTimeIsRecoveredOrRefused:
    def test_a_timeless_file_with_no_sibling_is_refused(self, tmp_path: Path) -> None:
        """39915_NeTe.mat carries no time field. Inventing one would misplace
        every measurement, so the mapping must fail instead."""
        _write_simple(tmp_path / "39915_NeTe.mat", te=10.0, ne=1e18, with_time=False)
        with pytest.raises(KeyError, match="Refusing to invent timestamps"):
            thomson_scattering(ODS(consistency_check=False), 39915, data_root=tmp_path)

    def test_a_matching_sibling_timebase_is_adopted(self, tmp_path: Path) -> None:
        _write_simple(tmp_path / "39915_NeTe.mat", te=10.0, ne=1e18, with_time=False)
        _write_v9(tmp_path / "NeTe_Shot39915.mat", te=20.0, ne=2e18)  # ranks above the bare file
        ods = ODS(consistency_check=False)
        thomson_scattering(ods, 39915, data_root=tmp_path)
        time = np.asarray(ods["thomson_scattering.time"], dtype=float)
        assert time.size == SAMPLES
        np.testing.assert_allclose(time, np.linspace(300.0, 309.0, SAMPLES) / 1e3)

    def test_a_sibling_of_the_wrong_length_is_not_adopted(self, tmp_path: Path) -> None:
        """A timebase that cannot be matched sample-for-sample is not a match.

        Exercised on the helper rather than through ``thomson_scattering``: every
        other layout outranks the bare ``{shot}_NeTe`` file, so a shot that has a
        sibling at all never resolves to the timeless one. The guard still has to
        hold for the day a sibling exists but does not line up.
        """
        source = tmp_path / "39915_NeTe.mat"
        _write_simple(source, te=10.0, ne=1e18, with_time=False)
        savemat(str(tmp_path / "NeTe_Shot39915_v1.mat"), {"time": np.linspace(0.0, 1.0, 3)})
        mat_data = loadmat(str(source))
        with pytest.raises(KeyError, match="Refusing to invent timestamps"):
            _recover_simple_time(mat_data, 39915, source)

    def test_disagreeing_siblings_are_refused_rather_than_picked_between(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "39915_NeTe.mat"
        _write_simple(source, te=10.0, ne=1e18, with_time=False)
        savemat(
            str(tmp_path / "NeTe_Shot39915_v1.mat"),
            {"time": np.linspace(300.0, 309.0, SAMPLES)},
        )
        savemat(
            str(tmp_path / "NeTe_Shot39915_v2.mat"),
            {"time": np.linspace(400.0, 409.0, SAMPLES)},
        )
        mat_data = loadmat(str(source))
        with pytest.raises(KeyError, match="disagree about"):
            _recover_simple_time(mat_data, 39915, source)

"""Unit tests for TokaMaker output collection and sidecar parsing (OFT-free)."""

import json
import shutil

import pytest

from vaft.code.tokamaker import collect_tokamaker_outputs, parse_stats_sidecar
from vaft.data.resources import data_path


def _sidecar_payload():
    return {
        "converged": True,
        "shot": 39915,
        "time_s": 0.319,
        "targets": {"Ip": 51.0e3},
        "coil_currents_A": {"PF1": -640.0, "PF2": 320.0},
        "f0": 0.06,
        "cocos": 2,
        "o_point": [0.4, 0.0],
        "diverted": False,
        "stats": {"Ip": 50.9e3, "q_95": 5.1, "kappa": 1.6, "beta_pol": 0.4},
    }


def test_collect_parses_real_gfile_and_sidecar(tmp_path):
    sample = data_path("efit/g039915.00319")
    shutil.copy(sample, tmp_path / "g039915.00319")
    (tmp_path / "tokamaker_result.json").write_text(json.dumps(_sidecar_payload()))

    result = collect_tokamaker_outputs(tmp_path)

    assert result.returncode is None          # collect-only; the runner fills this in
    assert result.gfile is not None and result.gfile.name == "g039915.00319"
    assert result.stats_file is not None
    assert len(result.geqdsk) == 1
    ip_from_ods = float(result.ods["equilibrium.time_slice.0.global_quantities.ip"])
    assert ip_from_ods == pytest.approx(float(result.geqdsk[0]["CURRENT"]))
    assert result.scalars["converged"] is True
    assert result.scalars["q_95"] == pytest.approx(5.1)
    assert result.scalars["coil_currents_A"]["PF2"] == pytest.approx(320.0)
    assert result.scalars["targets"]["Ip"] == pytest.approx(51.0e3)


def test_garbage_gfile_is_best_effort(tmp_path):
    (tmp_path / "g012345.00100").write_text("this is not a gEQDSK file\n")
    (tmp_path / "tokamaker_result.json").write_text(json.dumps(_sidecar_payload()))

    result = collect_tokamaker_outputs(tmp_path)

    assert result.gfile is not None
    assert result.geqdsk == ()
    assert result.ods is None
    assert "_geqdsk_error" in result.scalars
    assert result.scalars["converged"] is True   # sidecar still parsed


def test_corrupt_sidecar_is_best_effort(tmp_path):
    (tmp_path / "tokamaker_result.json").write_text("{not json")

    result = collect_tokamaker_outputs(tmp_path)

    assert "_parse_error" in result.scalars


def test_empty_workdir_yields_inert_result(tmp_path):
    result = collect_tokamaker_outputs(tmp_path)

    assert result.gfile is None
    assert result.stats_file is None
    assert result.mesh_file is None
    assert result.ods is None
    assert result.scalars == {}
    assert not result.ok


def test_unrelated_g_prefixed_files_are_ignored(tmp_path):
    (tmp_path / "geometry.json").write_text("{}")
    (tmp_path / "gpec_control_output_n1.nc").write_bytes(b"")

    result = collect_tokamaker_outputs(tmp_path)

    assert result.gfile is None


def test_parse_stats_sidecar_promotes_stats_keys(tmp_path):
    path = tmp_path / "tokamaker_result.json"
    path.write_text(json.dumps(_sidecar_payload()))

    scalars = parse_stats_sidecar(path)

    assert scalars["Ip"] == pytest.approx(50.9e3)
    assert scalars["kappa"] == pytest.approx(1.6)
    assert "beta_tor" not in scalars          # absent stats keys stay absent
    assert scalars["o_point"] == [0.4, 0.0]
    assert scalars["shot"] == 39915

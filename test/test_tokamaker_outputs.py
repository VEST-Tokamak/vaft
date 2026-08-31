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


def test_collect_evolution_outputs_round_trip(tmp_path):
    from vaft.code.tokamaker import collect_tokamaker_evolution_outputs

    sample = data_path("efit/g039915.00319")
    shutil.copy(sample, tmp_path / "g039915.00319")
    payload = {
        "shot": 39915,
        "vacuum": False,
        "times": [0.319, 0.321],
        "probes": [],
        "steps": [
            {"index": 0, "time": 0.319, "converged": True, "error": "",
             "gfile": "g039915.00319", "stats": {"Ip": 51.0e3},
             "coil_currents_A": {"PF1": -600.0},
             "vessel_currents_A": {"W1": 120.0}, "probe_fields": {}},
            {"index": 1, "time": 0.321, "converged": False, "error": "diverged",
             "gfile": None, "stats": {}, "coil_currents_A": {"PF1": -580.0},
             "vessel_currents_A": {}, "probe_fields": {}},
        ],
    }
    (tmp_path / "tokamaker_evolution.json").write_text(json.dumps(payload))

    result = collect_tokamaker_evolution_outputs(tmp_path)

    assert result.returncode is None                 # collect-only
    assert result.times == (0.319, 0.321)
    assert [rec.converged for rec in result.steps] == [True, False]
    assert result.steps[0].vessel_currents_A == {"W1": 120.0}
    assert [path.name for path in result.gfiles] == ["g039915.00319"]
    assert result.scalars["n_failed"] == 1
    # the real g-file merges into a one-slice equilibrium IDS
    assert result.ods is not None
    assert float(result.ods["equilibrium.time_slice.0.time"]) == pytest.approx(0.319)


def test_collect_evolution_outputs_empty_dir(tmp_path):
    from vaft.code.tokamaker import collect_tokamaker_evolution_outputs

    result = collect_tokamaker_evolution_outputs(tmp_path)
    assert result.sidecar_file is None
    assert result.steps == ()
    assert result.ods is None
    assert not result.ok


def test_collect_stability_outputs_round_trip(tmp_path):
    from vaft.code.tokamaker import collect_tokamaker_stability_outputs

    payload = {
        "wall": {"tau_wall_s": [0.007, 0.004], "tau_wall_max_s": 0.007, "converged": True},
        "vertical": {"gamma_s": 812.0, "stable": False, "converged": True},
    }
    (tmp_path / "tokamaker_stability.json").write_text(json.dumps(payload))

    result = collect_tokamaker_stability_outputs(tmp_path)

    assert result.tau_wall_s == pytest.approx((0.007, 0.004))
    assert result.gamma_s == pytest.approx(812.0)
    assert result.scalars["stable"] is False
    assert result.scalars["tau_wall_max_s"] == pytest.approx(0.007)

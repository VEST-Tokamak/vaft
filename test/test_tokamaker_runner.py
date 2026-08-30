"""Unit tests for the in-process TokaMaker runner using a fake OpenFUSIONToolkit.

The fake harness lives in ``test/tokamaker_fakes.py`` (shared with the
evolution/stability tests). These tests pin the adapter's lifecycle contract:
call order, ``reset()`` in all paths, the ``OFT_env`` singleton handling,
error mapping, and the g-file/sidecar outputs.
"""

import json
import sys
from pathlib import Path

import pytest

from tokamaker_fakes import make_fake_oft, make_inputs

from vaft.code.tokamaker import TokaMakerConfig, run_tokamaker
from vaft.code.tokamaker._oft import get_oft_env, import_oft


def test_run_lifecycle_order_and_outputs(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    config = TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path, maxits=60)

    result = run_tokamaker(make_inputs(tmp_path), config)

    names = [entry[0] for entry in calls]
    assert names == [
        "init", "setup_mesh", "setup_regions", "setup", "set_coil_currents",
        "set_targets", "set_profiles", "init_psi", "solve", "save_eqdsk", "reset",
    ]
    setup = calls[names.index("setup")]
    assert setup[1:] == (config.order, pytest.approx(0.06))
    assert calls[names.index("set_targets")][1] == {"Ip": pytest.approx(51.0e3)}

    save_name, save_kwargs = calls[names.index("save_eqdsk")][1:]
    assert Path(save_name).name == "g039915.00325"
    assert save_kwargs["cocos"] == config.eqdsk_cocos
    assert save_kwargs["run_info"] == "# 39915 325ms"

    assert result.ok
    assert result.returncode == 0
    assert result.gfile is not None and result.gfile.name == "g039915.00325"
    assert result.scalars["converged"] is True
    assert result.scalars["q_95"] == pytest.approx(5.0)
    assert result.scalars["coil_currents_A"]["PF1"] == pytest.approx(-640.0)
    # the fake g-file is not parseable; that stays best-effort
    assert "_geqdsk_error" in result.scalars


def test_vsc_coil_wires_the_stability_pair(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    config = TokaMakerConfig(
        shot=39915, time=0.325, workdir=tmp_path, vsc_coil="PF9", vsc_weight=0.5
    )

    result = run_tokamaker(make_inputs(tmp_path), config)

    names = [entry[0] for entry in calls]
    assert result.ok
    assert calls[names.index("set_coil_vsc")][1] == {"PF9_U": 1.0, "PF9_L": -1.0}
    reg_call = calls[names.index("coil_reg_term")]
    assert reg_call[1] == {"#VSC": 1.0}
    assert reg_call[3] == pytest.approx(0.5)
    # VSC is wired before the coil currents/targets, mirroring the OFT examples
    assert names.index("set_coil_vsc") < names.index("set_coil_currents")
    assert "set_coil_reg" in names


def test_no_vsc_calls_without_vsc_coil(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    run_tokamaker(make_inputs(tmp_path), TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path))
    names = [entry[0] for entry in calls]
    assert "set_coil_vsc" not in names
    assert "set_coil_reg" not in names


def test_failed_solve_reports_error_and_still_resets(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch, solve_error="boom: no convergence")
    config = TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path)

    result = run_tokamaker(make_inputs(tmp_path), config)

    names = [entry[0] for entry in calls]
    assert "reset" in names and names[-1] == "reset"
    assert "save_eqdsk" not in names
    assert not result.ok
    assert result.returncode == 1
    assert "boom: no convergence" in result.error
    sidecar = json.loads((tmp_path / "tokamaker_result.json").read_text())
    assert sidecar["converged"] is False
    assert "boom" in sidecar["error"]


def test_two_consecutive_runs_share_the_singleton_env(tmp_path, monkeypatch):
    calls, fake_env_cls = make_fake_oft(monkeypatch)
    config = TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path)

    first = run_tokamaker(make_inputs(tmp_path), config)
    second = run_tokamaker(make_inputs(tmp_path), config)

    assert first.ok and second.ok
    inits = [entry for entry in calls if entry[0] == "init"]
    assert len(inits) == 2
    assert inits[0][1] is inits[1][1] is fake_env_cls.instance


def test_get_oft_env_reuses_existing_instance(monkeypatch):
    _, fake_env_cls = make_fake_oft(monkeypatch)
    env = get_oft_env(nthreads=2)
    assert env is fake_env_cls.instance
    assert get_oft_env(nthreads=8) is env  # second call reuses, never reconstructs


def test_import_oft_error_message_is_actionable(monkeypatch):
    # None entries make ``import OpenFUSIONToolkit`` fail even when the real
    # package is installed in this environment.
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", None)
    for name in list(sys.modules):
        if name.startswith("OpenFUSIONToolkit."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delenv("OFT_ROOTPATH", raising=False)

    with pytest.raises(ImportError) as excinfo:
        import_oft()

    message = str(excinfo.value)
    assert "pip install -e" in message
    assert "OFT_LIBRARY_DIR" in message
    assert "OFT_ROOTPATH" in message

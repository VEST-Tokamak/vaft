"""Unit tests for the wall-eigenmode and vertical-stability wrappers (OFT-free)."""

import json

import numpy as np
import pytest

from tokamaker_fakes import make_fake_oft, make_inputs

from vaft.code.tokamaker import (
    TokaMakerConfig,
    run_tokamaker_vertical_stability,
    run_tokamaker_wall_eigenmodes,
)


def _config(tmp_path, **overrides):
    kwargs = dict(shot=39915, time=0.325, workdir=tmp_path, include_vessel=True)
    kwargs.update(overrides)
    return TokaMakerConfig(**kwargs)


def test_wall_eigenmodes_need_no_equilibrium(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    result = run_tokamaker_wall_eigenmodes(make_inputs(tmp_path), _config(tmp_path, wall_neigs=4))

    names = [entry[0] for entry in calls]
    assert result.ok
    assert "solve" not in names and "init_psi" not in names and "set_targets" not in names
    assert names[-1] == "reset"
    # fake eig_wall: 1/tau = 100, 200, 300, 400 -> tau descending from 10 ms
    assert result.tau_wall_s == pytest.approx((0.01, 0.005, 1 / 300.0, 0.0025))
    assert result.scalars["tau_wall_max_s"] == pytest.approx(0.01)

    # modes are replottable without OFT: npz carries eigvecs + mesh arrays
    assert result.eig_file is not None and result.eig_file.is_file()
    saved = np.load(result.eig_file)
    assert saved["eig_vals"].shape == (4, 2)
    assert saved["eig_vecs"].shape[0] == 4
    assert saved["mesh_r"].shape[1] == 3

    payload = json.loads(result.stats_file.read_text())
    assert payload["wall"]["tau_wall_max_s"] == pytest.approx(0.01)


def test_wall_eigenmodes_guard_non_arpack_builds(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch, eig_wall_vals=np.zeros((4, 2)))
    result = run_tokamaker_wall_eigenmodes(make_inputs(tmp_path), _config(tmp_path))

    assert not result.ok
    assert "ARPACK" in result.error
    assert result.tau_wall_s == ()


def test_wall_eigenmodes_require_vessel(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    with pytest.raises(ValueError, match="include_vessel"):
        run_tokamaker_wall_eigenmodes(
            make_inputs(tmp_path), TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path)
        )


def test_vertical_stability_solves_then_eigensolves(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    config = _config(tmp_path, vsc_coil="PF9", td_neigs=4, td_omega=-2.0e4)
    result = run_tokamaker_vertical_stability(make_inputs(tmp_path), config)

    names = [entry[0] for entry in calls]
    assert result.ok
    # full forward sequence, VSC wired before currents, eig_td after the solve
    assert names.index("set_coil_vsc") < names.index("set_coil_currents")
    assert names.index("solve") < names.index("eig_td")
    assert names[-1] == "reset"
    eig_call = calls[names.index("eig_td")]
    assert eig_call[1] == pytest.approx(-2.0e4)
    assert eig_call[3] is False                      # include_bounds default

    # fake eig_td leads with -50 1/s -> gamma = +50, unstable
    assert result.gamma_s == pytest.approx(50.0)
    assert result.scalars["stable"] is False
    assert result.gfile is not None and result.gfile.name == "g039915.00325"
    payload = json.loads(result.stats_file.read_text())
    assert payload["vertical"]["gamma_s"] == pytest.approx(50.0)
    assert payload["vertical"]["vsc_coil"] == "PF9"


def test_vertical_stability_failed_solve_skips_eig(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch, solve_error="no convergence")
    result = run_tokamaker_vertical_stability(make_inputs(tmp_path), _config(tmp_path))

    names = [entry[0] for entry in calls]
    assert not result.ok
    assert "eig_td" not in names
    assert names[-1] == "reset"
    assert result.gamma_s is None
    assert "no convergence" in result.error


def test_vertical_stability_guards_non_arpack(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch, eig_td_vals=np.zeros((4, 2)))
    result = run_tokamaker_vertical_stability(make_inputs(tmp_path), _config(tmp_path))

    assert not result.ok
    assert "ARPACK" in result.error
    # the equilibrium itself succeeded; its g-file is still exported
    assert result.gfile is not None and result.gfile.is_file()


def test_stability_sidecar_sections_merge(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    config = _config(tmp_path)
    run_tokamaker_wall_eigenmodes(make_inputs(tmp_path), config)
    result = run_tokamaker_vertical_stability(make_inputs(tmp_path), config)

    payload = json.loads(result.stats_file.read_text())
    assert "wall" in payload and "vertical" in payload   # sections coexist


def test_wall_eigenmodes_ignore_unconverged_trailing_zeros(tmp_path, monkeypatch):
    vals = np.array([[100.0, 0.0], [200.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    make_fake_oft(monkeypatch, eig_wall_vals=vals)
    result = run_tokamaker_wall_eigenmodes(make_inputs(tmp_path), _config(tmp_path))

    assert result.ok
    assert result.tau_wall_s == pytest.approx((0.01, 0.005))   # no inf entries
    assert all(np.isfinite(result.tau_wall_s))


def test_vertical_stability_ignores_unconverged_leading_zeros(tmp_path, monkeypatch):
    # unconverged zeros sort ahead of the negative (unstable) eigenvalue
    vals = np.array([[0.0, 0.0], [0.0, 0.0], [-50.0, 0.0], [10.0, 0.0]])
    make_fake_oft(monkeypatch, eig_td_vals=vals)
    result = run_tokamaker_vertical_stability(make_inputs(tmp_path), _config(tmp_path))

    assert result.ok
    assert result.gamma_s == pytest.approx(50.0)

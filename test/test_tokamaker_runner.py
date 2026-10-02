"""Unit tests for the in-process TokaMaker runner using a fake OpenFUSIONToolkit.

The fake harness lives in ``test/tokamaker_fakes.py`` (shared with the
evolution/stability tests). These tests pin the adapter's lifecycle contract:
call order, ``reset()`` in all paths, the ``OFT_env`` singleton handling,
error mapping, and the g-file/sidecar outputs.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
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
    # RBBBS/ZBBBS must be the LCFS itself, not the 1 - lcfs_pad surface (#882, #1469)
    assert save_kwargs["truncate_eq"] is False
    assert save_kwargs["lcfs_pad"] == config.eqdsk_lcfs_pad

    assert result.ok
    assert result.returncode == 0
    assert result.gfile is not None and result.gfile.name == "g039915.00325"
    assert result.scalars["converged"] is True
    assert result.scalars["q_95"] == pytest.approx(5.0)
    assert result.scalars["coil_currents_A"]["PF1"] == pytest.approx(-640.0)
    assert result.scalars["diverted"] is False
    assert result.scalars["lim_point"] == pytest.approx([0.105, 0.0])
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
    sidecar = json.loads((tmp_path / "tokamaker_result.json").read_text(encoding="utf-8"))
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


def test_reused_workdir_never_yields_a_stale_gfile(tmp_path, monkeypatch):
    # first run at t=0.200 leaves g039915.00200 behind
    calls, _ = make_fake_oft(monkeypatch)
    config_a = TokaMakerConfig(shot=39915, time=0.200, workdir=tmp_path)
    first = run_tokamaker(make_inputs(tmp_path, time=0.200), config_a)
    assert first.gfile.name == "g039915.00200"

    # second run in the SAME workdir must carry ITS OWN g-file, not the older
    # (alphabetically first) one
    config_b = TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path)
    second = run_tokamaker(make_inputs(tmp_path, time=0.325), config_b)
    assert second.ok
    assert second.gfile.name == "g039915.00325"

    # and a FAILED run must not surface any earlier run's equilibrium
    make_fake_oft(monkeypatch, solve_error="boom")
    failed = run_tokamaker(make_inputs(tmp_path, time=0.400), config_b)
    assert not failed.ok
    assert failed.gfile is None
    assert failed.geqdsk == ()
    assert failed.ods is None


def test_limiter_search_excludes_the_vest_chambers_by_default(tmp_path, monkeypatch):
    # The chamber corners at |Z| = 1.185 m carry the most-interior wall flux but
    # are not connected to the core; with them as limiter candidates the
    # 39915 @ 325 ms "LCFS" floated 6 cm off the inboard limiter (#1469).
    make_fake_oft(monkeypatch)
    fake = sys.modules["OpenFUSIONToolkit.TokaMaker"].TokaMaker

    run_tokamaker(make_inputs(tmp_path), TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path))
    assert fake.settings_at_setup["lim_zmax"] == pytest.approx(0.6)

    run_tokamaker(
        make_inputs(tmp_path),
        TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path, lim_zmax=None),
    )
    assert fake.settings_at_setup["lim_zmax"] == pytest.approx(1.0e99)


VEST_LIMITER = [
    [0.105, 0.575], [0.1337, 0.7279], [0.1337, 1.185], [0.6, 1.185], [0.6, 0.6],
    [0.7, 0.6], [0.7, 0.585], [0.761, 0.585], [0.761, -0.585], [0.7, -0.585],
    [0.7, -0.6], [0.6, -0.6], [0.6, -1.185], [0.1337, -1.185], [0.1337, -0.7279],
    [0.105, -0.575], [0.105, 0.575],
]


def test_neck_faces_above_lim_zmax_stay_limiter_points():
    from vaft.code.tokamaker.runner import neck_limiter_points

    config = TokaMakerConfig()
    points = neck_limiter_points(VEST_LIMITER, config)

    z = np.abs(points[:, 1])
    assert np.all((z > config.lim_zmax) & (z <= config.neck_limiter_zmax))
    # the inboard slant (0.105, 0.575) -> (0.1337, 0.7279), both halves
    slant = points[(points[:, 0] > 0.105) & (points[:, 0] < 0.1337)]
    assert slant[:, 1].max() > 0.7 and slant[:, 1].min() < -0.7
    # nothing from the chamber interior far above the neck
    assert z.max() <= 0.73
    assert len(neck_limiter_points(VEST_LIMITER, TokaMakerConfig(neck_limiter_zmax=None))) == 0


def test_neck_points_reach_tokamaker_as_a_limiter_file(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    fake = sys.modules["OpenFUSIONToolkit.TokaMaker"].TokaMaker

    run_tokamaker(
        make_inputs(tmp_path, {"limiter": VEST_LIMITER}),
        TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path),
    )

    path = Path(fake.settings_at_setup["limiter_file"])
    lines = path.read_text().split("\n")
    assert path.parent == tmp_path
    assert int(lines[0]) == len([ln for ln in lines[1:] if ln.strip()]) > 0


def test_lcfs_leaving_the_wall_is_flagged(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    fake = sys.modules["OpenFUSIONToolkit.TokaMaker"].TokaMaker
    config = TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path)

    inside = run_tokamaker(make_inputs(tmp_path, {"limiter": VEST_LIMITER}), config)
    assert inside.scalars["lcfs_inside_wall"] is True

    # an LCFS pushed 2 cm through the center stack
    original = fake.trace_surf
    monkeypatch.setattr(fake, "trace_surf", lambda self, psi: original(self, psi) - [0.165, 0.0])
    crossed = run_tokamaker(make_inputs(tmp_path, {"limiter": VEST_LIMITER}), config)
    assert crossed.scalars["lcfs_inside_wall"] is False
    assert crossed.scalars["lcfs_wall_excursion_m"] == pytest.approx(0.02, abs=2e-3)


def test_failed_lcfs_trace_reports_the_wall_check_as_unknown(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    fake = sys.modules["OpenFUSIONToolkit.TokaMaker"].TokaMaker
    monkeypatch.setattr(fake, "trace_surf", lambda self, psi: None)

    result = run_tokamaker(
        make_inputs(tmp_path, {"limiter": VEST_LIMITER}),
        TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path),
    )

    assert "lcfs_inside_wall" in result.scalars
    assert result.scalars["lcfs_inside_wall"] is None


def test_long_workdir_falls_back_to_a_short_limiter_file(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    fake = sys.modules["OpenFUSIONToolkit.TokaMaker"].TokaMaker
    original_init = fake.__init__

    def init(self, env):
        original_init(self, env)

        def path2c(path):
            if len(path) > 200:   # OFT_PATH_SLEN
                raise ValueError("path too long")
            return path

        self._oft_env = types.SimpleNamespace(path2c=path2c)

    monkeypatch.setattr(fake, "__init__", init)
    deep = tmp_path / ("d" * 120) / ("e" * 120)
    deep.mkdir(parents=True)
    mesh = deep / "mesh.h5"
    mesh.write_bytes(b"")

    run_tokamaker(
        make_inputs(deep, {"limiter": VEST_LIMITER}, mesh_file=mesh),
        TokaMakerConfig(shot=39915, time=0.325, workdir=deep),
    )

    limiter_file = fake.settings_at_setup["limiter_file"]
    assert len(limiter_file) <= 200
    assert Path(limiter_file).read_text().split("\n")[0].isdigit()

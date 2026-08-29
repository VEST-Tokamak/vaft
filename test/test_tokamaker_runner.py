"""Unit tests for the in-process TokaMaker runner using a fake OpenFUSIONToolkit.

The real toolkit is a ctypes shim over a compiled library, so these tests
substitute a recording fake into ``sys.modules`` (``import_oft`` resolves
modules through ``importlib``, which honours the patched entries). This pins
the adapter's lifecycle contract: call order, ``reset()`` in all paths, the
``OFT_env`` singleton handling, error mapping, and the g-file/sidecar outputs.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vaft.code.tokamaker import TokaMakerConfig, run_tokamaker
from vaft.code.tokamaker._oft import get_oft_env, import_oft
from vaft.code.tokamaker.config import TokaMakerInputs


class FakeSettings:
    def __init__(self):
        self.pm = True
        self.maxits = 40


def _make_fake_oft(monkeypatch, solve_error=None):
    """Install a recording fake OpenFUSIONToolkit into sys.modules."""
    calls = []

    class FakeOFTEnv:
        def __new__(cls, *args, **kwargs):
            if hasattr(cls, "instance"):
                raise RuntimeError("Only one instance of `OFT_env` can be created per python kernel")
            cls.instance = super().__new__(cls)
            return cls.instance

        def __init__(self, nthreads=2):
            self.nthreads = nthreads

    class FakeTokaMaker:
        def __init__(self, env):
            calls.append(("init", env))
            self.settings = FakeSettings()
            self.o_point = np.array([0.4, 0.0])
            self.diverted = False

        def setup_mesh(self, pts, lc, reg):
            calls.append(("setup_mesh",))

        def setup_regions(self, cond_dict=None, coil_dict=None):
            calls.append(("setup_regions", coil_dict))

        def setup(self, order=2, F0=0.0):
            calls.append(("setup", order, F0))

        def set_coil_currents(self, currents):
            calls.append(("set_coil_currents", dict(currents)))

        def set_targets(self, **targets):
            calls.append(("set_targets", targets))

        def set_profiles(self, ffp_prof=None, pp_prof=None):
            calls.append(("set_profiles",))

        def init_psi(self, r0, z0, a, kappa, delta):
            calls.append(("init_psi", r0, z0, a, kappa, delta))

        def solve(self):
            calls.append(("solve",))
            if solve_error is not None:
                raise ValueError(solve_error)

        def get_stats(self):
            return {"Ip": 51.0e3, "q_95": 5.0, "kappa": 1.6}

        def get_coil_currents(self):
            return {"PF1": -640.0, "PF2": 320.0}, np.zeros(4)

        def save_eqdsk(self, filename, **kwargs):
            calls.append(("save_eqdsk", filename, kwargs))
            Path(filename).write_text("fake gEQDSK\n")

        def reset(self):
            calls.append(("reset",))

    root = types.ModuleType("OpenFUSIONToolkit")
    root.OFT_env = FakeOFTEnv
    tokamaker_mod = types.ModuleType("OpenFUSIONToolkit.TokaMaker")
    tokamaker_mod.TokaMaker = FakeTokaMaker
    meshing = types.ModuleType("OpenFUSIONToolkit.TokaMaker.meshing")
    meshing.load_gs_mesh = lambda path: ("pts", "lc", "reg", {"PF1": {}}, {"AIR": {}})
    util = types.ModuleType("OpenFUSIONToolkit.TokaMaker.util")
    util.create_power_flux_fun = lambda npts, alpha, gamma: {"type": "linterp", "alpha": alpha}

    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", root)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker", tokamaker_mod)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker.meshing", meshing)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker.util", util)
    return calls, FakeOFTEnv


def _make_inputs(tmp_path):
    mesh_file = tmp_path / "vest_gs_mesh_test.h5"
    mesh_file.write_bytes(b"")
    return TokaMakerInputs(
        workdir=tmp_path,
        geometry={"limiter": [[0.2, -0.4], [0.6, 0.0], [0.2, 0.4]], "coils": {}},
        mesh_file=mesh_file,
        mesh_exists=True,
        coil_currents={"PF1": -640.0, "PF2": 320.0},
        targets={"Ip": 51.0e3},
        f0=0.06,
        shot=39915,
        time=0.325,
    )


def test_run_lifecycle_order_and_outputs(tmp_path, monkeypatch):
    calls, _ = _make_fake_oft(monkeypatch)
    config = TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path, maxits=60)

    result = run_tokamaker(_make_inputs(tmp_path), config)

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


def test_failed_solve_reports_error_and_still_resets(tmp_path, monkeypatch):
    calls, _ = _make_fake_oft(monkeypatch, solve_error="boom: no convergence")
    config = TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path)

    result = run_tokamaker(_make_inputs(tmp_path), config)

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
    calls, fake_env_cls = _make_fake_oft(monkeypatch)
    config = TokaMakerConfig(shot=39915, time=0.325, workdir=tmp_path)

    first = run_tokamaker(_make_inputs(tmp_path), config)
    second = run_tokamaker(_make_inputs(tmp_path), config)

    assert first.ok and second.ok
    inits = [entry for entry in calls if entry[0] == "init"]
    assert len(inits) == 2
    assert inits[0][1] is inits[1][1] is fake_env_cls.instance


def test_get_oft_env_reuses_existing_instance(monkeypatch):
    _, fake_env_cls = _make_fake_oft(monkeypatch)
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

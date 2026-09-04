"""Shared fake OpenFUSIONToolkit harness for the tokamaker adapter tests.

The real toolkit is a ctypes shim over a compiled library, so tests substitute
a recording fake into ``sys.modules`` (``import_oft`` resolves modules through
``importlib``, which honours the patched entries). Every method appends a
tuple to the returned ``calls`` list; assertions are made on call names,
arguments, and ordering.
"""

from __future__ import annotations

import types
import sys
from pathlib import Path

import numpy as np

from vaft.code.tokamaker.config import TokaMakerInputs

FAKE_NP = 12  # fake mesh point count


class FakeSettings:
    def __init__(self):
        self.pm = True
        self.maxits = 40
        self.urf = 0.2
        self.nl_tol = 1e-6


class FakeFieldInterpolator:
    """Position-dependent fake so probe evaluations are checkable."""

    def __init__(self, field_type):
        self.field_type = field_type

    def eval(self, pt):
        r, z = float(pt[0]), float(pt[1])
        if self.field_type == "B":
            return np.array([0.01 * r, 0.0, 0.02 * z])
        return np.array([0.05 * r * z])


def make_fake_oft(
    monkeypatch,
    solve_error=None,
    solve_error_at=None,
    eig_wall_vals=None,
    eig_td_vals=None,
    conductor_entries=None,
):
    """Install a recording fake OpenFUSIONToolkit into sys.modules.

    Parameters
    ----------
    solve_error : str, optional
        Message for a ``ValueError`` raised by ``solve()``. With
        ``solve_error_at=k`` only the k-th solve (1-based) raises; otherwise
        every solve raises.
    eig_wall_vals / eig_td_vals : array-like, optional
        Override the ``(neigs, 2)`` eigenvalue returns (e.g. all-zeros to
        emulate a non-ARPACK build).
    conductor_entries : dict, optional
        Conductor entries for the fake ``load_gs_mesh`` cond_dict (default one
        region ``W1`` with reg_id 5).
    """
    calls = []
    conductor_entries = conductor_entries or {
        "W1": {"reg_id": 5, "cond_id": 1, "eta": 7.8e-7, "noncontinuous": False},
    }

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
            self.np = FAKE_NP
            self.r = np.zeros((FAKE_NP, 3))
            self.lc = np.zeros((6, 3), dtype=int)
            self.reg = np.ones(6, dtype=int)
            self._solve_count = 0

        # --- setup ---
        def setup_mesh(self, pts, lc, reg):
            calls.append(("setup_mesh",))

        def setup_regions(self, cond_dict=None, coil_dict=None):
            # emulate the real mutation: vacuum entries are removed
            for key in [k for k, v in (cond_dict or {}).items() if "eta" not in v]:
                del cond_dict[key]
            calls.append(("setup_regions", coil_dict))

        def setup(self, order=2, F0=0.0):
            calls.append(("setup", order, F0))

        def update_settings(self):
            calls.append(("update_settings", self.settings.nl_tol))

        # --- configuration ---
        def set_coil_currents(self, currents):
            calls.append(("set_coil_currents", dict(currents)))

        def set_targets(self, **targets):
            calls.append(("set_targets", targets))

        def set_profiles(self, ffp_prof=None, pp_prof=None):
            calls.append(("set_profiles",))

        def set_coil_vsc(self, coil_gains):
            calls.append(("set_coil_vsc", dict(coil_gains)))

        def coil_reg_term(self, coffs, target=0.0, weight=1.0):
            calls.append(("coil_reg_term", dict(coffs), target, weight))
            return (dict(coffs), target, weight)

        def set_coil_reg(self, reg_terms=None):
            calls.append(("set_coil_reg", reg_terms))

        def init_psi(self, r0, z0, a, kappa, delta):
            calls.append(("init_psi", r0, z0, a, kappa, delta))

        # --- state / solves ---
        def get_psi(self, normalized=True):
            calls.append(("get_psi", normalized))
            # stamped with the solve count so psi0 chaining is checkable
            return np.full(self.np, float(self._solve_count))

        def set_psi(self, psi, update_bounds=False):
            calls.append(("set_psi", np.asarray(psi).copy()))

        def set_psi_dt(self, psi0, dt):
            calls.append(("set_psi_dt", np.asarray(psi0).copy(), float(dt)))

        def solve(self):
            self._solve_count += 1
            calls.append(("solve", self._solve_count))
            if solve_error is not None and (
                solve_error_at is None or self._solve_count == solve_error_at
            ):
                raise ValueError(solve_error)

        def vac_solve(self, psi=None):
            self._solve_count += 1
            calls.append(("vac_solve", self._solve_count, None if psi is None else np.asarray(psi).copy()))
            return np.full(self.np, float(self._solve_count))

        # --- eigenvalue solves ---
        def eig_wall(self, neigs=4, pm=False):
            calls.append(("eig_wall", neigs))
            if eig_wall_vals is not None:
                vals = np.asarray(eig_wall_vals, dtype=float)
            else:
                vals = np.column_stack([(np.arange(neigs) + 1) * 100.0, np.zeros(neigs)])
            return vals, np.ones((len(vals), self.np))

        def eig_td(self, omega=-1.0e4, neigs=4, include_bounds=True, pm=False, damping_scale=-1.0):
            calls.append(("eig_td", omega, neigs, include_bounds))
            if eig_td_vals is not None:
                vals = np.asarray(eig_td_vals, dtype=float)
            else:
                growth = np.r_[-50.0, (np.arange(neigs - 1) + 1) * 10.0]
                vals = np.column_stack([growth, np.zeros(neigs)])
            return vals, np.ones((len(vals), self.np))

        # --- diagnostics ---
        def get_stats(self):
            return {"Ip": 51.0e3, "q_95": 5.0, "kappa": 1.6}

        def get_coil_currents(self):
            return {"PF1": -640.0, "PF2": 320.0}, np.zeros(4)

        def get_xpoints(self):
            calls.append(("get_xpoints",))
            return np.array([[0.45, -0.35]]), True

        def get_conductor_currents(self, psi, cell_centered=False):
            calls.append(("get_conductor_currents",))
            return np.ones(6, dtype=bool), np.full(self.np, 2.0)

        def area_integral(self, field, reg_mask=-1):
            calls.append(("area_integral", int(reg_mask)))
            return 100.0 * float(reg_mask)

        def get_field_eval(self, field_type):
            calls.append(("get_field_eval", field_type))
            return FakeFieldInterpolator(field_type)

        def save_eqdsk(self, filename, **kwargs):
            calls.append(("save_eqdsk", filename, kwargs))
            Path(filename).write_text("fake gEQDSK\n", encoding="utf-8")

        def reset(self):
            calls.append(("reset",))

    root = types.ModuleType("OpenFUSIONToolkit")
    root.OFT_env = FakeOFTEnv
    tokamaker_mod = types.ModuleType("OpenFUSIONToolkit.TokaMaker")
    tokamaker_mod.TokaMaker = FakeTokaMaker
    meshing = types.ModuleType("OpenFUSIONToolkit.TokaMaker.meshing")
    meshing.load_gs_mesh = lambda path: (
        "pts", "lc", "reg",
        {"PF1": {"reg_id": 3, "coil_id": 0, "nturns": 16.0, "coil_set": "PF1"}},
        {"AIR": {"reg_id": 2, "vac_id": 1}, **{k: dict(v) for k, v in conductor_entries.items()}},
    )
    util = types.ModuleType("OpenFUSIONToolkit.TokaMaker.util")
    util.create_power_flux_fun = lambda npts, alpha, gamma: {"type": "linterp", "alpha": alpha}

    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", root)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker", tokamaker_mod)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker.meshing", meshing)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker.util", util)
    return calls, FakeOFTEnv


def make_inputs(tmp_path, geometry_extra=None, **overrides):
    """A minimal prepared TokaMakerInputs bundle with an existing mesh file."""
    mesh_file = tmp_path / "vest_gs_mesh_test.h5"
    mesh_file.write_bytes(b"")
    geometry = {"limiter": [[0.2, -0.4], [0.6, 0.0], [0.2, 0.4]], "coils": {}}
    geometry.update(geometry_extra or {})
    kwargs = dict(
        workdir=tmp_path,
        geometry=geometry,
        mesh_file=mesh_file,
        mesh_exists=True,
        coil_currents={"PF1": -640.0, "PF2": 320.0},
        targets={"Ip": 51.0e3},
        f0=0.06,
        shot=39915,
        time=0.325,
    )
    kwargs.update(overrides)
    return TokaMakerInputs(**kwargs)

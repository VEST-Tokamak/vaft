"""Small OFT-free checks of the native fixed-boundary PF inverse path."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS
from types import SimpleNamespace

from vaft.code.tokamaker import fit_vfixed_samples
from vaft.process.electromagnetics import compute_point_response_matrices
from vaft.process.equilibrium import solovev_example


def _machine() -> ODS:
    machine = ODS(consistency_check=False)
    for index, (r, z) in enumerate(((0.8, 0.5), (0.8, -0.5))):
        base = f"pf_active.coil.{index}"
        machine[f"{base}.name"] = f"PF{index + 1}"
        machine[f"{base}.element.0.geometry.rectangle.r"] = r
        machine[f"{base}.element.0.geometry.rectangle.z"] = z
        machine[f"{base}.element.0.turns_with_sign"] = 10.0
    return machine


def test_native_vfixed_flux_conversion_gauge_and_bounds():
    points = np.array([[.3, -.2], [.4, -.1], [.5, .1], [.4, .3], [.3, .2]])
    response = compute_point_response_matrices(
        points[:, 0], points[:, 1], np.array([.8, .8]), np.array([.5, -.5]),
        turns=np.array([10., 10.]), groups=np.array([0, 1]), n_groups=2,
        components=("psi",),
    )[0]
    truth = np.array([1234., -876.])
    # Independent of the direct path's plasma-field target construction:
    # get_vfixed is the vacuum flux, in OFT's per-radian orientation.
    vfixed = -(response @ truth) / (2 * np.pi)
    bounds = {"PF1": (-2000., 2000.), "PF2": (-2000., 2000.)}
    fit = fit_vfixed_samples(points, vfixed, _machine(), flux_scale_Wb=.01,
                              current_bounds=bounds)
    np.testing.assert_allclose(list(fit.currents_A.values()), truth, atol=1e-6)
    assert fit.accepted and fit.status == "accepted"
    assert fit.bounds_complete and fit.rank == 2
    assert fit.rms_relative_flux < 1e-12
    shifted = fit_vfixed_samples(points, vfixed + .2, _machine(),
                                  flux_scale_Wb=.01, current_bounds=bounds)
    np.testing.assert_allclose(list(shifted.currents_A.values()), truth, atol=1e-6)
    np.testing.assert_allclose(shifted.relative_flux_residual_Wb,
                               fit.relative_flux_residual_Wb, atol=1e-13)
    clipped = fit_vfixed_samples(points, vfixed, _machine(), flux_scale_Wb=.01,
                                  current_bounds={"PF1": (-100., 100.)})
    assert not clipped.accepted and clipped.currents_A["PF1"] <= 100.000001
    assert "PF1" in clipped.active_bounds


def test_vfixed_input_and_rank_diagnostics():
    machine = _machine()
    points = np.array([[.3, -.1], [.4, .0], [.3, .1]])
    with pytest.raises(ValueError, match="points"):
        fit_vfixed_samples(points, np.ones(2), machine, flux_scale_Wb=.01)
    with pytest.raises(ValueError, match="scale"):
        fit_vfixed_samples(points, np.ones(3), machine, flux_scale_Wb=0)
    with pytest.raises(ValueError, match="unknown current bounds"):
        fit_vfixed_samples(points, np.ones(3), machine, flux_scale_Wb=.01,
                           current_bounds={"missing": (-1, 1)})


def test_oft_green_orientation_if_installed():
    util = pytest.importorskip("OpenFUSIONToolkit.TokaMaker.util")
    points = np.array([[.4, -.1], [.4, .0], [.4, .1]])
    source = np.array([.8, .4])
    oft_psi = util.eval_green(points, source)
    vaft_psi = compute_point_response_matrices(
        points[:, 0], points[:, 1], source[:1], source[1:], components=("psi",),
    )[0][:, 0]
    # OFT's elliptic-integral approximation differs at about 1e-9 here.
    np.testing.assert_allclose(-2 * np.pi * oft_psi, vaft_psi, rtol=2e-8)


@pytest.mark.parametrize("fail", [False, True])
def test_fixed_solver_lifecycle_is_independent_of_direct_plasma_integral(monkeypatch, tmp_path, fail):
    from vaft.code.tokamaker import bridge
    from vaft.process import _equilibrium_coil_fit

    def forbidden(*args, **kwargs):
        raise AssertionError("direct plasma integral must not enter the OFT route")

    monkeypatch.setattr(_equilibrium_coil_fit, "_plasma_filaments", forbidden)
    calls = []
    points = np.array([[.3, -.2], [.4, -.1], [.5, .1], [.4, .3], [.3, .2]])
    samples = np.linspace(0., .001, len(points))

    class Mesh:
        def define_region(self, *args):
            pass

        def add_polygon(self, *args):
            pass

        def build_mesh(self):
            return points, np.array([[0, 1, 2]]), np.array([1])

    class Solver:
        settings = SimpleNamespace()

        def __init__(self, env):
            pass

        def setup_mesh(self, *args):
            calls.append("mesh")

        def setup(self, **kwargs):
            assert self.settings.free_boundary is False
            calls.append("fixed_setup")

        def set_targets(self, **kwargs):
            assert kwargs["Ip"] == pytest.approx(1e5)

        def set_profiles(self, **kwargs):
            calls.append("profiles")

        def init_psi(self, *args):
            pass

        def solve(self):
            calls.append("solve")
            if fail:
                raise RuntimeError("fixed solve failed")

        def get_vfixed(self):
            calls.append("get_vfixed")
            return points, samples

        def get_stats(self):
            return {"Ip": 1e5}

        def reset(self):
            calls.append("reset")
            # OFT owns these arrays; they must be copied before reset.
            samples[:] = np.nan

    oft = SimpleNamespace(TokaMaker=Solver, meshing=SimpleNamespace(gs_Domain=Mesh),
                          util=SimpleNamespace(create_power_flux_fun=lambda *args: {}))
    monkeypatch.setattr(bridge, "import_oft", lambda: oft)
    monkeypatch.setattr(bridge, "get_oft_env", lambda *args: object())
    workdir = tmp_path / "fixed"
    if fail:
        with pytest.raises(RuntimeError, match="fixed solve failed"):
            bridge.fit_free_boundary_coils_vfixed(solovev_example(resolution=33), _machine(), workdir)
        assert calls[-1] == "reset" and "get_vfixed" not in calls
    else:
        result = bridge.fit_free_boundary_coils_vfixed(solovev_example(resolution=33), _machine(), workdir)
        assert np.isfinite(result.required_vacuum_psi_Wb).all()
        assert result.fixed_stats == {"Ip": 1e5}
        assert result.fixed_profile_mode == "power_law"
        assert calls == ["mesh", "fixed_setup", "profiles", "solve", "get_vfixed", "reset"]
        assert (workdir / "vfixed_samples.npz").is_file()
        with pytest.raises(FileExistsError):
            bridge.fit_free_boundary_coils_vfixed(solovev_example(resolution=33), _machine(), workdir)

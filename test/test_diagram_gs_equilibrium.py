"""The Grad-Shafranov problem (#1052): the source formulas, and a toy flux whose topology is found, not drawn."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _gs_equilibrium as gs
from vaft.diagram._scene import Label
from vaft.formula.constants import MU0
from vaft.formula.equilibrium import grad_shafranov_source, toroidal_current_density_from_p_prime_ff_prime
from vaft.formula.green import green_psi_exact
from vaft.process.equilibrium import grad_shafranov_operator


def test_the_plasma_source_is_ampere_of_the_force_balance_current():
    R = np.array([0.5, 1.0, 2.0])
    p1, ff1 = 3e4, -0.2
    J = toroidal_current_density_from_p_prime_ff_prime(R, p1, ff1)
    np.testing.assert_allclose(J, R * p1 + ff1 / (MU0 * R))
    np.testing.assert_allclose(grad_shafranov_source(R, J), -MU0 * R**2 * p1 - ff1)
    for fn in (lambda: grad_shafranov_source(0.0, 1.0), lambda: toroidal_current_density_from_p_prime_ff_prime(-1, 1, 1)):
        with pytest.raises(ValueError):
            fn()


def test_a_solovev_flux_satisfies_the_grad_shafranov_equation():
    # psi = c1 R^4/8 + c2 Z^2 has Delta* psi = c1 R^2 + 2 c2: p' = -c1/mu0, FF' = -2 c2
    c1, c2 = -0.7, -0.3
    r, z = np.linspace(0.6, 1.4, 81), np.linspace(-0.5, 0.5, 81)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    psi = c1 * RR**4 / 8 + c2 * ZZ**2
    lhs = grad_shafranov_operator(psi, r, z)[2:-2, 2:-2]
    J = toroidal_current_density_from_p_prime_ff_prime(RR, -c1 / MU0, -2 * c2)
    np.testing.assert_allclose(lhs, grad_shafranov_source(RR, J)[2:-2, 2:-2], rtol=1e-4)  # 2nd-order FD


def test_a_ring_current_flux_is_homogeneous_in_vacuum():
    r, z = np.linspace(0.3, 0.5, 81), np.linspace(-0.1, 0.1, 81)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    psi = green_psi_exact(RR, ZZ, 1.0, 0.3)  # the ring is outside this box
    lap = grad_shafranov_operator(psi, r, z)[3:-3, 3:-3]
    scale = np.abs(np.gradient(psi, r, axis=0)).max() / 0.2
    assert np.abs(lap).max() < 1e-3 * scale


@pytest.mark.parametrize("configuration, bound", [("limited", "limiter"), ("diverted", "x_point")])
def test_the_topology_is_found_from_the_total_flux(configuration, bound):
    from scipy.interpolate import RectBivariateSpline

    m = gs.flux_model(configuration)
    np.testing.assert_allclose(m["psi"], m["psi_plasma"] + m["psi_coil"])
    assert m["limited_by"] == bound
    spline = RectBivariateSpline(gs._GRID_R, gs._GRID_Z, m["psi"])
    # the axis is a flux maximum
    ax = m["axis"]
    assert spline(*ax, dx=1)[0, 0] == pytest.approx(0.0, abs=1e-4) and spline(*ax, dy=1)[0, 0] == pytest.approx(0.0, abs=1e-4)
    assert spline(*ax, dx=2)[0, 0] < 0 and spline(*ax, dy=2)[0, 0] < 0
    boundary = gs.lcfs(m)
    on = spline(boundary[:, 0], boundary[:, 1], grid=False)
    assert np.ptp(on) < 2e-3 * (m["psi_axis"] - m["psi_boundary"])
    if configuration == "diverted":
        xp = m["x_point"]
        g = math.hypot(spline(*xp, dx=1)[0, 0], spline(*xp, dy=1)[0, 0])
        assert g < 1e-5
        assert spline(*xp, dx=2)[0, 0] * spline(*xp, dy=2)[0, 0] < 0  # a saddle
        assert np.hypot(*(boundary - xp).T).min() < 0.01  # the LCFS runs through it
        assert m["psi_x"] > m["psi_limiter"]
    else:
        assert m["x_point"] is None
        assert np.hypot(*(boundary - gs._LIMITER).T).min() < 0.01  # the LCFS touches the limiter tip


def test_only_the_total_has_the_topology():
    from scipy.interpolate import RectBivariateSpline

    m = gs.flux_model("diverted")
    coil = RectBivariateSpline(gs._GRID_R, gs._GRID_Z, m["psi_coil"])
    ax = m["axis"]
    # the coils alone have no extremum at the magnetic axis
    assert math.hypot(coil(*ax, dx=1)[0, 0], coil(*ax, dy=1)[0, 0]) > 1e-3
    plasma = RectBivariateSpline(gs._GRID_R, gs._GRID_Z, m["psi_plasma"])
    xp = m["x_point"]
    # the plasma alone has no null at the X-point
    assert math.hypot(plasma(*xp, dx=1)[0, 0], plasma(*xp, dy=1)[0, 0]) > 1e-3


def test_the_taxonomy_keeps_two_axes():
    cells = vaft.diagram.equilibrium_problem_taxonomy().model["cells"]
    assert set(cells) == {(r, c) for r in ("forward", "inverse") for c in ("fixed", "free")}
    assert "EFIT" in cells[("inverse", "free")] and "TokaMaker" in cells[("forward", "free")]
    assert "CHEASE" in cells[("forward", "fixed")]


def test_an_unknown_configuration_is_refused():
    with pytest.raises(ValueError):
        gs.flux_model("snowflake")


@pytest.mark.parametrize("name", ["grad_shafranov_domain_decomposition", "fixed_vs_free_boundary_equilibrium",
                                  "limiter_and_diverted_topologies", "equilibrium_problem_taxonomy",
                                  "poloidal_flux_source_decomposition"])
def test_every_gs_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    n_labels = lambda d: sum(isinstance(i, Label) for i in d.scene.items)  # noqa: E731
    assert fn().scene.role("note")
    assert not fn(labels=False).scene.role("note")
    assert n_labels(fn(labels=False)) <= n_labels(fn())
    with pytest.raises(ValueError):
        fn(labels="yes")

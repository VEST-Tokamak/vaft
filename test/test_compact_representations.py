"""Compact representations of existing equilibria (#1166).

``fit_solovev`` is the linear inverse of the Solov'ev family: an exact
Solov'ev input must come back exactly, and a reconstruction that is not a
Solov'ev equilibrium must come back as ``poor_fidelity``, not as a plausible
fit. ``fit_mxh_chebyshev`` follows Xie and Li (2026): an MXH shape per
surface, shifted-Chebyshev radial profiles, and a fidelity that must improve
with the number of parameters.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vaft.process.equilibrium import (
    as_equilibrium,
    evaluate_mxh_chebyshev,
    evaluate_solovev,
    fit_mxh_chebyshev,
    fit_solovev,
    solovev_example,
)


@pytest.fixture(scope="module")
def sample():
    from vaft.data.resources import sample_geqdsk

    return as_equilibrium(sample_geqdsk())


# --- fit_solovev -----------------------------------------------------------------


@pytest.mark.parametrize("topology", ["limited", "double_null", "single_null"])
def test_exact_solovev_round_trips(topology):
    eq = solovev_example(topology)
    fit = fit_solovev(eq)
    assert fit.status == "accepted" and fit.reason is None
    assert fit.metrics["psi_rms_error"] < 1e-9 and fit.metrics["topology_match"]
    # The recovered sources: the export stores p' per weber; the fit is per radian.
    assert fit.model.pprime == pytest.approx(2*np.pi*eq.pprime[0], rel=1e-6)
    assert fit.model.ffprime == pytest.approx(2*np.pi*eq.ffprime[0], rel=1e-6)


@pytest.mark.parametrize("convention", [1, 2, 3, 13, 17])
def test_fit_metrics_do_not_depend_on_the_convention(convention):
    base = fit_solovev(solovev_example("single_null"))
    other = fit_solovev(solovev_example("single_null", convention=convention))
    assert other.status == "accepted"
    assert abs(other.model.pprime) == pytest.approx(abs(base.model.pprime), rel=1e-6)
    for name, value in base.metrics.items():
        if isinstance(value, float):
            assert other.metrics[name] == pytest.approx(value, abs=1e-9), name
        else:
            assert other.metrics[name] == value, name


def test_a_reconstruction_is_reported_as_poor_not_hidden(sample):
    fit = fit_solovev(sample)
    assert fit.model is not None and fit.status == "poor_fidelity" and "tolerance" in fit.reason
    assert fit.metrics["psi_rms_error"] > 0.005 and fit.metrics["grad_psi_rms_error"] > 0.05
    assert fit_solovev(sample, tolerance=0.2).status == "accepted"      # same fit, looser tolerance


def test_mxh_recovers_an_exact_tilted_asymmetric_surface():
    """A surface that is exactly MXH, tilted and up-down asymmetric, must come back exactly."""
    from vaft.data.equilibrium import Contour
    from vaft.process._equilibrium_compact import _mxh_surface

    theta = np.linspace(0, 2*np.pi, 4000, endpoint=False) + 0.3
    c0, c, s_ = 0.1, (0.15, 0.05), (0.4, -0.1)
    bar = theta + c0 + sum(c[m]*np.cos((m+1)*theta) + s_[m]*np.sin((m+1)*theta) for m in range(2))
    p = _mxh_surface(Contour(1.0 + 0.3*np.cos(bar), 0.1 + 1.6*0.3*np.sin(theta), True), 2)
    assert p["c0"] == pytest.approx(c0, abs=2e-3)
    np.testing.assert_allclose(p["c"], c, atol=2e-3)
    np.testing.assert_allclose(p["s"], s_, atol=2e-3)


def test_fitted_model_is_an_exact_grad_shafranov_solution(sample):
    model = fit_solovev(sample).model
    r, z = np.meshgrid(np.linspace(0.3, 0.6, 5), np.linspace(-0.3, 0.3, 5), indexing="ij")
    v = evaluate_solovev(model, r, z)
    # d/dR psi_R - psi_R/R + psi_ZZ, by central differences on the analytic field.
    h = 1e-5
    dr = (evaluate_solovev(model, r + h, z)["dpsi_dr"] - evaluate_solovev(model, r - h, z)["dpsi_dr"])/(2*h)
    dz = (evaluate_solovev(model, r, z + h)["dpsi_dz"] - evaluate_solovev(model, r, z - h)["dpsi_dz"])/(2*h)
    np.testing.assert_allclose(dr - v["dpsi_dr"]/r + dz, v["grad_shafranov_source"], rtol=1e-5)


def test_fit_inputs_are_checked():
    eq = solovev_example("limited")
    with pytest.raises(ValueError, match="basis"):
        fit_solovev(eq, basis="spline")
    from vaft.data.equilibrium import EquilibriumConvention

    with pytest.raises(ValueError, match="flux unit"):
        fit_solovev(dataclasses.replace(eq, convention=EquilibriumConvention()))


# --- MXH-Chebyshev ---------------------------------------------------------------


def test_mxh_fidelity_improves_with_parameters(sample):
    errors = [fit_mxh_chebyshev(sample, harmonics=m, radial_order=l).metrics["psi_n_rms_error"]
              for m, l in ((1, 2), (2, 4), (3, 8))]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 3e-3


def test_mxh_parameter_count_and_edge_anchoring(sample):
    rep = fit_mxh_chebyshev(sample, harmonics=2, radial_order=4)
    assert rep.parameter_count == (5 + 2*2)*5
    assert set(rep.profiles) == {"h", "v", "kappa", "a", "c0", "c1", "c2", "s1", "s2"}
    # At rho = 1 every profile is its edge value, and h = v = 0 there by construction.
    assert rep.profiles["h"][0] == 0.0 and rep.profiles["v"][0] == 0.0
    r_edge, z_edge = evaluate_mxh_chebyshev(rep, np.ones(4), np.array([0.0, np.pi/2, np.pi, 3*np.pi/2]))
    assert r_edge[0] == pytest.approx(sample.lcfs.r.max(), abs=0.01)
    assert r_edge[2] == pytest.approx(sample.lcfs.r.min(), abs=0.01)


def test_mxh_of_an_ellipse_needs_no_distortion():
    eq = solovev_example("limited", aspect_ratio=4.0, elongation=1.5, triangularity=0.0)
    rep = fit_mxh_chebyshev(eq, harmonics=2, radial_order=3)
    assert rep.status == "accepted"
    for name in ("c0", "c1", "c2", "s2"):
        edge, coefficients = rep.profiles[name]
        assert abs(edge) < 0.02 and np.max(np.abs(coefficients)) < 0.02, name


def test_mxh_triangularity_enters_through_s1():
    """delta ~ sin(s1), as Miller's theta + arcsin(delta) sin(theta) gives; Xie & Li print it inverted."""
    rep = fit_mxh_chebyshev(solovev_example("limited", triangularity=0.7), harmonics=2, radial_order=3)
    assert rep.profiles["s1"][0] == pytest.approx(np.arcsin(0.7), abs=0.02)   # not 0.7, not sin(0.7)


def test_up_down_asymmetry_uses_the_cosine_terms():
    sym = fit_mxh_chebyshev(solovev_example("double_null"), harmonics=2, radial_order=3)
    asym = fit_mxh_chebyshev(solovev_example("single_null"), harmonics=2, radial_order=3)
    # The double null is symmetric to the separatrix tracing (~1e-3); the single null is not.
    assert abs(sym.profiles["c1"][0]) < 1e-2 and abs(asym.profiles["c1"][0]) > 0.1
    assert abs(asym.profiles["v"][1][0]) > 10*abs(sym.profiles["v"][1][0])


def test_diverted_corner_costs_boundary_fidelity():
    limited = fit_mxh_chebyshev(solovev_example("limited"), harmonics=2, radial_order=4)
    diverted = fit_mxh_chebyshev(solovev_example("double_null"), harmonics=2, radial_order=4)
    assert diverted.metrics["boundary_max_error"] > 3*limited.metrics["boundary_max_error"]


def test_mxh_inputs_are_checked(sample):
    with pytest.raises(ValueError, match="harmonics"):
        fit_mxh_chebyshev(sample, harmonics=0)
    with pytest.raises(ValueError, match="surfaces"):
        fit_mxh_chebyshev(sample, radial_order=6, surfaces=5)


def test_fit_solovev_takes_the_boundary_f_and_pressure_at_psi_boundary_not_at_the_last_sample(sample):
    """A boundary-first profile storage (legal for an ODS) used to hand the
    model the axis F and pressure while every psi metric stayed identical
    (cold review 0.8.0 equilibrium-representation F2)."""
    import warnings

    from vaft.data.equilibrium import EquilibriumData

    reversed_storage = EquilibriumData(**{
        **sample.__dict__,
        **{name: getattr(sample, name)[::-1]
           for name in ("psi_1d", "q", "f", "pressure", "pprime", "ffprime")},
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forward, backward = fit_solovev(sample), fit_solovev(reversed_storage)
    edge = int(np.argmin(np.abs(sample.psi_1d - sample.psi_boundary)))
    assert forward.model.f_boundary == float(sample.f[edge])
    assert forward.model.pressure_boundary == float(sample.pressure[edge])
    assert backward.model.f_boundary == forward.model.f_boundary
    assert backward.model.pressure_boundary == forward.model.pressure_boundary
    assert backward.model.f_sign == forward.model.f_sign
    assert backward.metrics["psi_rms_error"] == forward.metrics["psi_rms_error"]

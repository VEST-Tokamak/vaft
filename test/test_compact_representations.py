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


def test_fit_is_convention_independent():
    base = fit_solovev(solovev_example("limited"))
    other = fit_solovev(solovev_example("limited", convention=1))       # per-radian flux, other orientation
    assert other.status == "accepted"
    assert abs(other.model.pprime) == pytest.approx(abs(base.model.pprime), rel=1e-6)


def test_a_reconstruction_is_reported_as_poor_not_hidden(sample):
    fit = fit_solovev(sample)
    assert fit.model is not None and fit.status == "poor_fidelity" and "tolerance" in fit.reason
    assert fit.metrics["psi_rms_error"] > 0.005 and fit.metrics["bp_rms_error"] > 0.05
    assert fit_solovev(sample, tolerance=0.2).status == "accepted"      # same fit, looser tolerance


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
    rep = fit_mxh_chebyshev(solovev_example("limited", triangularity=0.4), harmonics=2, radial_order=3)
    assert rep.profiles["s1"][0] == pytest.approx(np.arcsin(0.4), abs=0.08)   # delta ~ sin(s1) (Xie & Li, Sec. 2.2)


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

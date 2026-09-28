"""Smooth-boundary shape benchmarks: Miller rejection is not representation failure (#942).

Seven deterministic contours, from a circle to a doublet, each fitted by the
semantic Miller model and by the general arc-length Fourier series, and
measured by the model-independent observables.  The point is the distinction
the issue asks for: "this boundary is not Miller-like" is a different
statement from "this boundary cannot be represented".
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.data.equilibrium import Contour, MillerSurface
from vaft.process.equilibrium import (
    contour_shape_parameters,
    contour_shaping_observables,
    evaluate_fourier_surface,
    evaluate_miller,
    fit_fourier_surface,
    fit_miller_surface,
)

THETA = np.linspace(0.0, 2.0*np.pi, 181)[:-1]
R0, A = 1.0, 0.3


def _miller(kappa=1.0, delta=0.0, zeta=0.0, indentation=0.0):
    return Contour(*evaluate_miller(MillerSurface(A, R0, 0.0, kappa, delta, zeta=zeta, indentation=indentation), THETA))


def _racetrack(p=6.0, kappa=1.6):
    """A superellipse ``|x|**p + |z/kappa|**p = 1``: flat sides, rounded corners."""
    c, s = np.cos(THETA), np.sin(THETA)
    return Contour(R0 + A*np.sign(c)*np.abs(c)**(2/p), kappa*A*np.sign(s)*np.abs(s)**(2/p))


def _doublet(waist=0.6, kappa=2.0):
    """Two lobes joined by a waist at the midplane, 40 % of the lobes' width."""
    return Contour(R0 + A*np.cos(THETA)*(1 - waist + waist*np.sin(2*THETA)**2), kappa*A*np.sin(THETA))


def _asymmetric_d(mean=0.3, half=0.25, kappa=1.7):
    """A D whose triangularity runs smoothly from 0.05 at the bottom to 0.55 at the top."""
    delta = mean + half*np.sin(THETA)
    return Contour(R0 + A*np.cos(THETA + np.arcsin(delta)*np.sin(THETA)), kappa*A*np.sin(THETA))


#: name -> (contour, Miller accepts, Miller + indentation accepts, Fourier modes, concave, asymmetric)
BENCHMARKS = {
    "circle": (_miller(), True, True, 8, False, False),
    "ellipse": (_miller(1.8), True, True, 8, False, False),
    "D": (_miller(1.7, 0.4), True, True, 8, False, False),
    "bean": (_miller(1.6, 0.4, indentation=0.35), True, True, 8, True, False),
    "racetrack": (_racetrack(), False, True, 8, False, False),
    "doublet": (_doublet(), False, False, 12, True, False),
    "asymmetric_d": (_asymmetric_d(), False, False, 8, False, True),
}


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_every_benchmark_is_representable_whether_or_not_miller_like(name):
    contour, miller_ok, indented_ok, modes, _, _ = BENCHMARKS[name]
    miller = fit_miller_surface(contour, squareness=True)
    indented = fit_miller_surface(contour, squareness=True, indentation=True)
    fourier = fit_fourier_surface(contour, modes=modes)
    assert miller.accepted is miller_ok, (name, miller.normalized_rms_error)
    assert indented.accepted is indented_ok, (name, indented.normalized_rms_error)
    # The general representation holds every one of them.
    assert fourier.accepted, (name, fourier.normalized_rms_error)
    assert fourier.normalized_rms_error < 0.02


def test_rejection_by_miller_is_a_statement_about_miller():
    contour = BENCHMARKS["asymmetric_d"][0]
    miller = fit_miller_surface(contour, squareness=True, indentation=True)
    fourier = fit_fourier_surface(contour, modes=8)
    assert not miller.accepted and fourier.accepted
    assert miller.normalized_rms_error > 10*fourier.normalized_rms_error


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_observables_measure_the_geometry_not_a_model(name):
    contour, _, _, _, concave, asymmetric = BENCHMARKS[name]
    observed = contour_shaping_observables(contour.r, contour.z)
    assert observed["inboard_concave"] is concave, (name, observed)
    assert (observed["up_down_asymmetry"] > 0.05) is asymmetric, (name, observed)
    assert observed["normalized_indentation_depth"] >= 0.0


def test_indentation_depth_follows_the_miller_coefficient():
    depths = [contour_shaping_observables(*(lambda c: (c.r, c.z))(_miller(1.6, 0.4, indentation=b)))
              ["normalized_indentation_depth"] for b in (0.0, 0.2, 0.35, 0.5)]
    assert depths[0] < 1e-3
    assert all(later > earlier for earlier, later in zip(depths[1:], depths[2:]))


def test_asymmetry_matches_the_two_triangularities():
    contour = BENCHMARKS["asymmetric_d"][0]
    shape = contour_shape_parameters(contour.r, contour.z)
    assert shape["triangularity_upper"] > shape["triangularity_lower"] + 0.3
    mirrored = contour_shaping_observables(contour.r, -contour.z)
    assert mirrored["up_down_asymmetry"] == pytest.approx(
        contour_shaping_observables(contour.r, contour.z)["up_down_asymmetry"], rel=1e-6)


def test_every_model_evaluates_back_to_a_contour_that_measures_the_same():
    # Contour is the interchange: fit, evaluate, measure again.  (Not the
    # racetrack: its flat top makes the height extremum, and so triangularity,
    # ill-defined, and any ripple moves it.)
    contour = BENCHMARKS["asymmetric_d"][0]
    # Its upper tip is sharp (delta_u = 0.55): eight harmonics round it to
    # 0.51 while holding the shape to 0.4 % rms, so the round trip keeps 24.
    fourier = fit_fourier_surface(contour, modes=24)
    back = Contour(*evaluate_fourier_surface(fourier.surface, THETA))
    before = contour_shape_parameters(contour.r, contour.z)
    after = contour_shape_parameters(back.r, back.z)
    for key in ("elongation", "triangularity_upper", "triangularity_lower", "area"):
        assert after[key] == pytest.approx(before[key], rel=0.02, abs=5e-3)


def test_observables_refuse_a_degenerate_contour():
    with pytest.raises(ValueError, match="four points"):
        contour_shaping_observables([1.0, 1.1, 1.0], [0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="degenerate"):
        contour_shaping_observables([1.0]*5, [0.0, 0.1, 0.2, 0.3, 0.4])

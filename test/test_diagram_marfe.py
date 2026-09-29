"""MARFE (#1209): Drake's condensation condition from vaft.formula.sol, a prescribed band on the equilibrium."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _marfe as mf
from vaft.diagram._scene import Label
from vaft.formula.constants import QE
from vaft.formula.sol import radiative_condensation_growth_rate, radiative_thermal_instability_growth_rate


def test_drake_growth_rates():
    n, T, L = 2e19, 5.0, 3e5
    # an independent SI evaluation of Drake's Eq. (2): T in joules, kappa per joule, (5/2) n dT/dt in J
    T_J, dLdT, k, kappa = T * QE, -1e4, 0.8, 50.0  # dL/dT per eV, kappa in W/(m eV)
    gamma_si = 2.0 / (5.0 * n) * (2 * L / T_J - dLdT / QE - k**2 * kappa / QE)
    assert radiative_condensation_growth_rate(n, T, L, dLdT, k, kappa) == pytest.approx(gamma_si, rel=1e-12)
    # a flat radiation curve and weak conduction still condense: the constant-pressure density rise drives it
    assert radiative_condensation_growth_rate(n, T, L, 0.0, k, 1e-6) > 0.0
    # parallel conduction stabilises: gamma crosses zero where k^2 kappa equals the drive
    drive = 2 * L / T - dLdT
    assert radiative_condensation_growth_rate(n, T, L, dLdT, k, drive / k**2) == pytest.approx(0.0, abs=1e-6)
    # k_parallel = 0 is the flute limit, not this formula
    with pytest.raises(ValueError):
        radiative_condensation_growth_rate(n, T, L, dLdT, 0.0, kappa)
    # Drake's Eq. (1): only a falling radiation curve is unstable at constant density
    assert radiative_thermal_instability_growth_rate(n, dLdT) == pytest.approx(-2.0 * dLdT / (3.0 * n * QE))
    assert radiative_thermal_instability_growth_rate(n, -dLdT) < 0.0


def test_the_chart_boundary_is_where_the_growth_rate_vanishes():
    x = np.linspace(-3, 1.9, 50)
    y = mf.condensation_boundary(x)
    np.testing.assert_allclose(y, 2.0 - x, rtol=1e-12)  # Drake: 2 - dlnL/dlnT
    chart = vaft.diagram.marfe().model["chart"]
    xs, ys = chart.curves["condensation"].T
    np.testing.assert_allclose(ys, mf.condensation_boundary(xs))
    # the flute segment lies on the k_parallel = 0 axis exactly where its growth rate is positive (L falls with T)
    fx, fy = chart.curves["flute"].T
    assert np.all(fx < 0) and fx.max() > -0.05 and np.allclose(fy, fy[0]) and fy[0] < 0.2


def test_the_hfs_band_straddles_the_boundary():
    m = mf.marfe_region(localization="hfs")
    axis, out = np.asarray(m["axis"]), m["outline"]
    theta = np.arctan2(out[:, 1] - axis[1], out[:, 0] - axis[0])
    rel = (theta - math.pi + math.pi) % (2 * math.pi) - math.pi
    assert np.max(np.abs(rel)) <= math.radians(15.0) + 1e-9  # Lipschultz: ~30 degrees wide
    assert np.all(out[:, 0] < axis[0])  # high-field side
    # "this radial extent straddles r = a" (Lipschultz p. 16): one edge inside the LCFS, the other outside
    assert m["inner_level"] < 1.0 < m["outer_level"]


def test_the_xpoint_band_sits_above_the_x_point_on_the_closed_side():
    m = mf.marfe_region(localization="xpoint")
    out, low = m["outline"], np.asarray(m["x_point_side"])
    assert m["centre_angle"] < 0
    assert np.min(np.hypot(out[:, 0] - low[0], out[:, 1] - low[1])) < 0.02 * m["minor_radius"]
    assert m["inner_level"] < m["outer_level"] <= 1.0


def test_inputs_and_labels():
    with pytest.raises(ValueError):
        mf.marfe_region(localization="lfs")
    with pytest.raises(ValueError):
        mf.marfe_region(poloidal_width_deg=0.0)
    assert not [i for i in vaft.diagram.marfe(labels=False).scene.items
                if isinstance(i, Label) and i.role not in ("axes", "ticks")]


@pytest.mark.parametrize("localization, topology", [("hfs", "limited"), ("xpoint", "single_null")])
def test_rendering_is_stable_under_last_bit_noise_in_psi(localization, topology):
    import dataclasses

    from vaft.process.equilibrium import solovev_example

    base = vaft.diagram.marfe(localization=localization).tikz
    rng = np.random.default_rng(2)
    eq = solovev_example(topology, a_parameter=0.0)
    eq = dataclasses.replace(eq, psi=np.asarray(eq.psi) * (1 + 1e-15 * rng.standard_normal(np.shape(eq.psi))))
    assert vaft.diagram.marfe(eq, localization=localization).tikz == base

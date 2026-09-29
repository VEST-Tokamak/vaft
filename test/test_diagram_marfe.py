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
    # flat radiation curve, no conduction: gamma = (2/5ne) 2L/T > 0 -- condensation needs no falling L(T)
    assert radiative_condensation_growth_rate(n, T, L, 0.0, 0.0, 0.0) == pytest.approx(4 * L / T / (5 * n * QE))
    # the flute limit is unstable only where L falls with T
    assert radiative_thermal_instability_growth_rate(n, 0.0) == 0.0
    assert radiative_thermal_instability_growth_rate(n, -1e4) > 0 > radiative_thermal_instability_growth_rate(n, 1e4)
    # parallel conduction stabilises: gamma falls linearly in k^2 kappa and crosses zero at the drive
    drive = 2 * L / T - 1e4
    k = 0.8
    assert radiative_condensation_growth_rate(n, T, L, 1e4, k, drive / k**2) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        radiative_condensation_growth_rate(n, 0.0, L, 0.0, 1.0, 1.0)


def test_the_chart_boundary_is_where_the_growth_rate_vanishes():
    x = np.linspace(-3, 1.9, 50)
    y = mf.condensation_boundary(x)
    np.testing.assert_allclose(y, 2.0 - x, rtol=1e-12)  # Drake: 2 - dlnL/dlnT
    chart = vaft.diagram.marfe().model["chart"]
    xs, ys = chart.curves["condensation"].T
    np.testing.assert_allclose(ys, mf.condensation_boundary(xs))
    assert chart.curves["flute"][0, 0] == 0.0


@pytest.mark.parametrize("localization", ["hfs", "xpoint"])
def test_the_band_sits_just_inside_the_boundary_where_asked(localization):
    m = mf.marfe_region(localization=localization)
    axis, out = np.asarray(m["axis"]), m["outline"]
    theta = np.arctan2(out[:, 1] - axis[1], out[:, 0] - axis[0])
    rel = (theta - m["centre_angle"] + math.pi) % (2 * math.pi) - math.pi
    assert np.max(np.abs(rel)) <= math.radians(15.0) + 1e-9  # Lipschultz: ~30 degrees wide
    if localization == "hfs":
        assert m["centre_angle"] == pytest.approx(math.pi)
        assert np.all(out[:, 0] < axis[0])  # high-field side
    else:
        assert m["centre_angle"] < 0  # below the axis, towards the X-point
    # inside the last closed surface: psi_N between the inner level and the boundary
    from scipy.interpolate import RectBivariateSpline

    eq = m["equilibrium"]
    sp = RectBivariateSpline(eq.r, eq.z, mf._psi_n(eq))
    psin = sp.ev(out[:, 0], out[:, 1])
    assert np.all(psin <= 1.0 + 1e-3) and np.all(psin >= m["inner_level"] - 1e-3)


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

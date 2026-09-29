"""Density-limit boundaries evaluated from the packaged VEST sample (#1297 step 5, #636).

This is the executable version of the recipe handed to lane D: take R_geo, B_T,
a, kappa_a and I_p from a reconstructed equilibrium, then draw the Greenwald
limit on the Hugill diagram and project the operating points onto it. The
sample carries no line-averaged density, so every point is placed at a chosen
Greenwald fraction; with a measured density, pass that instead.
"""

import numpy as np
import pytest

from _sample_fixtures import sample_ods
from vaft.formula import boundaries as B
from vaft.process.equilibrium import contour_shape_parameters


def _equilibrium_inputs(ods):
    """Per-slice R_geo [m], B_T at R_geo [T], a [m], kappa_a [-], |I_p| [MA] and q95 [-]."""
    eq = ods["equilibrium"]
    r0 = float(eq["vacuum_toroidal_field.r0"])
    b0 = np.asarray(eq["vacuum_toroidal_field.b0"], dtype=float)
    rows = []
    for i in range(len(eq["time_slice"])):
        ts = eq["time_slice"][i]
        ip = abs(float(ts["global_quantities.ip"])) / 1e6
        if ip <= 0:
            continue
        r = np.asarray(ts["boundary.outline.r"], dtype=float)
        z = np.asarray(ts["boundary.outline.z"], dtype=float)
        a = (r.max() - r.min()) / 2
        R_geo = (r.max() + r.min()) / 2
        kappa_a = contour_shape_parameters(r, z)["area"] / (np.pi * a**2)
        B_t = abs(b0[i]) * r0 / R_geo  # vacuum 1/R field carried to R_geo
        rows.append((R_geo, B_t, a, kappa_a, ip, float(ts["global_quantities.q_95"])))
    return np.array(rows)


@pytest.fixture(scope="module")
def vest_inputs():
    return _equilibrium_inputs(sample_ods())


def test_sample_gives_physical_inputs(vest_inputs):
    R_geo, B_t, a, kappa_a, ip, q95 = vest_inputs.T
    assert len(vest_inputs) >= 5
    assert np.all((R_geo > 0.2) & (R_geo < 0.6))
    assert np.all((a > 0.1) & (a < 0.4))
    assert np.all((kappa_a > 1.0) & (kappa_a < 2.5))
    assert np.all((ip > 0.01) & (ip < 0.2))


def test_greenwald_density_for_vest(vest_inputs):
    _, _, a, _, ip, _ = vest_inputs.T
    n_G = B.boundary_value(B.get_boundary("greenwald"), plasma_current=ip, minor_radius=a)
    # a few 1e19 m^-3 for 50-80 kA over a ~0.15-0.25 m minor radius
    assert np.all((n_G > 1.0) & (n_G < 20.0))


def test_operating_points_on_the_hugill_diagram(vest_inputs):
    R_geo, B_t, a, kappa_a, ip, q95 = vest_inputs.T
    n_G = B.boundary_value(B.get_boundary("greenwald"), plasma_current=ip, minor_radius=a)
    f_G = 0.3  # stand-in for a measured line-averaged density
    x, y = B.hugill_coordinates(f_G * n_G, R_geo, B_t, a, kappa_a, ip)

    line = B.get_boundary("greenwald_hugill")
    result = B.evaluate_boundary(line, x, inverse_cylindrical_q=y, area_elongation=kappa_a)
    np.testing.assert_allclose(result.ratio, f_G)
    assert np.all(result.allowed)

    # The Hugill y axis is 1/q_cyl, not 1/q95: at VEST aspect ratio q_cyl is far below q95.
    assert np.all(1.0 / y < q95)


def test_boundary_curve_for_overlay(vest_inputs):
    kappa_a = float(np.median(vest_inputs[:, 3]))
    curve = B.boundary_curve(B.get_boundary("greenwald_hugill"), "inverse_cylindrical_q",
                             np.linspace(0.0, 0.6, 61), swap_axes=True, area_elongation=kappa_a)
    assert curve.xy.shape == (61, 2)
    assert curve.x_quantity.unit == "1e19 m^-2 T^-1" and curve.allowed_side == "left"
    assert np.all(np.diff(curve.x) > 0)

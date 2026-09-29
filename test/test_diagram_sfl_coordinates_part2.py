"""SFL coordinates part 2 (#1074): toroidal shift, n != 0 spectra, action-angle, validity, COCOS."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _sfl_coordinates as sc
from vaft.formula.equilibrium import sfl_toroidal_angle_shift


def test_phi_plus_nu_is_straight_in_every_angle():
    # an independent field-line integration: d phi/d theta = (B_phi / R) |J| / psi', which also fixes q;
    # with the shift, zeta = phi + nu must be exactly q theta_sfl in every member of the family
    s = sc._surface(sc._R_SPECTRUM)
    r = sc._R_SPECTRUM
    B_phi = np.sqrt(np.maximum(s["B"] ** 2 - s["B_p"] ** 2, 0.0))
    rate = B_phi / s["R"] * np.abs(s["jacobian"]) / (2.0 * sc._PSI_SCALE * r)
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(s["theta"]))])
    assert phi[-1] / (2 * math.pi) == pytest.approx(s["q"], rel=1e-9)
    for name in sc.COORDINATES:
        nu = np.asarray(sfl_toroidal_angle_shift(s["q"], s["angles"][name], s["angles"]["PEST"]))
        np.testing.assert_allclose(phi + nu, s["q"] * s["angles"][name], atol=1e-6)
    with pytest.raises(ValueError):
        sfl_toroidal_angle_shift(np.nan, 1.0, 1.0)


def test_a_field_aligned_mode_moves_to_m0_without_broadening():
    q = sc._surface(sc._R_SPECTRUM)["q"]
    base = sc.spectra_with_toroidal_mode(0)["field_aligned"]
    for n in (1, 2, 4):
        spec = sc.spectra_with_toroidal_mode(n)["field_aligned"]
        m0 = round(n * q)
        for name in sc.COORDINATES:
            assert spec[name]["centroid"] == pytest.approx(m0, abs=0.5)
            assert spec[name]["width"] == pytest.approx(base[name]["width"], rel=1e-6)


def test_a_structure_fixed_in_geometric_phi_is_shifted_by_nu_and_not_in_pest():
    base = sc.spectra_with_toroidal_mode(0)["geometric"]
    for n in (2, 4):
        spec = sc.spectra_with_toroidal_mode(n)["geometric"]
        assert spec["PEST"]["centroid"] == pytest.approx(0.0, abs=1e-6)  # nu = 0
        assert abs(spec["Hamada"]["centroid"]) > 1.0  # shifted, not broadened (nu is nearly linear in theta)
        for name in sc.COORDINATES:
            assert spec[name]["width"] == pytest.approx(base[name]["width"], rel=0.02)
    with pytest.raises(ValueError):
        vaft.diagram.sfl_fourier_convergence(n=-1)


def test_the_field_line_is_straight_only_in_the_action_angle_coordinates():
    chart = vaft.diagram.field_line_action_angle().model
    q = chart.parameters["q"]
    xg, yg = chart.curves["geometric"].T
    x, y = chart.curves["straight"].T
    np.testing.assert_allclose(xg, x)  # the same field line, at the same points
    assert np.max(np.abs(yg - xg / q)) > 0.2  # the geometric angle bends away from the line
    assert yg[0] == pytest.approx(0.0) and yg[-1] == pytest.approx(2 * math.pi, abs=1e-6)


def test_q_diverges_toward_an_x_point_and_settles_when_limited():
    prof = sc.separatrix_q_profiles()
    lim, sn = prof["limited"], prof["single_null"]
    # toward the separatrix (last entries): single null keeps growing, about linearly in ln(1 - psi_N)
    assert np.all(np.diff(sn[-5:]) > 0)
    slope = np.diff(sn[-4:]) / np.diff(np.log(prof["distance"][-4:]))
    assert np.all(slope < -0.05)
    # limited: flat over the last decade
    assert abs(lim[-1] - lim[-4]) < 0.05
    assert sn[-1] > lim[-1] + 0.3


def test_coordinates_and_cocos_are_orthogonal_axes():
    diagram = vaft.diagram.coordinates_vs_cocos()
    roles = {getattr(i, "role", "") for i in diagram.scene.items}
    for name in diagram.model["coordinates"]:
        for cocos in diagram.model["cocos"]:
            assert f"combination:{name}:{cocos}" in roles


def test_labels_off():
    from vaft.diagram._scene import Label

    for build in (vaft.diagram.field_line_action_angle, vaft.diagram.sfl_coordinate_validity,
                  vaft.diagram.coordinates_vs_cocos):
        assert not [i for i in build(labels=False).scene.items
                    if isinstance(i, Label) and i.text and i.role not in ("axes", "ticks")]

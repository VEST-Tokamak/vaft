"""SFL coordinates part 2 (#1074): toroidal shift, n != 0 spectra, action-angle, validity, COCOS."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _sfl_coordinates as sc
from vaft.formula.equilibrium import sfl_toroidal_angle_shift


def test_the_toroidal_shift_vanishes_for_pest_and_closes_on_a_turn():
    s = sc._surface(sc._R_SPECTRUM)
    theta_p = s["angles"]["PEST"]
    assert np.allclose(sfl_toroidal_angle_shift(s["q"], theta_p, theta_p), 0.0)
    for name in ("Boozer", "Hamada", "equal-arc"):
        nu = np.asarray(sfl_toroidal_angle_shift(s["q"], s["angles"][name], theta_p))
        # every angle starts and ends a poloidal turn at 0 and 2 pi: nu is periodic
        assert nu[0] == pytest.approx(0.0, abs=1e-9) and nu[-1] == pytest.approx(0.0, abs=1e-6)
    assert sfl_toroidal_angle_shift(2.0, 1.5, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sfl_toroidal_angle_shift(np.nan, 1.0, 1.0)


def test_a_toroidal_mode_number_broadens_every_shifted_angle_but_not_pest():
    base = {k: v["m99"] for k, v in sc.perturbation_spectra().items()}
    for n in (1, 2, 4):
        spec = {k: v["m99"] for k, v in sc.perturbation_spectra(n=n).items()}
        assert spec["PEST"] == base["PEST"]  # nu = 0
        assert spec["Hamada"] > base["Hamada"] and spec["equal-arc"] > base["equal-arc"]
    assert sc.perturbation_spectra(n=4)["Hamada"]["m99"] > sc.perturbation_spectra(n=1)["Hamada"]["m99"]
    with pytest.raises(ValueError):
        vaft.diagram.sfl_fourier_convergence(n=-1)


def test_the_field_line_is_straight_only_in_the_action_angle_coordinates():
    chart = vaft.diagram.field_line_action_angle().model
    q = chart.parameters["q"]
    x, y = chart.curves["straight"].T
    np.testing.assert_allclose(y, x / q, atol=1e-12)
    xg, yg = chart.curves["geometric"].T
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

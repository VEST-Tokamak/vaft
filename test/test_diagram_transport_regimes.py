"""Collisionality-regime diagrams (#1111): boundaries from vaft.formula, regimes kept apart."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _transport_regimes as tr
from vaft.diagram._scene import Label
from vaft.formula.neoclassical import neoclassical_regime_boundaries
from vaft.formula.ntv import ntv_precession_frequency


def _region_labels(diagram):
    return {i.role for i in diagram.scene.items if isinstance(i, Label) and i.role.startswith("region_")}


def _inside(chart, xy):
    (x0, x1), (y0, y1) = chart.x_range, chart.y_range
    return (x0 - 1e-9 <= xy[0] <= x1 + 1e-9) and (y0 - 1e-9 <= xy[1] <= y1 + 1e-9)


@pytest.mark.parametrize("epsilon", [0.1, 0.3])
def test_neoclassical_boundaries_come_from_the_formula(epsilon):
    diagram = vaft.diagram.neoclassical_collisionality(epsilon=epsilon)
    chart = diagram.model
    banana_plateau, plateau_ps = neoclassical_regime_boundaries(epsilon)
    assert chart.curves["banana_plateau"][0, 0] == pytest.approx(np.log10(banana_plateau))
    assert chart.curves["plateau_pfirsch_schlueter"][0, 0] == pytest.approx(np.log10(plateau_ps))
    # the composite is flat exactly between the two boundaries
    x, y = chart.curves["diffusivity"].T
    between = (x > np.log10(banana_plateau) + 1e-6) & (x < np.log10(plateau_ps) - 1e-6)
    np.testing.assert_allclose(y[between], 0.0, atol=1e-12)
    assert _region_labels(diagram) == {"region_banana", "region_plateau", "region_pfirsch_schlueter"}
    for name, xy in chart.labels.items():
        assert _inside(chart, xy), name


def test_ntv_regimes_are_the_shaing_exponents_on_two_branches():
    chart = vaft.diagram.ntv_collisionality().model
    slopes = {}
    for branch in ("non_resonant", "resonant"):
        x, y = chart.curves[branch].T
        slopes[branch] = set(np.round(np.diff(y) / np.diff(x), 9))
    # non-resonant ends on the sqrt(nu) boundary layer; only the resonant branch reaches superbanana nu
    assert slopes["non_resonant"] == {-1.0, 0.5}
    assert slopes["resonant"] == {-1.0, 0.0, 1.0}
    assert set(tr.NTV_EXPONENTS.values()) == {-1.0, 0.5, 0.0, 1.0}
    # the resonant branch lies above the non-resonant one below the precession ordering
    x, res = chart.curves["resonant"].T
    nonres = chart.curves["non_resonant"][:, 1]
    assert np.all(res[x < 0.0] > nonres[x < 0.0])
    assert {"region_sbp", "region_sqrt_nu", "region_one_over_nu"} <= _region_labels(vaft.diagram.ntv_collisionality())


@pytest.mark.parametrize("omega_magnetic", [1.0, -2.5])
def test_the_superbanana_line_is_the_zero_of_the_precession(omega_magnetic):
    chart = vaft.diagram.ntv_precession_regimes(omega_magnetic=omega_magnetic).model
    y_res = chart.curves["resonance"][0, 1]
    assert ntv_precession_frequency(y_res * omega_magnetic, omega_magnetic) == pytest.approx(0.0, abs=1e-12)
    # the dashed V is nu_eff = |omega_d| from the same formula, drawn outside the resonant band
    band = chart.parameters["band"]
    for name in ("ordering_upper", "ordering_lower"):
        x, y = chart.curves[name].T
        omega_d = ntv_precession_frequency(y * omega_magnetic, omega_magnetic)
        np.testing.assert_allclose(10.0**x, np.abs(omega_d / omega_magnetic), rtol=1e-12)
        assert np.all(np.abs(y - y_res) >= band - 1e-12)
    # the superbanana plateau sits on the resonance at low collisionality, 1/nu at high
    sbp_x, sbp_y = chart.labels["sbp"]
    assert abs(sbp_y - y_res) < band and 10.0**sbp_x < band
    one_x, one_y = chart.labels["one_over_nu"]
    assert 10.0**one_x > abs(ntv_precession_frequency(one_y * omega_magnetic, omega_magnetic) / omega_magnetic)
    for name, xy in chart.labels.items():
        assert _inside(chart, xy), name


def test_diagrams_are_deterministic_and_validate_inputs():
    for build in (vaft.diagram.neoclassical_collisionality, vaft.diagram.ntv_collisionality,
                  vaft.diagram.ntv_precession_regimes):
        assert build().scene == build().scene
        assert not [i for i in build(labels=False).scene.items if isinstance(i, Label) and i.role.startswith("region")]
    with pytest.raises(ValueError):
        vaft.diagram.neoclassical_collisionality(epsilon=1.0)
    with pytest.raises(ValueError):
        vaft.diagram.ntv_precession_regimes(omega_magnetic=0.0)

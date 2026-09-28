"""Linear ideal-MHD waves (#1063): the drawn polarizations and speeds are the formulas'."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._scene import Label
from vaft.formula.stability import magnetosonic_phase_speeds, shear_alfven_frequency


def test_the_magnetosonic_speeds_solve_the_dispersion_relation():
    vA, cs = 1.3, 0.7
    theta = np.linspace(0.0, math.pi, 37)
    fast, slow = magnetosonic_phase_speeds(theta, vA, cs)
    for v in (fast, slow):
        residual = v**4 - v**2 * (vA**2 + cs**2) + vA**2 * cs**2 * np.cos(theta) ** 2
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    alfven = shear_alfven_frequency(np.cos(theta), vA)
    assert np.all(slow <= alfven + 1e-12) and np.all(alfven <= fast + 1e-12)
    assert (fast[0], slow[0]) == pytest.approx((max(vA, cs), min(vA, cs)))
    assert (fast[18], slow[18]) == pytest.approx((math.hypot(vA, cs), 0.0), abs=1e-12)
    # cold plasma: the fast wave is the compressional Alfven wave, isotropic at v_A
    f, s = magnetosonic_phase_speeds(theta, vA, 0.0)
    np.testing.assert_allclose(f, vA)
    np.testing.assert_allclose(s, 0.0)
    with pytest.raises(ValueError):
        magnetosonic_phase_speeds(0.0, -1.0, 1.0)


def test_the_shear_alfven_frequency_is_k_parallel_v_A():
    assert shear_alfven_frequency(-2.0, 3.0) == pytest.approx(6.0)
    np.testing.assert_allclose(shear_alfven_frequency(np.array([0.0, 1.0, -1.0]), 2.0), [0.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        shear_alfven_frequency(1.0, -1.0)


def test_the_shear_alfven_wave_bends_without_compressing():
    m = vaft.diagram.shear_alfven_wave().model
    assert m["omega"] == pytest.approx(float(shear_alfven_frequency(m["k"], 1.0)))
    lines = m["lines"]
    spacing = np.diff([line[:, 1] for line in lines], axis=0)
    np.testing.assert_allclose(spacing, spacing[0, 0])  # equal spacing: |B| unchanged at first order
    for _, dv, dB in m["samples"]:
        assert dB == pytest.approx(-dv / 1.0, abs=1e-12)  # delta B = -(B0/v_A) delta v, B0 = v_A = 1


def test_the_fast_wave_compresses_the_field_lines():
    m = vaft.diagram.fast_magnetosonic_wave().model
    fast, slow = magnetosonic_phase_speeds(0.5 * math.pi, 1.0, 0.6)
    assert m["v_fast"] == pytest.approx(float(fast)) and m["v_slow"] == pytest.approx(0.0, abs=1e-12)
    gaps = np.diff(m["positions"])
    assert gaps.max() / gaps.min() > 1.3  # visibly bunched and spread


def test_the_wave_family_orders_the_branches():
    m = vaft.diagram.mhd_wave_family().model
    assert np.all(m["slow"] <= m["alfven"] + 1e-12) and np.all(m["alfven"] <= m["fast"] + 1e-12)


@pytest.mark.parametrize("name", ["shear_alfven_wave", "fast_magnetosonic_wave", "mhd_wave_family"])
def test_every_wave_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("equations") and not fn(labels=False).scene.role("equations")
    assert not any(isinstance(i, Label) for i in fn(labels=False).scene.items)

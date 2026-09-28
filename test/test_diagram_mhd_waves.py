"""Linear ideal-MHD waves (#1063): the drawn polarizations and speeds are the formulas'."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._scene import Arrow, Label
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


def _arrows(diagram, role):
    return [a for a in diagram.scene.items if isinstance(a, Arrow) and a.role == role]


def test_the_shear_alfven_arrows_are_walen_antiparallel_and_on_the_bending():
    d = vaft.diagram.shear_alfven_wave()
    m = d.model
    assert m["omega"] == pytest.approx(float(shear_alfven_frequency(m["k"], 1.0)))
    dv, dB = _arrows(d, "delta_v"), _arrows(d, "delta_B")
    assert len(dv) == len(dB) >= 4
    for a, b in zip(dv, dB):
        assert a.start[0] == b.start[0]
        # delta v = d(xi)/dt of xi0 cos(kz - wt): positive where the line is about to rise
        z = a.start[0]
        assert np.sign(a.end[1] - a.start[1]) == np.sign(math.sin(m["k"] * z))
        # delta B = -(B0/v_A) delta v: antiparallel, and along the local tilt dy/dz of the drawn line
        assert np.sign(b.end[1] - b.start[1]) == -np.sign(a.end[1] - a.start[1])
        line = m["lines"][0]
        i = int(np.argmin(np.abs(line[:, 0] - z)))
        tilt = (line[i + 1, 1] - line[i - 1, 1]) / (line[i + 1, 0] - line[i - 1, 0])
        assert np.sign(tilt) == np.sign(b.end[1] - b.start[1])


def test_the_fast_wave_compresses_the_field_lines():
    m = vaft.diagram.fast_magnetosonic_wave().model
    fast, slow = magnetosonic_phase_speeds(0.5 * math.pi, 1.0, 0.6)
    assert m["v_fast"] == pytest.approx(float(fast)) and m["v_slow"] == pytest.approx(0.0, abs=1e-12)
    gaps = np.diff(m["positions"])
    assert gaps.max() / gaps.min() > 1.3  # visibly bunched and spread
    # the red (compressed) bands hold the tightest gap, the grey (rarefied) the widest
    d = vaft.diagram.fast_magnetosonic_wave()
    def inside(role, x):
        return any(min(p[1] for p in it.points) <= x <= max(p[1] for p in it.points)
                   for it in d.scene.role(role))
    mids = 0.5 * (m["positions"][1:] + m["positions"][:-1])
    assert inside("compressed", mids[np.argmin(gaps)]) and inside("rarefied", mids[np.argmax(gaps)])
    # the velocity arrows are in phase with the compression (wave along +k)
    for a in _arrows(d, "delta_v"):
        compressed = inside("compressed", a.start[1])
        if compressed:
            assert a.end[1] > a.start[1]


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

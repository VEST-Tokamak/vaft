"""Slab parity and harmonic coupling (#1071): contours of the formula's flux, spectra by FFT."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._slab_parity import harmonic_coupling_spectrum
from vaft.formula.stability import island_pendulum_hamiltonian, slab_perturbed_flux


def test_tearing_flux_is_the_island_pendulum_and_has_the_stated_width():
    shear, psi0, ky = 2.0, 0.05, 1.3
    x = np.linspace(-0.5, 0.5, 7)
    y = np.linspace(0, 5, 7)
    X, Y = np.meshgrid(x, y)
    w = 4 * math.sqrt(psi0 / shear)
    np.testing.assert_allclose(slab_perturbed_flux(X, Y, shear, psi0, ky) / shear,
                               island_pendulum_hamiltonian(X, ky * Y + np.pi, w), atol=1e-12)


def test_the_parities_have_their_symmetry_and_normal_field_on_the_surface():
    x, y = 0.3, 0.7
    for parity, sign in (("tearing", 1), ("twisting", -1)):
        pert = lambda xx: slab_perturbed_flux(xx, y, 1.0, 0.1, 1.0, parity) - 0.5 * xx * xx  # noqa: E731
        assert pert(-x) == pytest.approx(sign * pert(x))
    dbx = lambda parity: -(slab_perturbed_flux(0.0, y + 1e-6, 1.0, 0.1, 1.0, parity)  # noqa: E731
                           - slab_perturbed_flux(0.0, y - 1e-6, 1.0, 0.1, 1.0, parity)) / 2e-6
    assert abs(dbx("tearing")) > 1e-3 and dbx("twisting") == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        slab_perturbed_flux(0.0, 0.0, 0.0, 0.1, 1.0)
    with pytest.raises(ValueError):
        slab_perturbed_flux(0.0, 0.0, 1.0, 0.1, 1.0, "kink")


def test_the_tearing_panel_marks_true_o_and_x_points_on_its_separatrix():
    m = vaft.diagram.slab_parity("tearing").model
    amp = m["amplitude"]
    for y, x in m["x_points"]:
        assert slab_perturbed_flux(x, y, 1.0, amp, 1.0) == pytest.approx(m["separatrix_level"])
    for y, x in m["o_points"]:
        assert slab_perturbed_flux(x, y, 1.0, amp, 1.0) == pytest.approx(-amp)  # the minimum
    assert m["width"] == pytest.approx(4 * math.sqrt(amp))
    assert any(abs(b) > 1e-3 for _, b in m["delta_Bx_on_x0"])


def test_the_twisting_panel_keeps_x0_a_flux_surface():
    m = vaft.diagram.slab_parity("twisting").model
    ys = np.linspace(0, 4 * math.pi, 50)
    np.testing.assert_allclose(slab_perturbed_flux(0.0, ys, 1.0, m["amplitude"], 1.0, "twisting"), 0.0, atol=1e-15)
    assert all(abs(b) < 1e-9 for _, b in m["delta_Bx_on_x0"])
    assert not vaft.diagram.slab_parity("twisting").scene.role("delta_Bx")[:1] or all(
        not hasattr(it, "end") for it in vaft.diagram.slab_parity("twisting").scene.role("delta_Bx"))


@pytest.mark.parametrize("m", [1, 3, 7])
def test_the_coupling_spectrum_is_the_fourier_identity(m):
    c1, c2 = 0.4, 0.25
    spec = harmonic_coupling_spectrum(m, c1, c2)
    expected = {m: 1.0, m + 1: c1 / 2, m - 1: c1 / 2, m + 2: c2 / 2, m - 2: c2 / 2}
    assert set(spec) == set(expected)
    for k, v in expected.items():
        assert spec[k] == pytest.approx(v, abs=1e-12)
    chart = vaft.diagram.poloidal_harmonic_coupling(m).model
    assert chart.parameters["spectrum"] == spec


def test_the_matching_schematic_has_both_channels_on_every_surface():
    d = vaft.diagram.resonant_layer_matching()
    for m in d.model["surfaces"]:
        assert d.scene.role(f"layer:{m}:tearing") and d.scene.role(f"layer:{m}:twisting")
        assert d.scene.role(f"edge:outer->{m}")


@pytest.mark.parametrize("name, kw", [("slab_parity", {"parity": "tearing"}), ("slab_parity", {"parity": "twisting"}),
                                      ("slab_parity_comparison", {}), ("poloidal_harmonic_coupling", {}),
                                      ("resonant_layer_matching", {})])
def test_every_slab_diagram_is_deterministic_and_exported(name, kw):
    fn = getattr(vaft.diagram, name)
    assert fn(**kw).tikz == fn(**kw).tikz
    assert name in vaft.diagram.__all__


@pytest.mark.parametrize("fn, kw", [(vaft.diagram.slab_parity, {"parity": "kink"}),
                                    (vaft.diagram.poloidal_harmonic_coupling, {"m": 0})])
def test_bad_arguments_fail(fn, kw):
    with pytest.raises(ValueError):
        fn(**kw)

"""``rho_tor_norm`` is the toroidal-flux coordinate, or it is not written (#276).

Before this, two adapters wrote ``sqrt(psi_N)`` into ``profiles_1d.rho_tor_norm``
-- that is ``rho_pol``, not ``rho_tor`` -- and the plot layer had two more ways to
present something else under a toroidal label. Every packaged sample is still in
that state on disk, and ``rho_tor_norm`` is ``ProfileRecipe.default_coordinate``,
so it was the abscissa of every 1-D equilibrium profile plot drawn from them.

The invariant these tests defend: nothing anywhere presents ``sqrt(psi_N)`` as a
canonical ``rho_tor_norm``. Producers derive it or leave it unset; readers detect
the historical proxy and refuse it.
"""
from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

pytest.importorskip("omas")

from omas import ODS, load_omas_json

import vaft.omas as vomas
from vaft.data import read_geqdsk
from vaft.data.eqdsk import to_omas
from vaft.data._derived import RHO_POL_PROXY_TOLERANCE, is_rho_pol_proxy, rho_tor_profile
from vaft.data.resources import data_path, sample_geqdsk

TWO_PI = 2.0 * np.pi


def _psi_norm(profiles):
    psi = np.asarray(profiles["psi"], float)
    return (psi - psi[0]) / (psi[-1] - psi[0])


# --- the detector -------------------------------------------------------------


def test_the_proxy_detector_separates_the_three_known_populations():
    """The margin is ten orders of magnitude, so the threshold is not a knob."""
    from vaft.omas.sample import sample_ods

    packaged = np.asarray(
        sample_ods()["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"], float
    )
    assert is_rho_pol_proxy(packaged)
    assert np.max(np.abs(packaged - np.sqrt(np.linspace(0, 1, packaged.size)))) < 1e-12

    reference = np.asarray(
        load_omas_json(
            str(data_path("kineticEfit/ods_48224_300ms.json")), consistency_check=False
        )["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"],
        float,
    )
    assert not is_rho_pol_proxy(reference)
    assert np.max(np.abs(reference - np.sqrt(np.linspace(0, 1, reference.size)))) > 0.1

    fresh = np.asarray(
        read_geqdsk(data_path("efit/g039915.00319")).to_omas()[
            "equilibrium.time_slice.0.profiles_1d.rho_tor_norm"
        ],
        float,
    )
    assert not is_rho_pol_proxy(fresh)
    assert RHO_POL_PROXY_TOLERANCE < 1e-3  # nowhere near either population


def test_the_detector_answers_only_the_question_it_was_asked():
    """A profile that is not a plausible coordinate at all is not 'the proxy'."""
    assert not is_rho_pol_proxy([np.nan, 0.5, 1.0])
    assert not is_rho_pol_proxy([0.0])
    assert not is_rho_pol_proxy(np.sqrt(np.linspace(0, 1, 9)), psi_norm=np.zeros(4))


# --- producers ----------------------------------------------------------------


def test_the_two_adapters_share_one_derivation():
    """`eqdsk` and `vfit` diverged because each had its own; consolidating must
    not move a number, so this pins the refactor as a no-op."""
    from vaft.compat import cumtrapz_compat

    for name in ("efit/g039915.00319", "efit/g040330.00320", "kineticEfit/g048224.00300"):
        ods = read_geqdsk(data_path(name)).to_omas()
        profiles = ods["equilibrium.time_slice.0.profiles_1d"]
        psi_wb = np.asarray(profiles["psi"], float)
        b0 = abs(float(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).ravel()[0]))

        expected_phi = np.asarray(
            cumtrapz_compat(np.asarray(profiles["q"], float), x=psi_wb), float
        )
        np.testing.assert_allclose(
            np.asarray(profiles["phi"], float), expected_phi, rtol=1e-12, atol=1e-18
        )
        expected_rho = np.sqrt(np.abs(expected_phi) / (np.pi * b0))
        np.testing.assert_allclose(
            np.asarray(profiles["rho_tor_norm"], float),
            expected_rho / expected_rho[-1],
            rtol=1e-12,
        )
        assert not is_rho_pol_proxy(profiles["rho_tor_norm"], _psi_norm(profiles))


def test_a_producer_that_cannot_derive_writes_psi_norm_and_no_rho_tor_norm():
    """The DD gives equilibrium profiles_1d `psi_norm` but no `rho_pol_norm`, so
    that is the leaf to fall back to -- never a mislabelled rho_tor_norm."""
    gfile = sample_geqdsk("efit/g039915.00319")
    broken = copy.deepcopy(gfile.mapping)
    # A q that changes sign integrates to a non-monotonic Phi, which is not a
    # radial coordinate at all.
    broken["QPSI"] = np.linspace(-2.0, 2.0, len(np.asarray(broken["QPSI"])))

    ods = to_omas(broken)
    profiles = ods["equilibrium.time_slice.0.profiles_1d"]
    assert "rho_tor_norm" not in profiles
    assert "psi_norm" in profiles
    np.testing.assert_allclose(
        np.asarray(profiles["psi_norm"], float)[[0, -1]], [0.0, 1.0], atol=1e-12
    )


def test_the_ods_updater_uses_the_same_routine():
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    assert is_rho_pol_proxy(ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"])
    vomas.update_equilibrium_profiles_1d_toroidal_flux(ods, time_slice=0)
    profiles = ods["equilibrium.time_slice.0.profiles_1d"]
    assert not is_rho_pol_proxy(profiles["rho_tor_norm"], _psi_norm(profiles))

    expected = rho_tor_profile(
        np.asarray(profiles["q"], float), np.asarray(profiles["psi"], float) * TWO_PI
    )
    np.testing.assert_allclose(
        np.asarray(profiles["rho_tor_norm"], float), expected.rho_tor_norm, rtol=1e-12
    )


# --- readers ------------------------------------------------------------------


def test_a_legacy_sample_is_never_plotted_against_the_proxy():
    """The regression that matters: the packaged samples still hold the proxy on
    disk, and this is what stops it reaching an axis labelled rho_N."""
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    stored = np.asarray(
        ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"], float
    )
    assert is_rho_pol_proxy(stored), "fixture no longer exercises the legacy state"

    _figure, axes = vomas.plot_equilibrium_profile_pressure(ods, show=False)
    drawn = np.asarray(axes.get_lines()[0].get_xdata(), float)
    assert not is_rho_pol_proxy(drawn)
    assert "Toroidal" in axes.get_xlabel()
    # It is the coordinate integrated from q, and it really differs from the proxy.
    assert np.max(np.abs(drawn - stored)) > 0.1


def test_an_unresolvable_coordinate_relabels_rather_than_pretends():
    """Without `q` no toroidal coordinate exists; the axis must say so."""
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    del ods["equilibrium.time_slice.0.profiles_1d.q"]
    _figure, axes = vomas.plot_equilibrium_profile_pressure(ods, show=False)
    assert "Poloidal" in axes.get_xlabel()
    drawn = np.asarray(axes.get_lines()[0].get_xdata(), float)
    np.testing.assert_allclose(drawn[[0, -1]], [0.0, 1.0], atol=1e-12)


def test_a_bare_index_axis_is_labelled_as_one():
    """It used to be `linspace(0, 1, n)` under a toroidal-flux label."""
    ods = ODS(consistency_check=False)
    ods["equilibrium.time_slice.0.profiles_1d.pressure"] = np.linspace(100.0, 0.0, 6)
    _figure, axes = vomas.plot_equilibrium_profile_pressure(ods, show=False)
    assert axes.get_xlabel() == "Profile sample index"
    np.testing.assert_allclose(
        np.asarray(axes.get_lines()[0].get_xdata(), float), np.arange(6.0)
    )

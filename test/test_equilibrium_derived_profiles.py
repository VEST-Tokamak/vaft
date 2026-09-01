"""Equilibrium quantities an EFIT-sourced ODS omits, and the updaters that derive them.

An EFIT g-file stores ``p``, ``q``, ``f``, ``ff'``, ``p'`` and ``psi`` and nothing
else, so the packaged VEST samples carry seven ``profiles_1d`` leaves against the
thirty-seven an OMFIT-produced ODS carries. Issue #290 asked for ``j_tor``; the
same flux-surface trace supplies the geometry family around it.

``vaft/data/kineticEfit/ods_48224_300ms.json`` is the reference: OMFIT wrote it
from an independent surface solve, so agreeing with it is real evidence rather
than a restatement of our own arithmetic. Tolerances below were measured, and
each one's cause is named -- a tolerance nobody can explain is a tolerance that
will be loosened silently the next time it fails.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("omas")
pytest.importorskip("skimage")

from omas import load_omas_json

import vaft.omas.update as update
from vaft.data.resources import data_path

TWO_PI = 2.0 * np.pi

#: Leaves the updaters derive, deleted from the reference before comparing.
DERIVED = (
    "gm1", "gm5", "gm8", "gm9", "volume", "area", "surface",
    "dvolume_dpsi", "darea_dpsi", "elongation",
    "triangularity_upper", "triangularity_lower",
    "b_field_max", "b_field_min", "phi", "rho_tor", "rho_tor_norm", "j_tor",
)

#: ``leaf -> max |derived - stored| / peak(stored)``, for psi_N >= 0.05.
#:
#: The innermost couple of surfaces are excluded throughout: ``|grad psi|`` is
#: small and varies fastest there, and the *reference* is unreliable in that
#: region too -- its stored ``dvolume_dpsi`` runs 8.53, 22.85, 29.59, 29.05 from
#: the axis out, a ramp ``dV/dpsi`` cannot physically have.
INTERIOR_TOLERANCE = {
    "gm1": 2e-3,
    "gm5": 2e-3,
    "gm8": 1e-3,
    "gm9": 1e-3,
    "surface": 1e-3,
    "elongation": 2e-3,
    "b_field_max": 2e-3,
    "b_field_min": 2e-3,
    # Tracing a 129x129 map against OMFIT's own finer surface solve: a
    # systematic ~0.4%, not scatter.
    "volume": 6e-3,
    "area": 6e-3,
    # Same trace, differentiated: 3e-3 away from the axis.
    "dvolume_dpsi": 5e-3,
    "darea_dpsi": 5e-3,
    # The issue's own criterion.
    "j_tor": 5e-4,
}


@pytest.fixture(scope="module")
def reference():
    path = data_path("kineticEfit/ods_48224_300ms.json")
    if not path.exists():  # pragma: no cover - repository-only sample
        pytest.skip("kineticEfit reference ODS is not packaged in this build")
    return load_omas_json(str(path), consistency_check=False)


@pytest.fixture(scope="module")
def derived(reference):
    """The reference with every derived leaf deleted and then re-derived."""
    work = copy.deepcopy(reference)
    ts = work["equilibrium.time_slice.0"]
    for leaf in DERIVED:
        del ts[f"profiles_1d.{leaf}"]
    for leaf in ("volume", "area"):
        if leaf in ts["global_quantities"]:
            del ts[f"global_quantities.{leaf}"]
    update.update_equilibrium_derived_profiles(work)
    return work


def _pair(reference, derived, leaf):
    stored = np.asarray(
        reference["equilibrium.time_slice.0.profiles_1d"][leaf], float
    ).reshape(-1)
    mine = np.asarray(
        derived["equilibrium.time_slice.0.profiles_1d"][leaf], float
    ).reshape(-1)
    assert mine.size == stored.size
    return stored, mine


def _interior(size):
    return np.linspace(0.0, 1.0, size) >= 0.05


@pytest.mark.parametrize("leaf,tolerance", sorted(INTERIOR_TOLERANCE.items()))
def test_derived_profile_matches_the_omfit_reference(reference, derived, leaf, tolerance):
    stored, mine = _pair(reference, derived, leaf)
    interior = _interior(stored.size)
    error = np.abs(mine - stored)[interior] / np.max(np.abs(stored))
    assert np.max(error) < tolerance, f"{leaf}: worst {np.max(error):.2e}"


def test_triangularity_is_within_the_contour_sampling_error(reference, derived):
    """Triangularity is the loosest of the shape profiles, for the same reason
    ``test_eqdsk_derived_quantities`` allows it 0.05: it depends on where the
    contour happens to be sampled near its Z extremum, which on an interior
    surface of a coarse map is a few centimetres of R."""
    for leaf in ("triangularity_upper", "triangularity_lower"):
        stored, mine = _pair(reference, derived, leaf)
        interior = _interior(stored.size)
        assert np.max(np.abs(mine - stored)[interior]) < 0.05, leaf


def test_toroidal_flux_reproduces_the_reference_exactly(reference, derived):
    """``phi`` is ``integral q dpsi`` -- the same integral OMFIT performs, so this
    is an equality, not an approximation."""
    stored, mine = _pair(reference, derived, "phi")
    np.testing.assert_allclose(mine, stored, rtol=1e-12, atol=1e-15)


def test_rho_tor_is_the_dd_function_of_its_own_phi(reference, derived):
    """The DD defines ``rho_tor = sqrt(phi / (pi b0))``. The reference's stored
    ``rho_tor`` is *not* that function of its stored ``phi`` near the axis --
    the two came from different routes there, and its ``q[0] = 8.07`` against a
    neighbourhood of 1.9 is why. What is written here is self-consistent, matches
    at the edge, and is well inside the 0.126 the ``sqrt(psi_N)`` proxy it
    replaces was off by."""
    ts = derived["equilibrium.time_slice.0"]
    phi = np.asarray(ts["profiles_1d.phi"], float)
    rho = np.asarray(ts["profiles_1d.rho_tor"], float)
    b0 = abs(float(np.asarray(derived["equilibrium.vacuum_toroidal_field.b0"], float).ravel()[0]))
    np.testing.assert_allclose(rho, np.sqrt(np.abs(phi) / (np.pi * b0)), rtol=1e-12)

    stored, mine = _pair(reference, derived, "rho_tor_norm")
    assert mine[-1] == pytest.approx(1.0)
    assert np.max(np.abs(mine - stored)) < 0.05
    proxy = np.sqrt(np.linspace(0.0, 1.0, stored.size))
    assert np.max(np.abs(mine - stored)) < np.max(np.abs(proxy - stored))


def test_j_tor_is_exact_given_the_reference_own_flux_surface_averages(reference):
    """Separates the physics from the numerics: fed the reference's own ``gm1``
    and ``gm9``, the Grad-Shafranov expression must reproduce its ``j_tor`` to
    round-off. Anything worse is a wrong formula, not a coarse contour."""
    from vaft.formula.constants import MU0

    p = reference["equilibrium.time_slice.0.profiles_1d"]
    stored = np.asarray(p["j_tor"], float)
    j_tor = -TWO_PI * (
        np.asarray(p["dpressure_dpsi"], float)
        + np.asarray(p["gm1"], float) * np.asarray(p["f_df_dpsi"], float) / MU0
    ) / np.asarray(p["gm9"], float)
    np.testing.assert_allclose(j_tor, stored, rtol=1e-6, atol=1e-3)


def test_j_tor_sign_follows_sigma_bp_not_a_hard_coded_minus(reference):
    """Both available references are ``sigma_Bp = +1``, so ``-sigma_Bp`` and a
    bare ``-1`` are indistinguishable on them. Flipping the flux orientation --
    negating psi and, with it, the two psi-derivatives -- must flip
    ``sigma_Bp`` and leave ``j_tor`` pointing the same way as ``Ip``."""
    from vaft.data.cocos import cocos_spec

    assert cocos_spec(11).sigma_bp == 1
    assert cocos_spec(13).sigma_bp == -1
    assert cocos_spec(1).sigma_bp == cocos_spec(2).sigma_bp
    assert cocos_spec(1).exp_bp == cocos_spec(2).exp_bp

    flipped = copy.deepcopy(reference)
    ts = flipped["equilibrium.time_slice.0"]
    for leaf in ("global_quantities.psi_axis", "global_quantities.psi_boundary"):
        ts[leaf] = -float(ts[leaf])
    for leaf in ("profiles_1d.psi", "profiles_2d.0.psi"):
        ts[leaf] = -np.asarray(ts[leaf], float)
    for leaf in ("profiles_1d.dpressure_dpsi", "profiles_1d.f_df_dpsi"):
        ts[leaf] = -np.asarray(ts[leaf], float)
    for leaf in ("j_tor", "gm1", "gm9", "phi", "rho_tor"):
        del ts[f"profiles_1d.{leaf}"]

    update.update_equilibrium_profiles_1d_j_tor(flipped)
    j_tor = np.asarray(ts["profiles_1d.j_tor"], float)
    ip = float(ts["global_quantities.ip"])
    assert np.sign(j_tor[0]) == np.sign(ip), "j_tor must carry sign(Ip) in either orientation"
    stored = np.asarray(reference["equilibrium.time_slice.0.profiles_1d.j_tor"], float)
    error = np.abs(j_tor - stored)[_interior(stored.size)] / np.max(np.abs(stored))
    assert np.max(error) < 5e-4


def test_global_scalars_follow_from_the_derived_profiles(reference, derived):
    """``volume``, ``area`` and ``energy_mhd`` have had updaters all along; they
    were empty only because ``profiles_1d.volume``/``area`` did not exist."""
    g = derived["equilibrium.time_slice.0.global_quantities"]
    # The reference stores global area but no global volume, so both are checked
    # against the edge of its own profiles.
    edge = reference["equilibrium.time_slice.0.profiles_1d"]
    assert float(g["volume"]) == pytest.approx(float(np.asarray(edge["volume"], float)[-1]), rel=6e-3)
    assert float(g["area"]) == pytest.approx(float(np.asarray(edge["area"], float)[-1]), rel=6e-3)
    assert float(g["energy_mhd"]) > 0.0

    boundary = derived["equilibrium.time_slice.0.boundary"]
    for leaf in ("elongation", "triangularity_upper", "triangularity_lower"):
        assert leaf in boundary


# --- the packaged samples, which have no stored j_tor to compare against ------


@pytest.fixture(scope="module")
def packaged():
    from vaft.omas.sample import sample_ods

    try:
        ods = sample_ods()
    except Exception as exc:  # pragma: no cover - sample not packaged
        pytest.skip(f"39915 sample unavailable: {exc}")
    update.update_equilibrium_derived_profiles(ods)
    return ods


def test_packaged_sample_gains_the_missing_profiles(packaged):
    profiles = packaged["equilibrium.time_slice.0.profiles_1d"]
    for leaf in DERIVED:
        assert leaf in profiles, leaf


def test_derived_j_tor_integrates_to_the_stored_plasma_current(packaged):
    """The only check available where no reference ``j_tor`` exists, and the one
    that pins the 2*pi and the sign together:

        Ip = integral dpsi (dV/dpsi / 2pi) <1/R> j_tor

    A storage-convention slip would land this a factor 6.28 out, and a sign slip
    would invert it.
    """
    for index in range(len(packaged["equilibrium.time_slice"])):
        ts = packaged["equilibrium.time_slice"][index]
        if "profiles_1d.j_tor" not in ts:
            continue  # a degenerate EFIT slice; skipped by the updater
        p = ts["profiles_1d"]
        current = np.trapezoid(
            np.asarray(p["dvolume_dpsi"], float)
            / TWO_PI
            * np.asarray(p["gm9"], float)
            * np.asarray(p["j_tor"], float),
            np.asarray(p["psi"], float),
        )
        stored = float(ts["global_quantities.ip"])
        assert current == pytest.approx(stored, rel=0.02), f"time slice {index}"


def test_degenerate_slice_is_skipped_rather_than_filled_with_nonsense(packaged):
    """Slice 8 of shot 39915 is an EFIT solution with psi_axis == psi_boundary."""
    degenerate = packaged["equilibrium.time_slice.8"]
    assert "profiles_1d.j_tor" not in degenerate
    assert "profiles_1d.volume" not in degenerate


def test_volume_is_the_integral_of_its_own_dvolume_dpsi(packaged):
    """Near the axis there is no reference worth matching, so the derivatives are
    held to their own profile instead."""
    p = packaged["equilibrium.time_slice.0.profiles_1d"]
    psi = np.asarray(p["psi"], float)
    volume = np.asarray(p["volume"], float)
    integrated = np.concatenate(
        [[0.0], np.cumsum(np.diff(psi) * 0.5 * (np.asarray(p["dvolume_dpsi"], float)[1:]
                                                + np.asarray(p["dvolume_dpsi"], float)[:-1]))]
    )
    assert np.max(np.abs(integrated - volume)) / volume[-1] < 0.02

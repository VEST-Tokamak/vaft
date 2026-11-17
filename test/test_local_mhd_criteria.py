"""Equilibrium-native Mercier / GGJ interchange indices against DCON.

``test/data/gpec/mercier_reference_1902/dcon_di_dr.json`` holds DCON's own
``di`` and ``dr`` (mpsi 512, mtheta 1024, GPEC e68d7ac2) for the packaged VEST
g-file ``vaft/data/efit/g039915.00319`` and, when GPEC's example tree is
present, the upstream DIII-D-like example.  Nothing here runs a solver.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from vaft.process.local_mhd_criteria import interchange_comparison, mercier_ggj_from_ods, mercier_ggj_profile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "test" / "data" / "gpec" / "mercier_reference_1902" / "dcon_di_dr.json").read_text())


def _gfile(path):
    from vaft.data.eqdsk import read_geqdsk

    m = read_geqdsk(path).mapping
    nw, nh = int(m["NW"]), int(m["NH"])
    grid = dict(
        psi_grid=np.asarray(m["PSIRZ"]),  # the reader returns it indexed (R, Z)
        R=m["RLEFT"] + np.linspace(0.0, m["RDIM"], nw),
        Z=m["ZMID"] - m["ZDIM"] / 2 + np.linspace(0.0, m["ZDIM"], nh),
        psi_axis=float(m["SIMAG"]), psi_boundary=float(m["SIBRY"]),
    )
    profiles = dict(profile_psi_n=np.linspace(0.0, 1.0, nw), f=np.asarray(m["FPOL"]), pressure=np.asarray(m["PRES"]),
                    q=np.asarray(m["QPSI"]), axis_rz=(float(m["RMAXIS"]), float(m["ZMAXIS"])))
    return grid, profiles


VEST = ROOT / "vaft" / "data" / "efit" / "g039915.00319"
# Dense, so a pass does not depend on where the levels happen to sit (cold review).
LEVELS = np.round(np.arange(0.05, 0.951, 0.01), 2)


@pytest.fixture(scope="module")
def vest():
    grid, profiles = _gfile(VEST)
    return grid, profiles, mercier_ggj_profile(**grid, levels_norm=LEVELS, **profiles)


def test_vest_slice_matches_dcon(vest):
    _, _, profile = vest
    case = REFERENCE["cases"]["vest_39915_319"]
    c = interchange_comparison(profile, case["psi_n"], case["di"], case["dr"], label="DCON 39915@319")
    relative = np.abs(c["dD_I"] / c["di"])  # D_I runs from -19 to -0.25: compare relatively
    psi = profile.psi_n
    assert np.nanmax(relative[psi < 0.1]) < 0.05  # measured 3.3 %
    assert np.nanmax(relative[(psi >= 0.1) & (psi < 0.3)]) < 0.01  # measured 0.5 %
    assert np.nanmax(relative[psi >= 0.3]) < 3e-3  # measured 0.09 %
    assert np.nanmax(np.abs(c["dD_R"][psi >= 0.3])) < 2e-3
    # The map's q agrees with the file's away from its own edge spline.
    assert profile.provenance["q_map_vs_given_max_relative"] < 0.02
    assert profile.provenance["unresolved_levels"] == 0
    # Universal criterion only: this slice is Mercier stable everywhere evaluated.
    assert not profile.mercier_unstable.any()


def test_the_derivative_and_contour_resolution_are_converged(vest):
    grid, profiles, profile = vest
    finer = mercier_ggj_profile(**grid, levels_norm=LEVELS, **profiles, oversample=4)
    # measured: <= 0.6 % between oversample 2 and 4 from psi_N 0.10 out
    np.testing.assert_allclose(finer.D_I[LEVELS >= 0.1], profile.D_I[LEVELS >= 0.1], rtol=1e-2)
    np.testing.assert_allclose(finer.H[LEVELS >= 0.1], profile.H[LEVELS >= 0.1], atol=2e-3)


@pytest.mark.parametrize("flip", ["psi", "F", "q"])
def test_the_indices_do_not_depend_on_any_one_cocos_sign(vest, flip):
    grid, profiles, profile = vest
    if flip == "psi":
        grid = {**grid, "psi_grid": -grid["psi_grid"], "psi_axis": -grid["psi_axis"],
                "psi_boundary": -grid["psi_boundary"]}
    else:
        profiles = {**profiles, flip.lower(): -profiles[flip.lower()]}
    flipped = mercier_ggj_profile(**grid, levels_norm=LEVELS, **profiles)
    np.testing.assert_allclose(flipped.D_I, profile.D_I, rtol=1e-10)
    np.testing.assert_allclose(flipped.D_R, profile.D_R, rtol=1e-10)


def test_a_reversed_pressure_gradient_reverses_h(vest):
    """H is linear in p': a pressure rising outward must flip it (p' is not taken by magnitude)."""
    grid, profiles, profile = vest
    rising = mercier_ggj_profile(**grid, levels_norm=LEVELS,
                                 **{**profiles, "pressure": profiles["pressure"].max() - profiles["pressure"]})
    np.testing.assert_allclose(rising.H, -profile.H, rtol=1e-10, atol=1e-14)


def test_without_a_pressure_gradient_only_the_shear_term_is_left(vest):
    grid, profiles, _ = vest
    flat = mercier_ggj_profile(**grid, levels_norm=LEVELS, **{**profiles, "pressure": np.full_like(profiles["pressure"], 1e3)})
    np.testing.assert_allclose(flat.D_I, -0.25, atol=1e-12)  # D_I = -1/4: shear alone is stabilizing
    np.testing.assert_allclose(flat.H, 0.0, atol=1e-12)
    np.testing.assert_allclose(flat.D_R, 0.0, atol=1e-12)  # marginal for resistive interchange
    np.testing.assert_allclose(flat.E_plus_F, 0.0, atol=1e-12)


def test_the_ods_path_is_the_array_path(vest):
    from vaft.data.eqdsk import read_geqdsk

    _, _, profile = vest
    ods = read_geqdsk(VEST).to_omas()
    from_ods = mercier_ggj_from_ods(ods, 0, levels_norm=LEVELS)
    np.testing.assert_allclose(from_ods.D_I, profile.D_I, rtol=1e-6)
    assert from_ods.provenance["time_slice"] == 0


def test_the_code_h_is_the_root_nearer_the_equilibrium():
    from vaft.process.local_mhd_criteria import InterchangeProfile

    profile = InterchangeProfile(psi_n=np.array([0.5]), D_I=np.array([-0.3]), D_R=np.array([-0.05]), H=np.array([0.0]),
                                 E_plus_F=np.array([-0.05]), dV_dchi=np.array([1.0]), q=np.array([2.0]), dq_dchi=np.array([1.0]),
                                 averages={})
    c = interchange_comparison(profile, [0.0, 1.0], [-0.3, -0.3], [-0.05, -0.05], label="synthetic")
    assert c["H_code"][0] == pytest.approx(0.0)  # (H - 1/2)^2 = 0.25: roots 0 and 1, 0 is nearer


def test_inputs_are_refused_rather_than_guessed(vest):
    grid, profiles, _ = vest
    with pytest.raises(ValueError, match="inside"):
        mercier_ggj_profile(**grid, levels_norm=[0.0, 0.2, 0.4, 0.6, 0.8], **profiles)
    with pytest.raises(ValueError, match="indexed"):
        mercier_ggj_profile(**{**grid, "psi_grid": grid["psi_grid"][:, :-1]}, levels_norm=LEVELS, **profiles)
    with pytest.raises(ValueError, match="oversample"):
        mercier_ggj_profile(**grid, levels_norm=LEVELS, **profiles, oversample=0)


DIIID = Path(os.environ.get("GPECHOME", Path.home() / "git" / "GPEC")) / "docs" / "examples" / "DIIID_ideal_example" / "TkMkr_D3Dlike_Hmode.geqdsk"


@pytest.mark.skipif(not DIIID.is_file(), reason="GPEC's DIII-D-like example is not installed here")
def test_diiid_like_example_matches_dcon():
    grid, profiles = _gfile(DIIID)
    levels = np.round(np.arange(0.05, 0.981, 0.01), 2)
    profile = mercier_ggj_profile(**grid, levels_norm=levels, **profiles)
    case = REFERENCE["cases"]["diiid_like"]
    c = interchange_comparison(profile, case["psi_n"], case["di"], case["dr"], label="DCON DIII-D-like")
    relative = np.abs(c["dD_I"] / c["di"])
    assert np.nanmax(relative[levels <= 0.9]) < 5e-3  # measured 0.2 %
    assert np.nanmax(relative) < 0.05  # the steep pedestal, 0.95-0.96: measured 3.4 %


def test_differentiating_the_file_q_is_what_degrades_the_core(vest):
    """q_from_map is the method choice the comparison supports; the profile derivative is noisier."""
    grid, profiles, profile = vest
    from_profile = mercier_ggj_profile(**grid, levels_norm=LEVELS, **profiles, q_from_map=False)
    case = REFERENCE["cases"]["vest_39915_319"]
    worse = interchange_comparison(from_profile, case["psi_n"], case["di"], case["dr"], label="x")
    better = interchange_comparison(profile, case["psi_n"], case["di"], case["dr"], label="x")
    core = (LEVELS >= 0.1) & (LEVELS < 0.3)
    assert np.nanmax(np.abs(worse["dD_I"][core])) > 3 * np.nanmax(np.abs(better["dD_I"][core]))



# ------------------------------------------------ curvature and local shear (part 2)

from vaft.process.local_mhd_criteria import curvature_map, flux_surface_curvature  # noqa: E402

CURVATURE_LEVELS = [0.2, 0.4, 0.6, 0.8, 0.95]


@pytest.fixture(scope="module")
def vest_curvature(vest):
    grid, profiles, _ = vest
    return flux_surface_curvature(**grid, levels_norm=CURVATURE_LEVELS, **profiles)


def test_curvature_satisfies_its_identities(vest_curvature):
    checks = vest_curvature.checks
    assert np.nanmax(checks["kappa_dot_b"]) < 1e-8  # curvature is perpendicular to b
    assert np.nanmax(checks["kappa_g_force_balance"]) < 1e-6  # (b x n).grad(B^2/2)/B^2
    # n.grad(B^2/2)/B^2 + mu0 p'|grad psi|/B^2: limited by the file's p' spline; measured <= 0.5 %
    assert np.nanmax(checks["kappa_n_force_balance"]) < 1e-2
    # closed(local_shear R dl/|grad psi|) = 2 pi dq/dpsi: measured <= 1.6e-5
    assert np.nanmax(checks["shear_identity"]) < 1e-4


def test_bad_curvature_is_outboard_and_geodesic_curvature_is_up_down_odd(vest_curvature):
    surface = vest_curvature.surfaces[CURVATURE_LEVELS.index(0.6)]
    outboard = np.argmin(np.abs(np.mod(surface["theta"] + np.pi, 2 * np.pi) - np.pi))
    inboard = np.argmin(np.abs(surface["theta"] - np.pi))
    assert surface["kappa_n"][outboard] < 0 < surface["kappa_n"][inboard]
    assert np.corrcoef(surface["Z"], surface["kappa_g"])[0, 1] < -0.9


def test_curvature_and_shear_do_not_depend_on_cocos_signs(vest, vest_curvature):
    grid, profiles, _ = vest
    flipped = flux_surface_curvature(
        **{**grid, "psi_grid": -grid["psi_grid"], "psi_axis": -grid["psi_axis"], "psi_boundary": -grid["psi_boundary"]},
        levels_norm=CURVATURE_LEVELS, **{**profiles, "f": -profiles["f"], "q": -profiles["q"]},
    )
    for a, b in zip(flipped.surfaces, vest_curvature.surfaces):
        for name in ("kappa_n", "kappa_g", "local_shear"):
            np.testing.assert_allclose(a[name], b[name], rtol=1e-6, atol=1e-9 * np.max(np.abs(b[name])))
    # The checks must hold in the flipped orientation too (cold review: a doubled
    # orientation sign broke the kappa_n force balance only when psi falls outward).
    for name, value in flipped.checks.items():
        np.testing.assert_allclose(value, vest_curvature.checks[name], rtol=1e-6, atol=1e-12)


def test_the_map_masks_outside_the_requested_band(vest):
    grid, profiles, _ = vest
    maps = curvature_map(**grid, profile_psi_n=profiles["profile_psi_n"], f=profiles["f"], pressure=profiles["pressure"],
                         axis_rz=profiles["axis_rz"], psi_n_min=0.1, psi_n_max=0.9)
    kept = np.isfinite(maps["kappa_n"])
    band = (maps["psi_n"] >= 0.1) & (maps["psi_n"] <= 0.9)
    assert kept.any() and band[kept].all()
    # Vacuum nodes whose flux falls in the band (cold review: about half of them on this
    # limited map) are not kept: every kept node is near the axis's side of the edge.
    assert band.sum() > kept.sum()
    r_kept = maps["R"][kept]
    assert r_kept.min() > 0.1  # nothing from the centre-stack column at R = 0.05 m
    assert maps["kappa_n"].shape == grid["psi_grid"].shape


# ---------------------------------------------------------------- s-alpha (part 3)

from vaft.process.local_mhd_criteria import s_alpha_from_ods, s_alpha_profile  # noqa: E402

S_ALPHA_LEVELS = np.round(np.arange(0.10, 0.951, 0.05), 2)


@pytest.fixture(scope="module")
def vest_s_alpha(vest):
    grid, profiles, _ = vest
    return s_alpha_profile(**grid, levels_norm=S_ALPHA_LEVELS, **profiles)


def test_shear_is_dlnq_dln_volume_radius(vest_s_alpha):
    """An independent check: finite differences of the map's q against r_V = sqrt(V/2 pi^2 R0)."""
    from vaft.process.local_mhd_criteria import mercier_ggj_profile

    profile = vest_s_alpha
    r_v = np.sqrt(profile.volume / (2 * np.pi**2 * profile.provenance["R0"]))
    grid, profiles = _gfile(VEST)
    q = mercier_ggj_profile(**grid, levels_norm=S_ALPHA_LEVELS, **profiles).q
    s_fd = np.gradient(np.log(q), np.log(r_v))
    inner = slice(2, -2)  # one-sided differences at the ends
    np.testing.assert_allclose(profile.s[inner], s_fd[inner], rtol=0.05)


def test_alpha_is_positive_for_a_falling_pressure_and_zero_without_a_gradient(vest, vest_s_alpha):
    assert (vest_s_alpha.alpha > 0).all()
    grid, profiles, _ = vest
    flat = s_alpha_profile(**grid, levels_norm=S_ALPHA_LEVELS,
                           **{**profiles, "pressure": np.full_like(profiles["pressure"], 1e3)})
    np.testing.assert_allclose(flat.alpha, 0.0, atol=1e-12)
    assert not flat.s_alpha_unstable.any()


def test_the_volume_is_the_integral_of_v_prime(vest, vest_s_alpha):
    from scipy.integrate import trapezoid

    from vaft.process.local_mhd_criteria import mercier_ggj_profile

    grid, profiles, _ = vest
    fine = np.round(np.arange(0.10, 0.951, 0.01), 2)
    dV = mercier_ggj_profile(**grid, levels_norm=fine, **profiles).dV_dchi
    chi_span = 2 * np.pi * abs(grid["psi_boundary"] - grid["psi_axis"])
    integral = trapezoid(dV, fine) * chi_span
    volume = vest_s_alpha.volume
    assert integral == pytest.approx(volume[-1] - volume[0], rel=2e-3)


def test_s_alpha_from_ods_is_the_array_path(vest_s_alpha):
    from vaft.data.eqdsk import read_geqdsk

    ods = read_geqdsk(VEST).to_omas()
    from_ods = s_alpha_from_ods(ods, 0, levels_norm=S_ALPHA_LEVELS)
    np.testing.assert_allclose(from_ods.alpha, vest_s_alpha.alpha, rtol=1e-6)
    np.testing.assert_allclose(from_ods.s, vest_s_alpha.s, rtol=1e-6)
    assert from_ods.provenance["R0_source"] == "magnetic axis"


def test_s_and_alpha_reduce_to_connor_hastie_taylor_at_large_aspect_ratio():
    """External truth for the per-radian normalisation: a circular R0 = 20 m torus, q = 1 + 2 r^2,
    p = p0 (1 - r^2), where s = r q'/q and alpha = -2 mu0 R0 q^2 p'(r) / B0^2 exactly."""
    mu0, r0, b0, q0, q2, p0 = 4e-7 * np.pi, 20.0, 1.0, 1.0, 2.0, 1e3

    def psi_of_r(r):  # dpsi/dr = r B0 / q, per radian
        return b0 * np.log(1 + q2 * r**2 / q0) / (2 * q2)

    R = np.linspace(r0 - 1.3, r0 + 1.3, 201)
    Z = np.linspace(-1.3, 1.3, 201)
    rr, zz = np.meshgrid(R, Z, indexing="ij")
    psi = psi_of_r(np.hypot(rr - r0, zz))
    x = np.linspace(0.0, 1.0, 201)
    r_x = np.sqrt(q0 * (np.exp(2 * q2 * x * psi_of_r(1.0) / b0) - 1) / q2)
    levels = np.round(np.arange(0.1, 0.91, 0.05), 2)
    profile = s_alpha_profile(psi, R, Z, 0.0, psi_of_r(1.0), levels, profile_psi_n=x, f=np.full_like(x, b0 * r0),
                              pressure=p0 * (1 - r_x**2), q=q0 + q2 * r_x**2, axis_rz=(r0, 0.0), R0=r0)
    r = np.sqrt(q0 * (np.exp(2 * q2 * levels * psi_of_r(1.0) / b0) - 1) / q2)
    q = q0 + q2 * r**2
    np.testing.assert_allclose(profile.s, 2 * q2 * r**2 / q, rtol=3e-3)  # reviewer measured <= 1.7e-3
    np.testing.assert_allclose(profile.alpha, -2 * mu0 * r0 * q**2 * (-2 * p0 * r) / b0**2, rtol=2e-3)
    np.testing.assert_allclose(profile.volume, 2 * np.pi**2 * r0 * r**2, rtol=2e-3)


def test_the_map_refuses_a_band_or_axis_outside_the_plasma(vest):
    grid, profiles, _ = vest
    common = dict(profile_psi_n=profiles["profile_psi_n"], f=profiles["f"], pressure=profiles["pressure"])
    with pytest.raises(ValueError, match="inside the confined plasma"):
        curvature_map(**grid, **common, axis_rz=profiles["axis_rz"], psi_n_max=1.05)
    with pytest.raises(ValueError, match="encloses the axis"):
        curvature_map(**grid, **common, axis_rz=(5.0, 5.0))

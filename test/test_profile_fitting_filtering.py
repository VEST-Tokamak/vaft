"""Channel filtering / time-tolerance tests for the TS profile fitter and mappers."""

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.process.profile import (
    equilibrium_mapping_charge_exchange,
    equilibrium_mapping_thomson_scattering,
    profile_fitting_thomson_scattering,
)


def _fake_geq(r0=0.4, a_out=0.3):
    """Two circular flux surfaces (levels 0.5, 1.0) centred at (r0, 0)."""
    theta = np.linspace(0, 2 * np.pi, 181)
    surfaces = {}
    levels = [0.5, 1.0]
    radii = [a_out / 2, a_out]
    flux = {
        i: {"R": r0 + a * np.cos(theta), "Z": a * np.sin(theta)}
        for i, a in enumerate(radii)
    }
    return {"fluxSurfaces": {"levels": levels, "flux": flux}}


def _make_ts_ods(n_ch=7, times=(0.299, 0.300, 0.301)):
    """Synthetic TS ODS: Te ~ 300*(1-psiN^2)^1.5, ne ~ 4e18*(1-psiN^2)."""
    rng = np.random.default_rng(2)
    times = np.asarray(times, float)
    psi_n = np.linspace(0.05, 0.85, n_ch)
    ods = ODS()
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 1
    ods["thomson_scattering.time"] = times
    for i in range(n_ch):
        te = 300.0 * (1.0 - psi_n[i] ** 2) ** 1.5 + rng.normal(0, 3, times.size)
        ne = 4e18 * (1.0 - psi_n[i] ** 2) + rng.normal(0, 4e16, times.size)
        ods[f"thomson_scattering.channel.{i}.position.r"] = 0.25 + 0.05 * i
        ods[f"thomson_scattering.channel.{i}.position.z"] = 0.0
        ods[f"thomson_scattering.channel.{i}.t_e.data"] = te
        ods[f"thomson_scattering.channel.{i}.t_e.data_error_upper"] = np.full(times.size, 8.0)
        ods[f"thomson_scattering.channel.{i}.n_e.data"] = ne
        ods[f"thomson_scattering.channel.{i}.n_e.data_error_upper"] = np.full(times.size, 2e17)
    return ods, psi_n


def _fit(ods, rho, time_ms=300.0, **kwargs):
    return profile_fitting_thomson_scattering(
        ods, time_ms, rho, Te_order=2, Ne_order=2,
        fitting_function_te="polynomial", fitting_function_ne="polynomial",
        coordinate="psi_norm",   # the synthetic channels are placed in psi_N
        **kwargs,
    )


def test_zero_sigma_channel_does_not_hijack_fit():
    ods, psi_n = _make_ts_ods()
    _, te_ref, *_ = _fit(ods, psi_n)
    axis_ref = float(te_ref(np.array([0.0]))[0])

    # poison channel 3: dead 0.1 eV point with zero uncertainty at every time
    ods["thomson_scattering.channel.3.t_e.data"] = np.full(3, 0.1)
    ods["thomson_scattering.channel.3.t_e.data_error_upper"] = np.zeros(3)
    _, te_bad, *_ = _fit(ods, psi_n)
    axis_bad = float(te_bad(np.array([0.0]))[0])
    at_poisoned = float(te_bad(np.array([psi_n[3]]))[0])

    assert abs(axis_bad - axis_ref) / axis_ref < 0.3
    # with the dead channel dropped, the fit must not collapse onto 0.1 eV there
    assert at_poisoned > 100.0


def test_nan_rho_channel_is_dropped():
    ods, psi_n = _make_ts_ods()
    rho = psi_n.copy()
    rho[6] = np.nan  # e.g. outside-LCFS channel from the mapper
    n_e_f, te_f, *_ = _fit(ods, rho)
    assert np.isfinite(float(te_f(np.array([0.2]))[0]))
    assert np.isfinite(float(n_e_f(np.array([0.2]))[0]))


def test_nearest_time_within_tolerance():
    ods, psi_n = _make_ts_ods()
    _fit(ods, psi_n, time_ms=300.4)  # nearest 300 ms, within default 1.0 ms


def test_time_out_of_tolerance_raises():
    ods, psi_n = _make_ts_ods()
    with pytest.raises(ValueError):
        _fit(ods, psi_n, time_ms=305.0)


def test_returned_te_profile_clipped_nonnegative():
    ods, psi_n = _make_ts_ods()
    *_, t_e_rho = _fit(ods, psi_n)
    assert np.all(t_e_rho >= 0.0)


def test_ts_mapping_outside_lcfs_is_nan():
    geq = _fake_geq()
    ods = ODS()
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 1
    # channel 0 inside (near axis), channel 1 far outside the outer surface
    ods["thomson_scattering.channel.0.position.r"] = 0.45
    ods["thomson_scattering.channel.0.position.z"] = 0.0
    ods["thomson_scattering.channel.1.position.r"] = 0.85
    ods["thomson_scattering.channel.1.position.z"] = 0.0
    mapped = equilibrium_mapping_thomson_scattering(ods, geq)
    rho = mapped.psi_norm
    assert np.isfinite(rho[0]) and 0.0 <= rho[0] <= 1.0
    assert np.isnan(rho[1])
    assert np.isnan(mapped.rho_pol_norm[1])
    # a fluxSurfaces mapping carries no q: rho_tor_norm is unavailable, not guessed
    assert mapped.rho_tor_norm is None and "no q profile" in mapped.rho_tor_norm_unavailable


def test_ces_mapping_outside_lcfs_is_nan():
    geq = _fake_geq()
    ods = ODS()
    ods["charge_exchange.ids_properties.homogeneous_time"] = 1
    ods["charge_exchange.time"] = np.array([0.3])
    for i, r in enumerate([0.45, 0.85]):
        ods[f"charge_exchange.channel.{i}.position.r.time"] = np.array([0.3])
        ods[f"charge_exchange.channel.{i}.position.r.data"] = np.array([r])
        ods[f"charge_exchange.channel.{i}.position.z.time"] = np.array([0.3])
        ods[f"charge_exchange.channel.{i}.position.z.data"] = np.array([0.0])
    rho = equilibrium_mapping_charge_exchange(ods, geq).psi_norm
    assert np.isfinite(rho[0]) and 0.0 <= rho[0] <= 1.0
    assert np.isnan(rho[1])


# --- opt-in edge mapping on synthetic geometries (#2335 Part B) ------------------------


def _fake_gfile(with_outline=True):
    """psi_norm = ((R - 0.4)^2 + Z^2) / 0.3^2 on a grid; the LCFS outline is the upper half-disc.

    The lower half of the psi_norm <= 1 disc is outside the outline: a stand-in for
    the private-flux region below an X-point.
    """
    nw = nh = 65
    r = np.linspace(0.0, 1.0, nw) + 0.0
    z = np.linspace(0.0, 1.2, nh) - 0.6
    R, Z = np.meshgrid(r, z, indexing="ij")
    psi = ((R - 0.4) ** 2 + Z ** 2) / 0.09
    geq = {"NW": nw, "NH": nh, "RDIM": 1.0, "RLEFT": 0.0, "ZDIM": 1.2, "ZMID": 0.0,
           "PSIRZ": psi, "SIMAG": 0.0, "SIBRY": 1.0}
    if with_outline:
        theta = np.linspace(0.0, np.pi, 91)
        geq["RBBBS"] = np.append(0.4 + 0.3 * np.cos(theta), 0.7)
        geq["ZBBBS"] = np.append(0.3 * np.sin(theta), 0.0)
    return geq


def test_edge_mapping_labels_every_region():
    from vaft.process.profile import equilibrium_mapping_points

    r = np.array([0.4, 0.4, 0.8, 0.4, 2.0])
    z = np.array([0.1, -0.15, 0.0, 0.299, 0.0])
    edge = equilibrium_mapping_points(_fake_gfile(), r, z, edge_mapping="preserve")
    assert list(edge.region) == ["closed_flux", "private_flux", "sol", "closed_flux", "unmapped"]
    np.testing.assert_allclose(edge.psi_norm[:4], [0.1111, 0.25, 1.7778, 0.9933], atol=5e-3)
    assert np.isnan(edge.psi_norm[4])
    # private flux has a flux value but no closed-surface radius
    assert np.isnan(edge.rho_pol_norm[1])
    np.testing.assert_allclose(edge.rho_pol_norm[2], np.sqrt(edge.psi_norm[2]))
    assert "outline" in edge.region_basis
    # the legacy mapping cannot tell private flux from closed flux, and drops the SOL point
    legacy = equilibrium_mapping_points(_fake_gfile(), r, z)
    assert legacy.region is None
    assert np.isfinite(legacy.psi_norm[1]) and np.isnan(legacy.psi_norm[2])


def test_edge_mapping_without_an_outline_says_so():
    from vaft.process.profile import equilibrium_mapping_points

    edge = equilibrium_mapping_points(_fake_gfile(with_outline=False), [0.4, 0.8], [-0.15, 0.0],
                                      edge_mapping="preserve")
    assert list(edge.region) == ["closed_flux", "sol"]
    assert "cannot be told" in edge.region_basis


def test_edge_mapping_of_a_legacy_flux_surface_map_leaves_outside_points_unmapped():
    ods = ODS()
    for i, r in enumerate([0.45, 0.85]):
        ods[f"thomson_scattering.channel.{i}.position.r"] = r
        ods[f"thomson_scattering.channel.{i}.position.z"] = 0.0
    edge = equilibrium_mapping_thomson_scattering(ods, _fake_geq(), edge_mapping="preserve")
    assert list(edge.region) == ["closed_flux", "unmapped"]


def test_unknown_edge_mapping_is_refused():
    ods, _ = _make_ts_ods()
    with pytest.raises(ValueError, match="edge_mapping"):
        equilibrium_mapping_thomson_scattering(ods, _fake_geq(), edge_mapping="sol")


# --- independent Thomson channel validity (#2335 Part B) ------------------------------


def _report(ods, rho, **kwargs):
    from vaft.process.profile import profile_fit_report_thomson_scattering

    return profile_fit_report_thomson_scattering(
        ods, 300.0, rho, Te_order=2, Ne_order=2, coordinate="psi_norm",
        fitting_function_te="polynomial", fitting_function_ne="polynomial", **kwargs)


def test_independent_validity_keeps_a_channel_the_other_quantity_refuses():
    ods, psi_n = _make_ts_ods()
    ods["thomson_scattering.channel.3.n_e.data_error_upper"] = np.zeros(3)   # dead n_e sigma
    with pytest.warns(DeprecationWarning):
        joint = _report(ods, psi_n)
    with pytest.warns(DeprecationWarning):
        indep = _report(ods, psi_n, channel_validity="independent")
    assert not joint["t_e"].used[3]
    assert "refused with the channel's n_e" in joint["t_e"].rejected_reason[3]
    assert indep["t_e"].used[3] and indep["t_e"].rejected_reason[3] == ""
    assert not indep["n_e"].used[3] and indep["n_e"].rejected_reason[3] == "non-positive sigma"
    # the n_e fit is the same either way: it never had channel 3
    np.testing.assert_array_equal(joint["n_e"].curve, indep["n_e"].curve)
    # T_e gained one channel
    assert indep["t_e"].n_used == joint["t_e"].n_used + 1


def test_independent_validity_rejects_invalid_sigma_per_quantity():
    ods, psi_n = _make_ts_ods()
    ods["thomson_scattering.channel.1.t_e.data_error_upper"] = np.full(3, np.nan)
    ods["thomson_scattering.channel.5.t_e.data_error_upper"] = np.full(3, 1e-6)   # floored
    with pytest.warns(DeprecationWarning):
        indep = _report(ods, psi_n, channel_validity="independent")
    te, ne = indep["t_e"], indep["n_e"]
    assert te.rejected_reason[1] == "non-finite sigma" and ne.used[1]
    assert te.sigma_treatment[1] == "" and ne.sigma_treatment[1] == "measured"
    assert te.sigma_treatment[5].startswith("floored to")
    assert te.sigma_used[5] > te.sigma[5]


def test_unknown_channel_validity_is_refused():
    ods, psi_n = _make_ts_ods()
    with pytest.raises(ValueError, match="channel_validity"):
        _fit(ods, psi_n, channel_validity="either")


# --- separatrix density treatment (#2335 Part B, coordinated with #2144) ---------------


def _edge_ts_ods():
    """n_e = 4e18 (1 - 0.9 x^2): 4e17 at the separatrix, channels out to x = 0.9."""
    ods, _ = _make_ts_ods(n_ch=9)
    x = np.linspace(0.1, 0.9, 9)
    for i, xi in enumerate(x):
        ods[f"thomson_scattering.channel.{i}.n_e.data"] = np.full(3, 4e18 * (1 - 0.9 * xi ** 2))
        ods[f"thomson_scattering.channel.{i}.n_e.data_error_upper"] = np.full(3, 1e17)
    return ods, x


def test_a_nonzero_separatrix_prior_is_honoured_and_kept_apart_from_chi_square():
    from vaft.process.profile import MappedPositions, profile_fit_report_thomson_scattering

    ods, x = _edge_ts_ods()
    pos = MappedPositions(x, np.sqrt(x), None, "t", "t")
    common = dict(Te_order=2, Ne_order=3, coordinate="psi_norm",
                  fitting_function_te="polynomial", fitting_function_ne="free_exponential")
    free = profile_fit_report_thomson_scattering(ods, 300.0, pos, **common)["n_e"]
    prior = profile_fit_report_thomson_scattering(ods, 300.0, pos, ne_separatrix=(4e17, 2e16), **common)["n_e"]
    # free_exponential cannot reach the target exactly against nine channels at
    # sigma 1e17, but the prior pulls the unconstrained 9e17 edge most of the way
    edge = float(prior.function(np.array([1.0]))[0])
    edge_free = float(free.function(np.array([1.0]))[0])
    assert abs(edge - 4e17) < 0.25 * abs(edge_free - 4e17)
    assert prior.n_used == free.n_used                         # the prior is not a channel
    assert "separatrix_prior_penalty" in prior.domains
    assert any("separatrix prior" in a for a in prior.assumptions)


def test_a_nonzero_prior_on_an_edge_zero_basis_is_refused():
    ods, psi_n = _make_ts_ods()
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="zero at x = 1"):
        _fit(ods, psi_n, ne_separatrix=(4e17, 1e17))


def test_the_gp_zero_edge_anchor_is_an_explicit_assumption_and_can_be_dropped():
    from vaft.process.profile import MappedPositions, profile_fit_report_thomson_scattering

    ods, x = _edge_ts_ods()
    pos = MappedPositions(x, np.sqrt(x), None, "t", "t")
    common = dict(Te_order=2, Ne_order=3, coordinate="psi_norm",
                  fitting_function_te="polynomial", fitting_function_ne="gp")
    legacy = profile_fit_report_thomson_scattering(ods, 300.0, pos, **common)["n_e"]
    free = profile_fit_report_thomson_scattering(ods, 300.0, pos, ne_separatrix="none", **common)["n_e"]
    assert any("synthetic GP anchor n_e(1) = 0" in a for a in legacy.assumptions)
    assert not any("anchor n_e(1) = 0" in a for a in free.assumptions)
    # forced to ~0 by the anchor; left near the measured trend without it
    assert float(legacy.function(np.array([1.0]))[0]) < 0.1 * 4e17
    assert float(free.function(np.array([1.0]))[0]) > 0.5 * 4e17


def test_the_density_coefficients_record_their_scale():
    """The n_e coefficients describe n_e / 1e18; the callable is in m^-3 -- recorded, not changed."""
    from vaft.formula.utils import make_fit_function

    ods, psi_n = _make_ts_ods()
    with pytest.warns(DeprecationWarning):
        ne_fit, te_fit, coeffs_ne, *_ = _fit(ods, psi_n)
    assert (ne_fit.unit, ne_fit.coefficient_scale) == ("m^-3", 1e18)
    assert (te_fit.unit, te_fit.coefficient_scale) == ("eV", 1.0)
    x = np.linspace(0.0, 0.9, 7)
    model = make_fit_function("polynomial")(x, *coeffs_ne)
    np.testing.assert_allclose(ne_fit(x), model * ne_fit.coefficient_scale, rtol=1e-12)
    assert ne_fit.status == te_fit.status == "accepted"

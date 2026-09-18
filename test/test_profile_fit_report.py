"""The profile fits say how well they fit and which points they refused (#952).

``profile_fit_report_thomson_scattering`` / ``_charge_exchange`` return one
:class:`vaft.process.profile.FitReport` per fitted quantity: every channel,
its normalised residual ``r_i = (y_i - f(x_i)) / sigma_i``, the chi-square over
the channels the fit used, ``nu = N - k`` and the reason each refused channel
was refused.  The six-tuple the fitters return is unchanged.
"""

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.process.profile import (
    FitReport,
    MappedPositions,
    core_profiles,
    profile_fit_report_charge_exchange,
    profile_fit_report_thomson_scattering,
    profile_fitting_charge_exchange,
    profile_fitting_thomson_scattering,
)

X = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])


def _positions(x):
    x = np.asarray(x, dtype=float)
    return MappedPositions(x, np.sqrt(np.clip(x, 0, 1)), None, "synthetic", "synthetic")


def _thomson(te_sigma=5.0, ne_sigma=1e17):
    """Te = 200 (1 - x)(1 + x), ne = 3e18 (1 - x): exactly the polynomial model."""
    rng = np.random.default_rng(7)
    ods = ODS()
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 1
    ods["thomson_scattering.time"] = np.array([0.299, 0.300])
    te_true = 200.0 * (1 - X) * (1 + X)
    ne_true = 3e18 * (1 - X)
    for i, x in enumerate(X):
        te = te_true[i] + rng.normal(0, te_sigma, 2)
        ne = ne_true[i] + rng.normal(0, ne_sigma, 2)
        ods[f"thomson_scattering.channel.{i}.position.r"] = 0.2 + 0.03 * i
        ods[f"thomson_scattering.channel.{i}.position.z"] = 0.0
        ods[f"thomson_scattering.channel.{i}.t_e.data"] = te
        ods[f"thomson_scattering.channel.{i}.t_e.data_error_upper"] = np.full(2, te_sigma)
        ods[f"thomson_scattering.channel.{i}.n_e.data"] = ne
        ods[f"thomson_scattering.channel.{i}.n_e.data_error_upper"] = np.full(2, ne_sigma)
    return ods


def _report(ods, positions, **kwargs):
    options = dict(Te_order=2, Ne_order=1, fitting_function_te="polynomial",
                   fitting_function_ne="polynomial", coordinate="psi_norm")
    options.update(kwargs)
    return profile_fit_report_thomson_scattering(ods, 300.0, positions, **options)


def test_residuals_chi_squared_and_degrees_of_freedom_follow_their_definitions():
    ods = _thomson()
    reports = _report(ods, _positions(X))
    assert set(reports) == {"t_e", "n_e"}
    te = reports["t_e"]
    assert isinstance(te, FitReport)
    assert te.unit == "eV" and te.coordinate == "psi_norm"

    measured = np.array([ods[f"thomson_scattering.channel.{i}.t_e.data"][1] for i in range(X.size)])
    expected_r = (measured - te.function(X)) / 5.0
    np.testing.assert_allclose(te.normalized_residual, expected_r, rtol=1e-12)
    np.testing.assert_allclose(te.chi_squared, np.sum(expected_r**2), rtol=1e-12)
    # k = the number of coefficients of the order-2 polynomial actually fitted
    assert te.n_parameters == 2
    assert te.degrees_of_freedom == X.size - 2
    np.testing.assert_allclose(te.reduced_chi_squared, te.chi_squared / (X.size - 2))
    # the noise was drawn at the stated sigma, so chi2/nu is of order one
    assert 0.1 < te.reduced_chi_squared < 3.5
    assert te.used.all() and all(reason == "" for reason in te.rejected_reason)
    assert te.time == pytest.approx(0.300)


def test_density_residuals_are_in_the_measured_units_not_the_fit_scale():
    reports = _report(_thomson(), _positions(X))
    ne = reports["n_e"]
    assert ne.unit == "m^-3"
    np.testing.assert_allclose(ne.sigma_used, 1e17)
    assert np.all(np.abs(ne.normalized_residual) < 5)
    assert ne.n_parameters == 1 and ne.degrees_of_freedom == X.size - 1


def test_every_refused_channel_is_named_with_its_reason():
    ods = _thomson()
    ods["thomson_scattering.channel.2.t_e.data"] = np.array([np.nan, np.nan])
    ods["thomson_scattering.channel.4.t_e.data_error_upper"] = np.array([0.0, 0.0])
    ods["thomson_scattering.channel.6.t_e.data_error_upper"] = np.array([np.nan, np.nan])
    x = X.copy()
    x[8] = np.nan  # outside the LCFS: the mapper returns NaN there
    te = _report(ods, _positions(x))["t_e"]

    assert list(np.flatnonzero(~te.used)) == [2, 4, 6, 8]
    assert "non-finite value" in te.rejected_reason[2]
    assert "non-positive sigma" in te.rejected_reason[4]
    assert "non-finite sigma" in te.rejected_reason[6]
    assert "outside" in te.rejected_reason[8]
    assert np.isnan(te.normalized_residual[~te.used]).all()
    assert te.degrees_of_freedom == 6 - 2
    assert set(te.rejected) == {2, 4, 6, 8}


def test_the_report_and_the_six_tuple_are_the_same_fit():
    ods = _thomson()
    positions = _positions(X)
    kwargs = dict(Te_order=2, Ne_order=1, fitting_function_te="polynomial",
                  fitting_function_ne="polynomial", coordinate="psi_norm")
    ne_fit, te_fit, coeffs_ne, coeffs_te, ne_rho, te_rho = profile_fitting_thomson_scattering(
        ods, 300.0, positions, **kwargs
    )
    reports = profile_fit_report_thomson_scattering(ods, 300.0, positions, **kwargs)
    grid = np.linspace(0, 1, 11)
    np.testing.assert_allclose(reports["t_e"].function(grid), te_fit(grid))
    np.testing.assert_allclose(reports["n_e"].function(grid), ne_fit(grid))
    np.testing.assert_allclose(reports["t_e"].coefficients, coeffs_te)


def test_an_order_reduction_is_recorded_as_a_note():
    ods = _thomson()
    # a cubic through points confined to x < 0.3 can dive below zero before the edge
    x = X * 0.3
    te = _report(ods, _positions(x), Te_order=6)["t_e"]
    assert te.order_requested == 6
    assert te.order_used <= 6
    if te.order_used < 6:
        assert any("order" in note for note in te.notes)
    assert te.n_parameters == te.order_used


def test_a_gaussian_process_counts_its_effective_parameters():
    te = _report(_thomson(), _positions(X), fitting_function_te="gp")["t_e"]
    # the trace of the smoother matrix: more than a constant, fewer than the data
    assert 0.5 < te.n_parameters < X.size
    assert te.degrees_of_freedom == pytest.approx(X.size - te.n_parameters)
    assert "trace" in te.parameter_count_definition
    assert te.fitted_std is not None and np.all(np.isfinite(te.fitted_std))


def test_linear_interpolation_leaves_no_degrees_of_freedom():
    te = _report(_thomson(), _positions(X), fitting_function_te="linear")["t_e"]
    assert te.n_parameters == X.size
    assert te.degrees_of_freedom == 0
    assert np.isnan(te.reduced_chi_squared)


def _charge_exchange():
    ods = ODS()
    ods["charge_exchange.time"] = np.array([0.300])
    for i, x in enumerate(X):
        ods[f"charge_exchange.channel.{i}.position.r.data"] = np.array([0.2 + 0.03 * i])
        ods[f"charge_exchange.channel.{i}.position.z.data"] = np.array([0.0])
        ods[f"charge_exchange.channel.{i}.ion.0.t_i.data"] = np.array([50.0 * (1 - x)])
        ods[f"charge_exchange.channel.{i}.ion.0.t_i.data_error_upper"] = np.array([2.0])
        ods[f"charge_exchange.channel.{i}.ion.0.velocity_tor.data"] = np.array([1e4 * (1 - x)])
        ods[f"charge_exchange.channel.{i}.ion.0.velocity_tor.data_error_upper"] = np.array([500.0])
    return ods


def test_charge_exchange_report_matches_its_fitter():
    ods = _charge_exchange()
    ods["charge_exchange.channel.3.ion.0.t_i.data"] = np.array([np.nan])
    positions = _positions(X)
    kwargs = dict(Ti_order=1, Vtor_order=1, coordinate="psi_norm")
    reports = profile_fit_report_charge_exchange(ods, 300.0, positions, **kwargs)
    assert set(reports) == {"t_i", "velocity_tor"}
    ti = reports["t_i"]
    assert ti.unit == "eV" and reports["velocity_tor"].unit == "m/s"
    assert list(np.flatnonzero(~ti.used)) == [3]
    assert reports["velocity_tor"].used.all()
    _v, ti_fit, *_ = profile_fitting_charge_exchange(ods, 300.0, positions, **kwargs)
    np.testing.assert_allclose(ti.function(X), ti_fit(X))
    # the exact model: the residuals vanish to round-off
    np.testing.assert_allclose(ti.normalized_residual[ti.used], 0.0, atol=1e-6)
    assert ti.degrees_of_freedom == 9 - 1


def test_core_profiles_stores_the_electron_uncertainty_and_per_point_chi_squared():
    ods = _thomson()
    positions = MappedPositions(X, np.sqrt(X), X.copy(), None, "synthetic")
    ne_fit, te_fit, *_ = profile_fitting_thomson_scattering(
        ods, 300.0, positions, Te_order=2, Ne_order=1, coordinate="rho_tor_norm"
    )
    core_profiles(ods, 300.0, positions, ne_fit, te_fit, ti_te_fallback=False)
    base = "core_profiles.profiles_1d.0.electrons"
    np.testing.assert_allclose(ods[f"{base}.temperature_fit.measured_error_upper"], 5.0)
    np.testing.assert_allclose(ods[f"{base}.density_fit.measured_error_upper"], 1e17)
    measured = np.asarray(ods[f"{base}.temperature_fit.measured"])
    reconstructed = np.asarray(ods[f"{base}.temperature_fit.reconstructed"])
    np.testing.assert_allclose(
        ods[f"{base}.temperature_fit.chi_squared"], ((measured - reconstructed) / 5.0) ** 2
    )
    report = profile_fit_report_thomson_scattering(
        ods, 300.0, positions, Te_order=2, Ne_order=1, coordinate="rho_tor_norm"
    )["t_e"]
    np.testing.assert_allclose(
        np.sum(ods[f"{base}.temperature_fit.chi_squared"]), report.chi_squared, rtol=1e-10
    )


# ---------------------------------------------------------------------------
# shot 48224 at 300 ms: the packaged kinetic-EFIT inputs (stable loose files)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shot_48224():
    import omas
    from vaft.data import data_path

    ods = omas.load_omas_json(str(data_path("kineticEfit/ods_48224_300ms.json")), consistency_check=False)
    equilibria = {
        "magnetic": data_path("kineticEfit/g048224.00300"),
        "kinetic": data_path("kineticEfit/g048224.00300.kinetic_efit"),
    }
    return ods, equilibria


def test_two_reconstructions_place_the_same_thomson_channels_differently(shot_48224):
    from vaft.process.profile import compare_flux_mapping

    ods, equilibria = shot_48224
    mapped = compare_flux_mapping(ods, equilibria)
    assert list(mapped) == ["magnetic", "kinetic"]
    magnetic, kinetic = mapped["magnetic"], mapped["kinetic"]
    assert magnetic.n_channels == kinetic.n_channels == 7
    both = np.isfinite(magnetic.rho_tor_norm) & np.isfinite(kinetic.rho_tor_norm)
    assert both.sum() >= 5
    shift = kinetic.rho_tor_norm[both] - magnetic.rho_tor_norm[both]
    # the same (R, Z), a different flux map: a nonzero, sub-0.1 shift in rho_tor
    assert 1e-3 < np.max(np.abs(shift)) < 0.1
    # psi_norm moves too, and the ordering of the channels does not
    assert np.max(np.abs(kinetic.psi_norm[both] - magnetic.psi_norm[both])) > 1e-4
    with pytest.raises(ValueError, match="diagnostic"):
        compare_flux_mapping(ods, equilibria, diagnostic="interferometer")


def test_the_48224_thomson_report_names_its_refused_channels(shot_48224):
    from vaft.process.profile import compare_flux_mapping

    ods, equilibria = shot_48224
    mapped = compare_flux_mapping(ods, {"magnetic": equilibria["magnetic"]})["magnetic"]
    reports = profile_fit_report_thomson_scattering(ods, 300.0, mapped, coordinate="psi_norm")
    for report in reports.values():
        assert report.n_used + len(report.rejected) == 7
        assert all(report.rejected_reason[i] for i in report.rejected)
        assert np.isfinite(report.chi_squared) and report.degrees_of_freedom > 0
    # the physicality guard's order reduction is on record, not just printed
    ne = reports["n_e"]
    if ne.order_used < ne.order_requested:
        assert ne.notes and "physicality" in ne.notes[0]

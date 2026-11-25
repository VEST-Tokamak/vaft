"""Issue #2270 P4: legacy spectrogram retirement and the sweep's defects.

* the pre-redesign ``vaft.plot.mirnov.mirnov_spectrogram`` and
  ``vaft.plot.soft_x_rays.plot_soft_x_ray_spectrogram`` are gone; the SXR
  overview's spectrogram panel is the canonical one;
* ``impa_profile_field`` with no ``time=`` takes the middle sample, not the
  offset sample that made every voltage-mode value exactly 0;
* the power balance never matches a slice by its index as seconds, and a
  missing P_Br leaves the "synchrotron" residual unknown instead of 0;
* the equilibrium 2-D map contains cells by the boundary outline, as the
  core-profile maps do since #2253;
* the core-profile volume average replaces a sqrt(psi_N) stand-in radius.
"""

from __future__ import annotations

import copy
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from _sample_fixtures import sample_ods
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def sample():
    return sample_ods(39915)


# --- legacy spectrograms --------------------------------------------------------


def test_the_legacy_spectrograms_are_gone():
    import vaft.plot.mirnov as mirnov
    import vaft.plot.soft_x_rays as sxr

    assert not hasattr(mirnov, "mirnov_spectrogram")
    assert not hasattr(sxr, "plot_soft_x_ray_spectrogram")
    assert "soft_x_rays_spectrogram" not in sxr.__all__
    assert not hasattr(sxr, "scipy_spectrogram")


def test_the_migration_table_does_not_point_at_a_deleted_implementation():
    from vaft.plot._migration import DEPRECATED, LEGACY_MODULES, RETIRED_IMPLEMENTATIONS

    assert "plot_soft_x_ray_spectrogram" in DEPRECATED  # its replacement is still named
    assert not RETIRED_IMPLEMENTATIONS & set(LEGACY_MODULES)


def test_the_sxr_overview_draws_the_canonical_spectrogram(tmp_path):
    from test_soft_x_rays_plot import _tiny_sxr_ods
    from vaft.plot.backend.recipes import SPECTROGRAM_VALUE_SCALES
    from vaft.plot.soft_x_rays import plot_soft_x_ray_overview

    ods = _tiny_sxr_ods(tmp_path)
    figure, axes = plot_soft_x_ray_overview(ods, pattern_channels=[0, 1], show=False)
    assert axes[1, 0].collections
    labels = [axis.get_ylabel() for axis in figure.axes]
    assert any(SPECTROGRAM_VALUE_SCALES["stft"] in label for label in labels)


def test_a_mirnov_record_without_a_time_base_is_not_given_250_khz():
    from vaft.plot.mirnov import _sample_rate_from_time

    for time in (np.array([0.3]), np.array([0.3, 0.3, 0.3])):
        with pytest.raises(ValueError, match="pass sample_rate="):
            _sample_rate_from_time(time)
    assert _sample_rate_from_time(np.arange(5) / 1e6) == pytest.approx(1e6)


# --- IMPA voltage mode -------------------------------------------------------------


def test_the_impa_profile_without_time_is_not_all_zeros(sample):
    from _synthetic_inputs import make_impa_composed

    entries = normalize_entries(make_impa_composed(sample))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = build_model("impa_profile_field", entries)
    assert default.y_unit == "V"  # the packaged shots carry no valid calibration
    assert np.any(default.series[0].y != 0.0)
    assert len(default.series) == 2  # the 1/R shape is drawn again


def test_no_time_is_the_middle_sample_and_the_title_says_so(sample):
    from _synthetic_inputs import make_impa_composed
    from vaft.plot.backend.recipes import resolve_time_sample

    ods = make_impa_composed(sample)
    entries = normalize_entries(ods)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = build_model("impa_profile_field", entries)
        node = "magnetics.b_field_pol_probe"
        axis = next(
            np.asarray(ods[f"{node}.{i}.voltage.time"])
            for i in range(len(ods[node]))
            if "IMPA" in str(ods[f"{node}.{i}.identifier"]).upper()
            and f"{node}.{i}.voltage.time" in ods
        ) if any("IMPA" in str(ods[f"{node}.{i}.identifier"]).upper() for i in range(len(ods[node]))) else None
        if axis is None:
            pytest.skip("IMPA channels are not on b_field_pol_probe in this sample")
        _, middle, _ = resolve_time_sample(axis, None)
        explicit = build_model("impa_profile_field", entries, time=middle)
    np.testing.assert_array_equal(default.series[0].y, explicit.series[0].y)
    assert f"t = {middle * 1e3:.2f} ms" in default.title


# --- power balance -----------------------------------------------------------------


def test_an_untimed_slice_never_matches_and_never_hides_a_real_match():
    from vaft.plot.backend.recipes import _nearest_time_index

    reference = np.array([np.nan, 0.30, 0.31])
    assert _nearest_time_index(reference, 0.31, 1e-4) == 2
    assert _nearest_time_index(reference, 0.0, 1e-4) is None  # index 0 is not 0 s
    assert _nearest_time_index(np.array([np.nan, np.nan]), 0.3, 1e-4) is None


def _fake_power_balance(monkeypatch, *, with_br: bool):
    import vaft.omas.formula_wrapper as wrapper

    # 2.0 s is exactly where the old index-as-seconds fallback put slice 2.
    t = np.array([0.30, 0.31, 2.0])
    result = {
        "time": t,
        "dWdt": np.ones(3),
        "V_ind": np.ones(3),
        "P_heat": np.ones(3),
        "P_ohm_diss": np.ones(3),
        "P_rad": np.full(3, 10.0),
        "P_trans": np.ones(3),
        "P_rad_line": np.full(3, 4.0),
    }
    if with_br:
        result["P_Br"] = np.full(3, 1.0)
    monkeypatch.setattr(wrapper, "compute_power_balance", lambda ods: result)

    def _no_bremsstrahlung(ods, time_slice):
        raise RuntimeError("no profiles")

    monkeypatch.setattr(wrapper, "compute_bremsstrahlung_power", _no_bremsstrahlung)


def _power_balance_ods():
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods["equilibrium.time_slice.0.time"] = 0.30
    ods["equilibrium.time_slice.0.global_quantities.ip"] = 1e5
    ods["equilibrium.time_slice.1.time"] = 0.31
    ods["equilibrium.time_slice.1.global_quantities.ip"] = 2e5
    ods["equilibrium.time_slice.2.global_quantities.ip"] = 3e5  # no time stored
    return ods


def _series(model, panel, label):
    return next(s for s in model.models[panel].series if s.label == label).y


def test_power_balance_matches_by_time_only(monkeypatch):
    _fake_power_balance(monkeypatch, with_br=True)
    model = build_model("summary_time_power_balance", normalize_entries(_power_balance_ods()))
    dw_mag = _series(model, 1, "dW_mag,p/dt")
    # The untimed slice 2 is not "2 s" and matches nothing; the old fallback gave 3e5.
    np.testing.assert_allclose(dw_mag[:2], [1e5, 2e5])
    assert np.isnan(dw_mag[2])
    np.testing.assert_allclose(_series(model, 4, "P_sync"), 10.0 - 4.0 - 1.0)


def test_a_missing_bremsstrahlung_term_leaves_the_residual_unknown(monkeypatch):
    _fake_power_balance(monkeypatch, with_br=False)
    model = build_model("summary_time_power_balance", normalize_entries(_power_balance_ods()))
    assert np.all(np.isnan(_series(model, 4, "P_Br")))
    assert np.all(np.isnan(_series(model, 4, "P_sync")))  # not P_rad - P_line


def test_the_legacy_time_power_balance_does_the_same(monkeypatch):
    import vaft.plot.time as legacy

    _fake_power_balance(monkeypatch, with_br=False)  # imported inside the function
    figure = legacy.time_power_balance(_power_balance_ods())
    axes = figure.axes
    sync = next(line for line in axes[4].get_lines() if "sync" in line.get_label())
    assert np.all(np.isnan(sync.get_ydata()))
    mag = axes[1].get_lines()[0].get_ydata()
    np.testing.assert_allclose(mag[:2], [1e5, 2e5])
    assert np.isnan(mag[2])


# --- closed flux region ------------------------------------------------------------


def test_the_equilibrium_map_uses_the_core_maps_containment_rule(sample):
    """One rule for both 2-D maps: plasma_cell_weights >= 0.5 (#2253, #2270)."""
    from vaft.process.equilibrium import plasma_cell_weights

    base = "equilibrium.time_slice.0"
    r = np.asarray(sample[f"{base}.profiles_2d.0.grid.dim1"])
    z = np.asarray(sample[f"{base}.profiles_2d.0.grid.dim2"])
    psi = np.asarray(sample[f"{base}.profiles_2d.0.psi"])
    axis = float(sample[f"{base}.global_quantities.psi_axis"])
    edge = float(sample[f"{base}.global_quantities.psi_boundary"])
    plasma = plasma_cell_weights(r, z, (psi - axis) / (edge - axis),
                                 sample[f"{base}.boundary.outline.r"],
                                 sample[f"{base}.boundary.outline.z"]) >= 0.5
    model = build_model("equilibrium_field_2d", normalize_entries(sample), field="pressure", time_slice=0)
    # Exactly the plasma cells are filled: the edge-filled projection is finite
    # everywhere, so blank == outside.  The mask is not symmetric (R spans
    # 0.05-1.2 m, Z -1.5-1.5 m), so a transposed mask on this square grid fails.
    assert plasma.any() and not np.array_equal(plasma, plasma.T)
    np.testing.assert_array_equal(np.isfinite(model.values), plasma.T)


def test_the_equilibrium_map_no_longer_paints_the_coil_region(sample):
    base = "equilibrium.time_slice.0"
    r = np.asarray(sample[f"{base}.profiles_2d.0.grid.dim1"])
    z = np.asarray(sample[f"{base}.profiles_2d.0.grid.dim2"])
    psi = np.asarray(sample[f"{base}.profiles_2d.0.psi"])
    axis = float(sample[f"{base}.global_quantities.psi_axis"])
    edge = float(sample[f"{base}.global_quantities.psi_boundary"])
    psi_norm = ((psi - axis) / (edge - axis)).T  # (Z, R), the model's order
    model = build_model("equilibrium_field_2d", normalize_entries(sample), field="pressure", time_slice=0)
    far = np.abs(z)[:, None] > 1.0
    # psi_N <= 1 out there near the coils, and the map is blank there now.
    assert np.any((psi_norm <= 1.0) & far)
    assert not np.any(np.isfinite(model.values) & far)


def test_the_core_profile_map_no_longer_paints_the_coil_region():
    ods = sample_ods(48224)
    model = build_model("electron_temperature_field", normalize_entries(ods))
    z = np.asarray(model.z)
    assert np.isfinite(model.values).any()
    assert not np.any(np.isfinite(model.values)[np.abs(z) > 1.0, :])


# --- volume average radius ---------------------------------------------------------


def test_the_volume_average_replaces_the_stand_in_radius(sample):
    from vaft.data._derived import is_rho_pol_proxy
    from vaft.omas.update import _volume_average_rho_tor_norm
    from vaft.process.profile import equilibrium_rho_tor_norm_table

    profiles = sample["equilibrium.time_slice.3.profiles_1d"]
    stored = np.asarray(profiles["rho_tor_norm"])
    psi_norm, expected = equilibrium_rho_tor_norm_table(sample, 3)
    assert is_rho_pol_proxy(stored, psi_norm)  # the packaged state
    np.testing.assert_array_equal(_volume_average_rho_tor_norm(sample, 3, profiles), expected)


def test_an_authentic_radius_is_used_as_stored(sample):
    from vaft.omas.update import _volume_average_rho_tor_norm
    from vaft.process.profile import equilibrium_rho_tor_norm_table

    ods = copy.deepcopy(sample)
    _, authentic = equilibrium_rho_tor_norm_table(ods, 3)
    ods["equilibrium.time_slice.3.profiles_1d.rho_tor_norm"] = authentic
    result = _volume_average_rho_tor_norm(ods, 3, ods["equilibrium.time_slice.3.profiles_1d"])
    assert result is not None and np.array_equal(result, authentic)


def test_a_stand_in_radius_with_no_q_is_used_with_a_warning(sample, capsys):
    """Without q a sqrt(psi_N) leaf cannot be told from a flat-q radius."""
    from vaft.omas.update import _volume_average_rho_tor_norm

    ods = copy.deepcopy(sample)
    del ods["equilibrium.time_slice.3.profiles_1d.q"]
    stored = np.asarray(ods["equilibrium.time_slice.3.profiles_1d.rho_tor_norm"])
    result = _volume_average_rho_tor_norm(ods, 3, ods["equilibrium.time_slice.3.profiles_1d"])
    np.testing.assert_array_equal(result, stored)
    assert "no q to check it against" in capsys.readouterr().out


def test_the_bremsstrahlung_power_refuses_an_untimed_core_slice():
    from omas import ODS
    from vaft.omas.formula_wrapper import compute_bremsstrahlung_power

    ods = ODS(consistency_check=False)
    ods["core_profiles.profiles_1d.0.electrons.temperature"] = np.ones(3)
    ods["equilibrium.time_slice.0.time"] = 0.0
    with pytest.raises(KeyError, match="has no time"):
        compute_bremsstrahlung_power(ods, time_slice=0)

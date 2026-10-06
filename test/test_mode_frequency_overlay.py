"""Predicted mode-frequency tracks and the Mirnov ``mode_overlay=`` (issue #460).

:func:`vaft.process.mode_frequency.mode_frequency_tracks` locates ``|q| = m/n``
through the #506 resolver, evaluates the toroidal rotation there and returns
``f_pred = n f_phi`` per equilibrium slice; the Mirnov spectrogram only draws
``|f_pred|``.  No packaged sample holds Mirnov, equilibrium and rotation
together, so every input here is an analytic fixture.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from omas import ODS

from vaft.process.mode_frequency import (
    MODE_FREQUENCY_STATUSES,
    ModeFrequencyTracks,
    mode_frequency_tracks,
)

F0 = 5.0e3                     # rotation frequency of the fixture [Hz]
EQ_TIMES = np.round(np.arange(0.300, 0.3105, 0.002), 6)   # 0.300 ... 0.310
ROT_TIMES = (0.301, 0.305, 0.309)
PSI_N = np.linspace(0.0, 1.0, 51)


def r_out(psi_n):
    """Outboard-midplane radius of the fixture's surfaces [m]."""
    return 0.4 + 0.3 * np.asarray(psi_n, dtype=float)


def _equilibrium(ods, *, q=lambda t, psi_n: 1.0 + 3.0 * psi_n, times=EQ_TIMES, r_outboard=True):
    for i, t in enumerate(times):
        base = f"equilibrium.time_slice.{i}"
        ods[f"{base}.time"] = float(t)
        ods[f"{base}.profiles_1d.psi"] = -0.01 * PSI_N          # decreasing: sign is not the normaliser's business
        ods[f"{base}.profiles_1d.q"] = q(t, PSI_N)
        ods[f"{base}.profiles_1d.rho_tor_norm"] = PSI_N.copy()  # an authentic (not sqrt(psi_N)) coordinate
        if r_outboard:
            ods[f"{base}.profiles_1d.r_outboard"] = r_out(PSI_N)
        ods[f"{base}.global_quantities.psi_axis"] = 0.0
        ods[f"{base}.global_quantities.psi_boundary"] = -0.01
    ods["equilibrium.time"] = np.asarray(times, dtype=float)


def _rotation(ods, *, f_phi=lambda t, rho: F0 + 0.0 * rho, times=ROT_TIMES, grid=None,
              leaf="velocity.toroidal", coordinate="rho_tor_norm"):
    """Rotation profiles whose *angular* frequency is ``2 pi f_phi(t, rho)``."""
    rho = np.linspace(0.0, 1.0, 21) if grid is None else np.asarray(grid, dtype=float)
    for k, t in enumerate(times):
        base = f"core_profiles.profiles_1d.{k}"
        ods[f"{base}.time"] = float(t)
        ods[f"{base}.grid.{coordinate}"] = rho
        omega = 2.0 * np.pi * f_phi(t, rho)
        value = omega if leaf == "rotation_frequency_tor" else omega * r_out(rho)  # rho_tor = psi_N here
        ods[f"{base}.ion.0.{leaf}"] = value
    ods["core_profiles.time"] = np.asarray(times, dtype=float)


def _mirnov(ods, frequencies=(F0, 2 * F0), *, start=0.300, stop=0.310, rate=1.0e6):
    t = np.arange(start, stop, 1.0 / rate)
    ods["magnetics.b_field_pol_probe.0.name"] = "MP01"
    ods["magnetics.b_field_pol_probe.0.voltage.time"] = t
    ods["magnetics.b_field_pol_probe.0.voltage.data"] = sum(np.sin(2 * np.pi * f * t) for f in frequencies)


@pytest.fixture
def ods():
    ods = ODS()
    _equilibrium(ods)
    _rotation(ods)
    return ods


def _track(result, m, n, branch=0):
    (match,) = [t for t in result if (t.m, t.n, t.branch) == (m, n, branch)]
    return match


def _bracketed(times):
    return (times >= ROT_TIMES[0]) & (times <= ROT_TIMES[-1])


# --- process -------------------------------------------------------------------


def test_constant_rotation_gives_n_times_f_phi_on_each_surface(ods):
    result = mode_frequency_tracks(ods, [(2, 1), (4, 2), (3, 1)])
    assert isinstance(result, ModeFrequencyTracks) and result.model == "toroidal_rotation"
    assert [(t.m, t.n) for t in result] == [(2, 1), (4, 2), (3, 1)]
    inside = _bracketed(EQ_TIMES)
    for (m, n), f in (((2, 1), F0), ((4, 2), 2 * F0), ((3, 1), F0)):
        track = _track(result, m, n)
        assert np.array_equal(track.valid, inside)
        assert np.allclose(track.predicted_frequency[inside], f, rtol=1e-9)
        assert np.allclose(track.toroidal_rotation_frequency[inside], F0, rtol=1e-9)
    # linear q = 1 + 3 psi_N: q = 2 at psi_N = 1/3, q = 3 at 2/3
    assert np.allclose(_track(result, 2, 1).psi_norm, 1.0 / 3.0)
    assert np.allclose(_track(result, 3, 1).psi_norm, 2.0 / 3.0)
    assert np.allclose(_track(result, 2, 1).r_outboard, r_out(1.0 / 3.0))


def test_harmonics_of_one_ratio_share_the_surface_and_differ_by_n(ods):
    result = mode_frequency_tracks(ods, [(2, 1), (4, 2)])
    one, two = _track(result, 2, 1), _track(result, 4, 2)
    assert one.q == two.q == 2.0
    assert np.array_equal(one.psi_norm, two.psi_norm, equal_nan=True)
    assert np.array_equal(one.toroidal_rotation_frequency, two.toroidal_rotation_frequency, equal_nan=True)
    ok = one.valid
    assert np.allclose(two.predicted_frequency[ok], 2.0 * one.predicted_frequency[ok], rtol=0, atol=0)


def test_outside_the_rotation_time_span_is_a_gap_not_an_extrapolation(ods):
    track = _track(mode_frequency_tracks(ods, [(2, 1)]), 2, 1)
    outside = ~_bracketed(EQ_TIMES)
    assert outside.any()
    assert np.all(np.isnan(track.predicted_frequency[outside]))
    assert {track.status[i] for i in np.flatnonzero(outside)} == {"outside_rotation_time"}


def test_time_pairing_is_by_time_on_unequal_grids():
    ods = ODS()
    _equilibrium(ods)
    # f_phi rises linearly in time between the profiles; the slices fall between them
    _rotation(ods, f_phi=lambda t, rho: F0 * (1.0 + 100.0 * (t - 0.301)) + 0.0 * rho)
    track = _track(mode_frequency_tracks(ods, [(2, 1)]), 2, 1)
    inside = _bracketed(EQ_TIMES)
    expected = F0 * (1.0 + 100.0 * (EQ_TIMES - 0.301))
    assert np.allclose(track.predicted_frequency[inside], expected[inside], rtol=1e-9)
    # profile order in the file does not matter: pairing is by stored time
    reversed_ods = ODS()
    _equilibrium(reversed_ods)
    _rotation(reversed_ods, f_phi=lambda t, rho: F0 * (1.0 + 100.0 * (t - 0.301)) + 0.0 * rho,
              times=ROT_TIMES[::-1])
    again = _track(mode_frequency_tracks(reversed_ods, [(2, 1)]), 2, 1)
    assert np.allclose(again.predicted_frequency, track.predicted_frequency, equal_nan=True)


def test_a_time_tolerance_pairs_with_the_nearest_profile_only_within_it():
    ods = ODS()
    _equilibrium(ods)
    _rotation(ods, times=(0.3051,))
    strict = _track(mode_frequency_tracks(ods, [(2, 1)]), 2, 1)
    assert not strict.valid.any()
    loose = _track(mode_frequency_tracks(ods, [(2, 1)], time_tolerance=1.5e-3), 2, 1)
    assert np.array_equal(loose.valid, np.abs(EQ_TIMES - 0.3051) <= 1.5e-3)
    assert np.allclose(loose.predicted_frequency[loose.valid], F0)


def test_the_rotation_grid_bounds_the_radius_no_radial_extrapolation():
    ods = ODS()
    _equilibrium(ods)
    _rotation(ods, grid=np.linspace(0.0, 0.5, 11))
    result = mode_frequency_tracks(ods, [(2, 1), (3, 1)])
    inside = _bracketed(EQ_TIMES)
    assert np.array_equal(_track(result, 2, 1).valid, inside)       # psi_N = 1/3 is covered
    outer = _track(result, 3, 1)                                     # psi_N = 2/3 is not
    assert not outer.valid.any()
    assert {outer.status[i] for i in np.flatnonzero(inside)} == {"outside_rotation_radius"}


def test_reversed_shear_gives_two_branches_never_one():
    ods = ODS()
    _equilibrium(ods, q=lambda t, psi_n: 3.0 - 6.0 * psi_n * (1.0 - psi_n))
    # an angular frequency linear in rho: the radial interpolation is exact
    _rotation(ods, f_phi=lambda t, rho: F0 * (1.0 - 0.5 * rho), leaf="rotation_frequency_tor")
    result = mode_frequency_tracks(ods, [(2, 1)])
    assert [t.branch for t in result] == [0, 1]
    inner, outer = _track(result, 2, 1, 0), _track(result, 2, 1, 1)
    roots = (0.5 - np.sqrt(1.0 - 4.0 / 6.0) / 2.0, 0.5 + np.sqrt(1.0 - 4.0 / 6.0) / 2.0)
    ok = inner.valid
    assert ok.any() and np.array_equal(ok, outer.valid)
    for track, root in zip((inner, outer), roots):
        assert np.allclose(track.psi_norm, root, atol=2e-3)
        assert np.allclose(track.predicted_frequency[ok], F0 * (1.0 - 0.5 * track.rho_tor_norm[ok]), rtol=1e-9)
    assert np.all(inner.predicted_frequency[ok] > outer.predicted_frequency[ok])


def test_slices_where_the_surface_is_absent_are_gaps():
    ods = ODS()
    # q_axis rises through time: q = 2 exists only while q(0) = 1 + 300 (t - 0.300) < 2
    _equilibrium(ods, q=lambda t, psi_n: 1.0 + 300.0 * (t - 0.300) + 2.0 * psi_n)
    _rotation(ods)
    track = _track(mode_frequency_tracks(ods, [(2, 1)]), 2, 1)
    q_axis = 1.0 + 300.0 * (EQ_TIMES - 0.300)
    absent = q_axis > 2.0
    assert absent.any() and (~absent & _bracketed(EQ_TIMES)).any()
    assert all(track.status[i] == "no_surface" for i in np.flatnonzero(absent))
    assert np.all(np.isnan(track.predicted_frequency[absent]))
    assert np.all(track.valid[~absent & _bracketed(EQ_TIMES)])


def test_a_velocity_is_divided_by_the_outboard_radius_of_the_surface():
    ods = ODS()
    _equilibrium(ods)
    for k, t in enumerate(ROT_TIMES):
        ods[f"core_profiles.profiles_1d.{k}.time"] = t
        ods[f"core_profiles.profiles_1d.{k}.grid.rho_tor_norm"] = np.linspace(0.0, 1.0, 21)
        ods[f"core_profiles.profiles_1d.{k}.ion.0.velocity.toroidal"] = np.full(21, 1.0e4)
    track = _track(mode_frequency_tracks(ods, [(2, 1)]), 2, 1)
    ok = track.valid
    assert np.allclose(track.toroidal_rotation_frequency[ok], 1.0e4 / (2 * np.pi * r_out(1.0 / 3.0)))
    # never R_axis
    assert not np.allclose(track.toroidal_rotation_frequency[ok], 1.0e4 / (2 * np.pi * r_out(0.0)))


def test_a_velocity_without_an_outboard_radius_is_a_gap():
    ods = ODS()
    _equilibrium(ods, r_outboard=False)
    _rotation(ods)
    track = _track(mode_frequency_tracks(ods, [(2, 1)]), 2, 1)
    assert not track.valid.any()
    assert "no_major_radius" in track.status


def test_a_stored_angular_frequency_is_used_without_a_radius():
    ods = ODS()
    _equilibrium(ods, r_outboard=False)
    _rotation(ods, leaf="rotation_frequency_tor")
    result = mode_frequency_tracks(ods, [(2, 1)])
    track = _track(result, 2, 1)
    assert np.allclose(track.predicted_frequency[track.valid], F0)
    assert result.provenance["toroidal_rotation"]["source"] == [
        "core_profiles.profiles_1d.{i}.ion.0.rotation_frequency_tor"]


def test_a_rho_pol_grid_is_evaluated_at_the_root_rho_pol():
    ods = ODS()
    _equilibrium(ods)
    _rotation(ods, f_phi=lambda t, rho: F0 * (1.0 + rho), leaf="rotation_frequency_tor",
              coordinate="rho_pol_norm")
    track = _track(mode_frequency_tracks(ods, [(2, 1)]), 2, 1)
    assert np.allclose(track.predicted_frequency[track.valid], F0 * (1.0 + np.sqrt(1.0 / 3.0)))


def test_signs_follow_the_rotation_and_n():
    ods = ODS()
    _equilibrium(ods)
    _rotation(ods, f_phi=lambda t, rho: -F0 + 0.0 * rho)
    result = mode_frequency_tracks(ods, [(2, 1), (-2, -1), (2, -1)])
    ok = _track(result, 2, 1).valid
    assert np.allclose(_track(result, 2, 1).predicted_frequency[ok], -F0)
    assert np.allclose(_track(result, -2, -1).predicted_frequency[ok], F0)
    assert np.allclose(_track(result, 2, -1).predicted_frequency[ok], F0)
    # all three resonate on |q| = 2
    assert {t.q for t in result} == {2.0}


def test_every_status_is_in_the_vocabulary(ods):
    for track in mode_frequency_tracks(ods, [(2, 1), (7, 1)]):
        assert set(track.status) <= set(MODE_FREQUENCY_STATUSES)


def test_provenance_names_model_sources_and_policy(ods):
    provenance = mode_frequency_tracks(ods, [(2, 1)]).provenance
    assert provenance["model"] == "toroidal_rotation"
    assert provenance["rational_surface_resolver"] == "vaft.process.equilibrium.rational_surfaces"
    assert provenance["toroidal_rotation"]["source"] == ["core_profiles.profiles_1d.{i}.ion.0.velocity.toroidal"]
    assert "R_out" in provenance["toroidal_rotation"]["velocity_to_frequency"]
    assert "not mode identification" in provenance["interpretation"]


@pytest.mark.parametrize("kwargs, match", [
    ({"modes": [(2, 1)], "model": "exb_rotation"}, "model="),
    ({"modes": []}, "empty"),
    ({"modes": [(2, 0)]}, "undefined"),
    ({"modes": [(2.5, 1)]}, "whole"),
    ({"modes": "2/1"}, "pairs"),
    ({"modes": [(2, 1)], "time_tolerance": -1.0}, "time_tolerance"),
])
def test_bad_requests_are_refused(ods, kwargs, match):
    with pytest.raises(ValueError, match=match):
        mode_frequency_tracks(ods, **kwargs)


def test_missing_inputs_raise():
    only_eq = ODS()
    _equilibrium(only_eq)
    with pytest.raises(ValueError, match="toroidal rotation"):
        mode_frequency_tracks(only_eq, [(2, 1)])
    only_rotation = ODS()
    _rotation(only_rotation)
    with pytest.raises(ValueError, match="equilibrium"):
        mode_frequency_tracks(only_rotation, [(2, 1)])


def test_a_bare_pair_is_one_mode(ods):
    assert [(t.m, t.n) for t in mode_frequency_tracks(ods, (2, 1))] == [(2, 1)]


# --- view ----------------------------------------------------------------------

import vaft.omas as vo  # noqa: E402
from vaft.plot.backend import recipes as R  # noqa: E402
from vaft.plot.backend.options import validate_options  # noqa: E402


@pytest.fixture
def shot():
    ods = ODS()
    _equilibrium(ods)
    _rotation(ods)
    _mirnov(ods)
    return ods


def test_the_overlay_draws_one_line_per_mode_at_its_predicted_frequency(shot):
    model = vo.extract_mirnov_spectrogram(shot, mode_overlay=[(2, 1), (4, 2)])
    assert [t.label for t in model.tracks] == ["2/1 (q = 2): 1 × f_φ", "4/2 (q = 2): 2 × f_φ"]
    assert "toroidal_rotation" in model.tracks_title
    one, two = model.tracks
    inside = _bracketed(one.time)
    assert np.allclose(one.frequency[inside], F0) and np.allclose(two.frequency[inside], 2 * F0)
    assert np.all(np.isnan(one.frequency[~inside]))
    assert one.style["color"] != two.style["color"]
    # each predicted line runs along a bright ridge of the map it is drawn on
    for track in model.tracks:
        row = int(np.argmin(np.abs(model.frequency - track.frequency[inside][0])))
        column = model.magnitude[:, model.magnitude.shape[1] // 2]
        assert column[row] == pytest.approx(column.max(), rel=0.5)
    record = model.metadata["mode_overlay"]
    assert record["model"] == "toroidal_rotation" and record["modes"] == [[2, 1], [4, 2]]
    assert [t["drawn"] for t in record["tracks"]] == [True, True]


def test_a_reversed_shear_surface_draws_two_branches_in_one_colour():
    ods = ODS()
    _equilibrium(ods, q=lambda t, psi_n: 3.0 - 6.0 * psi_n * (1.0 - psi_n))
    _rotation(ods, f_phi=lambda t, rho: F0 * (1.0 - 0.5 * rho))
    _mirnov(ods)
    model = vo.extract_mirnov_spectrogram(ods, mode_overlay=[(2, 1)])
    assert [t.label for t in model.tracks] == ["2/1 (q = 2): 1 × f_φ, root 1", "2/1 (q = 2): 1 × f_φ, root 2"]
    assert model.tracks[0].style["color"] == model.tracks[1].style["color"]
    assert model.tracks[0].style["linestyle"] != model.tracks[1].style["linestyle"]


def test_negative_rotation_is_drawn_as_magnitude():
    ods = ODS()
    _equilibrium(ods)
    _rotation(ods, f_phi=lambda t, rho: -F0 + 0.0 * rho)
    _mirnov(ods)
    model = vo.extract_mirnov_spectrogram(ods, mode_overlay=[(2, 1)])
    (track,) = model.tracks
    assert np.allclose(track.frequency[np.isfinite(track.frequency)], F0)
    assert model.metadata["mode_overlay"]["tracks"][0]["predicted_frequency"][1] == pytest.approx(-F0)


def test_without_the_option_the_spectrogram_is_unchanged(shot):
    plain = vo.extract_mirnov_spectrogram(shot)
    base = R._build_spectrogram(shot, R.RECIPES["mirnov_spectrogram"])
    assert plain.tracks == () and plain.metadata == {} and plain.tracks_title == ""
    assert plain.to_xarray().identical(base.to_xarray())
    with_overlay = vo.extract_mirnov_spectrogram(shot, mode_overlay=[(2, 1)])
    assert np.array_equal(with_overlay.magnitude, plain.magnitude)
    assert np.array_equal(with_overlay.time, plain.time) and np.array_equal(with_overlay.frequency, plain.frequency)


def test_the_rendered_figure_adds_only_the_tracks(shot):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, axes = vo.plot_mirnov_spectrogram(shot)
    assert not axes.lines and axes.get_legend() is None
    _, axes2 = vo.plot_mirnov_spectrogram(shot, mode_overlay=[(2, 1), (4, 2)])
    assert [line.get_label() for line in axes2.lines] == ["2/1 (q = 2): 1 × f_φ", "4/2 (q = 2): 2 × f_φ"]
    assert "toroidal_rotation" in axes2.get_legend().get_title().get_text()
    assert len(axes2.collections) == len(axes.collections)
    plt.close("all")


def test_extract_equals_plot(shot):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = vo.extract_mirnov_spectrogram(shot, mode_overlay=[(2, 1)])
    _, axes = vo.plot_mirnov_spectrogram(shot, mode_overlay=[(2, 1)])
    (line,) = axes.lines
    assert np.array_equal(line.get_ydata(), model.tracks[0].frequency, equal_nan=True)
    dataset = model.to_xarray()
    assert dataset["track_frequency"].shape[0] == 1 and "mode_overlay" in dataset.attrs["metadata"]
    plt.close("all")


def test_missing_rotation_warns_and_draws_the_base_spectrogram():
    ods = ODS()
    _equilibrium(ods)
    _mirnov(ods)
    with pytest.warns(UserWarning, match="mode_overlay is not drawn: .*toroidal rotation"):
        model = vo.extract_mirnov_spectrogram(ods, mode_overlay=[(2, 1)])
    assert model.tracks == () and model.metadata == {}


def test_missing_equilibrium_warns_and_draws_the_base_spectrogram():
    ods = ODS()
    _rotation(ods)
    _mirnov(ods)
    with pytest.warns(UserWarning, match="mode_overlay is not drawn: .*equilibrium"):
        model = vo.extract_mirnov_spectrogram(ods, mode_overlay=[(2, 1)])
    assert model.tracks == ()


def test_an_absent_surface_warns_and_draws_no_line(shot):
    with pytest.warns(UserWarning, match=r"7/1: \|q\| = 7 does not occur"):
        model = vo.extract_mirnov_spectrogram(shot, mode_overlay=[(2, 1), (7, 1)])
    assert [t.label for t in model.tracks] == ["2/1 (q = 2): 1 × f_φ"]
    assert [t["drawn"] for t in model.metadata["mode_overlay"]["tracks"]] == [True, False]


def test_a_window_without_prediction_warns_with_the_reasons(shot):
    with pytest.warns(UserWarning, match="outside_spectrogram_window"):
        model = vo.extract_mirnov_spectrogram(shot, mode_overlay=[(2, 1)], time_range=(0.3092, 0.3099))
    assert model.tracks == ()


def test_the_warning_points_at_the_caller():
    ods = ODS()
    _equilibrium(ods)
    _mirnov(ods)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vo.extract_mirnov_spectrogram(ods, mode_overlay=[(2, 1)])
    (warning,) = [w for w in caught if "mode_overlay" in str(w.message)]
    assert warning.filename == __file__


def test_a_malformed_request_is_an_error_not_a_warning(shot):
    with pytest.raises(ValueError, match="mode_overlay="):
        vo.extract_mirnov_spectrogram(shot, mode_overlay=[(2, 0)])


@pytest.mark.parametrize("name", ["soft_x_rays_spectrogram", "interferometer_spectrogram",
                                  "mirnov_spectrum", "equilibrium_profile_q"])
def test_a_plot_that_does_not_declare_the_option_refuses_it(name):
    with pytest.raises(ValueError, match="does not take an option named 'mode_overlay'"):
        validate_options(name, {"mode_overlay": [(2, 1)]})


def test_the_mirnov_spectrogram_declares_the_option():
    for name in R.MODE_OVERLAY_PLOTS:
        validate_options(name, {"mode_overlay": [(2, 1)]})
        assert R.declares_option(name, "mode_overlay")


def test_discovery_lists_the_overlay_its_model_and_its_reads(shot):
    catalog = {record.name: record for record in vo.available_plots(shot)}
    annotation = catalog["mirnov_spectrogram"].annotations["mode_overlay"]
    assert annotation["options"] == ("mode_overlay",) and annotation["model"] == "toroidal_rotation"
    for leaf in ("equilibrium.time_slice.{i}.profiles_1d.q", "equilibrium.time_slice.{i}.profiles_1d.r_outboard",
                 "core_profiles.profiles_1d.{i}.ion.{j}.velocity.toroidal",
                 "core_profiles.profiles_1d.{i}.grid.rho_tor_norm"):
        assert leaf in annotation["reads"]


def test_the_base_spectrogram_needs_none_of_the_overlay_inputs():
    ods = ODS()
    _mirnov(ods)
    catalog = {record.name: record for record in vo.available_plots(ods)}
    assert "mirnov_spectrogram" in catalog
    model = vo.extract_mirnov_spectrogram(ods)
    assert model.tracks == ()

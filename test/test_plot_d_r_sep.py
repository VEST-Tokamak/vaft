"""Signed dRsep in vaft.plot: the time history and the cross-section (#2325 phase F).

The plot layer draws :func:`vaft.process.equilibrium.separatrix_balance_series`
and :func:`derive_boundary_representation` and computes nothing of its own.
The unit tests below replace the series with hand-made records so every state
sits at a known time; the integration tests run the real analysis on the
analytic family of ``test_separatrix_balance`` written into ODSs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import omas  # noqa: E402
import pytest  # noqa: E402

import vaft.omas as vaft_omas  # noqa: E402
import vaft.process.equilibrium as process_equilibrium  # noqa: E402
from vaft.data.equilibrium import SeparatrixBalanceSeries  # noqa: E402
from vaft.plot.backend.options import validate_options  # noqa: E402
from vaft.plot.backend.recipes import (  # noqa: E402
    D_R_SEP_CHANNEL,
    D_R_SEP_LIMITED_CHANNEL,
    D_R_SEP_NATIVE_CHANNEL,
    POLOIDAL_OVERLAYS,
)
from vaft.plot.equilibrium_section import build_equilibrium_section  # noqa: E402
from vaft.process.equilibrium import extract_flux_surface_contours  # noqa: E402

PLOT = "equilibrium_time_d_r_sep"


# -- hand-made series ------------------------------------------------------------


def _series(time, states, values, native=None):
    """A record whose d_r_sep is a number only where the state is valid."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    states = tuple(states)
    d_r_sep = np.where(np.array(states) == "valid", values, np.nan)
    # The limited slices keep a geometric value: the plot must never draw it.
    geometric = np.where(np.array(states) == "limited", -0.03, np.nan)
    native = np.full(time.shape, np.nan) if native is None else np.asarray(native, dtype=float)
    return SeparatrixBalanceSeries(
        time=time, d_r_sep=d_r_sep, state=states,
        topology=tuple("limited" if s == "limited" else None for s in states),
        reason=(None,) * time.size, resolution=np.full(time.shape, 1e-4),
        geometric_d_r_sep=geometric, native_d_r_sep=native,
        native_configuration=(None,) * time.size,
    )


#: Entry A: every state, a valid sample between two gaps, and a genuine zero.
SERIES_A = _series(
    [0.300, 0.302, 0.304, 0.306, 0.308, 0.310, 0.312, 0.314, 0.316],
    ["valid", "valid", "limited", "valid", "unavailable", "valid", "ambiguous", "valid", "no_plasma"],
    [-0.004, -0.002, 0.0, 0.0, 0.0, 0.003, 0.0, 0.001, 0.0],
    native=[np.nan, -0.0025, np.nan, 0.0005, np.nan, 0.004, np.nan, np.nan, np.nan],
)
#: Entry B: its own time base, a failed slice and a limited one.
SERIES_B = _series(
    [0.301, 0.305, 0.309, 0.313],
    ["limited", "valid", "failed", "valid"],
    [0.0, 0.006, 0.0, 0.007],
)


def _stub_ods(pulse):
    """An ODS that passes the plot's availability test; the series is stubbed."""
    ods = omas.ODS()
    ods["dataset_description.data_entry.pulse"] = pulse
    ods["equilibrium.time"] = np.array([0.3])
    ods["equilibrium.time_slice.0.time"] = 0.3
    ods["equilibrium.time_slice.0.profiles_2d.0.psi"] = np.zeros((3, 3))
    return ods


@pytest.fixture
def stubbed(monkeypatch):
    records = {}

    def fake(source, **_):
        return records[int(source["dataset_description.data_entry.pulse"])]

    monkeypatch.setattr(process_equilibrium, "separatrix_balance_series", fake)
    return records


def _by_channel(model, entry=None):
    return {
        series.channel: series for series in model.series
        if series.channel and (entry is None or series.entry == entry)
    }


def test_valid_samples_are_drawn_signed_and_every_other_state_is_a_gap(stubbed):
    stubbed[1] = SERIES_A
    model = vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1))
    trace = _by_channel(model)[D_R_SEP_CHANNEL]
    np.testing.assert_array_equal(trace.x, SERIES_A.time)
    valid = np.array(SERIES_A.state) == "valid"
    np.testing.assert_allclose(trace.y[valid], SERIES_A.d_r_sep[valid])
    # A gap is NaN, never zero: limited, unavailable, ambiguous, no-plasma.
    assert np.isnan(trace.y[~valid]).all()
    # A genuinely balanced valid sample stays a number at zero.
    assert trace.y[3] == 0.0 and valid[3]


def test_limited_slices_are_categorical_markers_at_their_own_times(stubbed):
    stubbed[1] = SERIES_A
    model = vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1))
    limited = _by_channel(model)[D_R_SEP_LIMITED_CHANNEL]
    np.testing.assert_array_equal(limited.x, [0.304])
    np.testing.assert_array_equal(limited.y, [0.0])
    assert limited.style["marker"] == "x" and limited.style["linestyle"] == "none"
    # The geometric value of the limited slice is drawn nowhere.
    drawn = np.concatenate([series.y for series in model.series])
    assert not np.any(np.isclose(drawn, -0.03))
    # Nothing else marks the limited time on the value trace.
    assert np.isnan(_by_channel(model)[D_R_SEP_CHANNEL].y[2])


def test_no_line_joins_across_an_invalid_interval(stubbed):
    stubbed[1] = SERIES_A
    figure, axes = vaft_omas.plot_equilibrium_time_d_r_sep(_stub_ods(1))
    line = next(line for line in axes.lines if line.get_label() == D_R_SEP_CHANNEL)
    y = np.asarray(line.get_ydata(), dtype=float)
    finite = np.isfinite(y)
    # Matplotlib breaks a line at NaN: the drawn runs are exactly the
    # consecutive valid slices.
    runs = np.flatnonzero(np.diff(np.r_[0, finite.astype(int), 0]) == 1)
    assert runs.size == 4  # [0.300, 0.302], [0.306], [0.310], [0.314]
    assert line.get_marker() == "."  # an isolated valid sample stays visible
    plt.close(figure)


def test_limited_and_valid_zero_have_distinct_legend_entries(stubbed):
    stubbed[1] = SERIES_A
    figure, axes = vaft_omas.plot_equilibrium_time_d_r_sep(_stub_ods(1))
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert D_R_SEP_CHANNEL in labels and D_R_SEP_LIMITED_CHANNEL in labels
    plt.close(figure)


def test_entries_keep_their_own_time_bases(stubbed):
    stubbed[1], stubbed[2] = SERIES_A, SERIES_B
    model = vaft_omas.extract_equilibrium_time_d_r_sep(
        [_stub_ods(1), _stub_ods(2)], label=["A", "B"], xunit="ms")
    a, b = _by_channel(model, "A"), _by_channel(model, "B")
    np.testing.assert_allclose(a[D_R_SEP_CHANNEL].x, SERIES_A.time * 1e3)
    np.testing.assert_allclose(b[D_R_SEP_CHANNEL].x, SERIES_B.time * 1e3)
    np.testing.assert_allclose(b[D_R_SEP_LIMITED_CHANNEL].x, [301.0])
    assert np.isnan(b[D_R_SEP_CHANNEL].y[[0, 2]]).all()
    assert a[D_R_SEP_CHANNEL].style["color"] != b[D_R_SEP_CHANNEL].style["color"]
    assert model.x_unit == "ms"
    assert model.x_limits == pytest.approx((300.0, 316.0))


def test_units_and_the_zero_reference(stubbed):
    stubbed[1] = SERIES_A
    metres = vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1))
    millimetres = vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1), yunit="mm")
    assert metres.y_unit == "m" and millimetres.y_unit == "mm"
    np.testing.assert_allclose(
        _by_channel(millimetres)[D_R_SEP_CHANNEL].y, _by_channel(metres)[D_R_SEP_CHANNEL].y * 1e3)
    zero = [series for series in metres.series if not series.label and not series.channel]
    assert len(zero) == 1 and np.all(zero[0].y == 0.0)
    with pytest.raises(ValueError, match="unsupported display unit"):
        vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1), yunit="km")


def test_the_dn_band_appears_only_when_requested(stubbed):
    stubbed[1] = SERIES_A
    plain = vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1), yunit="mm")
    assert not any("tolerance" in series.label for series in plain.series)
    assert all(series.yerr is None for series in plain.series)
    banded = vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1), yunit="mm", dn_tolerance=0.002)
    band = next(series for series in banded.series if "tolerance" in series.label)
    assert band.label == "DN tolerance ±2 mm"
    assert np.nanmax(band.y) == pytest.approx(2.0) and np.nanmin(band.y) == pytest.approx(-2.0)
    for bad in (0.0, -1e-3, np.nan, "wide"):
        with pytest.raises(ValueError, match="dn_tolerance"):
            vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1), dn_tolerance=bad)


def test_native_overlay_is_optional_and_kept_apart(stubbed):
    stubbed[1] = SERIES_A
    assert D_R_SEP_NATIVE_CHANNEL not in _by_channel(
        vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1)))
    model = vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1), native_overlay=True)
    native = _by_channel(model)[D_R_SEP_NATIVE_CHANNEL]
    np.testing.assert_allclose(native.y, SERIES_A.native_d_r_sep, equal_nan=True)
    # The derived trace is not merged with the native one.
    np.testing.assert_allclose(_by_channel(model)[D_R_SEP_CHANNEL].y[1], -0.002)


def test_time_range_windows_every_trace(stubbed):
    stubbed[1] = SERIES_A
    model = vaft_omas.extract_equilibrium_time_d_r_sep(_stub_ods(1), time_range=(0.303, 0.311))
    channels = _by_channel(model)
    np.testing.assert_allclose(channels[D_R_SEP_CHANNEL].x, [0.304, 0.306, 0.308, 0.310])
    np.testing.assert_allclose(channels[D_R_SEP_LIMITED_CHANNEL].x, [0.304])
    assert model.x_limits == (0.303, 0.311)


def test_the_options_belong_to_the_d_r_sep_plot_only():
    validate_options(PLOT, {"dn_tolerance": 0.002, "native_overlay": True})
    for option in ("dn_tolerance", "native_overlay"):
        with pytest.raises(ValueError, match=option):
            validate_options("equilibrium_time_q95", {option: 1})


# -- the real analysis on analytic equilibria ---------------------------------------

A = 0.65
R = np.linspace(0.5, 1.5, 121)
Z = np.linspace(-0.8, 0.8, 161)


def _saddle_fluxes(skew):
    roots = sorted(np.roots([4.0 / A**2, -3.0 * skew, -2.0]))
    return [float(zz**2 - zz**4 / A**2 + skew * zz**3) for zz in roots]  # lower, upper


def _put_slice(ods, index, time, skew, *, fraction=1.0, ip=1e5):
    """psi = (R-1)^2 + z^2 - z^4/a^2 + skew z^3, boundary at ``fraction`` of the primary saddle flux."""
    rm, zm = np.meshgrid(R, Z, indexing="ij")
    psi = (rm - 1) ** 2 + zm**2 - zm**4 / A**2 + skew * zm**3
    psi_lower, psi_upper = _saddle_fluxes(skew)
    boundary = min(psi_lower, psi_upper) * fraction
    rb, zb = max(extract_flux_surface_contours(psi, R, Z, 0.0, boundary, [1.0])[1.0], key=lambda c: c[0].size)
    base = f"equilibrium.time_slice.{index}"
    ods[f"{base}.time"] = time
    ods[f"{base}.profiles_2d.0.grid.dim1"] = R
    ods[f"{base}.profiles_2d.0.grid.dim2"] = Z
    ods[f"{base}.profiles_2d.0.psi"] = psi
    ods[f"{base}.global_quantities.psi_axis"] = 0.0
    ods[f"{base}.global_quantities.psi_boundary"] = boundary
    ods[f"{base}.global_quantities.ip"] = ip
    ods[f"{base}.global_quantities.magnetic_axis.r"] = 1.0
    ods[f"{base}.global_quantities.magnetic_axis.z"] = 0.0
    ods[f"{base}.boundary.outline.r"] = rb
    ods[f"{base}.boundary.outline.z"] = zb
    return np.sqrt(psi_lower) - np.sqrt(psi_upper), float(np.max(rb))


@pytest.fixture(scope="module")
def diverted():
    """LSN, USN and a no-plasma slice, as one discharge."""
    ods = omas.ODS()
    ods["dataset_description.data_entry.pulse"] = 1
    exact = [_put_slice(ods, 0, 0.30, 0.08)[0], _put_slice(ods, 1, 0.31, -0.08)[0]]
    _put_slice(ods, 2, 0.32, 0.08, ip=0.0)
    ods["equilibrium.time"] = np.array([0.30, 0.31, 0.32])
    return ods, exact


@pytest.fixture(scope="module")
def limited():
    """A boundary inset from its X-point and resting on an outboard limiter."""
    ods = omas.ODS()
    ods["dataset_description.data_entry.pulse"] = 2
    _, r_touch = _put_slice(ods, 0, 0.305, 0.08, fraction=0.6)
    ods["equilibrium.time"] = np.array([0.305])
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = r_touch + 0.2 * (np.cos(theta) - 1.0)
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = 0.6 * np.sin(theta)
    return ods


def test_real_analysis_draws_signed_values_markers_and_never_writes_the_input(diverted, limited):
    ods, exact = diverted
    before = {key: np.array(value, copy=True) for key, value in ods.flat().items()}
    before_limited = set(limited.flat())
    model = vaft_omas.extract_equilibrium_time_d_r_sep([ods, limited], label=["div", "lim"], yunit="mm")
    after = ods.flat()
    assert set(after) == set(before)
    for key, value in before.items():
        np.testing.assert_array_equal(np.asarray(after[key]), value, err_msg=key)
    assert set(limited.flat()) == before_limited

    div, lim = _by_channel(model, "div"), _by_channel(model, "lim")
    trace = div[D_R_SEP_CHANNEL]
    np.testing.assert_allclose(trace.x, [0.30, 0.31, 0.32])
    np.testing.assert_allclose(trace.y[:2], np.array(exact) * 1e3, atol=0.2)
    assert trace.y[0] < 0 < trace.y[1]  # LSN negative, USN positive
    assert np.isnan(trace.y[2])  # no plasma: a gap
    assert D_R_SEP_LIMITED_CHANNEL not in div
    np.testing.assert_allclose(lim[D_R_SEP_LIMITED_CHANNEL].x, [0.305])
    assert np.isnan(lim[D_R_SEP_CHANNEL].y).all()  # its geometric value is not drawn


def test_section_shows_separatrices_roles_and_midplane_crossings(diverted):
    ods, exact = diverted
    before = set(ods.flat())
    section = build_equilibrium_section(ods, time_slice=0, flux_surface_levels=(), separatrix_balance=True)
    labels = {layer.label: layer for layer in section.rz.layers}
    assert {"Primary separatrix (lower X-point)", "Secondary separatrix (upper X-point)",
            "Lower X-point (primary)", "Upper X-point (secondary)"} <= set(labels)
    reference = next(layer for label, layer in labels.items() if label.startswith("OMP reference"))
    np.testing.assert_allclose(reference.z, [0.0, 0.0])
    assert f"dRsep = {exact[0] * 1e3:+.1f} mm" in reference.label
    r_u = next(layer for label, layer in labels.items() if label.startswith("R_U"))
    r_l = next(layer for label, layer in labels.items() if label.startswith("R_L"))
    psi_lower, psi_upper = _saddle_fluxes(0.08)
    assert float(r_u.r[0]) == pytest.approx(1.0 + np.sqrt(psi_upper), abs=1e-3)
    assert float(r_l.r[0]) == pytest.approx(1.0 + np.sqrt(psi_lower), abs=1e-3)
    assert float(r_l.r[0]) - float(r_u.r[0]) == pytest.approx(exact[0], abs=2e-4)
    # Off by default: the section is unchanged unless asked.
    plain = build_equilibrium_section(ods, time_slice=0, flux_surface_levels=())
    assert not any("separatrix" in layer.label for layer in plain.rz.layers)
    assert set(ods.flat()) == before


def test_flux_map_overlay_selects_the_slice(diverted):
    ods, _ = diverted
    assert "d_r_sep" in POLOIDAL_OVERLAYS
    model = vaft_omas.extract_equilibrium_field_psi(ods, time_slice=1, overlay=["d_r_sep"])
    labels = {layer.label for layer in model.overlays}
    assert {"Primary separatrix (upper X-point)", "Lower X-point (secondary)"} <= labels
    assert any(label.startswith("R_L") for label in labels)


def test_flux_map_overlay_warns_rather_than_fails_without_separatrices():
    # Nested ellipses with no saddle anywhere: nothing to draw, and the map stays.
    rm, zm = np.meshgrid(R, Z, indexing="ij")
    psi = (rm - 1) ** 2 + zm**2
    rb, zb = max(extract_flux_surface_contours(psi, R, Z, 0.0, 0.04, [1.0])[1.0], key=lambda c: c[0].size)
    ods = omas.ODS()
    base = "equilibrium.time_slice.0"
    ods["equilibrium.time"] = np.array([0.3])
    ods[f"{base}.time"] = 0.3
    ods[f"{base}.profiles_2d.0.grid.dim1"] = R
    ods[f"{base}.profiles_2d.0.grid.dim2"] = Z
    ods[f"{base}.profiles_2d.0.psi"] = psi
    ods[f"{base}.global_quantities.psi_axis"] = 0.0
    ods[f"{base}.global_quantities.psi_boundary"] = 0.04
    ods[f"{base}.global_quantities.magnetic_axis.r"] = 1.0
    ods[f"{base}.global_quantities.magnetic_axis.z"] = 0.0
    ods[f"{base}.boundary.outline.r"] = rb
    ods[f"{base}.boundary.outline.z"] = zb
    with pytest.warns(UserWarning, match="d_r_sep overlay"):
        model = vaft_omas.extract_equilibrium_field_psi(ods, time_slice=0, overlay=["d_r_sep"])
    assert not any("separatrix" in layer.label for layer in model.overlays)

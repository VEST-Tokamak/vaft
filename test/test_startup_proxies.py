"""Startup proxies, midplane profiles, vacuum field lines on the camera, and the summary (issue #888).

Every helper here is a reading of an evaluator that already exists -- the
point response, the cached vacuum map, the connection-length trace, the
plasma timing -- so each test pins the new function against the old one on
VEST's packaged discharge 39915 rather than against a remembered number.
"""

import copy
import warnings

import numpy as np
import pytest
from omas import ODS

import vaft
import vaft.omas
from vaft.formula.equilibrium import poloidal_field_magnitude, toroidal_electric_field
from vaft.formula.startup import electron_cyclotron_resonance_radius


@pytest.fixture(scope="module")
def solved():
    """The packaged shot with its vessel currents solved, once."""
    ods = vaft.omas.sample_ods()
    vaft.omas.compute_eddy_currents(ods, [], [])
    return ods


@pytest.fixture(scope="module")
def t_breakdown(solved):
    return vaft.omas.find_breakdown_onset(solved)


@pytest.fixture(scope="module")
def proxies(solved):
    return vaft.omas.compute_startup_proxies_ods(solved, rz=(0.4, 0.0))


def _light_copy(ods, roots):
    private = ODS(consistency_check=False)
    for root in roots:
        private[root] = copy.deepcopy(ods[root])
    return private


# ---------------------------------------------------------------------------
# compute_startup_proxies_ods
# ---------------------------------------------------------------------------

def test_proxies_are_one_value_per_pf_sample(solved, proxies):
    n_time = len(np.asarray(solved["pf_active.time"]))
    for key in ("time", "b_z", "b_r", "psi", "v_loop", "decay_index"):
        assert proxies[key].shape == (n_time,), key
    assert np.isfinite(proxies["b_z"]).all()
    assert (proxies["r"], proxies["z"]) == (0.4, 0.0)


def test_the_loop_voltage_is_the_existing_loop_voltage(solved, proxies):
    """Same sign, same derivative; only the Green's function differs.

    The proxies use the exact-elliptic response and the old wrapper the
    polynomial one, which agree to ~2e-7 relative in psi on this shot; through
    a 40 us central difference that is ~1e-6 V against a 4.8 V peak, so
    1e-4 V is a hundredfold margin that still catches a sign or a factor.
    """
    time, v_loop = vaft.omas.compute_startup_loop_voltage_ods(solved, rz=(0.4, 0.0))
    np.testing.assert_array_equal(time, proxies["time"])
    np.testing.assert_allclose(proxies["v_loop"], v_loop, rtol=0, atol=1e-4)


def test_the_vertical_field_is_the_point_vacuum_field(solved, proxies):
    """Measured disagreement ~2e-9 T against a 26 mT peak; 1e-7 T bounds it."""
    _, _, _, bz = vaft.omas.compute_point_vacuum_fields_ods(solved, rz=[(0.4, 0.0)])
    np.testing.assert_allclose(proxies["b_z"], np.asarray(bz)[:, 0], rtol=0, atol=1e-7)


@pytest.mark.parametrize("instant", ["onset", 0.3307])
def test_the_decay_index_is_the_existing_decay_index_at_r0(solved, proxies, t_breakdown, instant):
    """At dr = 0.01 m the proxy is the central difference np.gradient takes at
    R = 0.4 m on the default 41-point 0.25-0.65 m line, so the two differ only
    by the Green's-function disagreement delta_B (<= 1e-8 T, 5x the measured):
    |delta n| <= R * delta_B / (dr |B_z|) + |n| delta_B / |B_z|.
    """
    t = t_breakdown if instant == "onset" else instant
    radius, legacy = vaft.omas.compute_decay_index_ods(
        solved, time=t, z=0.0, r_range=(0.25, 0.65), n_points=41
    )
    index = int(np.argmin(np.abs(proxies["time"] - t)))
    b_z = abs(proxies["b_z"][index])
    n = proxies["decay_index"][index]
    expected = float(np.interp(0.4, radius, legacy))
    delta_b = 1e-8
    tolerance = 0.4 * delta_b / (0.01 * b_z) + abs(n) * delta_b / b_z
    assert abs(n - expected) <= tolerance, (n, expected, tolerance)


def test_an_unknown_mode_is_refused(solved):
    with pytest.raises(ValueError, match="Invalid mode"):
        vaft.omas.compute_startup_proxies_ods(solved, mode="plasma")


# ---------------------------------------------------------------------------
# compute_vacuum_midplane_profiles_ods
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def profiles(solved, t_breakdown):
    return vaft.omas.compute_vacuum_midplane_profiles_ods(solved, time=t_breakdown, z=0.0)


def test_a_profile_is_a_row_of_the_vacuum_map(solved, t_breakdown, profiles):
    field = vaft.omas.compute_vacuum_field_map(solved, time=t_breakdown, resolution=65)
    row = int(np.argmin(np.abs(field["z"] - 0.0)))
    inside = profiles["inside"]
    assert profiles["row_index"] == row and profiles["z"] == field["z"][row]
    assert profiles["time"] == field["time"] and profiles["time_index"] == field["time_index"]
    np.testing.assert_array_equal(profiles["r"], field["r"])
    assert inside.sum() > 40 and not inside.all()

    b_r, b_z, dpsi = field["b_r"][:, row], field["b_z"][:, row], field["dpsi_dt"][:, row]
    np.testing.assert_array_equal(profiles["b_z"][inside], b_z[inside])
    np.testing.assert_array_equal(profiles["b_r"][inside], b_r[inside])
    np.testing.assert_allclose(
        profiles["b_poloidal"][inside], poloidal_field_magnitude(b_r, b_z)[inside]
    )
    e_phi = toroidal_electric_field(field["r"], dpsi)
    np.testing.assert_allclose(profiles["e_toroidal"][inside], np.abs(e_phi)[inside])
    # 2 pi R E_phi = -dpsi/dt, the sign compute_startup_loop_voltage_ods uses.
    np.testing.assert_allclose(
        profiles["v_loop"][inside], (2 * np.pi * field["r"] * e_phi)[inside]
    )
    for key in ("b_z", "e_toroidal", "breakdown_figure", "lloyd_margin"):
        assert np.isnan(profiles[key][~inside]).all(), key


def test_the_profile_loop_voltage_agrees_with_the_point_trace(solved, profiles):
    """The map's three-sample difference and time_derivative on a uniform PF
    clock are the same central difference, so at a grid node the two loop
    voltages differ only by rounding in the two response contractions."""
    at = int(np.argmin(np.abs(profiles["r"] - 0.4)))
    point = vaft.omas.compute_startup_proxies_ods(
        solved, rz=(float(profiles["r"][at]), profiles["z"])
    )
    assert profiles["v_loop"][at] == pytest.approx(
        point["v_loop"][profiles["time_index"]], rel=1e-6
    )


def test_the_breakdown_figure_is_e_times_b_toroidal_over_b_poloidal(solved, profiles):
    tf_time = np.asarray(solved["tf.time"], dtype=float)
    product = float(np.asarray(solved["tf.b_field_tor_vacuum_r.data"])[
        int(np.argmin(np.abs(tf_time - profiles["time"])))])
    inside = profiles["inside"]
    expected = profiles["e_toroidal"] * abs(product) / profiles["r"] / profiles["b_poloidal"]
    np.testing.assert_allclose(profiles["breakdown_figure"][inside], expected[inside])


def test_the_ecr_radius_is_the_formula_at_the_same_instant(solved, profiles):
    tf_time = np.asarray(solved["tf.time"], dtype=float)
    product = float(np.asarray(solved["tf.b_field_tor_vacuum_r.data"])[
        int(np.argmin(np.abs(tf_time - profiles["time"])))])
    assert profiles["b_field_tor_vacuum_r"] == product
    assert profiles["r_ecr"] == pytest.approx(electron_cyclotron_resonance_radius(product, 2.45e9))
    # 2.45 GHz resonates at 87.5 mT, so the ~0.06 T m product puts it near 0.69 m.
    assert 0.6 < profiles["r_ecr"] < 0.75


def test_the_row_connection_length_is_the_map_row(solved, t_breakdown):
    """Tracing only the row gives the lengths the full map traces there."""
    small = vaft.omas.compute_vacuum_midplane_profiles_ods(solved, time=t_breakdown, resolution=17)
    traced = vaft.omas.compute_connection_length_map_ods(solved, time=t_breakdown, resolution=17)
    row = small["row_index"]
    inside = small["inside"]
    np.testing.assert_allclose(
        small["connection_length_m"][inside], traced["length_m"][:, row][inside]
    )


def test_the_prefill_default_is_the_median_before_the_instant(solved, profiles):
    pressure = np.asarray(solved["barometry.gauge.0.pressure.data"], dtype=float)
    stamps = np.asarray(solved["barometry.gauge.0.pressure.time"], dtype=float)
    expected = float(np.median(pressure[stamps < profiles["time"]]))
    assert profiles["p_Pa"] == expected
    assert vaft.omas.compute_prefill_pressure_ods(solved, before=profiles["time"]) == expected


def test_a_higher_pressure_changes_the_margin_where_a_threshold_exists(solved, t_breakdown, profiles):
    higher = vaft.omas.compute_vacuum_midplane_profiles_ods(
        solved, time=t_breakdown, p_Pa=2 * profiles["p_Pa"]
    )
    both = np.isfinite(profiles["lloyd_margin"]) & np.isfinite(higher["lloyd_margin"])
    assert both.any()
    assert not np.allclose(profiles["lloyd_margin"][both], higher["lloyd_margin"][both])


def test_without_tf_or_barometry_the_dependent_profiles_are_nan(solved, t_breakdown):
    bare = _light_copy(solved, ("pf_active", "pf_passive", "wall"))
    result = vaft.omas.compute_vacuum_midplane_profiles_ods(bare, time=t_breakdown)
    inside = result["inside"]
    assert np.isfinite(result["b_z"][inside]).all()
    assert np.isnan(result["r_ecr"])
    for key in ("breakdown_figure", "lloyd_margin", "connection_length_m"):
        assert np.isnan(result[key]).all(), key
    assert np.isnan(result["p_Pa"])


# ---------------------------------------------------------------------------
# compute_camera_visible_vacuum_field_lines
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def field_lines(solved, t_breakdown, profiles):
    # No equilibrium in the copy: a vacuum trace must never reach for one.
    vacuum_only = _light_copy(solved, ("pf_active", "pf_passive", "wall", "tf"))
    return vaft.omas.compute_camera_visible_vacuum_field_lines(
        vacuum_only, shot=39915, time=t_breakdown,
        seeds=[(0.4, 0.0), (profiles["r_ecr"], 0.0), (0.05, 0.0)],
        max_length_m=10.0,
    )


def test_vacuum_field_lines_land_on_the_camera_image(field_lines):
    width, height = field_lines[0]["image_size"]
    assert (width, height) == (1024, 1280)
    for record in field_lines[:2]:
        n = record["r"].size
        assert n > 100, record["termination_reason"]
        assert record["u"].shape == record["v"].shape == (n,)
        in_image = record["in_image"]
        assert in_image.mean() > 0.2
        assert np.isfinite(record["u"][in_image]).all() and np.isfinite(record["v"][in_image]).all()
        assert (record["u"][in_image] >= 0).all() and (record["u"][in_image] < width).all()
        assert (record["v"][in_image] >= 0).all() and (record["v"][in_image] < height).all()
        assert np.isnan(record["u"][~record["valid"]]).all()
        assert 0.0 < record["length_m"] <= 2 * 10.0 + 0.1
        assert record["inside_wall"]


def test_a_seed_outside_the_wall_terminates_immediately(field_lines):
    record = field_lines[2]
    assert record["termination_reason"] == "seed_outside_wall"
    assert record["r"].size == 1 and record["length_m"] == 0.0
    assert not record["inside_wall"]


def test_max_turns_cuts_the_trace(solved, t_breakdown):
    vacuum_only = _light_copy(solved, ("pf_active", "pf_passive", "wall", "tf"))
    (record,) = vaft.omas.compute_camera_visible_vacuum_field_lines(
        vacuum_only, shot=39915, time=t_breakdown, seeds=[(0.4, 0.0)],
        max_length_m=30.0, max_turns=1.0,
    )
    assert np.all(np.abs(record["phi"]) <= 2 * np.pi + 1e-9)
    assert "max_turns" in record["termination_reason"]


def test_a_vacuum_field_line_needs_the_toroidal_field(solved, t_breakdown):
    no_tf = _light_copy(solved, ("pf_active", "pf_passive", "wall"))
    with pytest.raises(ValueError, match="tf.b_field_tor_vacuum_r"):
        vaft.omas.compute_camera_visible_vacuum_field_lines(
            no_tf, shot=39915, time=t_breakdown, seeds=[(0.4, 0.0)]
        )


# ---------------------------------------------------------------------------
# startup_summary
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def summary():
    ods = vaft.omas.sample_ods()
    before = sorted(ods.keys())
    result = vaft.omas.startup_summary(ods)
    assert sorted(ods.keys()) == before
    assert "time" not in ods["pf_passive"], "the summary must not store vessel currents"
    return result


def test_the_summary_reads_the_existing_evaluators(summary, solved, t_breakdown, proxies):
    assert summary["shot"] == 39915
    assert summary["t_breakdown"] == t_breakdown
    assert summary["onset_source"].startswith("h_alpha")
    assert summary["ip_peak_A"] > 5e4 and summary["t_ip_peak"] > t_breakdown
    assert summary["pressure_Pa"] == vaft.omas.compute_prefill_pressure_ods(solved, before=t_breakdown)

    index = int(np.argmin(np.abs(proxies["time"] - t_breakdown)))
    assert summary["v_loop_V"] == pytest.approx(proxies["v_loop"][index], rel=1e-9)
    assert summary["b_z_T"] == pytest.approx(proxies["b_z"][index], rel=1e-9)
    assert summary["decay_index"] == pytest.approx(proxies["decay_index"][index], rel=1e-9)

    tf_time = np.asarray(solved["tf.time"], dtype=float)
    product = float(np.asarray(solved["tf.b_field_tor_vacuum_r.data"])[
        int(np.argmin(np.abs(tf_time - t_breakdown)))])
    assert summary["r_ecr_m"] == pytest.approx(electron_cyclotron_resonance_radius(product, 2.45e9))
    assert summary["ec_power_W"] is None and "ec_power_W" in summary["unavailable"]


def test_the_null_area_is_the_session_02_definition(summary, solved, t_breakdown):
    from matplotlib.path import Path as MplPath

    limiter = MplPath(np.column_stack([
        solved["wall.description_2d.0.limiter.unit.0.outline.r"],
        solved["wall.description_2d.0.limiter.unit.0.outline.z"],
    ]))
    field = vaft.omas.compute_vacuum_field_map(solved, time=t_breakdown, resolution=33)
    mesh_r, mesh_z = np.meshgrid(field["r"], field["z"], indexing="ij")
    inside = limiter.contains_points(
        np.column_stack([mesh_r.ravel(), mesh_z.ravel()])).reshape(mesh_r.shape)
    b_p = poloidal_field_magnitude(field["b_r"], field["b_z"])
    assert summary["null_area_fraction"] == float(np.mean(b_p[inside] < 5e-4))
    assert 0.0 < summary["null_area_fraction"] < 1.0


def test_the_lloyd_fractions_are_the_session_02_definition(summary, solved, t_breakdown):
    from vaft.formula.startup import breakdown_margin, lloyd_breakdown_field

    traced = vaft.omas.compute_connection_length_map_ods(solved, time=t_breakdown, resolution=33)
    field = vaft.omas.compute_vacuum_field_map(solved, time=t_breakdown, resolution=33)
    interior = ~traced["outside"]
    e_local = np.abs(toroidal_electric_field(field["r"][:, None], field["dpsi_dt"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        threshold = lloyd_breakdown_field(summary["pressure_Pa"], traced["length_m"])
    margin = breakdown_margin(e_local, threshold)
    has_threshold = interior & np.isfinite(margin)
    assert summary["lloyd_threshold_fraction"] == pytest.approx(has_threshold.sum() / interior.sum())
    assert summary["lloyd_margin_fraction"] == pytest.approx(np.mean(margin[has_threshold] >= 1))


def test_the_summary_degrades_without_barometry(summary):
    ods = vaft.omas.sample_ods()
    del ods["barometry"]
    result = vaft.omas.startup_summary(ods)
    for key in ("pressure_Pa", "lloyd_threshold_fraction", "lloyd_margin_fraction"):
        assert result[key] is None and key in result["unavailable"], key
    for key in ("t_breakdown", "v_loop_V", "b_z_T", "decay_index", "null_area_fraction", "r_ecr_m"):
        assert result[key] == summary[key], key


def test_the_summary_of_an_empty_ods_is_all_none_not_an_exception():
    result = vaft.omas.startup_summary(ODS())
    assert result["t_breakdown"] is None and result["v_loop_V"] is None
    assert "t_breakdown" in result["unavailable"]


def test_the_ec_power_is_the_nan_aware_median_around_the_onset():
    from vaft.omas.startup_summary import _ec_power

    ods = ODS(consistency_check=False)
    stamps = np.arange(0.300, 0.312, 0.0005)
    power = np.full(stamps.shape, 5e3)
    power[stamps > 0.305] = 7e3
    power[np.argmin(np.abs(stamps - 0.3055))] = np.nan
    ods["ec_launchers.beam.0.power_launched.data"] = power
    ods["ec_launchers.beam.0.power_launched.time"] = stamps
    window = np.abs(stamps - 0.3055) <= 1e-3
    assert _ec_power(ods, 0.3055) == float(np.nanmedian(power[window]))
    assert _ec_power(ODS(), 0.3) is None

"""The cached vacuum-field evaluator behind the interactive startup maps.

One evaluation feeds every quantity the maps draw -- flux, |B_p|, the decay
index, the breakdown figure of merit -- so they cannot disagree with each other.
What has to hold is that the response matrices are built once per grid rather
than per frame, that the flux still matches the path it replaces, and that the
array orientation is the one the axes claim.
"""

import numpy as np
import pytest

import vaft.omas
from vaft.omas import process_wrapper as pw


@pytest.fixture(scope="module")
def solved():
    """The packaged shot with its vessel currents solved, once."""
    ods = vaft.omas.sample_ods()
    vaft.omas.compute_eddy_currents(ods, [], [])
    return ods


@pytest.fixture(autouse=True)
def _empty_cache():
    """No test may inherit another's cached matrices."""
    pw.clear_vacuum_field_cache()
    yield
    pw.clear_vacuum_field_cache()


COARSE = (np.linspace(0.12, 0.74, 9), np.linspace(-1.1, 1.1, 11))


# ---------------------------------------------------------------------------
# Shape and orientation
# ---------------------------------------------------------------------------

def test_every_field_is_indexed_r_then_z(solved):
    """The axes are named; the arrays must actually be laid out that way.

    ``compute_null_ods`` returns the transpose of this, which is exactly the
    kind of silent flip a plot inherits without complaint.
    """
    result = pw.compute_vacuum_field_map(solved, time=0.29, grid=COARSE)
    assert result["r"].size == 9 and result["z"].size == 11
    for key in ("psi", "b_r", "b_z", "dpsi_dt"):
        assert result[key].shape == (9, 11), key
        assert np.isfinite(result[key]).all(), key


def test_the_time_returned_is_a_stored_sample(solved):
    """The map is snapped, not interpolated, so it must say where it landed."""
    time_base = np.asarray(solved["pf_active.time"], dtype=float)
    result = pw.compute_vacuum_field_map(solved, time=0.2903177, grid=COARSE)
    assert result["time"] == pytest.approx(time_base[result["time_index"]])
    assert abs(result["time"] - 0.2903177) <= np.diff(time_base).max()


# ---------------------------------------------------------------------------
# The cache, which is the whole reason a time slider is affordable
# ---------------------------------------------------------------------------

def test_moving_only_in_time_does_not_rebuild_the_matrices(solved, monkeypatch):
    """A slider move must cost a contraction, not a rebuild."""
    calls = []
    original = pw.compute_point_response_matrices_ods

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pw, "compute_point_response_matrices_ods", counted)
    first = pw.compute_vacuum_field_map(solved, time=0.29, grid=COARSE)
    for t in (0.30, 0.31, 0.32):
        pw.compute_vacuum_field_map(solved, time=t, grid=COARSE)
    assert len(calls) == 1

    # ... but a different grid is a different machine-to-grid response.
    other = (COARSE[0], np.linspace(-1.0, 1.0, 11))
    pw.compute_vacuum_field_map(solved, time=0.29, grid=other)
    assert len(calls) == 2

    later = pw.compute_vacuum_field_map(solved, time=0.29, grid=COARSE)
    assert len(calls) == 2
    np.testing.assert_allclose(later["psi"], first["psi"])


def test_the_cache_is_keyed_on_the_geometry_it_describes(solved):
    """Move a coil and the cached response no longer describes this machine."""
    before = pw._machine_fingerprint(solved)
    original = float(solved["pf_active.coil.0.element.0.geometry.rectangle.r"])
    try:
        solved["pf_active.coil.0.element.0.geometry.rectangle.r"] = original + 0.01
        assert pw._machine_fingerprint(solved) != before
    finally:
        solved["pf_active.coil.0.element.0.geometry.rectangle.r"] = original
    assert pw._machine_fingerprint(solved) == before


def test_the_cache_does_not_grow_without_bound(solved):
    """Each entry is three dense matrices; keeping every grid ever asked for
    would be tens of gigabytes over a session."""
    for n in range(pw._VACUUM_MAP_CACHE_LIMIT + 2):
        grid = (COARSE[0], np.linspace(-1.0 - 0.01 * n, 1.0, 7))
        pw.compute_vacuum_field_map(solved, time=0.29, grid=grid)
    assert len(pw._VACUUM_MAP_CACHE) <= pw._VACUUM_MAP_CACHE_LIMIT


# ---------------------------------------------------------------------------
# Agreement with the path it replaces
# ---------------------------------------------------------------------------

def test_the_flux_matches_compute_null_ods(solved):
    """Re-pointing the vacuum psi map onto this evaluator must not move it.

    The tolerance is a percentile and a count, not a maximum, and deliberately
    so: the two paths differ only where a grid point sits on a source filament,
    where both answers are meaningless.  A max-based bound could not pass and
    would be demanding the wrong thing.  Measured on the packaged shot: median
    3e-8, 95th percentile 7e-7, and 1.4% of points past 1e-3.
    """
    psi_reference, mesh_r, mesh_z = vaft.omas.compute_null_ods(solved, 0.29)
    psi_reference = np.asarray(psi_reference, dtype=float)
    r_axis, z_axis = np.unique(np.asarray(mesh_r)), np.unique(np.asarray(mesh_z))

    # Every third point of the reference grid: the comparison is against its own
    # values at its own coordinates, but the whole 129x129 is past the response
    # budget -- which is the guard doing its job, not something to work around.
    stride = 3
    result = pw.compute_vacuum_field_map(
        solved, time=0.29, grid=(r_axis[::stride], z_axis[::stride])
    )
    # compute_null_ods lays its grid out (Z, R); this evaluator lays it out (R, Z).
    reference = psi_reference[::stride, ::stride].T
    assert reference.shape == result["psi"].shape

    relative = np.abs(reference - result["psi"]) / np.maximum(np.abs(reference), 1e-12)
    assert np.percentile(relative, 95) < 1e-5
    # ... and the disagreement is confined to a handful of points, rather than
    # being a small bias spread over the map.
    assert np.mean(relative > 1e-3) < 0.03


# ---------------------------------------------------------------------------
# The default grid and the magnitudes on it
# ---------------------------------------------------------------------------

def test_the_default_grid_covers_the_limiter(solved):
    """The map is read to site a null, so it covers where a plasma can sit."""
    wall_r = np.asarray(solved["wall.description_2d.0.limiter.unit.0.outline.r"], float)
    wall_z = np.asarray(solved["wall.description_2d.0.limiter.unit.0.outline.z"], float)
    r_axis, z_axis = pw._vacuum_map_grid(solved, 17)
    assert r_axis.size == z_axis.size == 17
    assert r_axis[0] == pytest.approx(wall_r.min())
    assert r_axis[-1] == pytest.approx(wall_r.max())
    assert z_axis[0] == pytest.approx(wall_z.min())
    assert z_axis[-1] == pytest.approx(wall_z.max())


def test_the_fields_are_the_size_a_vest_startup_is(solved):
    """A guard on units and on the current assembly at once: a factor of 2*pi,
    a sign flip on a coil or a wrong column order all leave this range."""
    result = pw.compute_vacuum_field_map(solved, time=0.29, grid=COARSE)
    b_poloidal_gauss = np.hypot(result["b_r"], result["b_z"]) * 1e4
    assert 1.0 < np.median(b_poloidal_gauss) < 1000.0
    assert np.percentile(np.abs(result["psi"]), 50) < 1.0        # weber
    assert np.any(np.abs(result["dpsi_dt"]) > 0.1)               # a driven shot


def test_the_flux_derivative_is_the_slope_of_the_flux(solved):
    """dpsi/dt has to be this map's own psi differenced, not a separate model."""
    time_base = np.asarray(solved["pf_active.time"], dtype=float)
    middle = pw.compute_vacuum_field_map(solved, time=0.29, grid=COARSE)
    index = middle["time_index"]
    behind = pw.compute_vacuum_field_map(solved, time=time_base[index - 1], grid=COARSE)
    ahead = pw.compute_vacuum_field_map(solved, time=time_base[index + 1], grid=COARSE)
    expected = (ahead["psi"] - behind["psi"]) / (time_base[index + 1] - time_base[index - 1])
    np.testing.assert_allclose(middle["dpsi_dt"], expected, rtol=1e-10, atol=1e-12)


def test_a_geometry_only_sample_solves_its_own_eddy_currents():
    """compute_null_ods does this, so the evaluator replacing it must too."""
    ods = vaft.omas.sample_ods()
    assert "time" not in ods["pf_passive"]
    result = pw.compute_vacuum_field_map(ods, time=0.29, grid=COARSE)
    assert np.isfinite(result["psi"]).all()


def test_a_recalled_eddy_solve_is_the_solve_it_recalls():
    """A second fresh ODS with the same PF programme gets bit-identical currents."""
    first = vaft.omas.sample_ods()
    pw.compute_vacuum_field_map(first, time=0.29, grid=COARSE)
    assert len(pw._VACUUM_EDDY_CACHE) == 1
    second = vaft.omas.sample_ods()
    pw.compute_vacuum_field_map(second, time=0.29, grid=COARSE)
    assert len(pw._VACUUM_EDDY_CACHE) == 1, "the same programme must not be solved twice"
    np.testing.assert_array_equal(second["pf_passive.time"], first["pf_passive.time"])
    for index in range(len(first["pf_passive.loop"])):
        np.testing.assert_array_equal(
            second[f"pf_passive.loop.{index}.current"], first[f"pf_passive.loop.{index}.current"]
        )


def test_the_budget_counts_the_columns_that_are_cached(solved):
    """One column per coil or loop, not per filament (review of #690)."""
    src_r, _, _, groups = pw._vacuum_sources(solved)
    columns = len(set(groups))
    assert columns < len(src_r), "VEST coils have several elements each"
    r_axis, z_axis = pw._vacuum_map_grid(solved, 129)
    with pytest.raises(ValueError, match=f"over {columns} coils and loops") as info:
        pw._refuse_oversized_grid(solved, r_axis, z_axis)
    needed = 3 * 129 * 129 * columns * 8 / 1024 ** 2
    assert f"needs {needed:.0f} MB" in str(info.value)


def test_clearing_the_cache_is_public(solved):
    pw.compute_vacuum_field_map(solved, time=0.29, grid=COARSE)
    assert pw._VACUUM_MAP_CACHE
    pw.clear_vacuum_field_cache()
    assert not pw._VACUUM_MAP_CACHE and not pw._VACUUM_EDDY_CACHE


# ---------------------------------------------------------------------------
# The connection length and the two quantities it made drawable (#230)
# ---------------------------------------------------------------------------

BREAKDOWN_S = 0.3063  # the packaged shot's own onset, pinned in test_omas_general_finders


@pytest.fixture(scope="module")
def traced(solved):
    """One coarse connection-length map, shared: a trace costs seconds."""
    pw.clear_vacuum_field_cache()
    return pw.compute_connection_length_map_ods(solved, time=BREAKDOWN_S, resolution=17)


def test_the_connection_length_map_is_laid_out_like_the_field_it_is_traced_through(solved, traced):
    grid = pw.compute_vacuum_field_map(solved, time=BREAKDOWN_S, resolution=17)
    np.testing.assert_array_equal(traced["r"], grid["r"])
    np.testing.assert_array_equal(traced["z"], grid["z"])
    assert traced["time"] == grid["time"]
    for key in ("length_m", "forward_m", "backward_m", "saturated", "outside"):
        assert traced[key].shape == (grid["r"].size, grid["z"].size), key


def test_every_point_outside_the_limiter_has_no_length_and_every_point_inside_has_one(traced):
    """`nan` outside, finite inside: a map that blanked an interior point
    would be hiding a line that failed to trace, not a wall."""
    outside = traced["outside"]
    assert outside.any() and (~outside).any()
    assert np.isnan(traced["length_m"][outside]).all()
    assert np.isfinite(traced["length_m"][~outside]).all()


def test_the_lengths_are_the_size_a_vest_startup_has(traced):
    """Tens of metres, bounded by the two-branch ceiling -- the scale Lloyd's
    threshold is sensitive to on VEST, where no threshold exists below ~98 m."""
    interior = traced["length_m"][~traced["outside"]]
    assert 10.0 < float(np.median(interior)) < 300.0
    assert float(interior.max()) <= 300.0 + 1e-9


def test_a_second_request_for_the_same_trace_is_recalled_not_repeated(solved, monkeypatch):
    """The file clears every cache between tests, so this one fills its own."""
    import vaft.process.equilibrium as process

    first = pw.compute_connection_length_map_ods(solved, time=BREAKDOWN_S, resolution=9)

    def refuse(*args, **kwargs):
        raise AssertionError("the connection length was traced again")

    monkeypatch.setattr(process, "connection_length_map", refuse)
    again = pw.compute_connection_length_map_ods(solved, time=BREAKDOWN_S, resolution=9)
    np.testing.assert_array_equal(again["length_m"], first["length_m"])


def test_clearing_the_vacuum_cache_also_releases_the_traces(solved):
    pw.compute_connection_length_map_ods(solved, time=BREAKDOWN_S, resolution=9)
    assert pw._CONNECTION_LENGTH_CACHE
    pw.clear_vacuum_field_cache()
    assert not pw._CONNECTION_LENGTH_CACHE


def test_a_connection_length_needs_the_toroidal_field(solved):
    import copy

    bare = copy.deepcopy(solved)
    del bare["tf"]
    with pytest.raises(ValueError, match="tf.b_field_tor_vacuum_r is required"):
        pw.compute_connection_length_map_ods(bare, time=BREAKDOWN_S, resolution=9)


def test_the_toroidal_electric_field_map_is_the_magnitude_of_the_formula(solved):
    """Drawn as |E_phi|, which is what the breakdown chain consumes (#354)."""
    from vaft.formula.equilibrium import toroidal_electric_field
    from vaft.plot.backend.recipes import build_model

    model = build_model("vacuum_field", [("0", solved)], field="e_toroidal",
                        time=BREAKDOWN_S, resolution=17)
    grid = pw.compute_vacuum_field_map(solved, time=BREAKDOWN_S, resolution=17)
    expected = np.abs(toroidal_electric_field(grid["r"][:, None], grid["dpsi_dt"])).T
    finite = np.isfinite(model.values)
    assert finite.any()
    np.testing.assert_allclose(model.values[finite], expected[finite] * model.display.scale)
    assert (model.values[finite] >= 0.0).all()


def test_the_lloyd_margin_map_marks_the_threshold_with_its_own_contour(solved):
    from vaft.plot.backend.recipes import LLOYD_MARGIN_THRESHOLD, build_model

    model = build_model("vacuum_field", [("0", solved)], field="lloyd_margin",
                        time=BREAKDOWN_S, resolution=17)
    assert model.secondary_levels == LLOYD_MARGIN_THRESHOLD == (1.0,)
    assert (model.values[np.isfinite(model.values)] >= 0.0).all()


def test_the_lloyd_margin_blanks_where_no_threshold_exists(solved):
    """Below A p L = 1 there is no field that breaks the gas down, which is a
    region of the map and not a large number -- so it is blank, and the blank
    is part of the answer."""
    from vaft.plot.backend.recipes import build_model

    model = build_model("vacuum_field", [("0", solved)], field="lloyd_margin",
                        time=BREAKDOWN_S, resolution=17)
    grid = pw.compute_vacuum_field_map(solved, time=BREAKDOWN_S, resolution=17)
    traced_here = pw.compute_connection_length_map_ods(solved, time=BREAKDOWN_S, resolution=17)
    interior = (~traced_here["outside"]).T
    assert np.isfinite(model.values[interior]).any()
    assert np.isnan(model.values[interior]).any()


def test_an_explicit_pressure_moves_the_margin_the_way_a_threshold_should(solved):
    """A threshold exists only where A p L > 1, so more gas can only extend the
    region that has one -- the one direction Lloyd's expression guarantees
    whatever side of its Paschen minimum a point sits on."""
    from vaft.plot.backend.recipes import build_model

    thin = build_model("vacuum_field", [("0", solved)], field="lloyd_margin",
                       time=BREAKDOWN_S, resolution=17, p_Pa=2.0e-3)
    dense = build_model("vacuum_field", [("0", solved)], field="lloyd_margin",
                        time=BREAKDOWN_S, resolution=17, p_Pa=8.0e-3)
    both = np.isfinite(thin.values) & np.isfinite(dense.values)
    assert both.any()
    assert np.isfinite(dense.values).sum() >= np.isfinite(thin.values).sum()


def test_discovery_offers_the_lloyd_margin_only_to_an_input_that_can_draw_it(solved):
    import copy

    def offered(ods):
        record = vaft.omas.available_plots(ods, query="vacuum_field", detail=True)[0]
        return record.fields["options"]

    assert {"e_toroidal", "lloyd_margin"} <= set(offered(solved))
    no_gauge = copy.deepcopy(solved)
    del no_gauge["barometry"]
    assert "lloyd_margin" not in offered(no_gauge)
    assert "e_toroidal" in offered(no_gauge)

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
    pw._VACUUM_MAP_CACHE.clear()
    yield
    pw._VACUUM_MAP_CACHE.clear()


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

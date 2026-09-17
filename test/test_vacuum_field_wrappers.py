"""The ODS-level vacuum-field wrappers: what they return, and how fast.

Two defects motivated these tests. ``compute_point_vacuum_fields_ods`` built
its current array one time sample at a time -- 2.4 million IMAS path parses on
the packaged shot to read 960 waveforms -- and ``compute_grid_ods`` could not
run at all, so nothing had ever exercised it.
"""

import numpy as np
import pytest

import vaft.omas
import vaft.process
from vaft.omas.process_wrapper import (
    GEOMETRY_TYPE_POLYGON,
    GEOMETRY_TYPE_RECTANGLE,
)


@pytest.fixture(scope="module")
def solved():
    """The packaged shot with its vessel currents solved, once."""
    ods = vaft.omas.sample_ods()
    vaft.omas.compute_eddy_currents(ods, [], [])
    return ods


# ---------------------------------------------------------------------------
# The current assembly
# ---------------------------------------------------------------------------

def _assembled(ods, mode="vacuum"):
    """What compute_point_vacuum_fields_ods builds, extracted for comparison."""
    pf, pfp = ods["pf_active"], ods["pf_passive"]
    nbcoil, nbloop = len(pf["coil"]), len(pfp["loop"])
    nbt = len(np.asarray(pf["time"]))
    current = np.zeros((nbt, nbcoil + nbloop))
    if mode in ("vacuum", "pf_active"):
        current[:, :nbcoil] = np.column_stack(
            [np.asarray(pf[f"coil.{i}.current.data"], dtype=float) for i in range(nbcoil)]
        )
    if mode in ("vacuum", "pf_passive"):
        current[:, nbcoil:] = np.column_stack(
            [np.asarray(pfp[f"loop.{i}.current"], dtype=float) for i in range(nbloop)]
        )
    return current


def test_the_hoisted_assembly_is_bit_identical_to_the_per_sample_one(solved):
    """The whole contract of the rewrite: same numbers, far fewer path parses.

    Compared over a slice of the time base, because paying the full 26 s the
    change exists to remove would defeat the point of the test.
    """
    pf, pfp = solved["pf_active"], solved["pf_passive"]
    nbcoil, nbloop = len(pf["coil"]), len(pfp["loop"])
    rows = 200

    reference = np.zeros((rows, nbcoil + nbloop))
    for t in range(rows):
        reference[t, :nbcoil] = [pf[f"coil.{i}.current.data"][t] for i in range(nbcoil)]
        reference[t, nbcoil:] = [pfp[f"loop.{i}.current"][t] for i in range(nbloop)]

    assert np.array_equal(_assembled(solved)[:rows], reference)


def test_the_assembly_keeps_its_shape_and_column_order(solved):
    pf, pfp = solved["pf_active"], solved["pf_passive"]
    nbcoil, nbloop = len(pf["coil"]), len(pfp["loop"])
    current = _assembled(solved)
    assert current.shape == (len(np.asarray(pf["time"])), nbcoil + nbloop)
    # Coils first, then loops.
    np.testing.assert_allclose(
        current[:, 0], np.asarray(pf["coil.0.current.data"], dtype=float)
    )
    np.testing.assert_allclose(
        current[:, nbcoil], np.asarray(pfp["loop.0.current"], dtype=float)
    )


def test_pf_active_mode_never_reads_the_passive_currents(solved):
    """Reading everything and slicing afterwards would break this case.

    ``mode="pf_active"`` has to work on an ODS whose passive loops carry no
    current at all, which is why the two blocks stay under separate guards.
    """
    import copy

    stripped = copy.deepcopy(solved)
    for index in range(len(stripped["pf_passive.loop"])):
        del stripped[f"pf_passive.loop.{index}.current"]

    current = _assembled(stripped, mode="pf_active")
    nbcoil = len(stripped["pf_active.coil"])
    assert np.any(current[:, :nbcoil] != 0.0)
    assert np.all(current[:, nbcoil:] == 0.0)


def test_the_point_field_call_still_returns_what_it_always_did(solved):
    times, psi, br, bz = vaft.omas.compute_point_vacuum_fields_ods(
        solved, rz=[(0.4, 0.0), (0.5, 0.1)]
    )
    n_times = len(np.asarray(solved["pf_active.time"]))
    assert times.shape == (n_times,)
    for name, array in (("psi", psi), ("br", br), ("bz", bz)):
        assert array.shape == (n_times, 2), name
        assert np.isfinite(array).any(), name


def test_the_derived_startup_quantities_are_unchanged(solved):
    """The two callers that inherit the speedup, pinned by their physics.

    These are the numbers tutorial session 02 reports; they must not move
    because the current array is now assembled differently.
    """
    times, v_loop = vaft.omas.compute_startup_loop_voltage_ods(solved)
    assert np.nanmax(np.abs(v_loop)) == pytest.approx(4.79, abs=0.05)

    radius, decay_index = vaft.omas.compute_decay_index_ods(solved, time=0.3307)
    finite = np.isfinite(decay_index)
    assert finite.any()
    assert np.all((decay_index[finite] > 0.0) & (decay_index[finite] < 1.5))


# ---------------------------------------------------------------------------
# compute_grid_ods
# ---------------------------------------------------------------------------

def test_compute_grid_ods_runs_at_all(solved):
    """It could not: calc_grid was never imported, and the coil lists are ragged.

    The grid is deliberately tiny -- vaft.process.calc_grid prints a progress
    percentage every 100 points.
    """
    br, bz, phi = vaft.omas.compute_grid_ods(solved, [0.4, 0.5], [0.0, 0.1])

    n_sources = len(solved["pf_active.coil"]) + len(solved["pf_passive.loop"])
    for name, array in (("Br", br), ("Bz", bz), ("Phi", phi)):
        array = np.asarray(array, dtype=float)
        # Response matrices per unit source current -- one row per grid point,
        # one column per source -- not a field map. calc_grid's own docstring
        # says "response matrix"; this wrapper's says "field components".
        assert array.shape == (4, n_sources), name
        assert np.isfinite(array).all(), name


def test_compute_grid_ods_returns_br_bz_phi_in_that_order(solved):
    """Pinned against a direct call, because return orders differ across this
    family of functions and this one's docstring is only a claim until checked.
    """
    xvar, zvar = [0.4, 0.5], [0.0, 0.1]
    through_ods = vaft.omas.compute_grid_ods(solved, xvar, zvar)

    pf, pfp = solved["pf_active"], solved["pf_passive"]
    geometry_type = np.array(
        [pfp[f"loop.{i}.element[0].geometry.geometry_type"] for i in range(len(pfp["loop"]))]
    )
    direct = vaft.process.calc_grid(
        xvar,
        zvar,
        [[pf[f"coil.{i}.element.{j}.turns_with_sign"]
          for j in range(len(pf[f"coil.{i}.element"]))] for i in range(len(pf["coil"]))],
        [[pf[f"coil.{i}.element.{j}.geometry.rectangle.r"]
          for j in range(len(pf[f"coil.{i}.element"]))] for i in range(len(pf["coil"]))],
        [[pf[f"coil.{i}.element.{j}.geometry.rectangle.z"]
          for j in range(len(pf[f"coil.{i}.element"]))] for i in range(len(pf["coil"]))],
        geometry_type,
        [pfp[f"loop.{i}.element[0].geometry.outline.r"] if geometry_type[i] == GEOMETRY_TYPE_POLYGON else []
         for i in range(len(pfp["loop"]))],
        [pfp[f"loop.{i}.element[0].geometry.outline.z"] if geometry_type[i] == GEOMETRY_TYPE_POLYGON else []
         for i in range(len(pfp["loop"]))],
        np.array([pfp[f"loop.{i}.element[0].geometry.rectangle.r"] if geometry_type[i] == GEOMETRY_TYPE_RECTANGLE else 0.0
                  for i in range(len(pfp["loop"]))]),
        np.array([pfp[f"loop.{i}.element[0].geometry.rectangle.z"] if geometry_type[i] == GEOMETRY_TYPE_RECTANGLE else 0.0
                  for i in range(len(pfp["loop"]))]),
    )
    assert len(through_ods) == len(direct) == 3
    for index in range(3):
        np.testing.assert_allclose(
            np.asarray(through_ods[index], dtype=float),
            np.asarray(direct[index], dtype=float),
        )


# ---------------------------------------------------------------------------
# Where a passive loop is placed
# ---------------------------------------------------------------------------

def test_a_loop_is_placed_at_the_centre_of_its_own_outline():
    """calc_grid divided the vertex sum by len-1 while summing every vertex.

    Every VEST passive loop is a four-vertex open polygon, so every centroid
    came out inflated by n/(n-1) -- putting a filament up to 461 mm from the
    6 mm conductor it stands for, and the vessel field with it.
    """
    from vaft.process.electromagnetics import _outline_centroid

    # A 6 mm square at (0.803, -0.5975), the shape VEST's loops actually are.
    r = [0.800, 0.806, 0.806, 0.800]
    z = [-0.6005, -0.6005, -0.5945, -0.5945]
    centre = _outline_centroid(r, z)
    assert centre == pytest.approx((0.803, -0.5975))

    inflated = (sum(r) / (len(r) - 1), sum(z) / (len(z) - 1))
    assert abs(inflated[0] - centre[0]) > 0.25  # the old answer, a quarter metre out


def test_a_closed_outline_does_not_weight_its_corner_twice():
    from vaft.process.electromagnetics import _outline_centroid

    square = ([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0])
    closed = ([0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0])
    assert _outline_centroid(*square) == pytest.approx((0.5, 0.5))
    assert _outline_centroid(*closed) == pytest.approx((0.5, 0.5))


def test_every_packaged_loop_sits_inside_its_own_outline(solved):
    """The property the old code violated for all 950 of them."""
    from vaft.process.electromagnetics import _outline_centroid

    loops = solved["pf_passive.loop"]
    for index in range(len(loops)):
        r = np.asarray(solved[f"pf_passive.loop.{index}.element[0].geometry.outline.r"], float)
        z = np.asarray(solved[f"pf_passive.loop.{index}.element[0].geometry.outline.z"], float)
        centre_r, centre_z = _outline_centroid(r, z)
        assert r.min() <= centre_r <= r.max(), index
        assert z.min() <= centre_z <= z.max(), index


def test_calc_grid_agrees_with_the_exact_greens_path_away_from_sources(solved):
    """The comparison that exposed the centroid bug, kept as the guard.

    Points sitting on a filament are excluded: both paths are singular there,
    and the exact one has no shift-averaging, so they legitimately diverge
    within a few millimetres of a conductor.
    """
    from vaft.omas.process_wrapper import compute_point_response_matrices_ods

    axis_r = np.linspace(0.10, 0.90, 11)
    axis_z = np.linspace(0.05, 1.40, 11)
    _, bz_grid_response, _ = vaft.omas.compute_grid_ods(solved, list(axis_r), list(axis_z))

    mesh_r, mesh_z = np.meshgrid(axis_r, axis_z, indexing="ij")
    points = np.column_stack([mesh_r.ravel(), mesh_z.ravel()])
    _, bz_exact_response, _ = compute_point_response_matrices_ods(solved, points.tolist())

    coils = len(solved["pf_active.coil"])
    loops = len(solved["pf_passive.loop"])
    k_active = int(np.argmin(np.abs(np.asarray(solved["pf_active.time"], float) - 0.3307)))
    k_passive = int(np.argmin(np.abs(np.asarray(solved["pf_passive.time"], float) - 0.3307)))
    currents = np.concatenate([
        [float(np.asarray(solved[f"pf_active.coil.{i}.current.data"], float)[k_active])
         for i in range(coils)],
        [float(np.asarray(solved[f"pf_passive.loop.{i}.current"], float)[k_passive])
         for i in range(loops)],
    ])
    width = currents.size
    from_grid = np.asarray(bz_grid_response, float)[:, :width] @ currents
    from_exact = np.asarray(bz_exact_response, float)[:, :width] @ currents

    filaments = np.array([
        [float(np.mean(solved[f"pf_passive.loop.{i}.element[0].geometry.outline.r"])),
         float(np.mean(solved[f"pf_passive.loop.{i}.element[0].geometry.outline.z"]))]
        for i in range(loops)
    ])
    distance = np.min(
        np.linalg.norm(points[:, None, :] - filaments[None, :, :], axis=2), axis=1
    )
    away = distance > 0.03
    assert away.sum() > 0.5 * away.size

    scale = np.abs(from_exact).max()
    relative = np.abs(from_grid - from_exact)[away] / scale
    assert relative.max() < 1e-3, relative.max()

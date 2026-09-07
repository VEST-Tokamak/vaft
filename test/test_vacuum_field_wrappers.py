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

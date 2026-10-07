"""Magnetic response split by source with an explicit plasma drive (#1795).

The eddy stage drives the wall with the measured Rogowski current; an
identifiability study (#918 H2: the magnetics carry 0.5-0.9 x the Rogowski
current) must choose the drive itself.  These tests pin the contract on the
packaged 39915 product: the drive is whatever the caller passes, never
``magnetics.ip``; the parts add up; scaling the drive scales exactly the
plasma parts; and the routine Rogowski drive is reproduced when passed
explicitly.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import vaft
from vaft.formula.green import green_r
from vaft.omas.magnetic_decomposition import (
    decompose_magnetic_response,
    reduced_wall_response,
    wall_currents_for_drive,
)

T0 = 0.31
SOURCES = [(0.35, 0.25), (0.35, 0.0), (0.35, -0.25)]  # the eddy stage's default filaments


@pytest.fixture(scope="module")
def sample():
    ods = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    pf_time = np.asarray(ods["pf_active.time"], dtype=float)
    ip_time = np.asarray(ods["magnetics.ip.0.time"] if "magnetics.ip.0.time" in ods else ods["magnetics.time"], dtype=float)
    ip = np.interp(pf_time, ip_time, np.asarray(ods["magnetics.ip.0.data"], dtype=float))
    return ods, np.vstack([ip / 3.0] * 3)


@pytest.fixture(scope="module")
def rogowski_drive(sample):
    ods, drive = sample
    return decompose_magnetic_response(ods, T0, plasma_sources=SOURCES, plasma_currents=drive)


def _railed_probe(ods):
    """A probe valid before 0.30975 s and railed (validity -2, 10 T) from then on."""
    ods = copy.deepcopy(ods)
    index = len(ods["magnetics.b_field_pol_probe"])
    base = f"magnetics.b_field_pol_probe.{index}"
    t = np.round(np.arange(0.300, 0.3201, 0.0001), 6)
    healthy = 0.05 * np.sin(2 * np.pi * 50 * t)
    rail = t >= 0.30975
    ods[f"{base}.name"] = "probe-railed"
    ods[f"{base}.position.r"] = 0.9
    ods[f"{base}.position.z"] = 0.0
    ods[f"{base}.field.data"] = np.where(rail, 10.0, healthy)
    ods[f"{base}.field.time"] = t
    ods[f"{base}.field.validity_timed"] = np.where(rail, -2, 0)
    ods[f"{base}.field.validity"] = 0
    return ods, index, t, healthy


def test_a_channel_invalid_at_the_instant_is_refused_although_valid_elsewhere_in_the_window(sample):
    """Selection accepts a channel with any valid sample in the +-0.5 ms window; the
    point read must not then interpolate from invalid samples (the rail, 10 T,
    instead of the healthy 0 T) -- cold review 0.8.0 delta-absorb-19 F2."""
    ods, drive = sample
    ods, index, t, healthy = _railed_probe(ods)
    channel = [("b_field_pol_probe", index)]
    with pytest.raises(ValueError, match="probe-railed is invalid at 0.31 s"):
        decompose_magnetic_response(ods, 0.31, plasma_sources=SOURCES, plasma_currents=drive, channels=channel)
    # both samples bracketing 0.3096 s are valid: read, from the valid samples only
    d = decompose_magnetic_response(ods, 0.3096, plasma_sources=SOURCES, plasma_currents=drive, channels=channel)
    assert d.channels[0]["name"] == "probe-railed"
    assert d.measured[0] == pytest.approx(float(np.interp(0.3096, t, healthy)), abs=1e-12)
    assert abs(d.measured[0]) < 0.1


def test_the_parts_add_up(rogowski_drive):
    """Bookkeeping only (wall_plasma is wall - wall_pf by construction); the scaling
    test below is what shows the split is a superposition."""
    d = rogowski_drive
    np.testing.assert_allclose(d.wall_pf + d.wall_plasma, d.wall, rtol=0, atol=1e-12)
    np.testing.assert_allclose(d.total, d.pf + d.wall + d.plasma)
    np.testing.assert_allclose(d.residual, d.measured - d.total)
    assert d.plasma_current() == pytest.approx(float(np.sum(d.plasma_currents)))


def test_a_zero_drive_gives_no_plasma_wall_although_magnetics_ip_is_populated(sample, rogowski_drive):
    """#1795 validation C: the critical regression -- the wall must not read the diagnostic."""
    ods, drive = sample
    assert np.max(np.abs(np.asarray(ods["magnetics.ip.0.data"]))) > 1e4
    zero = decompose_magnetic_response(ods, T0, plasma_sources=SOURCES, plasma_currents=np.zeros_like(drive))
    np.testing.assert_array_equal(zero.plasma, 0.0)
    np.testing.assert_allclose(zero.wall_plasma, 0.0, atol=1e-15)
    pf_only = decompose_magnetic_response(ods, T0)
    np.testing.assert_allclose(pf_only.wall, rogowski_drive.wall_pf, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(pf_only.pf, rogowski_drive.pf)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_scaling_the_drive_scales_only_the_plasma_parts(sample, rogowski_drive, scale):
    """#1795 validation B: direct field and plasma-driven wall move together, nothing else."""
    ods, drive = sample
    scaled = decompose_magnetic_response(ods, T0, plasma_sources=SOURCES, plasma_currents=scale * drive)
    np.testing.assert_allclose(scaled.plasma, scale * rogowski_drive.plasma, rtol=1e-12)
    np.testing.assert_allclose(scaled.wall_plasma, scale * rogowski_drive.wall_plasma, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(scaled.pf, rogowski_drive.pf)
    np.testing.assert_allclose(scaled.wall_pf, rogowski_drive.wall_pf)


def test_the_routine_rogowski_drive_is_reproduced_when_passed_explicitly(sample):
    """The eddy stage's solve call (``compute_eddy_currents`` with Rogowski Ip x 1/3 on
    its three default filaments, as ``build_eddy_ods`` drives it) gives the same wall
    as the explicit API.  The packaged sample stores no eddy currents to compare with."""
    from vaft.omas.process_wrapper import compute_eddy_currents

    ods, drive = sample
    routine = copy.deepcopy(ods)
    compute_eddy_currents(routine, SOURCES, list(drive))
    stored = np.array([np.asarray(routine[f"pf_passive.loop.{i}.current"]) for i in range(len(routine["pf_passive.loop"]))])
    _, explicit = wall_currents_for_drive(ods, SOURCES, drive)
    np.testing.assert_allclose(explicit, stored, rtol=1e-12, atol=1e-9)


def test_the_measurement_is_never_read_as_the_drive(sample):
    ods, drive = sample
    garbage = copy.deepcopy(ods)
    garbage["magnetics.ip.0.data"] = np.full_like(np.asarray(ods["magnetics.ip.0.data"]), 7.0e6)
    _, a = wall_currents_for_drive(ods, SOURCES, drive)
    _, b = wall_currents_for_drive(garbage, SOURCES, drive)
    np.testing.assert_array_equal(a, b)
    # and through the public decomposition: every model part is unchanged
    da = decompose_magnetic_response(ods, T0, plasma_sources=SOURCES, plasma_currents=drive)
    db = decompose_magnetic_response(garbage, T0, plasma_sources=SOURCES, plasma_currents=drive)
    for part in ("pf", "wall_pf", "wall_plasma", "plasma"):
        np.testing.assert_array_equal(getattr(da, part), getattr(db, part))


def test_units_and_the_plasma_flux_convention(rogowski_drive):
    """Flux loops in Wb of full poloidal flux (green_r), probes in T."""
    d = rogowski_drive
    loops = d.is_flux_loop
    assert {d.units[i] for i in np.flatnonzero(loops)} == {"Wb"}
    assert {d.units[i] for i in np.flatnonzero(~loops)} == {"T"}
    i = int(np.flatnonzero(loops)[0])
    r, z = d.channels[i]["r"], d.channels[i]["z"]
    expected = [float(green_r(np.array([r]), np.array([z]), R, Z)[0]) for R, Z in SOURCES]
    np.testing.assert_allclose(d.response_plasma[i], expected, rtol=1e-4)
    assert np.all(d.response_plasma[i] > 0)  # co-current ring flux is positive


def test_a_source_list_needs_its_currents_and_the_right_shape(sample):
    ods, drive = sample
    with pytest.raises(ValueError, match="magnetics.ip is a measurement"):
        wall_currents_for_drive(ods, SOURCES)
    with pytest.raises(ValueError, match=r"\(n_sources, n_times\)"):
        wall_currents_for_drive(ods, SOURCES, drive[:2])
    with pytest.raises(ValueError, match="without plasma_sources"):
        wall_currents_for_drive(ods, (), drive)
    with pytest.raises(ValueError, match="outside the source grid"):
        decompose_magnetic_response(ods, 10.0)


def test_the_full_wall_basis_reproduces_the_wall_response(sample, rogowski_drive):
    """#1795 item 4: G_wall V a == G_wall I_w for the full basis, a = V^T R I_w."""
    from vaft.omas.process_wrapper import compute_impedance_matrices_ods, compute_wall_mode_basis_ods
    from vaft.process.wall_modes import project

    ods, _ = sample
    work = copy.deepcopy(ods)
    basis = compute_wall_mode_basis_ods(work, remap_em_coupling=True, on_cluster="warn")
    R_mat, _, _ = compute_impedance_matrices_ods(work, [])
    reduced = reduced_wall_response(rogowski_drive, basis)
    a = project(basis, rogowski_drive.wall_currents, R_mat)
    np.testing.assert_allclose(reduced @ a, rogowski_drive.wall, rtol=1e-6, atol=1e-9)


def test_a_truncated_basis_maps_through_the_retained_modes(sample, rogowski_drive):
    """keep from select_slowest: G_wall V_keep, one column per retained mode."""
    from vaft.omas.process_wrapper import compute_wall_mode_basis_ods
    from vaft.process.wall_modes import select_slowest

    ods, _ = sample
    basis = compute_wall_mode_basis_ods(copy.deepcopy(ods), remap_em_coupling=True, on_cluster="warn")
    keep = select_slowest(basis, 2)
    reduced = reduced_wall_response(rogowski_drive, basis, keep)
    V = basis.V(keep)
    assert reduced.shape == (len(rogowski_drive.channels), V.shape[1]) and 0 < V.shape[1] < basis.n_elements
    np.testing.assert_allclose(reduced, rogowski_drive.response_wall @ V, rtol=1e-12)

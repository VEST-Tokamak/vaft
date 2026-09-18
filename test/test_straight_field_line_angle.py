"""Relabelling a geometric poloidal angle into a straight-field-line one.

FLARE reports a Poincare puncture at the lab angle about the magnetic axis;
GPEC and DCON label their harmonics by a straight-field-line angle. The two
agree only on a circular concentric equilibrium, so comparing a puncture
against an (m, n) harmonic needs the relabelling -- and the angle arrives in
turns with nothing in the file saying so.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.equilibrium import (
    lab_to_straight_field_line,
    straight_field_line_tables,
)

TURNS = np.linspace(0.0, 1.0, 65)  # closed, as GPEC writes it


def concentric(turns=TURNS, *, radius=0.5, axis=(1.8, 0.0), surfaces=3):
    """Circular surfaces about the axis: here the two angles coincide."""
    angle = 2.0 * np.pi * np.asarray(turns)
    radii = np.linspace(radius / surfaces, radius, surfaces)
    r = axis[0] + np.outer(np.cos(angle), radii)
    z = axis[1] + np.outer(np.sin(angle), radii)
    return r, z, axis


def shaped(turns=TURNS, *, axis=(1.8, 0.0)):
    """One elongated, shifted surface, where they do not."""
    angle = 2.0 * np.pi * np.asarray(turns)
    r = axis[0] + 0.5 * np.cos(angle) + 0.08 * np.cos(2 * angle)
    z = axis[1] + 0.9 * np.sin(angle)
    return r[:, None], z[:, None], axis


# --------------------------------------------------------------------------
# What the tables are
# --------------------------------------------------------------------------


def test_one_table_per_flux_surface_each_spanning_one_period():
    r, z, axis = concentric()
    tables = straight_field_line_tables(TURNS, r, z, axis, jacobian="hamada")
    assert len(tables) == r.shape[1]
    for lab, sfl in tables:
        assert lab.size == sfl.size
        assert np.all(np.diff(lab) > 0.0)
        assert lab[-1] - lab[0] == pytest.approx(2.0 * np.pi)


def test_the_closing_duplicate_is_dropped():
    """GPEC repeats its first row last so the grid is periodic; keeping it
    would put a zero-length step in every table."""
    r, z, axis = concentric()
    lab, _ = straight_field_line_tables(TURNS, r, z, axis, jacobian="hamada")[0]
    # 65 rows closed -> 64 distinct, plus the period-closing repeat.
    assert lab.size == 65
    assert np.all(np.diff(lab) > 0.0)


def test_on_concentric_circles_the_two_angles_coincide():
    r, z, axis = concentric()
    table = straight_field_line_tables(TURNS, r, z, axis, jacobian="hamada")[-1]
    for angle in (0.0, 1.0, np.pi, 5.0):
        assert lab_to_straight_field_line(angle, table) == pytest.approx(angle, abs=1e-9)


def test_on_a_shaped_surface_they_do_not():
    """The whole reason the relabelling exists. An elongated surface moves the
    angle by a lot: measured on the DIII-D reference's outermost surface, a
    lab angle of pi lands at 2.458."""
    r, z, axis = shaped()
    table = straight_field_line_tables(TURNS, r, z, axis, jacobian="hamada")[0]
    assert abs(lab_to_straight_field_line(np.pi / 2, table) - np.pi / 2) > 0.1


def test_the_relabelling_is_monotone_and_covers_the_circle():
    r, z, axis = shaped()
    table = straight_field_line_tables(TURNS, r, z, axis, jacobian="hamada")[0]
    angles = np.linspace(0.0, 2.0 * np.pi, 361, endpoint=False)
    mapped = lab_to_straight_field_line(angles, table)
    assert mapped.shape == angles.shape
    assert np.all((mapped >= 0.0) & (mapped < 2.0 * np.pi))
    # One full turn in, one full turn out.
    assert np.unwrap(mapped)[-1] - np.unwrap(mapped)[0] == pytest.approx(
        2.0 * np.pi, rel=0.02
    )


def test_an_angle_on_any_branch_gives_the_same_answer():
    """The table starts wherever the surface's outboard midplane fell, not at
    zero, so wrapping to [0, 2pi) and interpolating would read off its end."""
    r, z, axis = shaped()
    table = straight_field_line_tables(TURNS, r, z, axis, jacobian="hamada")[0]
    base = lab_to_straight_field_line(0.3, table)
    for turn in (-2, -1, 1, 5):
        shifted = lab_to_straight_field_line(0.3 + turn * 2.0 * np.pi, table)
        assert shifted == pytest.approx(base, abs=1e-9)


def test_a_surface_traced_the_other_way_gives_the_same_table():
    """The table has to increase whichever way the geometry was written."""
    r, z, axis = shaped()
    forward = straight_field_line_tables(TURNS, r, z, axis, jacobian="hamada")[0]
    backward = straight_field_line_tables(
        TURNS, r[::-1], z[::-1], axis, jacobian="hamada"
    )[0]
    for table in (forward, backward):
        assert np.all(np.diff(table[0]) > 0.0)


def test_a_clockwise_surface_closes_its_period_downwards():
    """Traced clockwise, the straight-field-line angle DEcreases along the
    table, so the closing entry sits 2 pi below the first. Closing it with
    +2 pi made the last interval sweep 4 pi the wrong way (cold review
    process F1)."""
    turns = np.linspace(0.0, 1.0, 33)
    query = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)
    mapped = {}
    for sign in (+1, -1):
        theta = sign * 2.0 * np.pi * turns
        r = (0.4 + 0.2 * np.cos(theta))[:, None]
        z = (0.2 * np.sin(theta))[:, None]
        (table,) = straight_field_line_tables(turns, r, z, (0.4, 0.0), jacobian="pest")
        mapped[sign] = lab_to_straight_field_line(query, table)
    for sign in (+1, -1):
        error = np.angle(np.exp(1j * (mapped[sign] - np.mod(sign * query, 2.0 * np.pi))))
        assert np.max(np.abs(error)) < 1e-9, sign

# --------------------------------------------------------------------------
# The units trap
# --------------------------------------------------------------------------


def test_the_angle_is_taken_in_turns():
    """GPEC writes theta_dcon from 0 to 1 with NO units attribute. Feeding it
    as radians is wrong by 2*pi, and the result looks like a mis-shaped
    plasma rather than a unit mistake -- so the parameter is named for what
    it holds and the conversion happens once, here."""
    r, z, axis = concentric()
    table = straight_field_line_tables(TURNS, r, z, axis, jacobian="hamada")[-1]
    # A quarter turn in is a quarter circle out.
    assert lab_to_straight_field_line(np.pi / 2, table) == pytest.approx(
        np.pi / 2, abs=1e-9
    )
    # Had turns been read as radians the surface would wrap many times and the
    # table could not be built at all.
    with pytest.raises(ValueError):
        straight_field_line_tables(TURNS * 2.0 * np.pi, r[:, :1] * 0 + axis[0],
                                   z[:, :1] * 0 + axis[1], axis, jacobian="hamada")


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_the_jacobian_must_be_named():
    """The file does not carry it, and the mapping means nothing without
    knowing which straight-field-line angle it lands in."""
    r, z, axis = concentric()
    with pytest.raises(ValueError, match="jacobian must name"):
        straight_field_line_tables(TURNS, r, z, axis, jacobian="")


def test_mismatched_geometry_is_refused():
    r, z, axis = concentric()
    with pytest.raises(ValueError, match="one 2-D"):
        straight_field_line_tables(TURNS, r, z[:, :1], axis, jacobian="hamada")
    with pytest.raises(ValueError, match="poloidal angles against"):
        straight_field_line_tables(TURNS[:-5], r, z, axis, jacobian="hamada")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_input_is_refused(bad):
    r, z, axis = concentric()
    broken = r.copy()
    broken[3, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        straight_field_line_tables(TURNS, broken, z, axis, jacobian="hamada")


def test_a_bad_axis_is_refused():
    r, z, _ = concentric()
    with pytest.raises(ValueError, match=r"finite \(r, z\) pair"):
        straight_field_line_tables(TURNS, r, z, (1.8,), jacobian="hamada")


def test_a_degenerate_surface_is_refused():
    """A surface that collapses onto the axis has no geometric angle to
    relabel against."""
    r, z, axis = concentric()
    collapsed = np.full_like(r[:, :1], axis[0])
    with pytest.raises(ValueError, match="distinct geometric angles"):
        straight_field_line_tables(
            TURNS, collapsed, np.full_like(collapsed, axis[1]), axis, jacobian="hamada"
        )


def test_a_malformed_table_is_refused():
    with pytest.raises(ValueError, match="pairs"):
        lab_to_straight_field_line(0.0, (np.zeros(4), np.zeros(3)))
    with pytest.raises(ValueError, match="at least three"):
        lab_to_straight_field_line(0.0, (np.zeros(2), np.zeros(2)))

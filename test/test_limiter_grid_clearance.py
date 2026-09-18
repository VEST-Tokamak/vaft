"""The limiter must not sit on an EFIT grid line (#965).

EFIT decides which grid points may carry plasma current with ``zlim``, a
point-in-polygon test. A grid column a fraction of a millimetre from a
vertical limiter face turns that into a knife edge: with the inboard face at
R = 0.104 m the default grid's column at R = 0.10391 m sat 0.09 mm outside it,
and moving the face 0.01 mm inward (so that 49 points of that column joined the
plasma region) took 3 of 4 inboard-limited 39915 slices from converged to
diverged, with nothing else changed. The face moved to R = 0.105 m, which keeps
the same plasma grid and puts 1.09 mm between it and the column.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.code.efit.efund import EFUNDConfig
from vaft.data.resources import data_path
from vaft.machine_mapping.static_geometry import load_static_ods

#: Smallest distance a limiter face may have from a grid line of the default
#: EFIT grid. Chosen well above anything a coordinate rounding or a unit
#: conversion can move a face by, and below the 1.09 mm the inboard fix gives.
MIN_CLEARANCE_M = 5.0e-4

LIMITER_FILE = data_path("efit/lim.dat")
STATIC_GEOMETRY = data_path("geometry/VEST_static_geometry.json.gz")


def _static_outline():
    ods = load_static_ods(STATIC_GEOMETRY)
    r = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.r"], dtype=float)
    z = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.z"], dtype=float)
    return r, z


def _lim_dat():
    lines = LIMITER_FILE.read_text(encoding="utf-8").split("\n")
    count = int(lines[0].split()[-1])
    points = np.array([[float(v) for v in line.split()] for line in lines[1 : count + 1]])
    return points[:, 0], points[:, 1]


def _grid():
    config = EFUNDConfig()
    r = np.linspace(config.rleft, config.rright, config.nw)
    z = np.linspace(config.zbotto, config.ztop, config.nh)
    return r, z


def _faces(r, z):
    """(vertical face radii, horizontal face heights) of an axis-aligned outline."""
    vertical, horizontal = set(), set()
    for (r0, z0), (r1, z1) in zip(zip(r[:-1], z[:-1]), zip(r[1:], z[1:])):
        if r0 == r1 and z0 != z1:
            vertical.add(float(r0))
        elif z0 == z1 and r0 != r1:
            horizontal.add(float(z0))
    return sorted(vertical), sorted(horizontal)


def test_efit_reads_the_same_limiter_the_machine_description_has():
    """lim.dat is what EFIT limits against; the wall IDS is what every other code does."""
    static_r, static_z = _static_outline()
    lim_r, lim_z = _lim_dat()
    np.testing.assert_allclose(lim_r, static_r, rtol=0, atol=1e-9)
    np.testing.assert_allclose(lim_z, static_z, rtol=0, atol=1e-9)


def test_the_inboard_face_is_the_numerically_chosen_one():
    """0.105 m is a numerical choice, not a measured tile surface -- see the module docstring."""
    r, _ = _static_outline()
    assert r.min() == pytest.approx(0.105, abs=1e-12)


_VERTICAL, _HORIZONTAL = _faces(*_static_outline())
#: A face the guard has found but that has not been moved yet. strict: moving
#: it makes the test pass, and XPASS then fails until the entry is removed.
_KNOWN = {0.76: "outboard face is 0.23 mm from the column at R = 0.759766 m (inside the plasma region)"}


@pytest.mark.parametrize(
    "face",
    [
        pytest.param(face, marks=pytest.mark.xfail(strict=True, reason=_KNOWN[face]))
        if face in _KNOWN else face
        for face in _VERTICAL
    ],
)
def test_no_vertical_limiter_face_sits_on_a_grid_column(face):
    columns, _ = _grid()
    assert np.abs(columns - face).min() >= MIN_CLEARANCE_M


@pytest.mark.parametrize("face", _HORIZONTAL)
def test_no_horizontal_limiter_face_sits_on_a_grid_row(face):
    _, rows = _grid()
    assert np.abs(rows - face).min() >= MIN_CLEARANCE_M

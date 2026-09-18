"""The Ejiri mirror proxy: the process half of issue #676.

The two relations live in ``vaft.formula.startup`` and are tested there. What is
pinned here is the geometry read off a field line: the synthetic fields below are
built so that every field line has a known shape, which makes each of the four
Ejiri lengths known exactly.
"""

import warnings

import numpy as np
import pytest

from vaft.formula.startup import (
    ejiri_f3_from_alpha,
    ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max,
)
from vaft.process.equilibrium import ejiri_mirror_geometry

#: B_phi * R of a small tokamak's vacuum field, so the trace pitch is realistic.
R0B0 = 0.08
STEP = np.deg2rad(0.5)


def _field(slope):
    """Every field line satisfies dR/dZ = slope(Z), with B_Z uniform."""

    def b_field(R, Z):
        Z = np.asarray(Z, dtype=float)
        return 0.01 * slope(Z), np.full_like(Z, 0.01), R0B0 / np.asarray(R, dtype=float)

    return b_field


def _box(top=0.4, bottom=-0.4, inner=0.15, outer=0.85):
    return np.array([inner, outer, outer, inner]), np.array([bottom, bottom, top, top])


def _parabola(R_C):
    # R = c - Z^2/(2 R_C) for every line: the model's own shape, exactly.
    return _field(lambda Z: -Z / R_C)


# ---------------------------------------------------------------------------
# The geometry, on fields where it is known exactly
# ---------------------------------------------------------------------------

def test_an_exact_parabola_recovers_every_ejiri_length():
    R_S, R_C = 0.5, 0.6
    wall_r, wall_z = _box()
    out = ejiri_mirror_geometry(R_S, _parabola(R_C), wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    assert out["curvature_radius"] == pytest.approx(R_C, rel=1e-9)
    assert out["r_inboard_limiter"] == pytest.approx(0.15, rel=1e-12)
    # The wall stops each branch within one step of Z = +-0.4.
    assert out["z_max"] == pytest.approx(0.4, abs=1e-3)
    assert out["reason_upper"] == out["reason_lower"] == "wall"
    assert out["mirror"] is True
    assert out["saturated"] is False
    # And the result is the formula on the true inputs, not a re-derivation.
    alpha = ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max(R_S, 0.15, R_C, out["z_max"])
    assert out["alpha"] == pytest.approx(alpha, rel=1e-9)
    assert out["f3"] == pytest.approx(ejiri_f3_from_alpha(alpha), rel=1e-9)


def test_the_secant_curvature_is_the_parabola_through_the_mirror_point():
    # R_C = Z_max^2 / (2 (R_S - R_m)), which makes Ejiri's curvature term exactly
    # 1/sqrt(M - 1) for the line's true mirror ratio M = R_S / R_m.
    R_S = 0.5
    wall_r, wall_z = _box()
    out = ejiri_mirror_geometry(R_S, _parabola(0.6), wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    binding = out["binding_branch"]
    r_mirror = out[f"r_mirror_{binding}"]
    assert out["curvature_radius"] == pytest.approx(
        out["z_max"] ** 2 / (2.0 * (R_S - r_mirror)), rel=1e-12
    )
    curvature_term = np.sqrt(
        (2.0 * out["curvature_radius"] * R_S - out["z_max"] ** 2) / out["z_max"] ** 2
    )
    assert curvature_term == pytest.approx(np.sqrt(r_mirror / (R_S - r_mirror)), rel=1e-10)


def test_the_result_does_not_depend_on_the_fit_window():
    # On a real field the local parabola fit moves with its window; the secant
    # does not use it at all, so only the diagnostic can change.
    wall_r, wall_z = _box()
    field = _field(lambda Z: -Z / 0.6 + Z**3 / 0.05)
    results = [
        ejiri_mirror_geometry(0.5, field, wall_r=wall_r, wall_z=wall_z, dphi=STEP, z_fit=zf)
        for zf in (0.02, 0.05, 0.1, 0.2)
    ]
    for out in results[1:]:
        assert out["curvature_radius"] == results[0]["curvature_radius"]
        assert out["f3"] == results[0]["f3"]
    assert len({round(r["curvature_radius_local"], 6) for r in results}) > 1


def test_a_turning_line_stops_where_r_has_its_minimum():
    # dR/dZ = -Z/R_C + Z^3/k has a minimum of R at |Z| = sqrt(k / R_C).
    R_C, k = 0.6, 0.012
    wall_r, wall_z = _box()
    out = ejiri_mirror_geometry(
        0.5, _field(lambda Z: -Z / R_C + Z**3 / k), wall_r=wall_r, wall_z=wall_z, dphi=STEP
    )
    assert out["z_max"] == pytest.approx(np.sqrt(k / R_C), abs=2e-3)
    assert out["reason_upper"] == out["reason_lower"] == "turning"


def test_a_line_tilted_at_the_midplane_still_finds_its_mirror():
    # The VEST case in miniature.  dR/dZ = c1 - Z/R_C + Z^3/k: on the upper side
    # R first *rises* a hair (slope c1 > 0 at the seed), then dips to an
    # interior minimum near Z = 0.123, then climbs past R_S to the outer wall.
    # A first-local-minimum rule gives up on the initial rise, reads the wall
    # end -- which lies outboard of the start -- as the mirror point, and
    # reports no confinement.  The global minimum finds the dip.
    c1, R_C, k = 0.05, 0.6, 0.012
    field = _field(lambda Z: c1 - Z / R_C + Z**3 / k)
    wall_r, wall_z = _box()
    out = ejiri_mirror_geometry(0.5, field, wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    assert out["reason_upper"] == "turning"
    assert out["r_mirror_upper"] < 0.5
    # The dip sits where dR/dZ changes sign from negative to positive.
    z = np.linspace(0.05, 0.3, 200001)
    slope = c1 - z / R_C + z**3 / k
    z_dip = z[np.flatnonzero((slope[:-1] < 0) & (slope[1:] >= 0))[0]]
    assert out["z_max_upper"] == pytest.approx(z_dip, abs=2e-3)
    assert out["mirror"] is True
    assert out["f3"] > 0.0


def test_a_line_that_dips_twice_is_read_at_its_deeper_dip():
    # dR/dZ = -sin(40 Z)(0.2 + 2 Z): R falls, rises, falls again, and each dip is
    # deeper than the last because the amplitude grows with Z.  Passing the
    # first, shallower field peak is not escape -- the electron still has to get
    # over the deeper one before the wall -- so the mirror point is past it.
    field = _field(lambda Z: -np.sin(40.0 * Z) * (0.2 + 2.0 * Z))
    wall_r, wall_z = _box()
    out = ejiri_mirror_geometry(0.5, field, wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    first_dip = np.pi / 40.0
    assert out["z_max_upper"] > 1.5 * first_dip
    # And it is the smallest R anywhere on the branch.
    assert out["r_mirror_upper"] == pytest.approx(
        float(np.min(out["trace_upper"]["R"])), abs=1e-12
    )


def test_the_weaker_mirror_binds():
    # An asymmetric wall stops the upper branch sooner, so it rises less in |B|:
    # its mirror point sits at the larger R, and that branch sets Z_max.
    wall_r, wall_z = _box(top=0.3, bottom=-0.5)
    out = ejiri_mirror_geometry(0.5, _parabola(0.6), wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    assert out["binding_branch"] == "upper"
    assert out["r_mirror_upper"] > out["r_mirror_lower"]
    assert out["z_max"] == out["z_max_upper"]
    assert out["z_max"] == pytest.approx(0.3, abs=1e-3)


@pytest.mark.parametrize(
    "slope", [lambda Z: 0.0 * Z, lambda Z: Z / 0.6], ids=["straight", "outward"]
)
def test_a_line_that_never_dips_inward_confines_nothing(slope):
    # Not an error: |B| does not rise away from the midplane, so curvature
    # traps nothing, and the formulas -- which reject a non-positive R_C -- are
    # not called.
    wall_r, wall_z = _box()
    out = ejiri_mirror_geometry(0.5, _field(slope), wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    assert out["mirror"] is False
    assert out["alpha"] == np.inf
    assert out["f3"] == 0.0
    assert out["curvature_radius"] == np.inf


# ---------------------------------------------------------------------------
# Convergence and refusals
# ---------------------------------------------------------------------------

def test_a_trace_cut_short_is_flagged_and_warned():
    wall_r, wall_z = _box()
    with pytest.warns(RuntimeWarning, match="not converged") as record:
        out = ejiri_mirror_geometry(
            0.5, _parabola(0.6), wall_r=wall_r, wall_z=wall_z, dphi=STEP, max_length_m=0.5
        )
    assert out["saturated"] is True
    assert "max_length_m" in (out["reason_upper"], out["reason_lower"])
    # The warning blames the caller, not a line inside the module.
    assert record[0].filename == __file__


def test_a_converged_trace_does_not_warn():
    wall_r, wall_z = _box()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = ejiri_mirror_geometry(0.5, _parabola(0.6), wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    assert out["saturated"] is False


@pytest.mark.parametrize("r_start", [0.10, 0.95], ids=["inboard_of_the_wall", "outboard_of_the_wall"])
def test_a_start_outside_the_wall_is_refused(r_start):
    wall_r, wall_z = _box()
    with pytest.raises(ValueError, match="not inside the wall"):
        ejiri_mirror_geometry(r_start, _parabola(0.6), wall_r=wall_r, wall_z=wall_z, dphi=STEP)


@pytest.mark.parametrize("z_fit", [0.0, -0.1, np.nan])
def test_a_non_positive_fit_window_is_refused(z_fit):
    wall_r, wall_z = _box()
    with pytest.raises(ValueError, match="z_fit"):
        ejiri_mirror_geometry(
            0.5, _parabola(0.6), wall_r=wall_r, wall_z=wall_z, dphi=STEP, z_fit=z_fit
        )


def test_the_inboard_limiter_is_the_nearest_crossing_inboard_of_the_start():
    # A wall with a step on its inboard side crosses the midplane once; a
    # polygon that also has an inner notch must report the crossing nearest the
    # start, which is the one an electron drifting inward meets first.
    wall_r = np.array([0.15, 0.85, 0.85, 0.30, 0.30, 0.15])
    wall_z = np.array([-0.4, -0.4, 0.4, 0.4, 0.1, 0.1])
    # The midplane is crossed by the 0.15 edge only.
    out = ejiri_mirror_geometry(0.5, _parabola(0.6), wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    assert out["r_inboard_limiter"] == pytest.approx(0.15, rel=1e-12)


# ---------------------------------------------------------------------------
# Cold review of #958
# ---------------------------------------------------------------------------

def test_a_start_one_step_from_the_wall_reads_as_no_confinement():
    # A steep poloidal field and a coarse step: the first step on one side
    # already leaves the polygon, so that branch is a single point.  It used to
    # reach np.argmin of an empty array and raise an unexplained ValueError, and
    # a nanmedian of an empty slice warned on the way.  An electron heading that
    # way is lost at once, so nothing is confined.
    def steep(R, Z):
        Z = np.asarray(Z, dtype=float)
        return np.zeros_like(Z), np.full_like(Z, 5.0), R0B0 / np.asarray(R, dtype=float)

    wall_r, wall_z = _box()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = ejiri_mirror_geometry(0.5, steep, wall_r=wall_r, wall_z=wall_z, dphi=np.deg2rad(5.0))
    assert out["mirror"] is False
    assert out["f3"] == 0.0
    assert "wall" in (out["reason_upper"], out["reason_lower"])


def test_the_labels_do_not_depend_on_which_direction_rises():
    # Reversing the toroidal field swaps which trace direction climbs; the
    # upper/lower labels must follow the geometry, not the direction.
    wall_r, wall_z = _box(top=0.3, bottom=-0.5)

    def reversed_parabola(R, Z):
        b_r, b_z, b_phi = _parabola(0.6)(R, Z)
        return b_r, b_z, -b_phi

    a = ejiri_mirror_geometry(0.5, _parabola(0.6), wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    b = ejiri_mirror_geometry(0.5, reversed_parabola, wall_r=wall_r, wall_z=wall_z, dphi=STEP)
    for out in (a, b):
        assert out["z_max_upper"] == pytest.approx(0.3, abs=1e-3)
        assert out["z_max_lower"] == pytest.approx(0.5, abs=1e-3)
    assert a["f3"] == pytest.approx(b["f3"], rel=1e-9)

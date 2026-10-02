"""The island separatrix overlay (migration slice V5, convention C-20).

The islands GPEC's result implies, drawn in the poloidal plane on the mesh
the run itself solved on.  Both halves are under test: that the mapper writes
the mesh at all, and that the overlay derives its phase rather than assuming
one.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pytest
from omas import ODS

from gpec_nc_fixtures import write_control_nc, write_cylindrical_nc, write_profile_nc
from vaft.machine_mapping.gpec_ideal import gpec_ideal
from vaft.plot.backend.recipes import RECIPES, _gpec_resonant_table

MESH = "mhd_linear.time_slice.0.toroidal_mode.0.plasma.coordinate_system"


#: Poloidal samples the fixture's flux-surface mesh carries. A separatrix of
#: m lobes needs a circuit that can hold them: the default 7 cannot represent
#: an m = 3 island at all, so the lobe-count tests would be reading noise.
THETA_COUNT = 129


def _run(path, *, n=1, rational_q=(2.0, 3.0), theta_count=THETA_COUNT):
    write_control_nc(path, n=n)
    write_cylindrical_nc(path, n=n)
    return write_profile_nc(
        path, n=n, rational_q=rational_q, theta_count=theta_count
    )


@pytest.fixture
def mapped(tmp_path):
    native = _run(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1], "include_geometry": True})
    return ods, native


# --- the mesh the mapper now writes ------------------------------------------

def test_the_flux_surface_mesh_reaches_the_ids_untransformed(mapped):
    ods, native = mapped

    np.testing.assert_array_equal(ods[f"{MESH}.grid.dim1"], native["psi_n"])
    np.testing.assert_array_equal(ods[f"{MESH}.grid.dim2"], native["theta"])
    # (theta, psi) in the container, (dim1, dim2) = (psi, theta) in the IDS.
    assert ods[f"{MESH}.r"].shape == (native["psi_n"].size, native["theta"].size)


def test_the_mesh_shares_the_spectral_fields_radial_grid(mapped):
    """A renderer reads a harmonic at psi_n[k] and its position at psi_n[k].
    Two radial grids would draw the right number in the wrong place."""
    ods, _ = mapped
    np.testing.assert_array_equal(
        ods[f"{MESH}.grid.dim1"],
        ods["mhd_linear.time_slice.0.toroidal_mode.0.plasma.grid.dim1"],
    )


def test_the_mesh_grid_type_is_private_and_says_why(mapped):
    """The poloidal angle is DCON's, set by the run's jacobian; the IMAS
    identifier's (psi, theta) grids name three other angles."""
    ods, _ = mapped
    assert ods[f"{MESH}.grid_type.index"] < 0
    assert "jacobian" in ods[f"{MESH}.grid_type.description"]


def test_a_caller_that_wants_only_the_spectrum_can_refuse_the_mesh(tmp_path):
    _run(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1], "include_geometry": False})

    assert ods.get(f"{MESH}.r", None) is None
    # ... and the spectral field it did ask for is still there.
    assert ods["mhd_linear.time_slice.0.toroidal_mode.0.plasma.grid.dim1"].size


def test_the_mesh_is_opt_in(tmp_path):
    """It is the largest thing the mapper writes -- two FLT_2D arrays over
    the full radial grid -- and only a real-space figure needs it, so a
    default caller does not pay for it."""
    _run(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})

    assert ods.get(f"{MESH}.r", None) is None
    # ... and what the default does write is still there.
    assert ods["mhd_linear.time_slice.0.toroidal_mode.0.plasma.grid.dim1"].size


def test_the_mesh_can_still_be_asked_for_without_the_spectral_field(tmp_path):
    _run(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(
        ods, str(tmp_path),
        {"modes": [1], "include_spectral": False, "include_geometry": True},
    )

    assert ods[f"{MESH}.r"].size


# --- the overlay --------------------------------------------------------------

def test_one_island_and_one_rational_surface_are_drawn_per_resonance(mapped):
    ods, _ = mapped
    table = _gpec_resonant_table(ods)
    model = RECIPES["mhd_linear_geometry_island"].builder(ods)

    # the outermost surface, then a (surface, island) pair per resonance
    assert len(model.layers) == 1 + 2 * len(table["rows"])
    labelled = [layer.label for layer in model.layers if layer.label]
    assert all("m/n" in label for label in labelled[1:])


def test_each_separatrix_is_a_closed_curve_on_the_runs_own_mesh(mapped):
    ods, native = mapped
    model = RECIPES["mhd_linear_geometry_island"].builder(ods)

    island = model.layers[2]
    assert island.kind == "polygon"
    # the outward half of the separatrix and the inward half traced back
    assert island.r.size == 2 * native["theta"].size
    assert np.all(np.isfinite(island.r)) and np.all(np.isfinite(island.z))


def test_the_island_half_width_is_the_derived_one(mapped):
    """The excursion either side of the rational surface is w/2, and w is the
    table's, so the figure and the island-width profile cannot disagree."""
    ods, _ = mapped
    table = _gpec_resonant_table(ods)
    model = RECIPES["mhd_linear_geometry_island"].builder(ods)

    for index, row in enumerate(table["rows"]):
        assert f"{float(row['w_isl']):.4g}" in model.layers[2 + 2 * index].label


def test_the_slice_the_figure_drew_is_on_the_title(mapped):
    ods, _ = mapped
    assert "0°" in RECIPES["mhd_linear_geometry_island"].builder(ods).title
    assert "47°" in RECIPES["mhd_linear_geometry_island"].builder(ods, phi_deg=47).title


def test_a_result_with_no_mesh_says_how_to_get_one(tmp_path):
    """Refusing beats drawing the islands against an equilibrium's own flux
    surfaces. Those are the same surfaces, but nothing ties the run's radial
    grid to the equilibrium's, and the resonant phase is referenced to the
    run's own angle origin."""
    _run(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1], "include_geometry": False})

    with pytest.raises(ValueError, match="include_geometry=True"):
        RECIPES["mhd_linear_geometry_island"].builder(ods)


def test_the_island_phase_is_derived_and_never_assumed_to_be_zero(mapped):
    """C-20: the figure this replaces fell back to arg(I_res) = 0 whenever
    I_res was absent, silently rotating every island. The phase here comes
    from the same jump the flux does, so there is nothing to fall back from.
    """
    ods, _ = mapped
    table = _gpec_resonant_table(ods)

    phases = [row["phase"] for row in table["rows"]]
    assert all(np.isfinite(phase) for phase in phases)
    assert any(abs(phase) > 1e-9 for phase in phases)
    for row in table["rows"]:
        assert row["phase"] == pytest.approx(float(np.angle(row["Phi_res"])))


def test_a_run_in_non_straight_field_line_coordinates_is_refused(tmp_path):
    """`m*theta - n*phi` is the helical phase only in the PEST angle, whose
    toroidal partner is the machine `phi`. GPEC's `equal_arc` angle is not even
    a straight-field-line one, and relabelling it into PEST needs `B_p` on the
    mesh, which no `coordinate_system` carries.
    """
    _run(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1], "include_geometry": True})
    ods["mhd_linear.code.parameters"] = ods["mhd_linear.code.parameters"].replace(
        "<jacobian>hamada</jacobian>", "<jacobian>equal_arc</jacobian>"
    )

    with pytest.raises(ValueError, match="equal_arc"):
        RECIPES["mhd_linear_geometry_island"].builder(ods)


def test_a_poloidal_angle_in_radians_is_refused(mapped):
    """Every reader of this mesh multiplies by 2*pi, so a mesh that was
    already in radians would put the island phase out by that factor."""
    ods, _ = mapped
    ods[f"{MESH}.grid.dim2"] = ods[f"{MESH}.grid.dim2"] * (2.0 * np.pi)

    with pytest.raises(ValueError, match="normalized to 1"):
        RECIPES["mhd_linear_geometry_island"].builder(ods)


def test_a_separatrix_that_leaves_the_mesh_breaks_rather_than_flattening(mapped):
    """GPEC's grid stops short of psi_N = 1, so the outermost resonance of a
    strongly driven run routinely reaches past it -- it is the normal case,
    not an edge case. `np.interp` would clamp those points to the last mapped
    surface and draw an island with a flat side; there is no geometry there,
    so the curve breaks and the title says which resonance it happened to.
    """
    ods, _ = mapped
    table = _gpec_resonant_table(ods)
    outer = max(table["rows"], key=lambda row: row["psi_n"])
    # Widen the outermost island until its separatrix runs off the mesh.
    psi_max = float(ods[f"{MESH}.grid.dim1"][-1])
    width = 4.0 * (psi_max - float(outer["psi_n"])) + 0.01

    import vaft.plot.backend.recipes as recipes

    original = recipes._gpec_resonant_table

    def widened(ods_, **options):
        result = original(ods_, **options)
        for row in result["rows"]:
            if row is outer or row["psi_n"] == outer["psi_n"]:
                row["w_isl"] = width
        return result

    recipes._gpec_resonant_table = widened
    try:
        model = RECIPES["mhd_linear_geometry_island"].builder(ods)
    finally:
        recipes._gpec_resonant_table = original

    island = model.layers[-1]
    assert not np.isfinite(island.r).all(), "the off-mesh points were clamped"
    assert np.isfinite(island.r).any(), "the whole island was dropped"
    assert "outermost mapped surface" in model.title
    # Every drawn point is on the mesh, not pinned to its last surface.
    drawn = island.r[np.isfinite(island.r)]
    assert drawn.size < island.r.size


def test_an_island_entirely_off_the_mesh_is_refused(mapped):
    """Nothing to draw, and a figure with a silently missing resonance is
    worse than one that says why."""
    ods, _ = mapped
    import vaft.plot.backend.recipes as recipes

    original = recipes._gpec_resonant_table

    def off_mesh(ods_, **options):
        result = original(ods_, **options)
        for row in result["rows"]:
            row["psi_n"] = float(ods[f"{MESH}.grid.dim1"][-1]) + 10.0
        return result

    recipes._gpec_resonant_table = off_mesh
    try:
        with pytest.raises(ValueError, match="entirely outside the mapped mesh"):
            RECIPES["mhd_linear_geometry_island"].builder(ods)
    finally:
        recipes._gpec_resonant_table = original


# --- the shape itself ---------------------------------------------------------

def _branch_extrema(values):
    """(minima, maxima) counts of a periodic sequence, on the closed loop."""
    slope = np.sign(np.diff(np.concatenate([values, values[:1]])))
    minima = int(np.sum((slope > 0) & (np.roll(slope, 1) < 0)))
    maxima = int(np.sum((slope < 0) & (np.roll(slope, 1) > 0)))
    return minima, maxima


def _separation(layer):
    """Distance between the outward and inward branches, sample by sample."""
    half = layer.r.size // 2
    outer_r, outer_z = layer.r[:half], layer.z[:half]
    inner_r, inner_z = layer.r[half:][::-1], layer.z[half:][::-1]
    return np.hypot(outer_r - inner_r, outer_z - inner_z)


def _o_point(layer):
    """Where on the flux surface the island is widest, in (R, z).

    The circuit is drawn starting from an X-point, so a phase shift rotates
    the *sample order* with it and leaves the separation-versus-index profile
    almost unchanged. Where the lobes sit in the poloidal plane is what
    actually moves, which is why the phase tests read this and not that.
    """
    half = layer.r.size // 2
    index = int(np.argmax(_separation(layer)))
    return np.array([
        0.5 * (layer.r[index] + layer.r[half:][::-1][index]),
        0.5 * (layer.z[index] + layer.z[half:][::-1][index]),
    ])


def _surface_extent(layer):
    """The drawn curve's own size, to measure a displacement against."""
    return max(np.ptp(layer.r), np.ptp(layer.z))


def test_each_island_closes_into_m_lobes(mapped):
    """The separatrix is `(w/2)|cos(xi/2)|`, not `(w/2)cos(xi)`.

    The half-angle is the whole shape. `cos(xi)` crosses zero twice per
    helical period, so its two branches meet `2m` times per circuit rather
    than `m`, and the lobes it closes are centred on `xi = pi` -- which is
    where the X-points are. `|cos(xi/2)|` touches zero once per period, at
    the X-point, and peaks at the O-point.
    """
    ods, _ = mapped
    table = _gpec_resonant_table(ods)
    model = RECIPES["mhd_linear_geometry_island"].builder(ods)

    for index, row in enumerate(table["rows"]):
        separation = _separation(model.layers[2 + 2 * index])
        assert np.all(np.isfinite(separation)), row["m_pol"]
        pinches, lobes = _branch_extrema(separation[:-1])
        assert pinches == row["m_pol"], (row["m_pol"], pinches)
        assert lobes == row["m_pol"], (row["m_pol"], lobes)


def test_the_branches_meet_at_the_x_points_and_open_to_w_at_the_o_points(mapped):
    """The separation is zero where the branches cross and the island's full
    width where they are furthest apart; anything else is a different curve.
    """
    from vaft.formula.stability import island_separatrix_half_width

    ods, _ = mapped
    table = _gpec_resonant_table(ods)
    mesh_psi = ods[f"{MESH}.grid.dim1"]
    model = RECIPES["mhd_linear_geometry_island"].builder(ods)

    for index, row in enumerate(table["rows"]):
        separation = _separation(model.layers[2 + 2 * index])
        # The widest separation is the island's full width, mapped into (R, z)
        # through the local dpsi -> dR; comparing in psi keeps it exact.
        assert island_separatrix_half_width(0.0, row["w_isl"]) == pytest.approx(
            0.5 * row["w_isl"]
        )
        assert island_separatrix_half_width(np.pi, row["w_isl"]) == pytest.approx(0.0)
        # On the drawn curve the pinch is limited only by how near a theta
        # sample lands to the X-point, which is a grid property: with N
        # samples per circuit the worst case is half a sample either side.
        samples_per_lobe = (ods[f"{MESH}.grid.dim2"].size - 1) / row["m_pol"]
        assert separation.min() < separation.max() * (
            2.0 * np.pi / samples_per_lobe
        ), row["m_pol"]
        assert float(np.max(mesh_psi)) > row["psi_n"]


def test_shifting_the_island_phase_by_pi_moves_the_lobes(mapped):
    """C-20's phase reference has to be visible in the figure, or getting it
    right is untestable. Half a helical period puts the O-points where the
    X-points were, which moves them around the flux surface.
    """
    ods, _ = mapped

    import vaft.plot.backend.recipes as recipes

    original = recipes._gpec_resonant_table

    def rotated(ods_, **options):
        result = original(ods_, **options)
        for entry in result["rows"]:
            entry["phase"] = entry["phase"] + np.pi
        return result

    plain = RECIPES["mhd_linear_geometry_island"].builder(ods)
    recipes._gpec_resonant_table = rotated
    try:
        shifted = RECIPES["mhd_linear_geometry_island"].builder(ods)
    finally:
        recipes._gpec_resonant_table = original

    # Same island -- same lobe count and the same widest separation ...
    before, after = _separation(plain.layers[2]), _separation(shifted.layers[2])
    assert _branch_extrema(before[:-1]) == _branch_extrema(after[:-1])
    assert after.max() == pytest.approx(before.max(), rel=1e-6)
    # ... sitting somewhere else on the surface.
    moved = np.linalg.norm(_o_point(plain.layers[2]) - _o_point(shifted.layers[2]))
    assert moved > 0.1 * _surface_extent(plain.layers[2])


def test_the_toroidal_angle_moves_the_pattern_without_changing_it(mapped):
    """`n*phi` enters with the sign `helical_phase` gives it, so advancing phi
    slides the lobes around the circuit rather than reshaping them. The legacy
    `+n phi` agrees at its own default of phi = 0 and rotates the other way
    everywhere else, which is invisible unless the figure is drawn off-axis.
    """
    ods, _ = mapped

    plain = RECIPES["mhd_linear_geometry_island"].builder(ods).layers[2]
    moved = RECIPES["mhd_linear_geometry_island"].builder(ods, phi_deg=47.0).layers[2]

    assert _branch_extrema(_separation(plain)[:-1]) == _branch_extrema(
        _separation(moved)[:-1]
    )
    # Both maxima are samples of (w/2)|cos(xi/2)|, taken wherever the mesh
    # happens to fall relative to the O-point: sample spacing m*dtheta in xi puts
    # the nearest sample within m*dtheta/4 of the peak in the half-angle, so each
    # maximum lies in [(w/2)cos(m*dtheta/4), w/2]. That band -- not a fixed
    # relative 1e-6, which pytest.approx's default abs=1e-12 was silently
    # overriding at this 1e-9 scale -- is how far they may differ. layers[2] is
    # the m = 2 island (3.0e-4); measured drift over phi = 5..311 deg peaks at
    # 1.8e-4, and 2.3e-4 was seen on a CI runner.
    assert plain.label.startswith("m/n = 2/"), plain.label  # the m the bound uses
    sampling = 1.0 - np.cos(2 * (2.0 * np.pi / (THETA_COUNT - 1)) / 4.0)
    assert _separation(moved).max() == pytest.approx(
        _separation(plain).max(), rel=sampling, abs=0.0
    )
    displaced = np.linalg.norm(_o_point(plain) - _o_point(moved))
    assert displaced > 0.05 * _surface_extent(plain)


def test_the_slice_is_periodic_in_one_full_turn_of_the_pattern(mapped):
    """Advancing phi by 360/n degrees returns the same figure: that is what a
    single toroidal harmonic means, and it fails if n enters anywhere but
    through the helical phase.
    """
    ods, _ = mapped
    table = _gpec_resonant_table(ods)

    plain = RECIPES["mhd_linear_geometry_island"].builder(ods)
    turned = RECIPES["mhd_linear_geometry_island"].builder(
        ods, phi_deg=360.0 / table["n_tor"]
    )

    np.testing.assert_allclose(plain.layers[2].r, turned.layers[2].r, atol=1e-9)
    np.testing.assert_allclose(plain.layers[2].z, turned.layers[2].z, atol=1e-9)


def test_the_circuit_starts_at_the_same_x_point_whatever_the_rounding():
    """Two X-points tied to rounding (an m = 2 island on an even mesh) must not
    let the platform's cos decide where the circuit starts; which of them won
    used to differ between CI runners, rotating the drawn samples by a lobe."""
    from vaft.plot.backend.recipes import _first_island_x_point

    base = np.abs(np.cos(np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False) + 0.3))
    tied = base.copy()
    tied[[10, 74]] = 1e-3
    lower_first, lower_second = tied.copy(), tied.copy()
    lower_first[10] -= 1e-18
    lower_second[74] -= 1e-18
    assert _first_island_x_point(lower_first) == _first_island_x_point(lower_second) == 10
    distinct = tied.copy()
    distinct[74] = 5e-4  # a genuinely lower X-point still wins
    assert _first_island_x_point(distinct) == 74


def test_a_non_finite_excursion_is_refused_by_name():
    from vaft.plot.backend.recipes import _first_island_x_point

    with pytest.raises(ValueError, match="not finite"):
        _first_island_x_point(np.array([0.3, np.nan, 0.1]))


# --- where the O-points actually sit (C-44, slice V5P) -------------------------
#
# The tests above this point pin the separatrix's *shape* and how it *moves*
# when the phase or phi changes. None of them pins where it sits, which is how
# the figure came to place every O-point a quarter period -- and a whole angle
# convention -- away from where a Poincare trace puts it. These do.

#: The fixture's flux-surface mesh is radial rays from ``(1.7, 0)``, so a
#: drawn point's poloidal angle and its ``psi_n`` are both read straight off
#: its position. A constant rather than a re-derivation per test, so that a
#: change to the fixture's geometry breaks these loudly.
_FIXTURE_AXIS_R = 1.7

#: An ``arg(Phi_res)`` at which the two candidate laws are far apart.
#: ``arg(Phi) - pi/2`` and ``-arg(Phi)`` differ by ``2 arg(Phi) - pi/2``, which
#: vanishes at ``arg(Phi) = pi/4`` and at ``pi/4 - pi``: a phase near either
#: would let the old law pass these tests. This one separates them by ``pi``.
_SEPARATING_PHASE = 0.25 * np.pi + 0.5 * np.pi


def _circular_pest_angle(psi_r, theta_norm):
    r"""The PEST angle of the fixture's circular surface, in closed form [rad].

    Independent of the code under test, which is the point: the fixture's mesh
    is a circle ``R = R0 + a cos(theta)`` about ``(1.7, 0)`` with the mesh angle
    as the geometric angle, and on it the Hamada-to-PEST weight ``1/R**2`` has
    an antiderivative,

    ``int_0^t dx / (R0 + a cos x)**2
      = a sin t / ((a**2 - R0**2)(R0 + a cos t))
        + R0/(R0**2 - a**2) * 2/b * atan2(k sin(t/2), cos(t/2))``

    with ``b = sqrt(R0**2 - a**2)`` and ``k = sqrt((R0 - a)/(R0 + a))``; the
    ``atan2`` form is the branch that stays continuous across ``t = pi``.
    Checked against quadrature to 2.1e-13 over a full circuit.

    So the truth these tests compare the renderer against is an integral, not
    a second call to :func:`pest_angle_from_jacobian_angle`. The *relation*
    ``1/R**2`` itself is validated elsewhere and against something else again:
    ``test_magnetic_island.py`` builds both angles out of a Solov'ev
    equilibrium's own surface weights, and the slice measured the function
    against ``straight_field_line_map`` on the DIII-D GPEC example
    (7.5e-06 rad at ``psi_N = 0.594``).
    """
    radius, axis = float(psi_r), _FIXTURE_AXIS_R
    b = np.sqrt(axis ** 2 - radius ** 2)
    k = np.sqrt((axis - radius) / (axis + radius))

    def integral(angle):
        half = 0.5 * angle
        return (radius * np.sin(angle)
                / ((radius ** 2 - axis ** 2) * (axis + radius * np.cos(angle)))
                + axis / (axis ** 2 - radius ** 2) * (2.0 / b)
                * np.arctan2(k * np.sin(half), np.cos(half)))

    geometric = 2.0 * np.pi * np.asarray(theta_norm, dtype=float)
    return 2.0 * np.pi * integral(geometric) / integral(2.0 * np.pi)


def _fixture_surface_angles(psi_r, theta_norm, jacobian):
    """``(theta_mesh, theta_pest)`` of one fixture flux surface, in radians.

    ``jacobian`` says what the mesh's angle is *declared* to be, so that a test
    reads the drawn curve in the labelling the recipe was told to use: under
    ``"pest"`` the two angles are the same, and under ``"hamada"`` the second is
    :func:`_circular_pest_angle`.
    """
    mesh = 2.0 * np.pi * np.asarray(theta_norm, dtype=float)
    if jacobian == "pest":
        return mesh, mesh
    if jacobian != "hamada":
        raise AssertionError(f"the fixture has no closed-form angle for {jacobian!r}")
    return mesh, _circular_pest_angle(psi_r, theta_norm)


def test_the_conversion_converges_to_the_circles_closed_form_angle():
    """The quadrature, against the antiderivative, on the fixture's geometry.

    Second order in the step, as the trapezoid rule is, so this also says the
    129-sample circuit the other tests use is good to about 1e-04 rad -- well
    inside the 5e-03 those tests assert.
    """
    from vaft.process.equilibrium import pest_angle_from_jacobian_angle

    errors = []
    for count in (65, 129, 511, 2049):
        theta_norm = np.linspace(0.0, 1.0, count)
        radius = _FIXTURE_AXIS_R + 0.4 * np.cos(2.0 * np.pi * theta_norm)
        errors.append(float(np.max(np.abs(
            pest_angle_from_jacobian_angle(theta_norm, radius, "hamada")
            - _circular_pest_angle(0.4, theta_norm)
        ))))
    assert errors[0] < 1e-3
    assert errors[-1] < 1e-6
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < 0.35 * coarse, errors


def _drawn_angles(layer, psi_r, theta_norm, jacobian):
    """Each drawn sample's ``(mesh angle, PEST angle, radial excursion)``."""
    half = layer.r.size // 2
    outer_r, outer_z = layer.r[:half], layer.z[:half]
    mesh_angle, pest_angle = _fixture_surface_angles(psi_r, theta_norm, jacobian)
    geometric = np.mod(np.arctan2(outer_z, outer_r - _FIXTURE_AXIS_R), 2.0 * np.pi)
    return (
        np.interp(geometric, mesh_angle, mesh_angle, period=2.0 * np.pi),
        np.interp(geometric, mesh_angle, pest_angle, period=2.0 * np.pi),
        np.hypot(outer_r - _FIXTURE_AXIS_R, outer_z) - psi_r,
    )


def _drawn_o_point_phase(layer, m_pol, psi_r, theta_norm, jacobian):
    """The helical phase of the drawn separatrix's widest point.

    Fitted, not argmax-ed: ``excursion**2`` is
    ``(w/2)**2 (1 + cos(m theta* - xi_O)) / 2``, one cosine at ``m`` whose
    phase is the O-point's, so the whole outward branch votes and the answer
    is not limited to the nearest mesh sample. It is the same fit applied to
    the FLARE punctures (:data:`R01_PROVENANCE`'s ``measurement``), which is
    what makes the two numbers comparable at all.
    """
    _, pest_angle, excursion = _drawn_angles(layer, psi_r, theta_norm, jacobian)
    design = np.column_stack([np.ones(excursion.size),
                              np.cos(m_pol * pest_angle), np.sin(m_pol * pest_angle)])
    fit, *_ = np.linalg.lstsq(design, excursion ** 2, rcond=None)
    return float(np.mod(np.arctan2(fit[2], fit[1]), 2.0 * np.pi))


def _phase_miss(first, second):
    return abs((first - second + np.pi) % (2.0 * np.pi) - np.pi)


def _fix_phase(monkeypatch, phase):
    """Put one chosen ``arg(Phi_res)`` on every surface of the recipe's table."""
    import vaft.plot.backend.recipes as recipes

    original = recipes._gpec_resonant_table

    def fixed(ods_, **options):
        result = original(ods_, **options)
        for row in result["rows"]:
            row["phase"] = float(phase)
        return result

    monkeypatch.setattr(recipes, "_gpec_resonant_table", fixed)


def _recorded_as(ods, *, jacobian=None, helicity=None):
    parameters = ods["mhd_linear.code.parameters"]
    if jacobian is not None:
        parameters = re.sub(r"<jacobian>[^<]*</jacobian>",
                            f"<jacobian>{jacobian}</jacobian>", parameters)
    if helicity is not None:
        parameters = re.sub(r"<helicity>[^<]*</helicity>",
                            f"<helicity>{helicity}</helicity>", parameters)
    ods["mhd_linear.code.parameters"] = parameters
    return ods


@pytest.mark.parametrize("jacobian", ["pest", "hamada"])
@pytest.mark.parametrize("phase", [0.0, 1.2, _SEPARATING_PHASE])
def test_the_o_point_sits_a_quarter_period_below_the_resonant_flux_phase(
    mapped, monkeypatch, jacobian, phase
):
    """C-44. ``Phi_res`` is a normal-field harmonic -- GPEC writes it in tesla
    -- while the field-line Hamiltonian's potential is the flux function, and
    harmonic k of the two differ by ``i k m``. The modulus is already inside
    ``w_isl``; the ``1/i`` is this quarter period, and nothing in ``w_isl``
    carries it. So the O-point is at ``arg(Phi_res) - pi/2``.
    """
    ods, native = mapped
    _recorded_as(ods, jacobian=jacobian)
    _fix_phase(monkeypatch, phase)
    table = _gpec_resonant_table(ods)
    model = RECIPES["mhd_linear_geometry_island"].builder(ods)
    law = float(np.mod(phase - 0.5 * np.pi, 2.0 * np.pi))

    for index, row in enumerate(table["rows"]):
        measured = _drawn_o_point_phase(
            model.layers[2 + 2 * index], int(row["m_pol"]), float(row["psi_n"]),
            native["theta"], jacobian,
        )
        assert _phase_miss(measured, law) < 5e-3, (row["m_pol"], measured, law)


def test_the_phase_is_not_the_negated_argument_the_figure_used_to_draw(
    mapped, monkeypatch
):
    """The regression pin. At :data:`_SEPARATING_PHASE` the law this figure
    used -- the O-point at ``-arg(Phi_res)`` -- is half a helical period from
    the measured one, so no sampling tolerance can hide the difference.
    """
    ods, native = mapped
    _recorded_as(ods, jacobian="pest")
    _fix_phase(monkeypatch, _SEPARATING_PHASE)
    table = _gpec_resonant_table(ods)
    model = RECIPES["mhd_linear_geometry_island"].builder(ods)

    row = table["rows"][0]
    measured = _drawn_o_point_phase(
        model.layers[2], int(row["m_pol"]), float(row["psi_n"]), native["theta"], "pest",
    )
    superseded = float(np.mod(-_SEPARATING_PHASE, 2.0 * np.pi))
    assert _phase_miss(measured, superseded) == pytest.approx(np.pi, abs=5e-3)


def test_the_drawn_lobes_are_equally_spaced_in_the_pest_angle_and_not_in_the_mesh(
    tmp_path, monkeypatch
):
    """One island's lobes are at one helical phase by definition, so they are
    equally spaced in the angle that carries that phase and in no other.
    Hamada, PEST and Boozer all make field lines straight, but each pairs its
    poloidal angle with its own toroidal angle and only PEST's is the machine
    ``phi``; drawing the Hamada angle as though it were the helical one leaves
    the lobes scattered around the surface.
    """
    # A finer circuit than the shared fixture: a lobe's own maximum is located
    # only to the sample spacing, and the claim is about spacings between them.
    native = _run(tmp_path, theta_count=511)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1], "include_geometry": True})
    _fix_phase(monkeypatch, _SEPARATING_PHASE)
    table = _gpec_resonant_table(ods)
    model = RECIPES["mhd_linear_geometry_island"].builder(ods)
    spacing = 2.0 * np.pi / (native["theta"].size - 1)

    for index, row in enumerate(table["rows"]):
        m_pol = int(row["m_pol"])
        layer = model.layers[2 + 2 * index]
        mesh_angle, pest_angle, _ = _drawn_angles(
            layer, float(row["psi_n"]), native["theta"], "hamada"
        )
        separation = _separation(layer)[:-1]
        peaks = np.flatnonzero(
            (separation >= np.roll(separation, 1)) & (separation >= np.roll(separation, -1))
        )
        assert peaks.size == m_pol, (m_pol, peaks.size)

        def deviation(angle):
            sorted_angle = np.sort(np.mod(angle[peaks], 2.0 * np.pi))
            gaps = np.diff(np.append(sorted_angle, sorted_angle[0] + 2.0 * np.pi))
            return float(np.max(np.abs(gaps - 2.0 * np.pi / m_pol)))

        # Equal to within how well a peak can be located on this mesh ...
        assert deviation(pest_angle) < 1.5 * spacing, m_pol
        # ... and, in the mesh's own angle, an order of magnitude worse.
        assert deviation(mesh_angle) > 10.0 * spacing, m_pol


def test_a_jacobian_that_cannot_be_relabelled_from_the_mesh_is_refused(mapped):
    """``boozer`` is a straight-field-line system and used to be accepted for
    being one. Relabelling its poloidal angle into PEST needs ``|B|`` on the
    same mesh, which no ``coordinate_system`` carries.
    """
    ods, _ = mapped
    _recorded_as(ods, jacobian="boozer")

    with pytest.raises(ValueError, match="boozer"):
        RECIPES["mhd_linear_geometry_island"].builder(ods)


@pytest.mark.parametrize("recorded", ["1.0", "None", "0", "0.0"])
def test_a_helicity_the_phase_law_was_not_measured_at_is_refused(mapped, recorded):
    """C-44 ties the conjugation's sense to the orientation of the code's
    angles against the machine helicity, and the Poincare traces that settled
    it ran at helicity -1. The other sign, and a run that recorded none, are
    refused rather than drawn from a law nobody measured there.

    ``"0"`` is the case a run with no ``helicity`` attribute actually produces:
    ``vaft.code.gpec``'s reader defaults the attribute to the integer 0, and
    ``ipd * btd`` is never 0, so 0 means "absent" and must be reported that way
    rather than as a recorded helicity of zero.
    """
    ods, _ = mapped
    _recorded_as(ods, helicity=recorded)

    with pytest.raises(ValueError, match="helicity"):
        RECIPES["mhd_linear_geometry_island"].builder(ods)


def test_a_missing_helicity_is_not_reported_as_a_recorded_zero(mapped):
    ods, _ = mapped
    _recorded_as(ods, helicity="0")

    with pytest.raises(ValueError, match="recorded no helicity"):
        RECIPES["mhd_linear_geometry_island"].builder(ods)


# --- the R-01 reference value (D-13: GPEC's own DIII-D example) ----------------
#
# D-13 (2026-09-28) lets data shipped inside an open-source code's own
# repository, and outputs derived from it with that code, into public VAFT
# provided the provenance records the code version, the input SHA-256 and the
# generating command. :data:`R01_PROVENANCE` is that record. It states the
# command as the task, its arguments and the measurement method rather than as
# a script path, so that it stands on its own: a reader reproduces the number
# from this record and the two open-source codes, with nothing else to obtain.
# It is the record that
# every number below -- and every R-01 number quoted in
# ``vaft.process.equilibrium.pest_angle_from_jacobian_angle``,
# ``vaft.formula.stability.helical_phase`` and the island recipe -- rests on.

#: What produced every R-01 number in this file, verbatim, as D-13 requires.
#:
#: The case is DIII-D 147131 @ 2300 ms: GPEC's own kinetic-EFIT example
#: equilibrium, with the six C-coils on a pure ``cos(n = 1)`` pattern,
#: ``jac_type = "hamada"``, ``jac_out = "boozer"``, ``tmag_out = 1``,
#: ``helicity = -1``. Nothing here is a DIII-D physics result: the run is a
#: machinery reference.
R01_PROVENANCE: Mapping[str, str] = MappingProxyType({
    # Code versions.
    "gpec_version": "v1.5.5-378-gf06e6ab",
    "flare_commit": "7ad6d2dc",
    # Inputs, SHA-256.
    "equilibrium": "g147131.02300_DIIID_KEFIT",
    "equilibrium_sha256":
        "35bf902f1d02759ad655f7b5315e3e2b4bc96d88ffc9753f261d48e86dbfe579",
    "profile_output": "gpec_profile_output_n1.nc",
    "profile_output_sha256":
        "8c17f9675c7046331130d0e6315b38c9e6642c832d3598129c087939072a16b5",
    "vacuum_field": "gpec_cbrzphi_n1.out",
    "vacuum_field_sha256":
        "718da784ff534a4acd2fbc85958d3e60438370222a1c9babd6b1f195c305a8fa",
    # What produced the traced O-point, stated so that it can be reproduced
    # from this record alone: a reader needs the FLARE task and its arguments,
    # not the name of a driver script.
    "trace_task": (
        "FLARE poincare_map_psiN over psi_N in [0.533644, 0.653644] -- the "
        "q = 2 surface at psi_N = 0.593644 plus and minus a 0.06 span -- with "
        "45 field lines x 300 punctures per line and nsym = 1, launched from "
        "the equilibrium below with the geqdsk's own limiter contour as the "
        "FLARE Axisurf boundary"
    ),
    "trace_field": (
        "one FLARE Gpec perturbation element on gpec_cbrzphi_n1.out, GPEC's "
        "vacuum (coil) real-space field, at amplitude |c_1| = 1 and phase 0; "
        "the n = 3 element is absent (|c_3| = 0), so exactly one toroidal "
        "harmonic is live and no relative phase is imposed"
    ),
    "measurement": (
        "the punctures are relabelled to the PEST straight-field-line angle of "
        "the same equilibrium; the lines whose psi_N interval contains the "
        "rational surface are binned in that angle; the island's full width per "
        "bin is fitted to width**2 = a + b cos(m theta*) + c sin(m theta*), "
        "whose phase atan2(c, b) is the O-point's helical angle xi_O directly, "
        "because width = (w/2)|cos((m theta* - xi_O) / 2)| for a pendulum "
        "island. Repeated at 72, 108, 144 and 180 bins; the spread across "
        "those four is R01_Q2_TRACED_O_POINT_SPREAD"
    ),
    "decision": "D-13 (2026-09-28)",
})

#: ``arg(Phi_res_v)`` at ``q = 2``, read from the ``Phi_res_v`` variable of
#: ``gpec_profile_output_n1.nc`` [rad].
#:
#: The *vacuum* resonant flux, because the vacuum field is the one with a
#: traced island: an ideal response shields the resonant field the total
#: harmonic is derived from. The phase relation is a statement about
#: ``Phi_res`` as a quantity -- a normal field rather than a flux function --
#: so it holds for the total harmonic too, which is what the figure uses.
#: Provenance: :data:`R01_PROVENANCE`.
R01_Q2_RESONANT_FLUX_PHASE = 2.826799

#: Where FLARE traces that island's O-point, as ``xi = m theta*`` at
#: ``phi = 0`` in the PEST angle [rad].
#:
#: 45 field lines x 300 punctures over ``psi_N`` in [0.534, 0.654], 19 of them
#: interior, the island width per poloidal bin fitted against the
#: ``|cos(xi/2)|`` profile a pendulum island has, in the PEST angle of the
#: run's own equilibrium. Provenance, including the trace parameters and the
#: fit: :data:`R01_PROVENANCE`.
R01_Q2_TRACED_O_POINT = 1.219653

#: How far that fit moves across binnings from 72 to 180 bins [rad]: the
#: resolution any comparison with it is entitled to.
R01_Q2_TRACED_O_POINT_SPREAD = 0.0075


def test_the_provenance_record_is_complete_for_d13():
    """D-13 admits this data on the strength of the record, so the record is
    under test: code versions, every input's SHA-256, and the command."""
    for key in ("gpec_version", "flare_commit", "trace_task", "trace_field",
                "measurement", "decision"):
        assert R01_PROVENANCE[key]
    # Self-contained: the record names codes, inputs, parameters and method,
    # never a repository a reader of public VAFT cannot open.
    joined = " ".join(R01_PROVENANCE.values()).lower()
    for private in ("hsyun", "vaft-mastu", "vaft_mastu", "hongsik"):
        assert private not in joined, private
    for name in ("equilibrium", "profile_output", "vacuum_field"):
        assert R01_PROVENANCE[name]
        digest = R01_PROVENANCE[f"{name}_sha256"]
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef"), name


def test_the_phase_law_reproduces_the_flare_traced_o_point_on_the_gpec_example():
    """The measurement the law rests on, pinned as a number.

    Not a figure test: it is the convention itself. The law the figure used
    placed the O-point at ``-arg(Phi_res)``; applied to the harmonic of the
    field FLARE traced, that is 128 degrees from the traced O-point, where
    ``arg(Phi_res) - pi/2`` is 2.1 degrees -- five times the trace's own
    spread against nearly 300.
    """
    law = R01_Q2_RESONANT_FLUX_PHASE - 0.5 * np.pi
    superseded = -R01_Q2_RESONANT_FLUX_PHASE

    assert _phase_miss(law, R01_Q2_TRACED_O_POINT) == pytest.approx(0.03635, abs=5e-5)
    assert _phase_miss(superseded, R01_Q2_TRACED_O_POINT) == pytest.approx(2.2367, abs=5e-4)
    assert _phase_miss(law, R01_Q2_TRACED_O_POINT) < 5.0 * R01_Q2_TRACED_O_POINT_SPREAD
    assert (_phase_miss(superseded, R01_Q2_TRACED_O_POINT)
            > 100.0 * R01_Q2_TRACED_O_POINT_SPREAD)


def test_the_recipe_and_the_pinned_r01_law_are_the_same_expression(mapped, monkeypatch):
    """The number above and the figure must not be able to drift apart."""
    ods, native = mapped
    _recorded_as(ods, jacobian="pest")
    _fix_phase(monkeypatch, R01_Q2_RESONANT_FLUX_PHASE)
    table = _gpec_resonant_table(ods)
    model = RECIPES["mhd_linear_geometry_island"].builder(ods)

    row = next(row for row in table["rows"] if int(row["m_pol"]) == 2)
    index = table["rows"].index(row)
    measured = _drawn_o_point_phase(
        model.layers[2 + 2 * index], 2, float(row["psi_n"]), native["theta"], "pest",
    )
    expected = float(np.mod(R01_Q2_RESONANT_FLUX_PHASE - 0.5 * np.pi, 2.0 * np.pi))
    assert _phase_miss(measured, expected) < 5e-3
    assert (_phase_miss(measured, R01_Q2_TRACED_O_POINT)
            < 5.0 * R01_Q2_TRACED_O_POINT_SPREAD)

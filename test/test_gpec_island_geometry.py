"""The island separatrix overlay (migration slice V5, convention C-20).

The islands GPEC's result implies, drawn in the poloidal plane on the mesh
the run itself solved on.  Both halves are under test: that the mapper writes
the mesh at all, and that the overlay derives its phase rather than assuming
one.
"""

from __future__ import annotations

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
    surfaces: those are the same surfaces, but their points carry a different
    poloidal angle, and the island phase is in this one."""
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
    """`m*theta - n*phi` is the helical phase only where the coordinates make
    field lines straight. GPEC's `equal_arc` jacobian does not, and a curve of
    constant phase drawn in it is not an island separatrix.
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
    # The widest *drawn* separation is the excursion at the sample nearest the
    # O-point. Moving phi moves the O-point off the samples by up to half a
    # step, i.e. xi by m*dtheta/2, and the excursion is (w/2)|cos(xi/2)|, so
    # the sampled maximum may drop by up to 1 - cos(m*dtheta/4). The fixture's
    # surfaces are circles of radius psi_N, so separation is exactly twice
    # the excursion and nothing else enters. The island is ~1e-9 wide, far
    # below pytest's default 1e-12 absolute floor, which is why the bound is
    # relative and the floor is switched off.
    table = _gpec_resonant_table(ods)
    m_pol = int(table["rows"][0]["m_pol"])
    dtheta = 2.0 * np.pi / (ods[f"{MESH}.grid.dim2"].size - 1)
    # Either side may sit up to half a sample off its O-point, so the ratio of
    # the two sampled maxima lies in [c, 1/c] with c = cos(m*dtheta/4).
    floor = np.cos(m_pol * dtheta / 4.0)
    sampling = (1.0 - floor) / floor
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

    # A full turn changes the phase only by rounding, so the same samples are
    # drawn in the same order. The flake this used to show on CI was the
    # start X-point flipping between two bit-identical candidates. Positions agree to rounding; the
    # separation, which is the island itself at ~1e-9, agrees to a millionth of
    # the island's width -- an absolute 1e-9 on positions would have passed
    # with no island drawn at all.
    before, after = plain.layers[2], turned.layers[2]
    np.testing.assert_allclose(after.r, before.r, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(after.z, before.z, rtol=0.0, atol=1e-12)
    width = float(table["rows"][0]["w_isl"])
    np.testing.assert_allclose(
        _separation(turned.layers[2]), _separation(plain.layers[2]), rtol=0.0, atol=1e-6 * width
    )


@pytest.mark.parametrize("m_pol", [2, 8, 16, 32, 64])
def test_the_circuit_start_does_not_depend_on_rounding(m_pol):
    """A full toroidal turn changes the helical phase only by rounding, so the
    circuit must start at the same X-point sample. When m divides the sample
    count the m X-point samples tie exactly in exact arithmetic, and their
    rounding spread grows with |xi| ~ 2 pi m: a fixed tolerance on the
    excursion value, or plain argmin, lets high-m islands flip (the CI flake
    was the m = 2 case of this)."""
    from vaft.formula.stability import helical_phase, island_separatrix_half_width
    from vaft.plot.backend.recipes import _island_circuit_start

    rng = np.random.default_rng(1190 + m_pol)
    theta = 2.0 * np.pi * np.linspace(0.0, 1.0, 129)[:-1]  # the builder drops the closing sample
    for _ in range(40):
        n_tor = int(rng.integers(1, 4))
        offset = float(rng.uniform(-np.pi, np.pi))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        width = float(10.0 ** rng.uniform(-10, -2))
        phases = [helical_phase(theta, value, m_pol, n_tor, phase=offset)
                  for value in (phi, phi + 2.0 * np.pi / n_tor)]
        starts = [_island_circuit_start(phase) for phase in phases]
        assert starts[0] == starts[1], (n_tor, offset, phi)
        # ... and the start is an X-point sample: the smallest excursion, to rounding.
        excursion = island_separatrix_half_width(phases[0], width)
        assert excursion[starts[0]] <= excursion.min() + 1e-9 * width


def test_the_circuit_start_survives_a_phase_with_no_finite_value():
    from vaft.plot.backend.recipes import _island_circuit_start

    assert _island_circuit_start(np.full(8, np.nan)) == 0
    phase = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    phase[3] = np.nan
    assert np.isfinite(phase[_island_circuit_start(phase)])

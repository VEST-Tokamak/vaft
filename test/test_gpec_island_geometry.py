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


def _run(path, *, n=1, rational_q=(2.0, 3.0)):
    write_control_nc(path, n=n)
    write_cylindrical_nc(path, n=n)
    return write_profile_nc(path, n=n, rational_q=rational_q)


@pytest.fixture
def mapped(tmp_path):
    native = _run(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})
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


def test_refusing_the_spectral_field_refuses_the_mesh_with_it(tmp_path):
    """Both live in the profile file, and `include_spectral=False` is how a
    caller says "do not open the 144 MB file". A mesh that defaulted to on
    would open it anyway."""
    _run(tmp_path)
    ods = ODS(consistency_check=False)
    gpec_ideal(ods, str(tmp_path), {"modes": [1], "include_spectral": False})

    assert ods.get(f"{MESH}.r", None) is None


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


def test_the_toroidal_angle_advances_the_pattern_the_way_the_formula_layer_does(mapped):
    """The helical phase is m*theta - n*phi. The figure this replaces added
    +n*phi, which agrees at its own default of phi = 0 and rotates the
    pattern the other way anywhere else."""
    from vaft.formula.stability import helical_phase

    ods, native = mapped
    table = _gpec_resonant_table(ods)
    row = table["rows"][0]
    theta = 2.0 * np.pi * native["theta"]
    psi_n = ods[f"{MESH}.grid.dim1"]
    r_grid = ods[f"{MESH}.r"]

    for phi_deg in (0.0, 47.0):
        model = RECIPES["mhd_linear_geometry_island"].builder(ods, phi_deg=phi_deg)
        phase = helical_phase(
            theta, np.deg2rad(phi_deg), row["m_pol"], table["n_tor"], phase=-row["phase"],
        )
        target = row["psi_n"] + 0.5 * row["w_isl"] * np.cos(phase)
        expected = [
            np.interp(target[j], psi_n, r_grid[:, j]) for j in range(theta.size)
        ]
        np.testing.assert_allclose(
            model.layers[2].r[: theta.size], expected, rtol=0, atol=1e-12
        )


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
    gpec_ideal(ods, str(tmp_path), {"modes": [1]})
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

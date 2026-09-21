"""The two IMAS stand-ins for field-line tracing products (issue #1099).

The Data Dictionary has no home for field-line tracing, so these writers use
structures whose meaning only partly matches.  What is pinned here is mostly
what they *refuse*: a stand-in that quietly accepts what it cannot represent
is worse than no stand-in at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.data.flare_products import read_flare_product
from vaft.machine_mapping.field_line_topology import (
    FLARE_TRACE_STATUS,
    REFUSED_TARGET_LEAVES,
    b_field_lines_from_flare_connection,
    divertor_incident_fractions_from_flare_footprints,
    write_b_field_lines,
    write_divertor_incident_fractions,
)
from vaft.process.field_line_topology import (
    FOOTPRINT_PROXY_MODELS,
    footprint_heat_load_proxy,
)

GRID_R = np.array([1.0, 1.5, 2.0])
GRID_Z = np.array([-1.0, 0.0])


def plane(grid_r=GRID_R, grid_z=GRID_Z):
    """Flattened starting positions covering the grid, in FLARE's own order."""
    r, z = np.meshgrid(grid_r, grid_z)
    return r.ravel(), z.ravel()


def written(ods, **overrides):
    starting_r, starting_z = plane()
    arguments = dict(
        grid_r=GRID_R, grid_z=GRID_Z,
        starting_r=starting_r, starting_z=starting_z,
        lengths=np.arange(starting_r.size, dtype=float),
        open_fraction=0.5, time=0.1, phi_deg=30.0,
    )
    arguments.update(overrides)
    return write_b_field_lines(ods, **arguments)


# ---------------------------------------------------------------------------
# 1. plasma_initiation.b_field_lines
# ---------------------------------------------------------------------------


def test_one_plane_lands_where_the_owner_put_it():
    ods = ODS()
    assert written(ods) == 0
    entry = ods["plasma_initiation.b_field_lines.0"]
    np.testing.assert_allclose(entry["grid.dim1"], GRID_R)
    np.testing.assert_allclose(entry["grid.dim2"], GRID_Z)
    np.testing.assert_allclose(entry["lengths"], np.arange(6.0))
    assert entry["grid_type.index"] == 1 and entry["grid_type.name"] == "rectangular"
    assert entry["open_fraction"] == 0.5
    assert entry["time"] == 0.1
    np.testing.assert_allclose(ods["plasma_initiation.time"], [0.1])
    assert ods["plasma_initiation.ids_properties.homogeneous_time"] == 1


def test_the_starting_positions_are_written_even_though_the_dd_calls_them_redundant():
    """On a rectangular grid the Data Dictionary says the position arrays are
    redundant with dim1 and dim2.  For a flat list of lengths they are not:
    the axes alone leave the flattening order open, and taking the wrong one
    transposes the map."""
    ods = ODS()
    written(ods)
    entry = ods["plasma_initiation.b_field_lines.0"]
    np.testing.assert_allclose(entry["starting_positions.r"], [1.0, 1.5, 2.0] * 2)
    np.testing.assert_allclose(entry["starting_positions.z"], [-1.0] * 3 + [0.0] * 3)
    assert entry["starting_positions.r"].size == entry["lengths"].size


def test_the_angle_and_what_is_lost_are_recorded_in_the_provenance():
    ods = ODS()
    written(ods, phi_deg=-12.5)
    parameters = ods["plasma_initiation.code.parameters"]
    assert 'phi_deg="-12.5"' in parameters
    assert "turn counts have no slot" in parameters
    assert "forward" in parameters and "backward" in parameters
    assert ods["plasma_initiation.code.name"] == "FLARE"


def test_a_second_plane_at_the_same_time_is_refused():
    """The array of structures is indexed by time and carries no toroidal
    coordinate, so two entries sharing a time would make the time base a lie."""
    ods = ODS()
    written(ods, phi_deg=0.0)
    with pytest.raises(ValueError, match="one plane per time"):
        written(ods, phi_deg=60.0)


def test_two_times_become_two_entries_on_one_time_base():
    ods = ODS()
    written(ods, time=0.1)
    written(ods, time=0.3)
    np.testing.assert_allclose(ods["plasma_initiation.time"], [0.1, 0.3])
    assert ods["plasma_initiation.b_field_lines.0.time"] == 0.1
    assert ods["plasma_initiation.b_field_lines.1.time"] == 0.3
    assert ods["plasma_initiation.code.parameters"].count("<field_line_plane") == 2


def test_an_earlier_instant_is_refused_rather_than_written_over_an_entry():
    """Entries are appended, so a time that sorts before the last one would
    land on an index that is already taken.  The version that sorted the time
    base instead overwrote it, and the map that disappeared left no trace."""
    ods = ODS()
    written(ods, time=0.3, phi_deg=0.0)
    with pytest.raises(ValueError, match="increasing order"):
        written(ods, time=0.1, phi_deg=0.0)
    assert ods["plasma_initiation.b_field_lines.0.time"] == 0.3
    np.testing.assert_allclose(ods["plasma_initiation.time"], [0.3])


def test_two_lines_from_one_node_are_refused_by_the_writer_too():
    starting_r, starting_z = plane()
    starting_r = starting_r.copy()
    starting_r[1] = starting_r[0]
    with pytest.raises(ValueError, match="same grid node"):
        write_b_field_lines(
            ODS(), grid_r=GRID_R, grid_z=GRID_Z,
            starting_r=starting_r, starting_z=starting_z,
            lengths=np.zeros(6), open_fraction=0.0, time=0.0, phi_deg=0.0,
        )


def test_an_earlier_instant_is_refused_for_the_divertor_signals_too():
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.5)
    with pytest.raises(ValueError, match="increasing order"):
        write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.2)


def test_an_absent_open_fraction_is_omitted_and_the_omission_recorded():
    """It is a required argument whose None is the decision, so a caller who
    has no status data says so rather than leaving it out by accident."""
    ods = ODS()
    written(ods, open_fraction=None)
    assert "open_fraction" not in ods["plasma_initiation.b_field_lines.0"]
    assert "no status was supplied" in ods["plasma_initiation.code.parameters"]


@pytest.mark.parametrize("overrides, message", [
    ({"lengths": [1.0, 2.0]}, "one non-empty entry"),
    ({"grid_r": [2.0, 1.0, 1.5]}, "strictly increasing"),
    ({"grid_r": [1.0]}, "strictly increasing"),
    ({"open_fraction": 1.4}, r"\[0, 1\]"),
    ({"open_fraction": float("nan")}, r"\[0, 1\]"),
])
def test_inputs_that_do_not_describe_one_plane_are_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        written(ODS(), **overrides)


def test_lines_that_do_not_cover_the_declared_grid_are_refused():
    starting_r, starting_z = plane()
    with pytest.raises(ValueError, match="do not cover the declared"):
        write_b_field_lines(
            ODS(), grid_r=GRID_R, grid_z=GRID_Z,
            starting_r=starting_r[:5], starting_z=starting_z[:5],
            lengths=np.zeros(5), open_fraction=0.0, time=0.0, phi_deg=0.0,
        )


def test_a_starting_position_off_the_declared_grid_is_refused():
    starting_r, starting_z = plane()
    starting_r = starting_r.copy()
    starting_r[0] = 1.25
    with pytest.raises(ValueError, match="do not lie on grid_r"):
        write_b_field_lines(
            ODS(), grid_r=GRID_R, grid_z=GRID_Z,
            starting_r=starting_r, starting_z=starting_z,
            lengths=np.zeros(6), open_fraction=0.0, time=0.0, phi_deg=0.0,
        )


# ---------------------------------------------------------------------------
# The FLARE adapter
# ---------------------------------------------------------------------------


def flare_connection(tmp_path, *, swept=False, status=True):
    """A synthetic fieldline_connection product on a 3 x 2 plane."""
    mesh = tmp_path / "m.grid"
    if swept:
        mesh.write_text(
            "# TYPE rmesh3d\n# NODES 3 2\n# MAP3D 1 2 3\n"
            "# COORDINATES cylindrical\n# UNITS m, deg\n"
            "  -3.0E+01\n  0.0E+00\n  3.0E+01\n"
            "  1.0E+00  -1.0E+00  0.0E+00\n  1.5E+00  -0.5E+00  1.0E+01\n"
        )
    else:
        mesh.write_text(
            "# TYPE rmesh\n# NODES 3 2\n# U-AXIS r [m]\n# V-AXIS z [m]\n"
            "# MAP3D 1 2 30.0\n# COORDINATES cylindrical\n# UNITS m, deg\n"
            + "".join(f"  {value:.6E}\n" for value in (1.0, 1.5, 2.0, -1.0, 0.0))
        )
    columns = ["Lc_bwd", "Lc_fwd"] + (["ierr_bwd", "ierr_fwd"] if status else [])
    rows = [(3.0, 4.0) + ((-1001.0, -1002.0) if status else ()) for _ in range(6)]
    if status:
        rows[0] = (3.0, 4.0, -1002.0, -1002.0)
        rows[1] = (3.0, 4.0, -1002.0, -1002.0)
    product = tmp_path / "p.dat"
    product.write_text(
        "# TYPE dataset\n# GEOMETRY m.grid\n"
        + "".join(f"# COLUMN_{index + 1} {name}\n" for index, name in enumerate(columns))
        + "".join("  ".join(f"{value:.6E}" for value in row) + "\n" for row in rows)
    )
    return read_flare_product(product)


def test_a_flare_plane_maps_with_its_total_length_and_open_fraction(tmp_path):
    ods = ODS()
    b_field_lines_from_flare_connection(ods, flare_connection(tmp_path), time=0.2)
    entry = ods["plasma_initiation.b_field_lines.0"]
    np.testing.assert_allclose(entry["lengths"], 7.0)
    # Four of six lines reached the wall in at least one direction.
    assert entry["open_fraction"] == pytest.approx(4.0 / 6.0)
    assert 'phi_deg="30.0"' in ods["plasma_initiation.code.parameters"]
    assert "rmesh 3 x 2" in ods["plasma_initiation.code.parameters"]


def test_a_plane_written_in_centimetres_reaches_the_ids_in_metres(tmp_path):
    """A FlareMesh keeps its axes in the file's own units and converts only
    the node positions, so the adapter must take the grid off the positions."""
    product = flare_connection(tmp_path)
    mesh = product.source.parent / "m.grid"
    mesh.write_text(mesh.read_text().replace("UNITS m, deg", "UNITS cm, deg"))
    ods = ODS()
    b_field_lines_from_flare_connection(ods, read_flare_product(product.source), time=0.0)
    entry = ods["plasma_initiation.b_field_lines.0"]
    np.testing.assert_allclose(entry["grid.dim1"], [0.01, 0.015, 0.02])
    np.testing.assert_allclose(entry["grid.dim2"], [-0.01, 0.0])
    np.testing.assert_allclose(
        np.unique(entry["starting_positions.r"]), [0.01, 0.015, 0.02]
    )


def test_a_toroidally_swept_map_is_refused_rather_than_flattened(tmp_path):
    """Its lines start at different toroidal angles, and starting_positions
    has nowhere to say so -- the coordinate that carries an RMP footprint's
    whole structure would be the one discarded."""
    with pytest.raises(ValueError, match="sweeps the toroidal angle"):
        b_field_lines_from_flare_connection(
            ODS(), flare_connection(tmp_path, swept=True), time=0.0
        )


def test_a_status_code_this_flare_does_not_define_is_refused(tmp_path):
    """The committed reference outputs carry ``1`` on about a third of their
    traced directions -- which is about half their lines -- and no installed
    FLARE defines it.  Counting those lines as "did not reach the wall" would
    put a silent half into a number that reads as a measurement."""
    product = flare_connection(tmp_path)
    path = product.source
    path.write_text(path.read_text().replace("-1.002000E+03", "1.000000E+00"))
    with pytest.raises(ValueError, match="does not define"):
        b_field_lines_from_flare_connection(ODS(), read_flare_product(path), time=0.0)


def test_the_known_status_codes_are_the_ones_flare_documents():
    assert set(FLARE_TRACE_STATUS) == {0, -1001, -1002, -1003, -1004}


def test_a_product_without_status_columns_is_refused(tmp_path):
    """Without ierr there is no way to tell which lines reached the wall, and
    an open fraction would be a guess."""
    with pytest.raises(ValueError, match="ierr"):
        b_field_lines_from_flare_connection(
            ODS(), flare_connection(tmp_path, status=False), time=0.0
        )


def test_the_writer_serves_vests_own_startup_tracer_unchanged():
    """The writer is written against the quantity, not against FLARE.
    ``vaft.process.equilibrium.connection_length_map`` is the VEST startup
    tracer (#783); it produces the same thing on the same kind of grid, and
    a breakdown study is what plasma_initiation is actually for."""
    from vaft.plot.backend.recipes import RECIPES
    from vaft.process.equilibrium import connection_length_map

    def b_field(r, z):
        return np.zeros_like(r), np.full_like(r, 0.02), np.full_like(r, 0.5)

    grid_r = np.linspace(0.3, 0.7, 5)
    grid_z = np.linspace(-0.4, 0.4, 7)
    seed_r, seed_z = np.meshgrid(grid_r, grid_z)
    # A wall that leaves the outermost seeds outside it, so the tracer writes
    # NaN for them -- which it documents doing, and which a map has to carry.
    traced = connection_length_map(
        seed_r, seed_z, b_field,
        wall_r=np.array([0.35, 0.65, 0.65, 0.35, 0.35]),
        wall_z=np.array([-0.45, -0.45, 0.45, 0.45, -0.45]),
        max_length_m=50.0,
    )
    assert traced["outside"].any(), "the fixture must exercise the NaN path"
    reached = ~traced["saturated"] & ~traced["outside"]

    ods = ODS()
    write_b_field_lines(
        ods, grid_r=grid_r, grid_z=grid_z,
        starting_r=seed_r.ravel(), starting_z=seed_z.ravel(),
        lengths=traced["length_m"].ravel(),
        open_fraction=float(np.count_nonzero(reached)) / reached.size,
        time=0.0, phi_deg=0.0,
        provenance={"source": "vaft.process.equilibrium.connection_length_map"},
    )
    drawn = RECIPES["field_line_topology_field_connection_length"].builder(ods)
    np.testing.assert_allclose(drawn.values, traced["length_m"])


# ---------------------------------------------------------------------------
# 2. divertors ... power_incident_fraction
# ---------------------------------------------------------------------------


FRACTIONS = {"inner_lower": 0.7, "outer_lower": 0.3}


def test_the_fractions_land_on_their_targets():
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.4,
                                      divertor_name="lower")
    root = ods["divertors.divertor.0"]
    assert root["name"] == "lower"
    assert root["target.0.identifier"] == "inner_lower"
    np.testing.assert_allclose(root["target.0.power_incident_fraction.data"], [0.7])
    np.testing.assert_allclose(root["target.1.power_incident_fraction.data"], [0.3])
    np.testing.assert_allclose(ods["divertors.time"], [0.4])


@pytest.mark.parametrize("leaf", sorted(REFUSED_TARGET_LEAVES))
def test_the_leaves_that_do_not_fit_the_proxy_stay_unwritten(leaf):
    """power_flux_peak is in W/m^2 and the proxy has no heat-flux units;
    wetted_area is defined through lambda_q; tilt_angle_pol is poloidal-only.
    Writing any of them would give a relative number a meaning it has not
    earned."""
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.0)
    for position in (0, 1):
        assert leaf not in ods[f"divertors.divertor.0.target.{position}"]


def test_the_model_and_the_deliberate_omissions_are_recorded():
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.0,
                                      provenance="some_model: lambda_psi=0.02")
    parameters = ods["divertors.code.parameters"]
    assert "lambda_psi=0.02" in parameters
    assert "no heat-flux units" in parameters
    assert "power_flux_peak" in parameters


def test_a_single_target_is_refused_because_its_share_is_one_by_construction():
    with pytest.raises(ValueError, match="something to be a share of"):
        write_divertor_incident_fractions(ODS(), fractions={"only": 1.0}, time=0.0)


def test_shares_that_do_not_close_are_refused():
    with pytest.raises(ValueError, match="does not close"):
        write_divertor_incident_fractions(
            ODS(), fractions={"a": 0.5, "b": 0.2}, time=0.0
        )


def test_a_share_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match=r"not in \[0, 1\]"):
        write_divertor_incident_fractions(
            ODS(), fractions={"a": 1.5, "b": -0.5}, time=0.0
        )


def test_a_second_value_for_the_same_instant_is_refused_not_replaced():
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.4)
    with pytest.raises(ValueError, match="already carries a fraction"):
        write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.4)


def test_a_refused_write_leaves_the_ods_exactly_as_it_found_it():
    """The refusals come before the first write, so a caller that catches one
    is not left with a half-updated divertor and a time base that moved."""
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.4)
    before = ods.pretty_paths()
    with pytest.raises(ValueError, match="already carries a fraction"):
        write_divertor_incident_fractions(
            ods, fractions={"inner_lower": 0.1, "baffle": 0.9}, time=0.4
        )
    assert ods.pretty_paths() == before
    np.testing.assert_allclose(ods["divertors.time"], [0.4])


def test_a_later_instant_extends_every_target_on_one_time_base():
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.1)
    write_divertor_incident_fractions(
        ods, fractions={"inner_lower": 0.4, "outer_lower": 0.6}, time=0.2
    )
    np.testing.assert_allclose(ods["divertors.time"], [0.1, 0.2])
    np.testing.assert_allclose(
        ods["divertors.divertor.0.target.0.power_incident_fraction.data"], [0.7, 0.4]
    )
    np.testing.assert_allclose(
        ods["divertors.divertor.0.target.1.power_incident_fraction.data"], [0.3, 0.6]
    )


def test_a_target_first_seen_late_is_padded_rather_than_left_short():
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.1)
    write_divertor_incident_fractions(
        ods, fractions={"inner_lower": 0.5, "outer_lower": 0.2, "baffle": 0.3},
        time=0.2,
    )
    late = ods["divertors.divertor.0.target.2.power_incident_fraction.data"]
    assert late.size == 2 and np.isnan(late[0]) and late[1] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Proxies straight through to fractions
# ---------------------------------------------------------------------------


def proxy(scale, *, model_name="gpec_research_inverse", incidence=True):
    shape = (4, 4)
    return footprint_heat_load_proxy(
        min_psi_norm=np.full(shape, 1.0),
        connection_length=np.full(shape, 0.0),
        area=np.full(shape, float(scale)),
        incidence_angle_deg=np.full(shape, 90.0) if incidence else None,
        model=FOOTPRINT_PROXY_MODELS[model_name],
    )


def test_proxies_become_shares_of_the_run_they_belong_to():
    ods = ODS()
    fractions = divertor_incident_fractions_from_flare_footprints(
        ods, {"inner": proxy(3.0), "outer": proxy(1.0)}, time=0.0,
    )
    assert fractions == pytest.approx({"inner": 0.75, "outer": 0.25})
    np.testing.assert_allclose(
        ods["divertors.divertor.0.target.0.power_incident_fraction.data"], [0.75]
    )
    assert "gpec_research_inverse" in ods["divertors.code.parameters"]


def test_targets_built_from_different_models_are_refused():
    """Their integrals are in different scales, so a share between them
    compares two different quantities."""
    with pytest.raises(ValueError, match="one model and one incidence"):
        divertor_incident_fractions_from_flare_footprints(
            ODS(),
            {"inner": proxy(1.0), "outer": proxy(1.0, model_name="gpec_research_exponential")},
            time=0.0,
        )


def test_targets_that_disagree_about_the_incidence_factor_are_refused():
    with pytest.raises(ValueError, match="one model and one incidence"):
        divertor_incident_fractions_from_flare_footprints(
            ODS(),
            {"inner": proxy(1.0), "outer": proxy(1.0, incidence=False)},
            time=0.0,
        )

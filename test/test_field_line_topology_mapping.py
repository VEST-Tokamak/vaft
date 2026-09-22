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
    FLARE_DOMAIN_ERROR,
    FLARE_TRACE_STATUS,
    REFUSED_TARGET_LEAVES,
    b_field_lines_from_flare_connection,
    divertor_incident_fractions_from_flare_footprints,
    WALL_BOUNDARY,
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


def test_a_time_base_laid_out_ahead_of_the_entries_is_refused(tmp_path):
    """The entry index comes from b_field_lines, not from the time base. A
    base laid out first used to send the first plane to an index the array
    of structures did not have yet, which OMAS raises on -- and for a time
    past the end it appended to `.time` before raising, so the refusal was
    not free either."""
    for requested in (0.2, 0.3):
        ods = ODS()
        ods["plasma_initiation.time"] = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="owns the time base"):
            written(ods, time=requested)
        np.testing.assert_allclose(ods["plasma_initiation.time"], [0.1, 0.2])


def test_the_entry_index_follows_the_entries_not_the_time_base():
    ods = ODS()
    assert written(ods, time=0.1) == 0
    assert written(ods, time=0.2) == 1
    np.testing.assert_allclose(ods["plasma_initiation.time"], [0.1, 0.2])
    assert len(ods["plasma_initiation.b_field_lines"]) == 2


def test_an_earlier_instant_is_refused_rather_than_written_over_an_entry():
    """Entries are appended, so a time that sorts before the last one would
    land on an index that is already taken.  The version that sorted the time
    base instead overwrote it, and the map that disappeared left no trace."""
    ods = ODS()
    written(ods, time=0.3, phi_deg=0.0)
    with pytest.raises(ValueError, match="increasing order"):
        written(ods, time=0.1, phi_deg=0.0)
    assert ods["plasma_initiation.b_field_lines.0.time"] == 0.3
    assert len(ods["plasma_initiation.b_field_lines"]) == 1
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


def test_a_length_the_tracer_could_not_follow_is_accepted_as_nan():
    """NaN is a real result -- the VEST startup tracer writes one for every
    seed outside the wall -- so it must reach the IDS rather than be refused
    with the genuinely impossible values."""
    lengths = np.arange(6.0)
    lengths[2] = np.nan
    ods = ODS()
    written(ods, lengths=lengths)
    assert np.isnan(ods["plasma_initiation.b_field_lines.0.lengths"][2])


@pytest.mark.parametrize("bad", [-1.0, np.inf, -np.inf])
def test_a_negative_or_infinite_length_is_refused(bad):
    """An arclength cannot be negative, and an infinite one is a claim about
    a line that never closed -- which the tracer's status codes make and
    this leaf cannot."""
    lengths = np.arange(6.0)
    lengths[1] = bad
    with pytest.raises(ValueError, match="non-negative and finite"):
        written(ODS(), lengths=lengths)


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
    """Whether such a line reached the wall is unknown, so counting it either
    way would put a guess into a number that reads as a measurement."""
    product = flare_connection(tmp_path)
    path = product.source
    path.write_text(path.read_text().replace("-1.002000E+03", "7.000000E+00"))
    with pytest.raises(ValueError, match="does not define"):
        b_field_lines_from_flare_connection(ODS(), read_flare_product(path), time=0.0)


def test_the_known_status_codes_are_the_ones_flare_and_moose_define():
    """``fieldline.f90:25``'s four terminal codes, ``moose_error``'s
    ``SUCCESS = 0``, and ``DOMAIN_ERROR = 1`` -- which ``fieldline.f90:41``
    binds to ``edom`` and returns from the derivative when the trace leaves
    the field's domain."""
    assert set(FLARE_TRACE_STATUS) == {0, 1, -1001, -1002, -1003, -1004}
    assert FLARE_DOMAIN_ERROR == 1
    assert FLARE_TRACE_STATUS[FLARE_DOMAIN_ERROR] == "left the field's domain"


# ---------------------------------------------------------------------------
# A trace that left the field's domain: a failed trace, not a verdict
# ---------------------------------------------------------------------------


def domain_exit_product(tmp_path):
    """One of the six lines leaves the field's domain in both directions."""
    product = flare_connection(tmp_path)
    path = product.source
    rows = path.read_text().splitlines()
    body = [row for row in rows if not row.startswith("#")]
    body[0] = "  3.000000E+00  4.000000E+00  1.000000E+00  1.000000E+00"
    path.write_text("\n".join([r for r in rows if r.startswith("#")] + body) + "\n")
    return read_flare_product(path)


#: A box containing the r = 1.5 and r = 2.0 columns of the fixture grid but
#: not the r = 1.0 one, so two of the six lines start outside it.
BOX = np.array([[1.2, -1.5], [2.2, -1.5], [2.2, 0.5], [1.2, 0.5]])


def test_a_domain_exit_needs_a_boundary_to_mean_anything(tmp_path):
    """FLARE does not check that a start point lies inside the wall, so a
    domain exit is a seeding artefact or a real anomaly depending on where
    the line began -- and nothing in the product says which."""
    with pytest.raises(ValueError, match="no boundary was supplied"):
        b_field_lines_from_flare_connection(
            ODS(), domain_exit_product(tmp_path), time=0.0
        )


def test_a_line_starting_outside_the_boundary_is_neither_open_nor_closed(tmp_path):
    """Owner decision, 2026-09-22: excluded from open_fraction on both sides
    of the ratio, written with a NaN length, and counted in the provenance."""
    ods = ODS()
    b_field_lines_from_flare_connection(
        ods, domain_exit_product(tmp_path), time=0.0, boundary=BOX
    )
    entry = ods["plasma_initiation.b_field_lines.0"]
    lengths = np.asarray(entry["lengths"])
    # The whole r = 1.0 column is outside the box -- both of its lines.
    assert np.isnan(lengths[0]) and np.isnan(lengths[3])
    assert np.count_nonzero(np.isnan(lengths)) == 2
    # Four lines are counted, three of which reached the wall.
    assert entry["open_fraction"] == pytest.approx(3.0 / 4.0)
    parameters = ods["plasma_initiation.code.parameters"]
    assert "2 of 6 lines started outside the boundary" in parameters
    assert "numerator and denominator alike" in parameters


def test_the_test_is_on_the_start_alone_not_on_the_status_code(tmp_path):
    """A line seeded just outside the wall strikes it at once and reports a
    clean intersection, so keying on the status would leave exactly those in
    the numerator -- counted open for hitting a wall they started on the
    wrong side of.  Nothing in this product carries a domain exit, and two
    lines are still excluded."""
    product = flare_connection(tmp_path)
    assert not (product.column("ierr_bwd") == FLARE_DOMAIN_ERROR).any()
    assert not (product.column("ierr_fwd") == FLARE_DOMAIN_ERROR).any()

    ods = ODS()
    b_field_lines_from_flare_connection(ods, product, time=0.0, boundary=BOX)
    entry = ods["plasma_initiation.b_field_lines.0"]
    assert np.count_nonzero(np.isnan(np.asarray(entry["lengths"]))) == 2
    assert entry["open_fraction"] == pytest.approx(3.0 / 4.0)
    assert "whatever their status" in ods["plasma_initiation.code.parameters"]

    # ... and without the boundary the same product counts all six.
    bare = ODS()
    b_field_lines_from_flare_connection(bare, product, time=0.0)
    assert bare["plasma_initiation.b_field_lines.0.open_fraction"] == pytest.approx(4.0 / 6.0)


def test_a_domain_exit_from_inside_the_boundary_is_refused(tmp_path):
    """A line that should have been traceable and was not is a real anomaly,
    not a seeding artefact."""
    product = flare_connection(tmp_path)
    path = product.source
    rows = path.read_text().splitlines()
    body = [row for row in rows if not row.startswith("#")]
    body[1] = "  3.000000E+00  4.000000E+00  1.000000E+00  1.000000E+00"
    path.write_text("\n".join([r for r in rows if r.startswith("#")] + body) + "\n")
    with pytest.raises(ValueError, match=r"from a start \*inside\* the"):
        b_field_lines_from_flare_connection(
            ODS(), read_flare_product(path), time=0.0, boundary=BOX
        )


def test_a_product_with_no_boundary_and_no_domain_exit_says_so(tmp_path):
    ods = ODS()
    b_field_lines_from_flare_connection(ods, flare_connection(tmp_path), time=0.0)
    parameters = ods["plasma_initiation.code.parameters"]
    assert "no boundary was supplied" in parameters


def test_a_plane_wholly_inside_the_boundary_excludes_nothing(tmp_path):
    ods = ODS()
    wide = np.array([[0.0, -9.0], [9.0, -9.0], [9.0, 9.0], [0.0, 9.0]])
    b_field_lines_from_flare_connection(
        ods, flare_connection(tmp_path), time=0.0, boundary=wide
    )
    entry = ods["plasma_initiation.b_field_lines.0"]
    assert not np.isnan(np.asarray(entry["lengths"])).any()
    assert entry["open_fraction"] == pytest.approx(4.0 / 6.0)
    assert "every line started inside the boundary" in ods["plasma_initiation.code.parameters"]


def test_the_boundary_can_come_from_the_ods_wall(tmp_path):
    ods = ODS()
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = BOX[:, 0]
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = BOX[:, 1]
    b_field_lines_from_flare_connection(
        ods, domain_exit_product(tmp_path), time=0.0, boundary=WALL_BOUNDARY
    )
    assert ods["plasma_initiation.b_field_lines.0.open_fraction"] == pytest.approx(0.75)


def test_asking_for_a_wall_the_ods_does_not_have_is_refused(tmp_path):
    with pytest.raises(ValueError, match="read_flare_boundary"):
        b_field_lines_from_flare_connection(
            ODS(), domain_exit_product(tmp_path), time=0.0, boundary=WALL_BOUNDARY
        )


@pytest.mark.parametrize("bad, message", [
    ("outline", "boundary must be a contour"),
    (np.array([[1.0, 0.0], [2.0, 0.0]]), r"\(N, 2\) contour"),
    (np.array([1.0, 2.0, 3.0]), r"\(N, 2\) contour"),
])
def test_a_boundary_that_is_not_a_contour_is_refused(tmp_path, bad, message):
    with pytest.raises(ValueError, match=message):
        b_field_lines_from_flare_connection(
            ODS(), domain_exit_product(tmp_path), time=0.0, boundary=bad
        )


def test_a_plane_entirely_outside_the_boundary_has_no_fraction_to_take(tmp_path):
    """Refused rather than written as zero, which would read as a closed
    field rather than as a plane nobody traced."""
    product = flare_connection(tmp_path)
    path = product.source
    rows = path.read_text().splitlines()
    body = ["  3.000000E+00  4.000000E+00  1.000000E+00  1.000000E+00"] * 6
    path.write_text("\n".join([r for r in rows if r.startswith("#")] + body) + "\n")
    with pytest.raises(ValueError, match="no line left to take|none left to take"):
        b_field_lines_from_flare_connection(
            ODS(), read_flare_product(path), time=0.0,
            boundary=np.array([[9.0, 9.0], [10.0, 9.0], [10.0, 10.0]]),
        )


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
    with pytest.raises(ValueError, match="already carries fractions"):
        write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.4)


def test_a_second_write_naming_fresh_targets_is_refused_too():
    """Each write closes to one on its own, so two writes at one instant
    leave the divertor summing to two while every individual call looks
    correct. The check is on the divertor, not on the targets named here."""
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions={"a": 0.5, "b": 0.5}, time=0.0)
    with pytest.raises(ValueError, match="sum past one"):
        write_divertor_incident_fractions(ods, fractions={"c": 0.3, "d": 0.7}, time=0.0)
    assert len(ods["divertors.divertor.0.target"]) == 2
    total = sum(
        float(ods[f"divertors.divertor.0.target.{i}.power_incident_fraction.data"][0])
        for i in range(2)
    )
    assert total == pytest.approx(1.0)


def test_a_second_divertor_at_the_same_instant_is_fine():
    """The shares close per divertor, so another one is a separate total."""
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.4)
    write_divertor_incident_fractions(
        ods, fractions={"upper_in": 0.2, "upper_out": 0.8}, time=0.4, divertor=1
    )
    np.testing.assert_allclose(
        ods["divertors.divertor.1.target.1.power_incident_fraction.data"], [0.8]
    )


def test_a_refused_write_leaves_the_ods_exactly_as_it_found_it():
    """The refusals come before the first write, so a caller that catches one
    is not left with a half-updated divertor and a time base that moved."""
    ods = ODS()
    write_divertor_incident_fractions(ods, fractions=FRACTIONS, time=0.4)
    before = ods.pretty_paths()
    with pytest.raises(ValueError, match="already carries fractions"):
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


def proxy(scale, *, model_name="sol_channel_inverse_loss", incidence=True):
    shape = (4, 4)
    return footprint_heat_load_proxy(
        min_psi_norm_backward=np.full(shape, 1.0),
        min_psi_norm_forward=np.full(shape, 1.0),
        connection_length_backward=np.full(shape, 0.0),
        connection_length_forward=np.full(shape, 0.0),
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
    assert "sol_channel_inverse_loss" in ods["divertors.code.parameters"]


def test_targets_built_from_different_models_are_refused():
    """Their integrals are in different scales, so a share between them
    compares two different quantities."""
    with pytest.raises(ValueError, match="one model and one incidence"):
        divertor_incident_fractions_from_flare_footprints(
            ODS(),
            {"inner": proxy(1.0),
             "outer": proxy(1.0, model_name="sol_channel_exponential_loss")},
            time=0.0,
        )


def test_targets_that_disagree_about_the_incidence_factor_are_refused():
    with pytest.raises(ValueError, match="one model and one incidence"):
        divertor_incident_fractions_from_flare_footprints(
            ODS(),
            {"inner": proxy(1.0), "outer": proxy(1.0, incidence=False)},
            time=0.0,
        )

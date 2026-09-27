"""Reading FLARE's ASCII products.

FLARE's header is self-describing -- the product names its type, its columns
carry labels and units, and a dataset points at its mesh -- so these tests
cover the reader against synthetic files shaped like the reference outputs,
and the properties measured on those outputs are pinned as facts rather than
re-derived here.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.data.flare_products import (
    FlareColumn,
    read_flare_grid,
    read_flare_mesh,
    read_flare_product,
    resolve_geometry,
)


def write(path, header_lines, rows):
    path.write_text(
        "".join(f"# {line}\n" for line in header_lines)
        + "".join("  ".join(f"{value:.14E}" for value in row) + "\n" for row in rows)
    )
    return path


# --------------------------------------------------------------------------
# The header
# --------------------------------------------------------------------------


def test_the_type_and_the_columns_come_from_the_header(tmp_path):
    product = read_flare_product(write(
        tmp_path / "p.dat",
        [
            "TYPE dataset",
            'COLUMN_1 Lc_bwd, label = "Backward connection length", units = m',
            'COLUMN_2 Ltt_fwd, label = "Forward connection length", units = "{toroidal turns}"',
        ],
        [(1.0, 2.0), (3.0, 4.0)],
    ))
    assert product.kind == "dataset"
    assert product.columns == (
        FlareColumn(1, "Lc_bwd", "Backward connection length", "m"),
        FlareColumn(2, "Ltt_fwd", "Forward connection length", "{toroidal turns}"),
    )
    np.testing.assert_allclose(product.column("Ltt_fwd"), [2.0, 4.0])


def test_a_braced_unit_keeps_its_braces(tmp_path):
    """FLARE quotes compound units like "{poloidal turns}"; the braces are
    part of its own notation, not quoting."""
    product = read_flare_product(write(
        tmp_path / "p.dat",
        ["TYPE dataset", 'COLUMN_1 Lpt, label = "L", units = "{poloidal turns}"'],
        [(1.0,)],
    ))
    assert product.columns[0].units == "{poloidal turns}"


def test_a_file_without_a_type_is_read_rather_than_refused(tmp_path):
    """One reference output carries no TYPE at all -- a bare wide matrix --
    and a reader that required the key would reject a file FLARE writes."""
    product = read_flare_product(write(tmp_path / "f.dat", [], [(1.0, 2.0), (3.0, 4.0)]))
    assert product.kind is None
    assert product.columns == ()
    assert product.values.shape == (2, 2)


def test_an_unnamed_column_keeps_whatever_the_header_said(tmp_path):
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "COLUMN_1 plain"], [(1.0,)]
    ))
    assert product.columns[0] == FlareColumn(1, "plain", "", "")


def test_columns_come_back_in_index_order_not_file_order(tmp_path):
    product = read_flare_product(write(
        tmp_path / "p.dat",
        ["TYPE dataset", "COLUMN_2 second", "COLUMN_1 first"],
        [(1.0, 2.0)],
    ))
    assert [c.name for c in product.columns] == ["first", "second"]


def test_a_column_name_that_is_not_there_says_what_is(tmp_path):
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "COLUMN_1 only"], [(1.0,)]
    ))
    with pytest.raises(KeyError, match="only"):
        product.column("missing")


# --------------------------------------------------------------------------
# Meshes, including the block-structured one
# --------------------------------------------------------------------------


def test_a_rectangular_mesh_reads_as_a_table(tmp_path):
    grid = read_flare_grid(write(
        tmp_path / "m.grid",
        ["TYPE rmesh", "NODES 2 2", "U-AXIS r [m]", "V-AXIS z [m]"],
        [(1.0,), (2.0,), (3.0,), (4.0,)],
    ))
    assert grid.kind == "rmesh"
    assert grid.nodes == (2, 2)
    assert grid.values.shape == (4, 1)


def test_a_block_structured_mesh_keeps_its_blocks(tmp_path):
    """A tpzmesh3d writes NODES n_u n_v as n_u blocks, each one scalar -- that
    block's U-axis value -- then n_v rows. Flattening it would lose the
    blocking and refusing it would reject a mesh FLARE really writes."""
    path = tmp_path / "m.grid"
    path.write_text(
        "# TYPE tpzmesh3d\n# NODES 2 3\n"
        + "".join(
            f"  {u:.5E}\n" + "".join(f"  {r:.5E}  {z:.5E}  {p:.5E}\n" for r, z, p in
                                     [(1.0, 2.0, 3.0)] * 3)
            for u in (0.0, 180.0)
        )
    )
    grid = read_flare_grid(path)
    assert grid.nodes == (2, 3)
    assert len(grid.rows) == 2 * (1 + 3)
    assert [row.size for row in grid.rows[:4]] == [1, 3, 3, 3]
    with pytest.raises(ValueError, match="Read rows and nodes"):
        grid.values


def test_a_ragged_data_product_is_refused(tmp_path):
    """A mesh may be blocked; a data product may not, because its columns are
    indexed by position."""
    path = tmp_path / "p.dat"
    path.write_text("# TYPE dataset\n# COLUMN_1 a\n  1.0\n  1.0  2.0\n")
    with pytest.raises(ValueError, match="columns are indexed by position"):
        read_flare_product(path)


# --------------------------------------------------------------------------
# A dataset and the mesh it names
# --------------------------------------------------------------------------


def test_a_dataset_loads_the_mesh_it_points_at(tmp_path):
    write(tmp_path / "m.grid", ["TYPE rmesh", "NODES 2 2"],
          [(1.0,), (2.0,), (3.0,), (4.0,)])
    product = read_flare_product(write(
        tmp_path / "p.dat",
        ["TYPE dataset", "GEOMETRY m.grid", "COLUMN_1 v"],
        [(1.0,), (2.0,), (3.0,), (4.0,)],
    ))
    assert product.geometry == "m.grid"
    assert product.grid is not None and product.grid.nodes == (2, 2)
    assert resolve_geometry(product) == tmp_path / "m.grid"


def test_a_dataset_separated_from_its_mesh_still_reads(tmp_path):
    """A product copied away from its mesh is a normal thing to be handed,
    and its numbers are still its numbers."""
    product = read_flare_product(write(
        tmp_path / "p.dat",
        ["TYPE dataset", "GEOMETRY absent.grid", "COLUMN_1 v"],
        [(1.0,)],
    ))
    assert product.geometry == "absent.grid"
    assert product.grid is None
    assert product.grid_shape == ()
    assert product.cell_centred is None


def test_loading_the_mesh_can_be_declined(tmp_path):
    write(tmp_path / "m.grid", ["TYPE rmesh", "NODES 1 1"], [(1.0,)])
    product = read_flare_product(
        write(tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY m.grid"], [(1.0,)]),
        load_grid=False,
    )
    assert product.grid is None


# --------------------------------------------------------------------------
# Node- against cell-centred, which is the one that bites
# --------------------------------------------------------------------------


def test_a_dataset_with_one_row_per_node_is_node_centred(tmp_path):
    """Measured on the reference outputs: an rmesh or rmesh3d dataset has
    exactly prod(nodes) rows -- 64 on 8x8, 441 on 21x21."""
    write(tmp_path / "m.grid", ["TYPE rmesh", "NODES 3 4"], [(0.0,)] * 7)
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY m.grid", "COLUMN_1 v"],
        [(float(i),) for i in range(12)],
    ))
    assert product.grid_shape == (4, 3)
    assert product.cell_centred is False
    assert product.column("v").reshape(product.grid_shape).shape == (4, 3)


def test_the_value_shape_is_the_reverse_of_the_header(tmp_path):
    """FLARE's mesh library writes ``# NODES`` as its own ``nodes_shape``
    reversed (moose ``grids/_grid.py:110``), so a mesh declaring ``3 4`` holds
    values running C-order over ``(4, 3)``.  Taking the header at face value
    transposes every mesh whose axes differ in length -- and succeeds
    silently on every square one, which is what the reference outputs
    mostly are."""
    write(tmp_path / "m.grid", ["TYPE rmesh", "NODES 3 4"], [(0.0,)] * 7)
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY m.grid", "COLUMN_1 v"],
        [(float(i),) for i in range(12)],
    ))
    assert product.grid_shape == (4, 3) != product.grid.nodes


def test_a_dataset_with_one_row_per_cell_is_cell_centred(tmp_path):
    """And a tpzmesh3d dataset has prod(n - 1) rows -- 6192 on 37x173,
    19368 on 73x270 -- because a strike density lives on the wall element,
    not on the node. Reshaping it onto the nodes does not merely mislabel
    the picture; it does not fit."""
    write(tmp_path / "m.grid", ["TYPE tpzmesh3d", "NODES 4 5"], [(0.0,)])
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY m.grid", "COLUMN_1 v"],
        [(float(i),) for i in range(3 * 4)],
    ))
    assert product.grid_shape == (4, 3)
    assert product.cell_centred is True
    with pytest.raises(ValueError):
        product.column("v").reshape(5, 4)


def test_a_row_count_matching_neither_is_left_undecided(tmp_path):
    """Better an empty answer than a shape that happens to divide."""
    write(tmp_path / "m.grid", ["TYPE rmesh", "NODES 3 3"], [(0.0,)])
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY m.grid", "COLUMN_1 v"],
        [(float(i),) for i in range(7)],
    ))
    assert product.grid_shape == ()
    assert product.cell_centred is None


# --------------------------------------------------------------------------
# Placing a mesh's nodes: which axis is which, and where they sit
# --------------------------------------------------------------------------


RMESH_HEADER = [
    "TYPE rmesh",
    "NODES 3 2",
    "U-AXIS r [m]",
    "V-AXIS z [m]",
    "MAP3D 1 2 17.5",
    "COORDINATES cylindrical",
    "UNITS m, deg",
]


def rmesh(tmp_path, header=None):
    return read_flare_grid(write(
        tmp_path / "m.grid", header or RMESH_HEADER,
        [(1.0,), (2.0,), (3.0,), (-1.0,), (1.0,)],
    ))


def test_an_rmesh_is_one_poloidal_plane_at_the_angle_map3d_names(tmp_path):
    """MAP3D spends two entries on the coordinates the axes are, and the
    third on the value of the one they are not -- here the toroidal angle."""
    mesh = read_flare_mesh(rmesh(tmp_path))
    assert mesh.kind == "rmesh"
    assert mesh.node_shape == (2, 3) and mesh.cell_shape == (1, 2)
    assert mesh.toroidally_swept is False
    np.testing.assert_allclose(mesh.u, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(mesh.v, [-1.0, 1.0])
    np.testing.assert_allclose(mesh.r, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(mesh.z, [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(mesh.phi_deg, np.full((2, 3), 17.5))
    assert mesh.u_label == "r [m]" and mesh.v_label == "z [m]"


def test_an_rmesh_in_radians_and_centimetres_is_converted(tmp_path):
    """``# UNITS`` is part of the header for a reason: the node positions come
    back in metres and degrees whatever the file chose."""
    header = [line for line in RMESH_HEADER if not line.startswith("UNITS")]
    header += ["UNITS cm, rad"]
    mesh = read_flare_mesh(rmesh(tmp_path, header))
    np.testing.assert_allclose(mesh.r[0], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(mesh.z[:, 0], [-0.01, 0.01])
    np.testing.assert_allclose(mesh.phi_deg, np.full((2, 3), 17.5 * 180.0 / np.pi))
    # The axes themselves stay in the file's own units, with their labels.
    np.testing.assert_allclose(mesh.u, [1.0, 2.0, 3.0])
    assert mesh.length_units == "cm" and mesh.angle_units == "rad"


def test_units_naming_only_a_length_still_mean_degrees(tmp_path):
    """FLARE's own reader defaults the angular half, so a mesh writing
    ``# UNITS m`` is in degrees, not radians."""
    header = [line for line in RMESH_HEADER if not line.startswith("UNITS")] + ["UNITS m"]
    assert read_flare_mesh(rmesh(tmp_path, header)).angle_units == "deg"


def rmesh3d(tmp_path, header=None):
    """One target curve (three points) swept over two toroidal angles."""
    path = tmp_path / "m.grid"
    path.write_text(
        "".join(f"# {line}\n" for line in (header or [
            "TYPE rmesh3d", "NODES 2 3",
            "U-AXIS Toroidal Angle [deg]", "V-AXIS Distance along target [cm]",
            "MAP3D 1 2 3", "COORDINATES cylindrical", "UNITS m, deg",
        ]))
        + "  -3.0E+01\n  3.0E+01\n"
        + "  1.5E+00  -1.0E+00  0.0E+00\n"
        + "  1.6E+00  -1.1E+00  1.0E+01\n"
        + "  1.7E+00  -1.2E+00  2.0E+01\n"
    )
    return read_flare_grid(path)


def test_an_rmesh3d_sweeps_one_target_curve_through_the_toroidal_angle(tmp_path):
    """Its (R, z) depends on the distance along the target alone, so the file
    stores the curve once and the U axis is the toroidal angle itself."""
    mesh = read_flare_mesh(rmesh3d(tmp_path))
    assert mesh.kind == "rmesh3d" and mesh.toroidally_swept is True
    assert mesh.node_shape == (3, 2)
    np.testing.assert_allclose(mesh.u, [-30.0, 30.0])
    np.testing.assert_allclose(mesh.v, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(mesh.r, [[1.5, 1.5], [1.6, 1.6], [1.7, 1.7]])
    np.testing.assert_allclose(mesh.z, [[-1.0, -1.0], [-1.1, -1.1], [-1.2, -1.2]])
    np.testing.assert_allclose(mesh.phi_deg, [[-30.0, 30.0]] * 3)


def tpzmesh3d(tmp_path, header=None):
    """A wall whose shape changes with the angle: one block per position."""
    path = tmp_path / "m.grid"
    body = ""
    for block, (u, shift) in enumerate(((0.0, 0.0), (90.0, 0.5))):
        body += f"  {u:.5E}\n"
        for point in range(3):
            body += (f"  {1.0 + point + shift:.5E}  {-1.0 * point:.5E}"
                     f"  {2.0 * point:.5E}\n")
    path.write_text(
        "".join(f"# {line}\n" for line in (header or [
            "TYPE tpzmesh3d", "NODES 2 3", "U-AXIS Toroidal angle [deg]",
            "MAP3D 1 2 3", "COORDINATES cylindrical", "UNITS m, deg",
        ])) + body
    )
    return read_flare_grid(path)


def test_a_tpzmesh3d_carries_a_different_curve_at_every_angle(tmp_path):
    mesh = read_flare_mesh(tpzmesh3d(tmp_path))
    assert mesh.kind == "tpzmesh3d" and mesh.node_shape == (3, 2)
    np.testing.assert_allclose(mesh.u, [0.0, 90.0])
    np.testing.assert_allclose(mesh.r, [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
    np.testing.assert_allclose(mesh.z, [[0.0, 0.0], [-1.0, -1.0], [-2.0, -2.0]])
    np.testing.assert_allclose(mesh.v, [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
    np.testing.assert_allclose(mesh.phi_deg, [[0.0, 90.0]] * 3)


# --------------------------------------------------------------------------
# What the mesh reader refuses
# --------------------------------------------------------------------------


def test_a_mesh_type_whose_nodes_cannot_be_placed_is_refused(tmp_path):
    grid = read_flare_grid(write(tmp_path / "m.grid", ["TYPE ugrid2d", "NODES 2 2"],
                                 [(0.0,)] * 4))
    with pytest.raises(ValueError, match="refuses to guess"):
        read_flare_mesh(grid)


def test_cartesian_coordinates_are_refused_rather_than_read_as_cylindrical(tmp_path):
    header = [line for line in RMESH_HEADER if not line.startswith("COORDINATES")]
    with pytest.raises(ValueError, match="cylindrical"):
        read_flare_mesh(rmesh(tmp_path, header + ["COORDINATES cartesian"]))


def test_a_mapping_the_type_does_not_imply_is_refused(tmp_path):
    """An rmesh mapped onto (R, phi) would put its V axis on the angle; the
    reader names the axes from the type, so it must refuse rather than
    mislabel them."""
    header = [line for line in RMESH_HEADER if not line.startswith("MAP3D")]
    with pytest.raises(ValueError, match="only mapping"):
        read_flare_mesh(rmesh(tmp_path, header + ["MAP3D 1 3 0.0"]))


def test_a_swept_mesh_whose_u_axis_is_not_the_angle_is_refused(tmp_path):
    header = ["TYPE rmesh3d", "NODES 2 3", "MAP3D 1 3 2",
              "COORDINATES cylindrical", "UNITS m, deg"]
    with pytest.raises(ValueError, match="only mapping"):
        read_flare_mesh(rmesh3d(tmp_path, header))


def test_a_body_that_is_not_the_shape_the_header_declares_is_refused(tmp_path):
    header = [line for line in RMESH_HEADER if not line.startswith("NODES")]
    with pytest.raises(ValueError, match="single values"):
        read_flare_mesh(rmesh(tmp_path, header + ["NODES 4 4"]))


def test_a_tpzmesh3d_block_of_the_wrong_length_is_refused(tmp_path):
    header = ["TYPE tpzmesh3d", "NODES 2 4", "MAP3D 1 2 3",
              "COORDINATES cylindrical", "UNITS m, deg"]
    with pytest.raises(ValueError, match="blocks of"):
        read_flare_mesh(tpzmesh3d(tmp_path, header))


def test_an_unknown_unit_is_refused_rather_than_assumed_to_be_metres(tmp_path):
    header = [line for line in RMESH_HEADER if not line.startswith("UNITS")]
    with pytest.raises(ValueError, match="unknown length unit"):
        read_flare_mesh(rmesh(tmp_path, header + ["UNITS furlong, deg"]))


def test_a_mesh_declaring_one_axis_is_refused(tmp_path):
    header = [line for line in RMESH_HEADER if not line.startswith("NODES")]
    with pytest.raises(ValueError, match="two are required"):
        read_flare_mesh(rmesh(tmp_path, header + ["NODES 5"]))


# --------------------------------------------------------------------------
# A product on its mesh
# --------------------------------------------------------------------------


def test_on_mesh_lays_a_column_out_as_the_mesh_orders_it(tmp_path):
    write(tmp_path / "m.grid", RMESH_HEADER,
          [(1.0,), (2.0,), (3.0,), (-1.0,), (1.0,)])
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY m.grid", "COLUMN_1 Lc"],
        [(float(i),) for i in range(6)],
    ))
    np.testing.assert_allclose(product.on_mesh("Lc"), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert product.mesh.node_shape == (2, 3)


def test_on_mesh_refuses_a_row_count_that_fits_neither_nodes_nor_cells(tmp_path):
    write(tmp_path / "m.grid", RMESH_HEADER,
          [(1.0,), (2.0,), (3.0,), (-1.0,), (1.0,)])
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY m.grid", "COLUMN_1 Lc"],
        [(float(i),) for i in range(5)],
    ))
    with pytest.raises(ValueError, match="layout on it is unknown"):
        product.on_mesh("Lc")


def test_a_product_with_no_mesh_says_so_rather_than_failing_on_a_shape(tmp_path):
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY absent.grid", "COLUMN_1 Lc"],
        [(1.0,)],
    ))
    with pytest.raises(ValueError, match="absent.grid"):
        product.mesh


# --------------------------------------------------------------------------
# The model boundary, whose unit is in a descriptor and not in the contour
# --------------------------------------------------------------------------


def boundary_model(tmp_path, *, descriptor=None, contour=None, name="wall.txt"):
    """A FLARE model directory with a ``.boundary`` beside its equilibrium."""
    root = tmp_path / ".boundary"
    root.mkdir(exist_ok=True)
    (root / ".boundary").write_text(descriptor if descriptor is not None else (
        "[axisurf]\nfilename: {}\nunits:    cm\n".format(name)
    ))
    points = contour if contour is not None else [
        (100.0, -50.0), (300.0, -50.0), (300.0, 50.0), (100.0, 50.0)
    ]
    (root / name).write_text(
        "# a wall\n" + "".join(f"{r}\t{z}\n" for r, z in points)
    )
    return tmp_path


def test_a_boundary_is_converted_by_the_unit_its_descriptor_declares(tmp_path):
    """The contour file is two bare columns with at most a title comment, so
    a reader that assumed metres would place a centimetre wall a hundred
    times too far out and find every point inside it."""
    from vaft.data.flare_products import read_flare_boundary

    boundary = read_flare_boundary(boundary_model(tmp_path))
    assert boundary.units == "cm"
    np.testing.assert_allclose(boundary.points,
                               [[1.0, -0.5], [3.0, -0.5], [3.0, 0.5], [1.0, 0.5]])


def test_a_boundary_is_found_from_the_model_the_directory_or_the_file(tmp_path):
    from vaft.data.flare_products import read_flare_boundary

    model = boundary_model(tmp_path)
    for path in (model, model / ".boundary", model / ".boundary" / ".boundary"):
        assert read_flare_boundary(path).points.shape == (4, 2)


def test_a_toroidal_only_boundary_is_named_and_refused(tmp_path):
    """A torosurf is a 3-D wall, not a poloidal polygon; flattening one to
    test a starting point against would be an invention."""
    from vaft.data.flare_products import read_flare_boundary

    model = boundary_model(tmp_path, descriptor=(
        "[DEFAULT]\nunits: cm\n\n[firstwall:torosurf]\nfilename: wall.txt\n"
    ))
    with pytest.raises(ValueError, match="torosurf"):
        read_flare_boundary(model)


def test_a_boundary_without_a_declared_unit_is_refused(tmp_path):
    from vaft.data.flare_products import read_flare_boundary

    model = boundary_model(tmp_path, descriptor="[axisurf]\nfilename: wall.txt\n")
    with pytest.raises(ValueError, match="declares no length unit"):
        read_flare_boundary(model)


def test_a_boundary_in_an_unknown_unit_is_refused(tmp_path):
    from vaft.data.flare_products import read_flare_boundary

    model = boundary_model(
        tmp_path, descriptor="[axisurf]\nfilename: wall.txt\nunits: cubit\n"
    )
    with pytest.raises(ValueError, match="cubit"):
        read_flare_boundary(model)


def test_a_boundary_naming_a_contour_that_is_not_there_is_refused(tmp_path):
    from vaft.data.flare_products import read_flare_boundary

    model = boundary_model(
        tmp_path, descriptor="[axisurf]\nfilename: absent.txt\nunits: cm\n"
    )
    with pytest.raises(FileNotFoundError, match="absent.txt"):
        read_flare_boundary(model)


def test_a_contour_too_short_to_close_is_refused(tmp_path):
    from vaft.data.flare_products import read_flare_boundary

    model = boundary_model(tmp_path, contour=[(100.0, 0.0), (200.0, 0.0)])
    with pytest.raises(ValueError, match="at least three"):
        read_flare_boundary(model)


def test_a_model_with_no_descriptor_says_where_one_lives(tmp_path):
    from vaft.data.flare_products import read_flare_boundary

    with pytest.raises(FileNotFoundError, match=".boundary"):
        read_flare_boundary(tmp_path)


def test_several_axisymmetric_surfaces_are_kept_apart(tmp_path):
    """Which of them bounds the region a caller means is the caller's
    question, so `points` refuses and `contours` hands them all over."""
    from vaft.data.flare_products import read_flare_boundary

    model = boundary_model(tmp_path, descriptor=(
        "[DEFAULT]\nunits: cm\n\n[firstwall:axisurf]\nfilename: wall.txt\n\n"
        "[target:axisurf]\nfilename: wall.txt\n"
    ))
    boundary = read_flare_boundary(model)
    assert len(boundary.contours) == 2
    with pytest.raises(ValueError, match="2 axisymmetric surfaces"):
        boundary.points

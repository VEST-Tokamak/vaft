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
    write(tmp_path / "m.grid", ["TYPE rmesh", "NODES 3 3"], [(0.0,)] * 9)
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY m.grid", "COLUMN_1 v"],
        [(float(i),) for i in range(9)],
    ))
    assert product.grid_shape == (3, 3)
    assert product.cell_centred is False
    assert product.column("v").reshape(product.grid_shape).shape == (3, 3)


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
    assert product.grid_shape == (3, 4)
    assert product.cell_centred is True
    with pytest.raises(ValueError):
        product.column("v").reshape(4, 5)


def test_a_row_count_matching_neither_is_left_undecided(tmp_path):
    """Better an empty answer than a shape that happens to divide."""
    write(tmp_path / "m.grid", ["TYPE rmesh", "NODES 3 3"], [(0.0,)])
    product = read_flare_product(write(
        tmp_path / "p.dat", ["TYPE dataset", "GEOMETRY m.grid", "COLUMN_1 v"],
        [(float(i),) for i in range(7)],
    ))
    assert product.grid_shape == ()
    assert product.cell_centred is None

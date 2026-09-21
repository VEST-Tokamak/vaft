"""Reading FLARE's ASCII products.

FLARE writes every product as a commented header over a numeric block, and
the header is self-describing: ``# TYPE`` names the product, ``# COLUMN_n``
gives each column a label and a unit, and a dataset points at the mesh it
lives on with ``# GEOMETRY``. So one reader covers all of them, and the
containers keep what the file said rather than what a caller expected.

Six product types and three mesh types appear in the reference outputs --
``dataset``, ``poincare_map``, ``rlist``, ``fieldline``, ``interp_curve``,
and one file with **no** ``TYPE`` at all (``fourier_transform.dat``, a bare
wide matrix). A reader that required the key would refuse that one, so
:attr:`FlareProduct.kind` is ``None`` there rather than an error.

Nothing here converts. FLARE's own units are in the header and stay there;
the legacy loaders in ``library/flare_plotting.py`` hard-coded a column
contract per product, which is what a self-describing header exists to avoid.

What a mesh file *means* -- which axis is which, where each node sits in
``(R, z, phi)``, and in what order a dataset's rows run over it -- is read by
:func:`read_flare_mesh`. That order is not a convention anyone chose: FLARE's
mesh library writes ``# NODES`` as its own ``nodes_shape`` reversed
(``moose/src/python/moose/grids/_grid.py:110``), and every ``nodes_shape``
there is ``(n_v, n_u)``. So the header reads ``n_u n_v`` while the values run
C-order over ``(n_v, n_u)``, with the U axis varying fastest -- and a reader
that takes the header at face value transposes every non-square mesh it is
handed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

__all__ = [
    "RECTANGULAR_MESHES",
    "FlareColumn",
    "FlareGrid",
    "FlareMesh",
    "FlareProduct",
    "read_flare_grid",
    "read_flare_mesh",
    "read_flare_product",
    "resolve_geometry",
]

#: Mesh types whose nodes this module can place in ``(R, z, phi)``.  Each is a
#: surface swept in the toroidal angle, which is what makes a node's position
#: derivable; FLARE's unstructured and flux-coordinate meshes are not.
RECTANGULAR_MESHES = ("rmesh", "rmesh3d", "tpzmesh3d")

#: ``# COLUMN_3 Lpt_bwd, label = "Backward connection length", units = "{poloidal turns}"``
_COLUMN = re.compile(
    r"^(?P<name>\S+)"
    r"(?:\s*,\s*label\s*=\s*(?P<label>\"[^\"]*\"|\S+))?"
    r"(?:\s*,\s*units\s*=\s*(?P<units>\"[^\"]*\"|\S+))?"
    r"\s*$"
)


def _unquote(value: str | None) -> str:
    return "" if value is None else value.strip().strip('"')


@dataclass(frozen=True)
class FlareColumn:
    """One column of a FLARE product, as its header described it."""

    index: int
    name: str
    label: str = ""
    units: str = ""


@dataclass(frozen=True)
class FlareGrid:
    """One FLARE mesh file."""

    kind: str | None
    header: Mapping[str, str]
    rows: tuple[np.ndarray, ...]
    source: Path

    @property
    def values(self) -> np.ndarray:
        """The rows as one array, when every row is the same width.

        Raises
        ------
        ValueError
            The mesh is block-structured -- read :attr:`rows` and
            :attr:`nodes` instead.
        """
        return _rectangular(
            list(self.rows), self.source,
            what="Read rows and nodes: a block-structured mesh writes one "
                 "scalar U-axis value per block, then that block's points.",
        )

    @property
    def nodes(self) -> tuple[int, ...]:
        """The ``# NODES`` shape, empty when the file does not declare one."""
        raw = self.header.get("NODES", "")
        return tuple(int(part) for part in raw.split()) if raw.strip() else ()


#: ``# UNITS`` length words, as metres per unit.  FLARE writes ``m`` on every
#: reference mesh; the rest are here so a file that uses one is converted
#: rather than read as metres.
_LENGTH_SCALE = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3}

#: ``# UNITS`` angle words, as degrees per unit.
_ANGLE_SCALE = {"deg": 1.0, "rad": 180.0 / np.pi}

#: Cylindrical coordinate ids, as ``# MAP3D`` numbers them: 1 = R, 2 = z,
#: 3 = the toroidal angle.
_R, _Z, _PHI = 1, 2, 3


@dataclass(frozen=True)
class FlareMesh:
    """A FLARE mesh with every node placed in ``(R, z, phi)``.

    The axes stay in the file's own coordinates -- ``u`` is whatever
    ``# U-AXIS`` says it is, in whatever ``# UNITS`` declares -- while
    :attr:`r`, :attr:`z` and :attr:`phi_deg` are the same nodes in metres and
    degrees, which is what an area or a plot needs. Both are kept, because
    the footprint axes a figure is drawn on (toroidal angle, distance along
    the target) are not recoverable from the positions alone.

    Every array of node positions has shape :attr:`node_shape`, and a
    dataset's values run over it in C order -- see
    :attr:`FlareProduct.grid_shape` for why that is the reverse of the
    header.
    """

    kind: str
    u: np.ndarray
    v: np.ndarray
    u_label: str
    v_label: str
    r: np.ndarray
    z: np.ndarray
    phi_deg: np.ndarray
    length_units: str
    angle_units: str
    source: Path

    @property
    def node_shape(self) -> tuple[int, int]:
        """``(n_v, n_u)`` -- the shape a node-centred dataset reshapes to."""
        return (int(self.r.shape[0]), int(self.r.shape[1]))

    @property
    def cell_shape(self) -> tuple[int, int]:
        """``(n_v - 1, n_u - 1)`` -- the shape a cell-centred dataset reshapes to."""
        rows, columns = self.node_shape
        return (rows - 1, columns - 1)

    @property
    def toroidally_swept(self) -> bool:
        """Whether the mesh spans a toroidal range rather than one plane.

        An ``rmesh`` is one ``(R, z)`` plane at a fixed angle; the two 3-D
        meshes sweep their U axis through the toroidal angle.
        """
        return self.kind != "rmesh"


def _units(header: Mapping[str, str]) -> tuple[str, str]:
    """``(length word, angle word)`` from ``# UNITS``.

    ``# UNITS m`` means metres and degrees: FLARE's own reader defaults the
    angular half when the field names only one unit.
    """
    parts = [part.strip() for part in header.get("UNITS", "m").split(",")]
    length = parts[0] or "m"
    angle = (parts[1] if len(parts) > 1 else "") or "deg"
    for word, table, what in ((length, _LENGTH_SCALE, "length"),
                              (angle, _ANGLE_SCALE, "angle")):
        if word not in table:
            raise ValueError(
                f"unknown {what} unit {word!r} in '# UNITS'; this reader "
                f"knows {sorted(table)}"
            )
    return length, angle


def _map3d(header: Mapping[str, str], source: Path) -> list[str]:
    """The three entries of ``# MAP3D``, unsplit.

    The field always holds three numbers. A 2-D domain spends two of them on
    coordinate ids and the third on the value of the coordinate it does not
    span; a 3-D one spends all three on ids. Which it is follows from whether
    the trailing entries are whole numbers, so this splits on the mesh's own
    dimensionality instead -- the caller knows it and passes the count.
    """
    raw = header.get("MAP3D", "").split()
    if len(raw) != 3:
        raise ValueError(
            f"{source.name} declares '# MAP3D {' '.join(raw)}'; three entries "
            "are required to place its nodes in 3-D."
        )
    return raw


def _split_map3d(raw: list[str], ndim: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    ids = tuple(int(float(value)) for value in raw[:ndim])
    reference = tuple(float(value) for value in raw[ndim:])
    return ids, reference


def _refuse_coordinates(header: Mapping[str, str], source: Path) -> None:
    coordinates = header.get("COORDINATES", "cylindrical").strip()
    if coordinates != "cylindrical":
        raise ValueError(
            f"{source.name} is in {coordinates!r} coordinates; placing its "
            "nodes in (R, z, phi) is only defined for cylindrical ones."
        )


def _axis_label(header: Mapping[str, str], key: str) -> str:
    return header.get(key, "").strip()


def read_flare_mesh(grid: FlareGrid) -> FlareMesh:
    """Interpret one FLARE mesh: its axes, and where its nodes sit.

    Parameters
    ----------
    grid : FlareGrid
        As :func:`read_flare_grid` returns it.

    Returns
    -------
    FlareMesh

    Raises
    ------
    ValueError
        The mesh is of a type this cannot place (:data:`RECTANGULAR_MESHES`
        lists the ones it can), is not in cylindrical coordinates, declares
        no ``# NODES``, maps its axes onto coordinates other than the ones
        its type implies, or holds a number of rows its declared shape does
        not account for.

    Notes
    -----
    The three supported types differ in what depends on what, and the
    difference is the whole reason a reader is needed:

    ``rmesh``
        One ``(R, z)`` plane. ``u`` is R and ``v`` is z, and the toroidal
        angle is the fixed third coordinate ``# MAP3D`` carries.
    ``rmesh3d``
        A target curve swept toroidally. ``u`` is the toroidal angle and
        ``v`` runs along the target, whose ``(R, z)`` depends on ``v``
        alone -- so the file stores the curve once.
    ``tpzmesh3d``
        The same, for a wall whose shape changes with the angle: ``(R, z)``
        and ``v`` both depend on both axes, and the file is written as one
        block per toroidal position.
    """
    source = grid.source
    kind = grid.kind
    if kind not in RECTANGULAR_MESHES:
        raise ValueError(
            f"{source.name} is a {kind!r} mesh; this reader places the nodes "
            f"of {list(RECTANGULAR_MESHES)} and refuses to guess for the rest."
        )
    _refuse_coordinates(grid.header, source)
    nodes = grid.nodes
    if len(nodes) != 2:
        raise ValueError(
            f"{source.name} declares '# NODES "
            f"{' '.join(str(n) for n in nodes)}'; two are required."
        )
    # The header prints (n_u, n_v); everything downstream works in (n_v, n_u).
    n_u, n_v = (int(nodes[0]), int(nodes[1]))
    length_units, angle_units = _units(grid.header)
    to_metres = _LENGTH_SCALE[length_units]
    to_degrees = _ANGLE_SCALE[angle_units]
    raw_map = _map3d(grid.header, source)
    rows = list(grid.rows)

    if kind == "rmesh":
        ids, reference = _split_map3d(raw_map, 2)
        if ids != (_R, _Z) or len(reference) != 1:
            raise ValueError(
                f"{source.name} maps its axes onto coordinates {ids}; an "
                "rmesh is read as one (R, z) plane, so (1, 2) is the only "
                "mapping whose axes this reader can name."
            )
        u, v = _mesh_axes(rows, n_u, n_v, source)
        r, z = np.meshgrid(u * to_metres, v * to_metres)
        phi_deg = np.full(r.shape, float(reference[0]) * to_degrees)
        return FlareMesh(
            kind=kind, u=u, v=v,
            u_label=_axis_label(grid.header, "U-AXIS"),
            v_label=_axis_label(grid.header, "V-AXIS"),
            r=r, z=z, phi_deg=phi_deg,
            length_units=length_units, angle_units=angle_units, source=source,
        )

    ids, reference = _split_map3d(raw_map, 3)
    if ids != (_R, _Z, _PHI) or reference:
        raise ValueError(
            f"{source.name} maps its axes onto coordinates {ids}; a {kind} "
            "is a surface swept in the toroidal angle, so (1, 2, 3) is the "
            "only mapping whose U axis is that angle."
        )
    if kind == "rmesh3d":
        u, curve = _rmesh3d_body(rows, n_u, n_v, source)
        r = np.tile((curve[:, 0] * to_metres)[:, None], (1, n_u))
        z = np.tile((curve[:, 1] * to_metres)[:, None], (1, n_u))
        v = curve[:, 2]
    else:
        u, r, z, v = _tpzmesh3d_body(rows, n_u, n_v, source)
        r = r * to_metres
        z = z * to_metres
    phi_deg = np.tile((u * to_degrees)[None, :], (n_v, 1))
    return FlareMesh(
        kind=kind, u=u, v=v,
        u_label=_axis_label(grid.header, "U-AXIS"),
        v_label=_axis_label(grid.header, "V-AXIS"),
        r=r, z=z, phi_deg=phi_deg,
        length_units=length_units, angle_units=angle_units, source=source,
    )


def _widths(rows: list[np.ndarray]) -> list[int]:
    return [int(row.size) for row in rows]


def _mesh_axes(rows, n_u, n_v, source) -> tuple[np.ndarray, np.ndarray]:
    """An ``rmesh`` body: ``n_u`` U values, then ``n_v`` V values."""
    if len(rows) != n_u + n_v or any(width != 1 for width in _widths(rows)):
        raise ValueError(
            f"{source.name} declares {n_u} x {n_v} nodes, so an rmesh body is "
            f"{n_u + n_v} single values; it holds {len(rows)} rows of "
            f"{sorted(set(_widths(rows)))}."
        )
    stacked = np.concatenate(rows)
    return stacked[:n_u], stacked[n_u:]


def _rmesh3d_body(rows, n_u, n_v, source) -> tuple[np.ndarray, np.ndarray]:
    """An ``rmesh3d`` body: ``n_u`` U values, then ``n_v`` ``(x1, x2, v)`` rows."""
    widths = _widths(rows)
    if (len(rows) != n_u + n_v or widths[:n_u] != [1] * n_u
            or widths[n_u:] != [3] * n_v):
        raise ValueError(
            f"{source.name} declares {n_u} x {n_v} nodes, so an rmesh3d body "
            f"is {n_u} single values then {n_v} rows of three; it holds "
            f"{len(rows)} rows of {sorted(set(widths))}."
        )
    return np.concatenate(rows[:n_u]), np.vstack(rows[n_u:])


def _tpzmesh3d_body(rows, n_u, n_v, source):
    """A ``tpzmesh3d`` body: ``n_u`` blocks of one U value and ``n_v`` triples."""
    expected = n_u * (1 + n_v)
    widths = _widths(rows)
    if len(rows) != expected:
        raise ValueError(
            f"{source.name} declares {n_u} x {n_v} nodes, so a tpzmesh3d body "
            f"is {n_u} blocks of {1 + n_v} rows ({expected} in all); it holds "
            f"{len(rows)} rows."
        )
    u = np.empty(n_u)
    r = np.empty((n_v, n_u))
    z = np.empty((n_v, n_u))
    v = np.empty((n_v, n_u))
    for block in range(n_u):
        start = block * (1 + n_v)
        if widths[start] != 1 or widths[start + 1:start + 1 + n_v] != [3] * n_v:
            raise ValueError(
                f"{source.name} block {block} is not one U value over {n_v} "
                "rows of three; the mesh is not the shape its header declares."
            )
        u[block] = rows[start][0]
        chunk = np.vstack(rows[start + 1:start + 1 + n_v])
        r[:, block], z[:, block], v[:, block] = chunk[:, 0], chunk[:, 1], chunk[:, 2]
    return u, r, z, v


@dataclass(frozen=True)
class FlareProduct:
    """One FLARE data file: what it says it is, and its numbers."""

    kind: str | None
    header: Mapping[str, str]
    columns: tuple[FlareColumn, ...]
    values: np.ndarray
    source: Path
    grid: FlareGrid | None = field(default=None)

    @property
    def geometry(self) -> str:
        """The mesh file this references, empty when it stands alone."""
        return self.header.get("GEOMETRY", "").strip()

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """The shape one of this product's columns takes on the mesh it names.

        ``(n_v, n_u)`` on a node-centred dataset and ``(n_v - 1, n_u - 1)`` on
        a cell-centred one -- **the order a reshape wants, which is the
        reverse of the order the header prints**. FLARE's mesh library writes
        ``# NODES`` as its own ``nodes_shape`` reversed
        (``moose/src/python/moose/grids/_grid.py:110``), so a header reading
        ``37 173`` describes values that run C-order over ``(173, 37)``.
        Reshaping to the header's own order transposes every mesh whose two
        axes differ in length, and silently succeeds on every square one.

        Returns an empty tuple when there is no mesh, or when the row count
        matches neither the mesh's nodes nor its cells.

        **Node against cell is the other half, and getting it wrong is not a
        rounding error.** Measured on the reference outputs: an ``rmesh`` or
        ``rmesh3d`` dataset has exactly ``prod(nodes)`` rows -- 64 on 8x8, 441
        on 21x21 -- so its values sit on the nodes. A ``tpzmesh3d`` dataset
        has ``prod(n - 1)`` rows -- 6192 on 37x173 and 19368 on 73x270 -- so
        its values sit on the cells the nodes bound, which is what a strike
        density on a wall element is.
        """
        nodes = self.grid.nodes if self.grid is not None else ()
        if not nodes or self.values.size == 0:
            return ()
        count = int(self.values.shape[0])
        # The header is (n_u, n_v); a value array runs over (n_v, n_u).
        shape = tuple(int(n) for n in nodes[::-1])
        if count == int(np.prod(shape)):
            return shape
        cells = tuple(n - 1 for n in shape)
        if all(n > 0 for n in cells) and count == int(np.prod(cells)):
            return cells
        return ()

    @property
    def cell_centred(self) -> bool | None:
        """Whether the values sit on cells rather than nodes.

        ``None`` when :attr:`grid_shape` could not decide.
        """
        shape = self.grid_shape
        if not shape or self.grid is None:
            return None
        return shape != tuple(int(n) for n in self.grid.nodes[::-1])

    @property
    def mesh(self) -> "FlareMesh":
        """The mesh this product names, with every node placed in ``(R, z, phi)``.

        Raises
        ------
        ValueError
            The product names no mesh, or the mesh is of a type whose nodes
            cannot be placed -- see :func:`read_flare_mesh`.
        """
        if self.grid is None:
            named = self.geometry or "nothing"
            raise ValueError(
                f"{self.source.name} carries no mesh (its GEOMETRY names "
                f"{named}); read it with read_flare_grid and pass it to "
                "read_flare_mesh."
            )
        return read_flare_mesh(self.grid)

    def on_mesh(self, name: str) -> np.ndarray:
        """One column laid out on the mesh, as ``values[i_v, i_u]``.

        Raises
        ------
        KeyError
            No column carries that name.
        ValueError
            The product names no mesh, or its row count fits neither the
            mesh's nodes nor its cells -- in which case the layout is
            genuinely unknown and a reshape that happened to divide would be
            a guess.
        """
        values = self.column(name)
        shape = self.grid_shape
        if not shape:
            nodes = self.grid.nodes if self.grid is not None else ()
            raise ValueError(
                f"{self.source.name} has {values.size} values, which is "
                f"neither the nodes nor the cells of a {nodes or 'missing'} "
                "mesh, so their layout on it is unknown."
            )
        return values.reshape(shape)

    def column(self, name: str) -> np.ndarray:
        """One column by the name its header gave it.

        Raises
        ------
        KeyError
            No column carries that name.
        """
        for entry in self.columns:
            if entry.name == name:
                return self.values[:, entry.index - 1]
        raise KeyError(
            f"{self.source.name} has no column named {name!r}; it declares "
            f"{[entry.name for entry in self.columns]}"
        )


def _read(path: Path) -> tuple[dict[str, str], list[np.ndarray]]:
    """Split a FLARE file into its ``# KEY value`` header and its rows.

    Rows are kept as they come, one array each, because not every product is
    rectangular: a ``tpzmesh3d`` mesh writes ``NODES n_u n_v`` as ``n_u``
    blocks, each a single scalar -- that block's U-axis value -- followed by
    ``n_v`` rows. Flattening that into a matrix would lose the blocking, and
    refusing it would reject a mesh FLARE really writes.
    """
    header: dict[str, str] = {}
    rows: list[np.ndarray] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                body = stripped.lstrip("#").strip()
                if not body:
                    continue
                key, _, value = body.partition(" ")
                # Later keys win: FLARE writes each once, and a duplicate is
                # the file disagreeing with itself rather than a list.
                header[key] = value.strip()
                continue
            rows.append(np.asarray([float(part) for part in stripped.split()]))
    return header, rows


def _rectangular(rows: list[np.ndarray], path: Path, *, what: str) -> np.ndarray:
    """``rows`` as a 2-D array, or a refusal naming the widths it found."""
    if not rows:
        return np.empty((0, 0))
    widths = sorted({int(row.size) for row in rows})
    if len(widths) > 1:
        raise ValueError(
            f"{path.name} has rows of {widths} values, so it is not a table. "
            f"{what}"
        )
    return np.vstack(rows)


def _columns(header: Mapping[str, str]) -> tuple[FlareColumn, ...]:
    found: list[FlareColumn] = []
    for key, value in header.items():
        if not key.startswith("COLUMN_"):
            continue
        try:
            index = int(key.removeprefix("COLUMN_"))
        except ValueError:  # pragma: no cover - FLARE numbers them
            continue
        match = _COLUMN.match(value.strip())
        if match is None:
            found.append(FlareColumn(index=index, name=value.strip()))
            continue
        found.append(
            FlareColumn(
                index=index,
                name=match.group("name"),
                label=_unquote(match.group("label")),
                units=_unquote(match.group("units")),
            )
        )
    return tuple(sorted(found, key=lambda entry: entry.index))


def read_flare_grid(path: str | Path) -> FlareGrid:
    """Read one FLARE mesh file."""
    source = Path(path)
    header, rows = _read(source)
    return FlareGrid(
        kind=header.get("TYPE"), header=header, rows=tuple(rows), source=source
    )


def resolve_geometry(product: FlareProduct) -> Path | None:
    """Where a product's ``# GEOMETRY`` mesh would be, beside the product."""
    name = product.geometry
    return None if not name else product.source.parent / name


def read_flare_product(path: str | Path, *, load_grid: bool = True) -> FlareProduct:
    """Read one FLARE data file, and the mesh it names.

    Parameters
    ----------
    path : str or path-like
        The ``.dat`` to read.
    load_grid : bool, optional
        Read the ``# GEOMETRY`` mesh too, when the header names one and the
        file is beside it. A missing mesh is left as ``None`` rather than
        raised on: the data is still the data, and a product copied away from
        its mesh is a normal thing to be handed.

    Returns
    -------
    FlareProduct
        ``kind`` is ``None`` for a file carrying no ``# TYPE``.

    Raises
    ------
    ValueError
        The numeric block is ragged.
    OSError
        The product itself cannot be read.
    """
    source = Path(path)
    header, rows = _read(source)
    values = _rectangular(
        rows, source,
        what="A data product's columns are indexed by position, which a "
             "ragged block cannot support.",
    )
    product = FlareProduct(
        kind=header.get("TYPE"),
        header=header,
        columns=_columns(header),
        values=values,
        source=source,
    )
    if not load_grid:
        return product
    geometry = resolve_geometry(product)
    if geometry is None or not geometry.exists():
        return product
    return FlareProduct(
        kind=product.kind,
        header=product.header,
        columns=product.columns,
        values=product.values,
        source=product.source,
        grid=read_flare_grid(geometry),
    )

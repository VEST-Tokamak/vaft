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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

__all__ = [
    "FlareColumn",
    "FlareGrid",
    "FlareProduct",
    "read_flare_grid",
    "read_flare_product",
    "resolve_geometry",
]

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
        """The shape this product's rows take on the mesh it names.

        Returns an empty tuple when there is no mesh, or when the row count
        matches neither its nodes nor its cells.

        **Which one it is depends on the mesh type, and getting it wrong is
        not a rounding error.** Measured on the reference outputs: an
        ``rmesh`` or ``rmesh3d`` dataset has exactly ``prod(nodes)`` rows --
        64 on 8x8, 441 on 21x21 -- so its values sit on the nodes. A
        ``tpzmesh3d`` dataset has ``prod(n - 1)`` rows -- 6192 on 37x173 and
        19368 on 73x270 -- so its values sit on the cells the nodes bound,
        which is what a strike density on a wall element is.
        """
        nodes = self.grid.nodes if self.grid is not None else ()
        if not nodes or self.values.size == 0:
            return ()
        count = int(self.values.shape[0])
        node_count = int(np.prod(nodes))
        if count == node_count:
            return tuple(int(n) for n in nodes)
        cells = tuple(int(n) - 1 for n in nodes)
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
        return shape != tuple(int(n) for n in self.grid.nodes)

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

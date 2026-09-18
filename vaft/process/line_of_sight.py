"""Line integrals of a poloidal-plane field along diagnostic sightlines (issue #886).

A line-integrated diagnostic measures ``B_i = int epsilon dl`` along sightline
``i``. On a grid of cells that is a matrix product ``B = G epsilon`` with
``G_ij`` the length of sightline ``i`` inside cell ``j``. This module builds
``G`` from exact segment-cell intersections, so it is a geometric statement
about the chords and the grid, not a resampled image of a mask, and it
converges as the grid is refined.

The operator knows nothing about which diagnostic the chords belong to or
what the field is: soft X-ray emissivity, radiated power or density all
integrate the same way. Machine geometry is loaded elsewhere, for VEST by
:func:`vaft.machine_mapping.soft_x_rays.sxr_sightlines`.

Notation
--------
G_ij      : length of chord i inside cell j                               [m]
epsilon_j : field value on cell j                          [any, per metre]
B_i       : line integral along chord i                         [any]

Conventions
-----------
**A grid node is a cell centre.** Cell edges sit halfway between nodes and
half a spacing beyond the outermost ones, so the cells tile the rectangle the
grid spans plus half a cell all round, and a field sampled on the nodes of an
equilibrium map is integrated as piecewise constant on those cells. Cells are
flattened in C order over ``(R, Z)``, the order ``field.reshape(-1)`` gives
for a field indexed ``(R, Z)``.

**A chord is a straight segment in one poloidal plane.** Its end points are
``(R, Z)`` at one toroidal angle; a sightline that crosses toroidal angle is
not representable here.

Provenance
----------
.. [1] Siddon, R. L., Med. Phys. 12, 252 (1985), the exact ray-grid
   intersection this follows.
.. [2] Jang, J. Y. et al., Rev. Sci. Instrum. 93, 093506 (2022), the VEST
   soft-X-ray arrays whose chords motivated the module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Sightlines",
    "build_line_integral_operator",
    "clip_segment_to_polygon",
    "project_emissivity",
]


@dataclass(frozen=True)
class Sightlines:
    """Straight chords in poloidal planes: end points, toroidal angle, labels.

    ``r1, z1`` and ``r2, z2`` are the end points in metres, ``phi`` the IMAS
    toroidal angle of each chord's plane in radians, and ``labels`` a name per
    chord. All arrays have one entry per chord.
    """

    r1: np.ndarray
    z1: np.ndarray
    r2: np.ndarray
    z2: np.ndarray
    phi: np.ndarray
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("r1", "z1", "r2", "z2", "phi"):
            value = np.atleast_1d(np.asarray(getattr(self, name), dtype=float)).reshape(-1)
            object.__setattr__(self, name, value)
            arrays[name] = value
        sizes = {value.size for value in arrays.values()}
        if len(sizes) != 1:
            raise ValueError(f"sightline arrays disagree in length: {sorted(sizes)}")
        (size,) = sizes
        labels = tuple(str(label) for label in self.labels) or tuple(str(i) for i in range(size))
        if len(labels) != size:
            raise ValueError(f"{len(labels)} labels for {size} sightlines")
        object.__setattr__(self, "labels", labels)
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("a sightline coordinate is not finite")

    def __len__(self) -> int:
        return self.r1.size

    def subset(self, index) -> "Sightlines":
        """The chords selected by an index or boolean mask [-]."""
        index = np.asarray(index)
        if index.dtype == bool:
            index = np.nonzero(index)[0]
        return Sightlines(self.r1[index], self.z1[index], self.r2[index], self.z2[index],
                          self.phi[index], tuple(self.labels[i] for i in index))


def clip_segment_to_polygon(p1, p2, polygon_r, polygon_z) -> list[tuple[float, float]]:
    """The parameter intervals of a segment that lie inside a closed polygon.

    Parameters
    ----------
    p1 : sequence of float
        ``(R, Z)`` of the segment start [m].
    p2 : sequence of float
        ``(R, Z)`` of the segment end [m].
    polygon_r : array_like
        Polygon vertex major radii, closed or open [m].
    polygon_z : array_like
        Polygon vertex heights [m].

    Returns
    -------
    list of tuple of float
        ``(t_start, t_end)`` intervals of the segment parameter
        ``p = p1 + t (p2 - p1)``, ``0 <= t <= 1``, lying inside the polygon,
        in increasing order [-].

    Raises
    ------
    ValueError
        The polygon has fewer than three distinct vertices.

    Convention
    ----------
    Inside is decided by even-odd parity at each sub-interval's midpoint, so a
    non-convex polygon, such as a limiter with a divertor recess, yields
    several intervals. A segment grazing a vertex or running along an edge
    splits there with zero-length pieces dropped.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The polygon is a single loop; holes are not supported.

    Provenance
    ----------
    .. [1] The even-odd rule for point-in-polygon, as
       :class:`matplotlib.path.Path` applies it.
    """
    from matplotlib.path import Path

    pr = np.asarray(polygon_r, dtype=float).reshape(-1)
    pz = np.asarray(polygon_z, dtype=float).reshape(-1)
    if pr.size != pz.size:
        raise ValueError("polygon_r and polygon_z must have the same length")
    if pr.size and (pr[0] != pr[-1] or pz[0] != pz[-1]):
        pr = np.append(pr, pr[0])
        pz = np.append(pz, pz[0])
    if np.unique(np.column_stack((pr, pz)), axis=0).shape[0] < 3:
        raise ValueError("a polygon needs at least three distinct vertices")
    a = np.asarray(p1, dtype=float)
    d = np.asarray(p2, dtype=float) - a
    e0 = np.column_stack((pr[:-1], pz[:-1]))
    e = np.column_stack((pr[1:], pz[1:])) - e0
    denom = d[0] * e[:, 1] - d[1] * e[:, 0]
    rel = e0 - a
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rel[:, 0] * e[:, 1] - rel[:, 1] * e[:, 0]) / denom
        u = (rel[:, 0] * d[1] - rel[:, 1] * d[0]) / denom
    hit = (denom != 0) & (u >= 0.0) & (u <= 1.0) & (t > 0.0) & (t < 1.0)
    cuts = np.unique(np.concatenate(([0.0, 1.0], t[hit])))
    mids = 0.5 * (cuts[:-1] + cuts[1:])
    inside = Path(np.column_stack((pr, pz))).contains_points(a + mids[:, None] * d)
    intervals: list[tuple[float, float]] = []
    for lo, hi, keep in zip(cuts[:-1], cuts[1:], inside):
        if not keep or hi <= lo:
            continue
        if intervals and intervals[-1][1] == lo:
            intervals[-1] = (intervals[-1][0], float(hi))
        else:
            intervals.append((float(lo), float(hi)))
    return intervals


def _cell_edges(axis: np.ndarray) -> np.ndarray:
    if axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise ValueError("a grid axis must have at least two strictly increasing values")
    mid = 0.5 * (axis[1:] + axis[:-1])
    return np.concatenate(([axis[0] - (mid[0] - axis[0])], mid, [axis[-1] + (axis[-1] - mid[-1])]))


def _cell_index(edges: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cell of each coordinate: ``[edge_k, edge_k+1)``, the last cell closed.

    A piece lying exactly on an interior edge belongs to the cell above it,
    and one on either outer edge to the outermost cell, so a chord along the
    grid's bottom and one along its top are counted alike.
    """
    index = np.searchsorted(edges, x, side="right") - 1
    return np.where(x == edges[-1], edges.size - 2, index)


def build_line_integral_operator(r, z, sightlines: Sightlines, *, domain=None):
    """The path-length matrix of chords through the cells of an ``(R, Z)`` grid.

    Parameters
    ----------
    r : array_like
        Major-radius grid axis, the cell centres, strictly increasing [m].
    z : array_like
        Height grid axis, the cell centres, strictly increasing [m].
    sightlines : Sightlines
        The chords, one row of the operator each [-].
    domain : tuple of array_like, optional
        ``(R, Z)`` vertices of a closed polygon, normally the limiter; each
        chord counts only where it is inside it [m].

    Returns
    -------
    scipy.sparse.csr_matrix
        ``G`` shaped ``(len(sightlines), len(r) * len(z))``; ``G @
        field.reshape(-1)`` is the line integral of a field indexed ``(R, Z)``
        [m].

    Raises
    ------
    ValueError
        A grid axis is not strictly increasing or has fewer than two values,
        or the domain polygon is degenerate.

    Convention
    ----------
    Nodes are cell centres, with edges halfway between them and half a spacing
    beyond the outermost node, and cells are flattened in C order over
    ``(R, Z)``. A chord running exactly along an interior edge is assigned to
    the cell above it, and along an outer edge to the outermost cell. Each entry is the exact length of the chord inside that cell,
    so a row sums to the chord's length inside the gridded rectangle and the
    domain. Chords are straight in the ``(R, Z)`` plane; their toroidal angle
    does not enter the operator.

    Applicability
    -------------
    Machine-independent.

    Processing steps
    ----------------
    1. Clip each chord to the domain polygon, if one is given.
    2. For each inside interval, collect the parameters at which it crosses a
       cell edge in either direction.
    3. Assign each piece between consecutive crossings to the cell containing
       its midpoint, with the piece's length as the entry.

    Limitations
    -----------
    A pencil beam: no finite aperture, no viewing cone, no etendue weighting
    and no reflection. Deterministic but not cached; build it once per grid
    and chord set and reuse it.

    Provenance
    ----------
    .. [1] Siddon, R. L., Med. Phys. 12, 252 (1985).
    """
    from scipy.sparse import csr_matrix

    r = np.asarray(r, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    r_edges, z_edges = _cell_edges(r), _cell_edges(z)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for i in range(len(sightlines)):
        a = np.array([sightlines.r1[i], sightlines.z1[i]])
        b = np.array([sightlines.r2[i], sightlines.z2[i]])
        d = b - a
        length = float(np.hypot(*d))
        if length == 0.0:
            continue
        if domain is None:
            intervals = [(0.0, 1.0)]
        else:
            intervals = clip_segment_to_polygon(a, b, domain[0], domain[1])
        for lo, hi in intervals:
            cuts = [np.array([lo, hi])]
            for k, edges in ((0, r_edges), (1, z_edges)):
                if d[k] != 0.0:
                    t = (edges - a[k]) / d[k]
                    cuts.append(t[(t > lo) & (t < hi)])
            t = np.unique(np.concatenate(cuts))
            mid = 0.5 * (t[1:] + t[:-1])
            seg = (t[1:] - t[:-1]) * length
            pr = a[0] + mid * d[0]
            pz = a[1] + mid * d[1]
            ir = _cell_index(r_edges, pr)
            iz = _cell_index(z_edges, pz)
            ok = (ir >= 0) & (ir < r.size) & (iz >= 0) & (iz < z.size) & (seg > 0.0)
            rows.append(np.full(int(ok.sum()), i))
            cols.append(ir[ok] * z.size + iz[ok])
            vals.append(seg[ok])
    shape = (len(sightlines), r.size * z.size)
    if not rows:
        return csr_matrix(shape)
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape)


def project_emissivity(emissivity, operator) -> np.ndarray:
    """Line integrals of one or many ``(R, Z)`` fields through a path-length operator.

    Parameters
    ----------
    emissivity : array_like
        Field indexed ``(..., R, Z)``; leading axes, such as time, are kept
        [any, per metre].
    operator : scipy.sparse matrix or ndarray
        From :func:`build_line_integral_operator` for the same grid [m].

    Returns
    -------
    numpy.ndarray
        Line integrals shaped ``(..., n_chords)`` [any].

    Raises
    ------
    ValueError
        The field's last two axes do not hold as many cells as the operator
        has columns.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Exactly linear: whatever the field's unit, the result is that unit times
    metres. No detector response, filter transmission or etendue is applied.
    """
    field = np.asarray(emissivity, dtype=float)
    if field.ndim < 2:
        raise ValueError("emissivity must be indexed (..., R, Z)")
    lead = field.shape[:-2]
    cells = field.shape[-2] * field.shape[-1]
    if cells != operator.shape[1]:
        raise ValueError(f"the field has {cells} cells but the operator {operator.shape[1]} columns")
    flat = field.reshape(-1, cells)
    return np.asarray(operator @ flat.T).T.reshape(*lead, operator.shape[0])

"""Relative heat-load proxies from field-line tracing products (issue #1099).

A field-line tracer answers *where does a line launched here go, and how far
does it travel before it hits something*. Turning that into a picture of
where the heat lands needs a model, and the model here is deliberately a
crude one: an upstream channel width in normalized flux, the projection of
the field onto the target, and a parallel-loss weight. It models no
cross-field transport, no radiation, no detachment, no sheath, and no
self-consistent scrape-off layer.

So **the result is relative and has no heat-flux units.** It says which part
of a target is hit harder than which other part, and how the targets compare
with each other. It does not say how many watts.

Nothing here reads a file. The arrays come from
:mod:`vaft.data.flare_products`, and the IDS stand-ins they are written into
are in :mod:`vaft.machine_mapping.field_line_topology`.

Notation
--------
A         : area of one target surface element                           [m^2]
psi_N     : normalized poloidal flux, 1 on the separatrix                  [-]
psi_N,min : the deepest flux surface a traced line reached                 [-]
lambda_psi: upstream channel width, in psi_N                               [-]
alpha     : angle between the field line and the target surface          [deg]
L_c       : connection length of a field line, reduced over its two
            directions by the model                                        [m]
w_psi     : upstream channel weight exp(-max(psi_N,min - psi_sep, 0)/lambda_psi)  [-]
w_L       : parallel-loss weight, inverse or exponential in L_c/L_0        [-]
q         : the proxy, w_psi |sin alpha| w_L                               [-]

Conventions
-----------
**The proxy is a surface density, and the cell areas are what integrates
it.** ``q`` stands in for the heat flux arriving per unit target area, so
the incident total on a target is ``sum(q A)`` -- and the areas enter there
and nowhere else. The donor implementation instead multiplied ``q`` by
``1/A``, a "hit density" borrowed from FLARE's own strike-density task. That factor belongs there and not here: a strike density counts lines
launched elsewhere and landing on a cell, whereas a footprint grid launches
exactly one line *from* every node, so ``1/A`` carries no strike information
at all. It is the Jacobian of the launch lattice -- on a grid uniform in
toroidal angle and distance along the target it is exactly proportional to
``1/R`` -- and carrying it makes the per-target total proportional to the
node count the caller chose. See :func:`footprint_incident_total`.

**A missing incidence column is not an incidence of ninety degrees.** FLARE
writes ``alphaS`` only when the tracing task was asked for it, and the
committed reference footprint does not have it. Omitting the factor is a
legitimate choice; making it silently is not, so
:func:`footprint_heat_load_proxy` takes the angles as a required argument
whose ``None`` is the explicit opt-out, and records which it was.

**A tracer reports two directions, and reducing them is part of the model.**
The shortest connection length and the total are different quantities, and
the donor used the shortest while calling the result ``Lc``. So
:func:`footprint_heat_load_proxy` takes both directions and reduces them
itself, by the rule :attr:`FootprintProxyModel.connection_length_reduction`
names -- a caller cannot hand it an already-reduced array and have the
wrong reduction labelled with a model that did not produce it. The flux is
always the *deepest* incursion, the smaller of the two, because that is the
surface the upstream channel weight is asking about; the model's
``describe()`` says so, so the choice reaches the IDS provenance too.

Provenance
----------
.. [legacy] ``flare_footprint_analysis.py``, the donor divertor-footprint
   analysis this layer replaces: ``footprint_cell_areas``,
   ``upstream_psi_weight``, ``incidence_factor``,
   ``connection_length_weight``, ``compute_heat_flux_proxy``.
.. [flare-area] ``moose/src/fortran/geometry/hypermesh3d.f90:563``
   (``tpzmesh3d_cell_area``), the quadrilateral area FLARE reports in its own
   strike-density product and which :func:`toroidal_surface_cell_areas`
   reproduces.
.. [flare-alpha] ``FLARE/src/fortran/tasks/fieldline_connection.f90:158``:
   ``alphaS = asin(|b.n| / |b|)`` in degrees, so ``|sin alpha|`` is the
   projection exactly rather than by approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

__all__ = [
    "CONNECTION_LENGTH_FORMS",
    "CONNECTION_LENGTH_REDUCTIONS",
    "FOOTPRINT_PROXY_MODELS",
    "FootprintProxy",
    "FootprintProxyModel",
    "connection_length_weight",
    "footprint_heat_load_proxy",
    "footprint_incident_total",
    "footprint_proxy_model",
    "incidence_factor",
    "reduce_traced_directions",
    "target_incident_fractions",
    "toroidal_surface_cell_areas",
    "toroidal_surface_node_areas",
    "upstream_flux_weight",
]

#: The parallel-loss weights :func:`connection_length_weight` implements.
CONNECTION_LENGTH_FORMS = ("inverse", "exponential", "none")

#: How a model reduces a field line's two traced directions to one length.
#: ``"shortest"`` is the nearer wall, ``"total"`` the whole line from wall to
#: wall. They are different quantities and weight the map differently, which
#: is why no default picks between them.
CONNECTION_LENGTH_REDUCTIONS = ("shortest", "total")


def _cartesian(r, z, phi_deg) -> np.ndarray:
    """``(..., 3)`` Cartesian positions from cylindrical node arrays."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    if not (r.shape == z.shape == phi.shape):
        raise ValueError(
            f"r, z and phi_deg must share a shape; got {r.shape}, {z.shape} "
            f"and {phi.shape}"
        )
    if r.ndim != 2 or min(r.shape) < 2:
        raise ValueError(
            f"a surface mesh needs at least two nodes on each axis; got {r.shape}"
        )
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def toroidal_surface_cell_areas(r, z, phi_deg) -> np.ndarray:
    """Area of every quadrilateral cell of a surface mesh in ``(R, z, phi)``.

    Parameters
    ----------
    r : array_like
        Node major radius, shape ``(n_v, n_u)`` [m].
    z : array_like
        Node height, same shape [m].
    phi_deg : array_like
        Node toroidal angle, same shape [deg].

    Returns
    -------
    ndarray
        Cell areas, shape ``(n_v - 1, n_u - 1)`` [m^2].

    Raises
    ------
    ValueError
        The three arrays disagree in shape, or the mesh is not two
        dimensional with at least two nodes on each axis.

    Processing steps
    ----------------
    The nodes are placed in Cartesian space, and each cell is split along the
    diagonal from ``[j, i+1]`` to ``[j+1, i]`` into two triangles whose areas
    are summed. No small-angle expansion is taken, so the result is exact for
    a flat quadrilateral. A cell that is not flat has two diagonals that give
    slightly different answers, and this is the one FLARE splits along --
    which is why the two agree to parts in a trillion rather than merely
    closely.

    Convention
    ----------
    Node order is ``[i_v, i_u]``, the order FLARE's own mesh files lay their
    values out in -- see :attr:`vaft.data.flare_products.FlareMesh.node_shape`.
    Cell ``[j, i]`` is bounded by nodes ``[j, i]``, ``[j, i+1]``,
    ``[j+1, i+1]`` and ``[j+1, i]``.

    Applicability
    -------------
    Machine-independent. Any surface described by node positions in
    cylindrical coordinates; nothing here is specific to a divertor target.

    Provenance
    ----------
    .. [1] ``moose/src/fortran/geometry/hypermesh3d.f90:563``, the same
       two-triangle construction FLARE uses for the ``area`` column of its
       strike-density product.
    """
    x = _cartesian(r, z, phi_deg)
    corner_00, corner_01 = x[:-1, :-1], x[:-1, 1:]
    corner_11, corner_10 = x[1:, 1:], x[1:, :-1]
    first = np.linalg.norm(np.cross(corner_01 - corner_00, corner_10 - corner_00), axis=-1)
    second = np.linalg.norm(np.cross(corner_01 - corner_11, corner_10 - corner_11), axis=-1)
    return (first + second) / 2.0


def toroidal_surface_node_areas(r, z, phi_deg) -> np.ndarray:
    """Area each node of a surface mesh represents, as a dual cell.

    Parameters
    ----------
    r : array_like
        Node major radius, shape ``(n_v, n_u)`` [m].
    z : array_like
        Node height, same shape [m].
    phi_deg : array_like
        Node toroidal angle, same shape [deg].

    Returns
    -------
    ndarray
        Node areas, shape ``(n_v, n_u)``, summing to the total surface area
        [m^2].

    Raises
    ------
    ValueError
        As :func:`toroidal_surface_cell_areas`.

    Processing steps
    ----------------
    Each cell's area is split equally between its four corners. A corner node
    therefore receives a quarter of one cell, an edge node a quarter of two,
    and an interior node a quarter of four -- so the node areas sum to the
    surface area exactly, which is what makes an integral over them the same
    integral whether the values sit on nodes or on cells.

    Convention
    ----------
    Node order is ``[i_v, i_u]``, as :func:`toroidal_surface_cell_areas`.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] The donor's ``footprint_cell_areas`` built the
       same quantity from midpoint axis widths and ``R dphi ds``, which
       assumes the second axis is parameterized by its own arc length. This
       form reads the arc from the node positions instead, so a target curve
       whose parameter is not arc length is measured rather than assumed.
    """
    cells = toroidal_surface_cell_areas(r, z, phi_deg)
    quarter = cells / 4.0
    nodes = np.zeros(np.asarray(r, dtype=float).shape)
    nodes[:-1, :-1] += quarter
    nodes[:-1, 1:] += quarter
    nodes[1:, :-1] += quarter
    nodes[1:, 1:] += quarter
    return nodes


def upstream_flux_weight(psi_norm, *, separatrix_flux: float, decay_width: float,
                         clip_inside: bool = True) -> np.ndarray:
    """Exponential upstream heat-channel weight in normalized flux.

    Parameters
    ----------
    psi_norm : array_like
        The flux label each field line is weighted by -- for a connection
        product, the deepest normalized flux it reached [-].
    separatrix_flux : float
        Where the channel starts, normally 1 [-].
    decay_width : float
        e-folding width of the channel, in the same normalized flux; must be
        positive [-].
    clip_inside : bool, optional
        Whether a line that reached inside ``separatrix_flux`` is given the
        full weight rather than one growing exponentially. ``True`` clips
        [n/a].

    Returns
    -------
    ndarray
        Weights in ``(0, 1]`` when clipped, unbounded above otherwise [-].

    Raises
    ------
    ValueError
        ``decay_width`` is not positive.

    Convention
    ----------
    ``psi_norm`` increases outward, so a line that stayed further out is
    weighted less. With ``clip_inside`` false the weight grows without bound
    inside the separatrix, which is the model being read outside the region
    it describes; it is offered because a caller comparing against the
    unclipped legacy form needs it, not because it is ever the better choice.

    Applicability
    -------------
    Machine-independent. The channel is an upstream scrape-off-layer width,
    so the weight is meaningful only for lines that reach the plasma edge.

    Provenance
    ----------
    .. [legacy] the donor's ``upstream_psi_weight``.
    """
    if not decay_width > 0.0:
        raise ValueError(f"decay_width must be positive, not {decay_width!r}")
    delta = np.asarray(psi_norm, dtype=float) - float(separatrix_flux)
    if clip_inside:
        delta = np.maximum(delta, 0.0)
    return np.exp(-delta / float(decay_width))


def incidence_factor(angle_deg) -> np.ndarray:
    """The projection of a field line onto the surface it strikes.

    Parameters
    ----------
    angle_deg : array_like
        Angle between the field line and the target surface, as FLARE's
        ``alphaS`` reports it [deg].

    Returns
    -------
    ndarray
        ``|sin alpha|``, in ``[0, 1]`` [-].

    Convention
    ----------
    The angle is measured from the *surface*, not from its normal, so a
    grazing line has a small angle and a small factor. That is FLARE's own
    definition: ``alphaS = asin(|b.n| / |b|)``, which makes ``|sin alpha|``
    the projection exactly rather than an approximation of it. A line that
    never struck the boundary is written as zero by the tracer and so
    receives zero weight, which is the right answer for a different reason --
    it deposits nothing because it does not arrive.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] ``FLARE/src/fortran/tasks/fieldline_connection.f90:158``.
    .. [legacy] the donor's ``incidence_factor``, which returned
       ones when the column was absent; see
       :func:`footprint_heat_load_proxy`.
    """
    alpha = np.deg2rad(np.abs(np.asarray(angle_deg, dtype=float)))
    return np.clip(np.abs(np.sin(alpha)), 0.0, 1.0)


def connection_length_weight(length, *, scale: float, form: str) -> np.ndarray:
    """Parallel-loss weight from a field line's connection length.

    Parameters
    ----------
    length : array_like
        Connection length per line; negative entries are treated as zero,
        and non-finite ones are given zero weight [m].
    scale : float
        The length the weight is measured against; must be positive [m].
    form : str
        ``"inverse"`` for ``1 / (1 + L / L_0)``, ``"exponential"`` for
        ``exp(-L / L_0)``, ``"none"`` for a flat one [n/a].

    Returns
    -------
    ndarray
        Weights in ``[0, 1]`` [-].

    Raises
    ------
    ValueError
        ``scale`` is not positive, or ``form`` is not one of
        :data:`CONNECTION_LENGTH_FORMS`.

    Convention
    ----------
    A longer line is weighted less: the length stands in for how much of the
    flux tube's content was lost on the way. The two forms are empirical and
    disagree by construction -- the inverse one keeps a long line at a small
    but finite weight, the exponential one takes it to zero -- so which is
    used is part of the model, never a default.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] the donor's ``connection_length_weight``, whose
       ``"exp_loss"`` is this ``"exponential"``.
    """
    if form not in CONNECTION_LENGTH_FORMS:
        raise ValueError(
            f"form must be one of {list(CONNECTION_LENGTH_FORMS)}, not {form!r}"
        )
    values = np.asarray(length, dtype=float)
    if form == "none":
        return np.ones(values.shape)
    if not scale > 0.0:
        raise ValueError(f"scale must be positive, not {scale!r}")
    finite = np.isfinite(values)
    positive = np.maximum(np.where(finite, values, 0.0), 0.0)
    if form == "inverse":
        weight = 1.0 / (1.0 + positive / float(scale))
    else:
        weight = np.exp(-positive / float(scale))
    return np.where(finite, weight, 0.0)


@dataclass(frozen=True)
class FootprintProxyModel:
    """The coefficients one relative footprint proxy is built from.

    Every field is a modelling choice with no defensible default, which is
    why this is a named preset rather than a set of keyword defaults: see
    :data:`FOOTPRINT_PROXY_MODELS` and :func:`footprint_proxy_model`.
    """

    name: str
    #: Upstream channel width in normalized flux [-].
    decay_width: float
    #: Where that channel starts, normally 1 [-].
    separatrix_flux: float
    #: The length the parallel-loss weight is measured against [m].
    connection_length_scale: float
    #: One of :data:`CONNECTION_LENGTH_FORMS`.
    connection_length_form: str
    #: One of :data:`CONNECTION_LENGTH_REDUCTIONS`: which of a line's two
    #: traced directions the weight is measured on.
    connection_length_reduction: str
    #: Whether a line reaching inside the separatrix keeps the full weight.
    clip_inside: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a proxy model must be named; the name is its provenance")
        if not self.decay_width > 0.0:
            raise ValueError(f"decay_width must be positive, not {self.decay_width!r}")
        if self.connection_length_form not in CONNECTION_LENGTH_FORMS:
            raise ValueError(
                f"connection_length_form must be one of "
                f"{list(CONNECTION_LENGTH_FORMS)}, not {self.connection_length_form!r}"
            )
        if (self.connection_length_form != "none"
                and not self.connection_length_scale > 0.0):
            raise ValueError(
                f"connection_length_scale must be positive, not "
                f"{self.connection_length_scale!r}"
            )
        if self.connection_length_reduction not in CONNECTION_LENGTH_REDUCTIONS:
            raise ValueError(
                f"connection_length_reduction must be one of "
                f"{list(CONNECTION_LENGTH_REDUCTIONS)}, not "
                f"{self.connection_length_reduction!r}"
            )

    def describe(self) -> str:
        """The model as one line, for a provenance record."""
        return (
            f"{self.name}: lambda_psi={self.decay_width!r}, "
            f"psi_sep={self.separatrix_flux!r}, "
            f"L_0={self.connection_length_scale!r} m, "
            f"form={self.connection_length_form}, "
            f"L_c={self.connection_length_reduction} of the two directions, "
            "psi_N=deepest of the two directions, "
            f"clip_inside={self.clip_inside}"
        )


#: The named footprint proxy models. Both carry the coefficients the donor
#: divertor-footprint scans were run with -- a 0.02-wide channel in psi_N, a
#: 50 m parallel-loss scale, and the shortest of a line's two directions --
#: and differ only in the loss form, which the donor offered as a switch.
#: Neither is a default: a caller names one.
FOOTPRINT_PROXY_MODELS: dict[str, FootprintProxyModel] = {
    model.name: model
    for model in (
        FootprintProxyModel(
            name="sol_channel_inverse_loss",
            decay_width=0.02,
            separatrix_flux=1.0,
            connection_length_scale=50.0,
            connection_length_form="inverse",
            connection_length_reduction="shortest",
            clip_inside=True,
        ),
        FootprintProxyModel(
            name="sol_channel_exponential_loss",
            decay_width=0.02,
            separatrix_flux=1.0,
            connection_length_scale=50.0,
            connection_length_form="exponential",
            connection_length_reduction="shortest",
            clip_inside=True,
        ),
    )
}


def footprint_proxy_model(name: str) -> FootprintProxyModel:
    """One named footprint proxy model.

    Parameters
    ----------
    name : str
        A key of :data:`FOOTPRINT_PROXY_MODELS` [n/a].

    Returns
    -------
    FootprintProxyModel
        The preset, which is frozen and safe to share [n/a].

    Raises
    ------
    KeyError
        No model carries that name; the message lists the ones that do.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] the donor's ``FootprintProxyConfig``, whose
       field defaults these presets carry as named values.
    """
    try:
        return FOOTPRINT_PROXY_MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown footprint proxy model {name!r}; "
            f"registered: {sorted(FOOTPRINT_PROXY_MODELS)}"
        ) from None


@dataclass(frozen=True)
class FootprintProxy:
    """One relative footprint proxy and the factors it was built from.

    ``density`` is the proxy itself, a relative surface density with no
    heat-flux units; ``relative`` is the same field scaled so its peak is
    one, which is what a colour map wants. The three factors are kept so a
    caller can see which one shaped the map.
    """

    density: np.ndarray
    area: np.ndarray
    flux_weight: np.ndarray
    incidence: np.ndarray
    connection_weight: np.ndarray
    model: FootprintProxyModel
    #: ``False`` when the caller opted out of the incidence factor.
    incidence_applied: bool

    @property
    def relative(self) -> np.ndarray:
        """The proxy scaled to a peak of one; a map with no peak stays zero [-]."""
        finite = np.isfinite(self.density)
        peak = float(np.max(self.density[finite])) if finite.any() else 0.0
        return self.density / peak if peak > 0.0 else np.zeros(self.density.shape)

    def describe(self) -> str:
        """Model and incidence choice as one line, for a provenance record."""
        incidence = "|sin(alphaS)|" if self.incidence_applied else "omitted by the caller"
        return f"{self.model.describe()}, incidence={incidence}"


def reduce_traced_directions(backward, forward, *, how: str) -> np.ndarray:
    """Reduce a quantity traced in both directions to one value per line.

    Parameters
    ----------
    backward : array_like
        The value the tracer reported going one way [any].
    forward : array_like
        The value it reported going the other [any].
    how : str
        ``"shortest"`` takes the smaller magnitude, ``"total"`` the sum,
        ``"deepest"`` the smaller signed value [n/a].

    Returns
    -------
    ndarray
        One value per line, in the inputs' own unit [any].

    Raises
    ------
    ValueError
        The arrays disagree in shape, or ``how`` is none of the three.

    Convention
    ----------
    ``"shortest"`` compares magnitudes, because a tracer may report a
    backward length as negative; ``"deepest"`` compares signed values,
    because a flux label that ran further inward is genuinely smaller.
    Non-finite entries propagate rather than being skipped: a direction the
    tracer could not follow leaves the reduction undefined, and that is the
    honest answer for that line.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] ``FLARE/src/fortran/tasks/fieldline_connection.f90:90-96``, whose
       own ``Lc = Lc_bwd + Lc_fwd``, ``Lcs = min(Lc_bwd, Lc_fwd)`` and
       ``minPsiN = min(minPsiN_bwd, minPsiN_fwd)`` are these three
       reductions. FLARE's arclengths are non-negative, so its plain ``min``
       and the magnitude comparison here agree on its own output.
    """
    first = np.asarray(backward, dtype=float)
    second = np.asarray(forward, dtype=float)
    if first.shape != second.shape:
        raise ValueError(
            f"the two directions must share a shape; got {first.shape} and "
            f"{second.shape}"
        )
    if how == "shortest":
        return np.where(np.abs(first) <= np.abs(second), first, second)
    if how == "total":
        return first + second
    if how == "deepest":
        return np.minimum(first, second)
    raise ValueError(
        f'how must be "shortest", "total" or "deepest", not {how!r}'
    )


def footprint_heat_load_proxy(
    *,
    min_psi_norm_backward,
    min_psi_norm_forward,
    connection_length_backward,
    connection_length_forward,
    area,
    incidence_angle_deg,
    model: FootprintProxyModel,
) -> FootprintProxy:
    """A relative heat-load proxy over a divertor target's footprint grid.

    Parameters
    ----------
    min_psi_norm_backward : array_like
        Deepest normalized flux each line reached tracing one way [-].
    min_psi_norm_forward : array_like
        The same tracing the other way; the two are reduced to the deeper
        incursion, which is the surface the channel weight asks about [-].
    connection_length_backward : array_like
        Length each line ran before it stopped, one way [m].
    connection_length_forward : array_like
        The same the other way; the two are reduced by
        :attr:`FootprintProxyModel.connection_length_reduction`, so the
        shortest and the total cannot be confused for one another [m].
    area : array_like
        Surface area each sample represents, from
        :func:`toroidal_surface_node_areas` or
        :func:`toroidal_surface_cell_areas`; carried through so an integral
        needs nothing else [m^2].
    incidence_angle_deg : array_like or None
        Angle between each line and the target, as FLARE's ``alphaS``
        reports it. ``None`` is the explicit opt-out, recorded on the result
        -- there is no implicit one [deg].
    model : FootprintProxyModel
        A named preset; see :func:`footprint_proxy_model`. Required, because
        every coefficient in it is a modelling choice [n/a].

    Returns
    -------
    FootprintProxy
        The proxy and its three factors, all on the caller's grid [-].

    Raises
    ------
    ValueError
        The arrays disagree in shape.

    Processing steps
    ----------------
    1. Reduce each line's two traced directions: the flux to the deeper
       incursion, the length by the model's own rule.
    2. ``w_psi`` from :func:`upstream_flux_weight`.
    3. ``|sin alpha|`` from :func:`incidence_factor`, or ones when the
       caller opted out -- and the opt-out is recorded either way.
    4. ``w_L`` from :func:`connection_length_weight`.
    5. Their product, which is the proxy.

    Output semantics
    ----------------
    A relative surface density, in no unit. Only ratios within one run and
    between targets of the same run mean anything; two runs are comparable
    only if they used the same model and the same tracing limits.

    Convention
    ----------
    The proxy is a density and ``area`` is only its measure -- it is not a
    factor. Multiplying by ``1/area``, as the donor did, imposes the launch
    lattice's own Jacobian on the map and makes the per-target total scale
    with the node count; see this module's ``Conventions``.

    Both directions are required rather than one reduced array, because the
    reduction is part of the model and a pre-reduced input would let the
    wrong one be labelled with a model that did not produce it.

    Assumptions
    -----------
    One traced line per grid sample, launched from the target -- which is how
    FLARE's ``equi2d_footprint_grid`` builds a footprint grid. A product
    whose lines were launched upstream and counted where they land is a
    strike density and is a different quantity.

    Applicability
    -------------
    Machine-independent. Any target whose footprint was sampled by field-line
    tracing; the model's coefficients are not.

    Limitations
    -----------
    No cross-field transport, radiation, detachment, sheath physics or
    self-consistent scrape-off layer. The result has no heat-flux units and
    must not be converted into any.

    Provenance
    ----------
    .. [legacy] the donor's ``compute_heat_flux_proxy``.
    """
    cells = np.asarray(area, dtype=float)
    shapes = {
        "min_psi_norm_backward": np.asarray(min_psi_norm_backward, dtype=float).shape,
        "min_psi_norm_forward": np.asarray(min_psi_norm_forward, dtype=float).shape,
        "connection_length_backward":
            np.asarray(connection_length_backward, dtype=float).shape,
        "connection_length_forward":
            np.asarray(connection_length_forward, dtype=float).shape,
        "area": cells.shape,
    }
    if incidence_angle_deg is not None:
        shapes["incidence_angle_deg"] = np.asarray(incidence_angle_deg, dtype=float).shape
    if len(set(shapes.values())) != 1:
        raise ValueError(
            "every input must be on the same grid; got "
            + ", ".join(f"{name} {shape}" for name, shape in shapes.items())
        )

    flux = reduce_traced_directions(
        min_psi_norm_backward, min_psi_norm_forward, how="deepest"
    )
    length = reduce_traced_directions(
        connection_length_backward, connection_length_forward,
        how=model.connection_length_reduction,
    )
    flux_weight = upstream_flux_weight(
        flux,
        separatrix_flux=model.separatrix_flux,
        decay_width=model.decay_width,
        clip_inside=model.clip_inside,
    )
    applied = incidence_angle_deg is not None
    incidence = (incidence_factor(incidence_angle_deg) if applied
                 else np.ones(flux.shape))
    connection = connection_length_weight(
        length,
        scale=model.connection_length_scale,
        form=model.connection_length_form,
    )
    return FootprintProxy(
        density=flux_weight * incidence * connection,
        area=cells,
        flux_weight=flux_weight,
        incidence=incidence,
        connection_weight=connection,
        model=model,
        incidence_applied=applied,
    )


def footprint_incident_total(proxy: FootprintProxy) -> float:
    """The proxy integrated over one target.

    Parameters
    ----------
    proxy : FootprintProxy
        As :func:`footprint_heat_load_proxy` returns it [-].

    Returns
    -------
    float
        ``sum(density * area)``, in the proxy's own relative scale [-].

    Processing steps
    ----------------
    Non-finite samples are skipped rather than propagated: a line the tracer
    could not follow contributes nothing, and one NaN would otherwise erase
    the target.

    Convention
    ----------
    This is a Riemann sum over the sample areas, so it converges as the grid
    is refined -- which is the property a fraction between targets needs, and
    the one the legacy's extra ``1/area`` factor destroyed. With it the sum
    reduces to ``sum(w)``, which grows roughly with the number of samples the
    caller chose to trace rather than settling on the target's own value.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] the donor's ``summarize_heat_flux_proxy``, whose
       ``integral`` column is this same sum -- taken over a density that
       carried an extra ``1/area`` factor, which is what reduced it to the
       bare sum of the weights.
    """
    values = proxy.density * proxy.area
    return float(np.nansum(np.where(np.isfinite(values), values, 0.0)))


def target_incident_fractions(totals: Mapping[str, float]) -> dict[str, float]:
    """Each target's share of the proxy incident on all of them.

    Parameters
    ----------
    totals : mapping of str to float
        Per-target integrals, from :func:`footprint_incident_total`; each
        must be finite and not negative [-].

    Returns
    -------
    dict of str to float
        The same keys, each divided by the sum over all of them, so the
        values add to one [-].

    Raises
    ------
    ValueError
        ``totals`` is empty, an entry is negative or not finite, or they sum
        to zero -- in which case there is no share to take, and returning
        equal shares would invent one.

    Convention
    ----------
    The denominator is the total over the targets *in this mapping*, so the
    fraction says what it says only if that set is every target the run
    covered. A fraction over a subset is a fraction of the subset.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [owner] The IMAS stand-in decided on vaft#1099 (2026-09-21): the proxy
       integrated over each target, divided by the total over all targets,
       written to ``divertors.divertor[].target[].power_incident_fraction``
       because it is dimensionless and so is the proxy.
    """
    if not totals:
        raise ValueError("no targets to take a share of")
    for key, value in totals.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"target {key!r} has an incident total of {value!r}; a share "
                "is defined only for finite, non-negative totals"
            )
    total = float(sum(float(value) for value in totals.values()))
    if total <= 0.0:
        raise ValueError(
            "every target's incident total is zero, so there is no share to "
            "take; equal shares would be an invention, not a result"
        )
    return {key: float(value) / total for key, value in totals.items()}

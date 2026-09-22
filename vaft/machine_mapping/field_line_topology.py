"""Field-line tracing results in the IDS the owner chose for them (issue #1099).

The Data Dictionary has no home for field-line tracing. Searching 3.41 for
one turned up three near misses and no match, so rather than force the
products into a structure whose meaning they do not share, the owner chose
two **stand-ins** and left the rest native (2026-09-21):

======================================= ==================================================
product                                 destination
======================================= ==================================================
connection-length map in a fixed-phi    ``plasma_initiation.b_field_lines``, one entry per
``(R, z)`` plane                        ``(time, plane)`` -- see :func:`write_b_field_lines`
footprint summary per divertor target   ``divertors.divertor[].target[].``
                                        ``power_incident_fraction`` -- see
                                        :func:`write_divertor_incident_fractions`
Poincare punctures, ``minPsiN``, the     nothing: they stay native, and
2-D footprint map                       :mod:`vaft.data.flare_products` reads them
======================================= ==================================================

Neither stand-in is a perfect fit, and both writers refuse what does not fit
rather than approximate it.

:func:`write_b_field_lines` is written against the *quantity*, not against
FLARE: :func:`vaft.process.equilibrium.connection_length_map`, the VEST
startup tracer (issue #783), produces the same thing on the same kind of grid
and writes through it unchanged. That is the structure's own subject -- its
siblings are Townsend fields and prefill pressure -- so a breakdown study is
the case it fits best, and an RMP footprint is the one it is being borrowed
for.

``plasma_initiation.b_field_lines``
    Its siblings (``e_field_townsend``, a prefill ``pressure``) frame the
    structure as breakdown analysis, and its starting positions carry no
    toroidal angle. So **one toroidal plane per entry**: a second plane at
    the same time is refused, because the array of structures is indexed by
    time and two entries sharing a time would make the time base a lie. The
    angle, the forward/backward split and the turn counts have no slot and
    are recorded in ``code.parameters`` together with the fact that they are
    lost.

``power_incident_fraction``
    Dimensionless, which is what the proxy is. Three neighbouring leaves are
    **deliberately not written**: ``power_flux_peak`` is in W/m^2 and the
    proxy has no heat-flux units; ``wetted_area`` is defined by the Data
    Dictionary through lambda_q and FLARE's footprint uses a tenth of the
    peak instead; and ``tilt_angle_pol`` is poloidal-only where the
    incidence angle the proxy uses is three dimensional.

Provenance
----------
.. [owner] The IMAS stand-ins, decided 2026-09-21 on vaft#1099.
.. [dd] ``plasma_initiation.b_field_lines[:].grid_type`` takes the standard
   ``poloidal_plane_coordinates_identifier``; index 1 is
   ``rectangular``, "Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2)".
"""

from __future__ import annotations

from typing import Any, Mapping, Optional
from xml.sax.saxutils import quoteattr

import numpy as np
from omas import ODS

from vaft.ods_access import path_count, path_value

__all__ = [
    "RECTANGULAR_GRID_TYPE",
    "b_field_lines_from_flare_connection",
    "divertor_incident_fractions_from_flare_footprints",
    "write_b_field_lines",
    "write_divertor_incident_fractions",
]

#: ``poloidal_plane_coordinates_identifier`` index 1: Cylindrical R, z ala
#: eqdsk, with R on ``dim1`` and z on ``dim2``.
RECTANGULAR_GRID_TYPE = 1

#: FLARE's status code for a line that ended on the wall
#: (``FLARE/src/fortran/analysis/fieldline.f90:25``).  A line is *open* when
#: it reached the wall in at least one direction.
FLARE_INTERSECT_BOUNDARY = -1001

#: FLARE's status code for a trace that left the field's domain:
#: ``fieldline.f90:41`` sets ``edom = DOMAIN_ERROR`` and returns it from the
#: derivative when ``bfield%out_of_bounds`` (``:332``, ``:364``), and
#: ``moose_error`` defines ``DOMAIN_ERROR = 1`` (``core/error.f90:9``).
#: ``FIELDLINE_SUCCESS`` is false for it, so such a line is a *failed* trace
#: whose reported length is only what the integrator had accumulated when it
#: left -- which is why whether it counts as open is a policy and not a fact.
FLARE_DOMAIN_ERROR = 1

#: Every status :func:`b_field_lines_from_flare_connection` can read: the
#: four terminal codes of ``fieldline.f90:25``, ``moose_error``'s
#: ``SUCCESS = 0``, and ``DOMAIN_ERROR``.  A code outside this set is
#: refused rather than counted as "did not reach the wall" -- guessing at
#: one would put a silent share into a number that reads as a measurement.
FLARE_TRACE_STATUS = {
    0: "success",
    FLARE_DOMAIN_ERROR: "left the field's domain",
    -1001: "reached the wall",
    -1002: "stopped at the arclength limit",
    -1003: "stopped at the poloidal-turn limit",
    -1004: "stopped below the minimum psi_N",
}

#: What :func:`b_field_lines_from_flare_connection` may do with a line that
#: left the field's domain.  ``"refuse"`` is the only default: the trace
#: failed, so whether the line is open is unknown, and calling it either way
#: is an owner's policy rather than a reading of the data.
DOMAIN_EXIT_POLICIES = ("refuse", "open", "closed")


def _append_code_parameters(ods: ODS, ids: str, fragment_xml: str, *,
                            code_name: str, repository: str = "") -> None:
    """Append one fragment to ``<ids>.code.parameters``.

    ``code.parameters`` is one IDS-global string while these writers add an
    entry at a time, so this appends rather than overwrites -- otherwise the
    second plane would erase the first one's provenance, which is the only
    record of the angle it was traced at.
    """
    ods[f"{ids}.code.name"] = code_name
    if repository:
        ods[f"{ids}.code.repository"] = repository
    path = f"{ids}.code.parameters"
    existing = path_value(ods, path, None)
    if not existing:
        ods[path] = f"<parameters>{fragment_xml}</parameters>"
        return
    existing = str(existing).rstrip()
    if existing.endswith("</parameters>"):
        ods[path] = existing[: -len("</parameters>")] + fragment_xml + "</parameters>"
    else:
        ods[path] = existing + fragment_xml


def _plane_times(ods: ODS) -> list[float]:
    """The time each existing ``b_field_lines`` entry records."""
    return [
        float(path_value(ods, f"plasma_initiation.b_field_lines.{position}.time", np.nan))
        for position in range(path_count(ods, "plasma_initiation.b_field_lines"))
    ]


def _time_index(ods: ODS, ids: str, time: float) -> Optional[int]:
    """Where ``time`` already sits in ``<ids>.time``, or ``None`` if it is new.

    Reads without writing, so a caller can check what an instant would
    collide with before anything is committed.
    """
    times = np.atleast_1d(np.asarray(path_value(ods, f"{ids}.time", []), dtype=float))
    times = times[np.isfinite(times)] if times.size else times
    match = np.flatnonzero(times == float(time))
    return int(match[0]) if match.size else None


def _time_base(ods: ODS, ids: str, time: float) -> int:
    """The index of ``time`` in ``<ids>.time``, appending it if it is new.

    Keeps the IDS homogeneous in time, which is what lets a consumer read
    every target's trace on one axis.

    **A new instant is appended, never inserted.** Both structures this
    module writes are indexed by position -- an array of structures in one
    case, a signal's samples in the other -- so inserting an earlier time
    would have to shift everything already written, and the version that
    sorted the time base instead silently overwrote whatever sat at the index
    the new instant sorted into. An out-of-order write is refused rather than
    reordered: the caller knows its own time order and the IDS cannot.
    """
    ods[f"{ids}.ids_properties.homogeneous_time"] = 1
    existing = _time_index(ods, ids, time)
    if existing is not None:
        return existing
    times = np.atleast_1d(np.asarray(path_value(ods, f"{ids}.time", []), dtype=float))
    times = times[np.isfinite(times)] if times.size else times
    if times.size and float(time) < float(times[-1]):
        raise ValueError(
            f"{ids}.time already reaches {float(times[-1])!r} s and this is "
            f"{float(time)!r} s. Entries are appended in the order they are "
            "written, so a time base stays sorted only if its instants do; "
            "write them in increasing order."
        )
    ods[f"{ids}.time"] = np.concatenate([times, [float(time)]])
    return int(times.size)


# ---------------------------------------------------------------------------
# 1. The connection-length map
# ---------------------------------------------------------------------------


def write_b_field_lines(
    ods: ODS,
    *,
    grid_r,
    grid_z,
    starting_r,
    starting_z,
    lengths,
    open_fraction: Optional[float],
    time: float,
    phi_deg: float,
    provenance: Mapping[str, Any] | None = None,
    code_name: str = "FLARE",
    code_repository: str = "",
) -> int:
    """Write one traced ``(R, z)`` plane into ``plasma_initiation.b_field_lines``.

    ``starting_r``/``starting_z``/``lengths`` are one flat entry per traced
    line, in whatever order the tracer produced them; the positions are what
    pins that order, which is why they are written even though the Data
    Dictionary calls them redundant on a rectangular grid. They are not: the
    grid axes alone leave the flattening ambiguous, and taking the wrong one
    transposes the map.

    ``open_fraction`` may be ``None``, which leaves the leaf unwritten and
    records the omission; it is a required argument so that the omission is
    a decision rather than an oversight.

    Returns the index of the entry it wrote.

    Raises
    ------
    ValueError
        The line arrays disagree in length, the grid axes are not strictly
        monotonic, a starting position does not lie on the declared grid, two
        lines start from one node, a length is negative or infinite (``NaN``
        is legitimate -- it is a line the tracer could not follow), the open
        fraction is outside ``[0, 1]``, an entry already exists at this time
        -- the array of structures is indexed by time, so one plane per time
        is the whole contract -- an earlier instant is written after a later
        one, or ``plasma_initiation.time`` disagrees with the entries already
        there.
    """
    grid_r = np.asarray(grid_r, dtype=float).ravel()
    grid_z = np.asarray(grid_z, dtype=float).ravel()
    r = np.asarray(starting_r, dtype=float).ravel()
    z = np.asarray(starting_z, dtype=float).ravel()
    length = np.asarray(lengths, dtype=float).ravel()
    if not (r.size == z.size == length.size) or r.size == 0:
        raise ValueError(
            "starting_r, starting_z and lengths must be one non-empty entry "
            f"per line; got {r.size}, {z.size} and {length.size}"
        )
    for name, axis in (("grid_r", grid_r), ("grid_z", grid_z)):
        if axis.size < 2 or not np.all(np.diff(axis) > 0):
            raise ValueError(
                f"{name} must be a strictly increasing axis of at least two "
                f"nodes; got {axis.size} nodes"
            )
    if r.size != grid_r.size * grid_z.size:
        raise ValueError(
            f"{r.size} traced lines do not cover the declared "
            f"{grid_r.size} x {grid_z.size} grid; this writer maps a full "
            "rectangular plane, and a partial one has no grid to declare"
        )
    for name, values, axis in (("r", r, grid_r), ("z", z, grid_z)):
        if not np.isin(values, axis).all():
            raise ValueError(
                f"some starting {name} positions do not lie on grid_{name}; "
                "the grid declared and the lines traced must be the same plane"
            )
    # Covering the node count and lying on the axes still leaves room for two
    # lines on one node and a node with none, which reads as a complete plane
    # and draws as a hole beside a value that overwrote another.
    if len(set(zip(r.tolist(), z.tolist()))) != r.size:
        raise ValueError(
            "two or more lines start from the same grid node; the plane would "
            "have a hole where one of them should be and a silently "
            "overwritten value where they collide"
        )
    finite = np.isfinite(length)
    if np.any(length[finite] < 0.0) or np.any(np.isinf(length)):
        raise ValueError(
            "a connection length must be non-negative and finite, or NaN for "
            "a line the tracer could not follow; this one carries "
            f"{int(np.count_nonzero(length[finite] < 0.0))} negative and "
            f"{int(np.count_nonzero(np.isinf(length)))} infinite value(s). "
            "An infinite length is a line that never closed, which is a "
            "statement the tracer's own status codes make and this leaf "
            "cannot."
        )
    if open_fraction is not None:
        open_fraction = float(open_fraction)
        if not np.isfinite(open_fraction) or not 0.0 <= open_fraction <= 1.0:
            raise ValueError(
                f"open_fraction must be a fraction in [0, 1], not {open_fraction!r}"
            )

    # Every refusal comes before the first write, so a rejected plane leaves
    # the ODS exactly as it found it rather than half updated.
    recorded = _plane_times(ods)
    if any(existing == float(time) for existing in recorded):
        raise ValueError(
            f"plasma_initiation.b_field_lines already holds an entry at "
            f"t = {float(time)!r} s. The array of structures is indexed "
            "by time and carries no toroidal coordinate, so one plane per "
            "time is all it can say; write a second angle into its own "
            "ODS rather than duplicating the time."
        )
    if recorded and float(time) < max(recorded):
        raise ValueError(
            f"plasma_initiation.b_field_lines already reaches "
            f"{max(recorded)!r} s and this is {float(time)!r} s. Entries are "
            "appended in the order they are written, so a time base stays "
            "sorted only if its instants do; write them in increasing order."
        )

    # The entry index comes from the array of structures, never from
    # `plasma_initiation.time`: a time base laid out ahead of the entries
    # would otherwise send the first plane to an index the AOS does not have
    # yet, and OMAS raises on the gap rather than filling it.
    index = len(recorded)
    times = np.atleast_1d(np.asarray(
        path_value(ods, "plasma_initiation.time", []), dtype=float))
    times = times[np.isfinite(times)] if times.size else times
    if times.size > index or (times.size and not np.allclose(
            times[:len(recorded)], recorded[:times.size], equal_nan=True)):
        raise ValueError(
            f"plasma_initiation.time reads {[float(t) for t in times]} while "
            f"its b_field_lines entries are at {[float(t) for t in recorded]}. "
            "This writer keeps the "
            "two in step and owns the time base; it will not write into one "
            "laid out by something else."
        )
    ods["plasma_initiation.ids_properties.homogeneous_time"] = 1
    ods["plasma_initiation.time"] = np.asarray(recorded + [float(time)], dtype=float)
    entry = f"plasma_initiation.b_field_lines.{index}"
    ods[f"{entry}.time"] = float(time)
    ods[f"{entry}.grid.dim1"] = grid_r
    ods[f"{entry}.grid.dim2"] = grid_z
    ods[f"{entry}.grid_type.index"] = RECTANGULAR_GRID_TYPE
    ods[f"{entry}.grid_type.name"] = "rectangular"
    ods[f"{entry}.grid_type.description"] = (
        "Cylindrical R,Z ala eqdsk (R=dim1, Z=dim2), at a single toroidal "
        f"angle phi = {float(phi_deg)!r} deg."
    )
    ods[f"{entry}.starting_positions.r"] = r
    ods[f"{entry}.starting_positions.z"] = z
    ods[f"{entry}.lengths"] = length
    if open_fraction is not None:
        ods[f"{entry}.open_fraction"] = open_fraction

    details = dict(provenance or {})
    details.setdefault("lengths", "total: forward plus backward")
    if open_fraction is None:
        details["open_fraction"] = "not written: no status was supplied"
    attributes = "".join(
        f" {key}={quoteattr(str(value))}" for key, value in sorted(details.items())
    )
    _append_code_parameters(
        ods, "plasma_initiation",
        f'<field_line_plane index="{index}" time="{float(time)!r}"'
        f' phi_deg="{float(phi_deg)!r}"{attributes}'
        ' lost="the toroidal angle, the separate forward and backward lengths,'
        ' and the poloidal and toroidal turn counts have no slot in'
        ' b_field_lines; only the total length is mapped"/>',
        code_name=code_name, repository=code_repository,
    )
    return index


def b_field_lines_from_flare_connection(
    ods: ODS, product, *, time: float, domain_exit: str = "refuse",
    code_name: str = "FLARE", code_repository: str = "",
) -> int:
    """Map one FLARE connection-length product at a fixed angle into the IDS.

    ``product`` is a :class:`~vaft.data.flare_products.FlareProduct` read
    from a ``fieldline_connection`` run on an ``rmesh`` grid -- one poloidal
    plane. A swept mesh is refused: its lines start at different toroidal
    angles and ``starting_positions`` has nowhere to say so.

    ``domain_exit`` says what to do with a line whose trace left the field's
    domain (:data:`FLARE_DOMAIN_ERROR`). Such a trace *failed*: its reported
    length is only what the integrator had accumulated before it left, and
    whether the line would have reached the wall is unknown. So the default
    is ``"refuse"``, and ``"open"`` or ``"closed"`` are an explicit policy
    the caller takes responsibility for -- both are written into the
    provenance beside the fraction they produced.

    Raises
    ------
    ValueError
        The mesh is not a single poloidal plane; the product carries no
        ``ierr_bwd``/``ierr_fwd`` columns, so which lines reached the wall is
        not recorded and ``open_fraction`` would be a guess; a status code
        appears that :data:`FLARE_TRACE_STATUS` does not define, which means
        the same thing for the lines that carry it; ``domain_exit`` is not
        one of :data:`DOMAIN_EXIT_POLICIES`; or it is ``"refuse"`` and the
        product holds a domain exit. Re-run the tracing task with its
        ``ierr`` output enabled, choose a policy, or call
        :func:`write_b_field_lines` directly with an explicit
        ``open_fraction``.
    """
    if domain_exit not in DOMAIN_EXIT_POLICIES:
        raise ValueError(
            f"domain_exit must be one of {list(DOMAIN_EXIT_POLICIES)}, not "
            f"{domain_exit!r}"
        )
    mesh = product.mesh
    if mesh.toroidally_swept:
        raise ValueError(
            f"{product.source.name} sits on a {mesh.kind} mesh, which sweeps "
            "the toroidal angle. plasma_initiation.b_field_lines carries only "
            "(r, z) starting positions, so a swept map cannot be written "
            "without discarding the coordinate that carries its structure."
        )
    names = {column.name for column in product.columns}
    missing = {"ierr_bwd", "ierr_fwd"} - names
    if missing:
        raise ValueError(
            f"{product.source.name} has no {sorted(missing)} column, so which "
            "lines reached the wall is not recorded and open_fraction cannot "
            "be derived. Re-run fieldline_connection with ierr enabled, or "
            "call write_b_field_lines directly with an explicit open_fraction."
        )
    total = product.column("Lc_bwd") + product.column("Lc_fwd")
    backward, forward = product.column("ierr_bwd"), product.column("ierr_fwd")
    status = np.concatenate([backward, forward])
    unknown = sorted({int(code) for code in np.unique(status)
                      if int(code) not in FLARE_TRACE_STATUS})
    if unknown:
        share = float(np.count_nonzero(~np.isin(status, list(FLARE_TRACE_STATUS))))
        raise ValueError(
            f"{product.source.name} carries trace status {unknown}, which "
            f"this FLARE does not define, on {share / status.size:.0%} of its "
            f"directions; known codes are {sorted(FLARE_TRACE_STATUS)}. "
            "Whether those lines reached the wall is unknown, and counting "
            "them either way would put a guess into open_fraction. Call "
            "write_b_field_lines directly with an explicit open_fraction if "
            "you know what the code means."
        )

    reached = ((backward == FLARE_INTERSECT_BOUNDARY)
               | (forward == FLARE_INTERSECT_BOUNDARY))
    # A domain exit is a failed trace, not a verdict on the line.
    left_domain = ((backward == FLARE_DOMAIN_ERROR)
                   | (forward == FLARE_DOMAIN_ERROR)) & ~reached
    if left_domain.any():
        if domain_exit == "refuse":
            raise ValueError(
                f"{product.source.name}: "
                f"{int(np.count_nonzero(left_domain))} of {left_domain.size} "
                f"lines left the field's domain (status {FLARE_DOMAIN_ERROR}, "
                f"{FLARE_TRACE_STATUS[FLARE_DOMAIN_ERROR]}) without reaching "
                "the wall. Those traces failed, so whether they are open is "
                "unknown and open_fraction cannot be derived from them. "
                'Pass domain_exit="open" or "closed" to state a policy -- it '
                "is recorded with the result -- or extend the field's domain "
                "and trace again."
            )
        if domain_exit == "open":
            reached = reached | left_domain
    phi_deg = float(np.unique(mesh.phi_deg)[0])
    # The axes come off `r` and `z` rather than off `u` and `v`: a FlareMesh
    # keeps its axes in the file's own units and converts only the node
    # positions, so a mesh written in centimetres would otherwise declare a
    # grid in centimetres under positions in metres.
    return write_b_field_lines(
        ods,
        grid_r=mesh.r[0], grid_z=mesh.z[:, 0],
        starting_r=mesh.r.ravel(), starting_z=mesh.z.ravel(),
        lengths=total,
        open_fraction=float(np.count_nonzero(reached)) / float(reached.size),
        time=time, phi_deg=phi_deg,
        provenance={
            "source": product.source.name,
            "mesh": f"{mesh.kind} {mesh.node_shape[1]} x {mesh.node_shape[0]}",
            "open_fraction_rule":
                "a line reaching the wall in either direction "
                f"(FLARE status {FLARE_INTERSECT_BOUNDARY})",
            "domain_exit_policy": (
                f"{int(np.count_nonzero(left_domain))} line(s) left the "
                f"field's domain and were counted as {domain_exit}"
                if left_domain.any() else "no line left the field's domain"
            ),
        },
        code_name=code_name, code_repository=code_repository,
    )


# ---------------------------------------------------------------------------
# 2. The footprint summary
# ---------------------------------------------------------------------------


#: Leaves this writer will not touch, with the reason each is wrong for a
#: relative proxy.  ``test_field_line_topology_mapping`` holds it to them.
REFUSED_TARGET_LEAVES = {
    "power_flux_peak": "W/m^2; the proxy has no heat-flux units",
    "wetted_area": "the Data Dictionary defines it through lambda_q, and "
                   "FLARE's footprint uses a tenth of the peak",
    "tilt_angle_pol": "poloidal-only, where the incidence angle is 3-D",
}


def write_divertor_incident_fractions(
    ods: ODS,
    *,
    fractions: Mapping[str, float],
    time: float,
    divertor: int = 0,
    divertor_name: str = "",
    provenance: str = "",
    code_name: str = "FLARE",
    code_repository: str = "",
) -> None:
    """Write one divertor's per-target incident fractions.

    ``fractions`` maps a target identifier to its share of the proxy
    incident on all of this divertor's targets; use
    :func:`vaft.process.field_line_topology.target_incident_fractions` to
    build it, which is where the definition lives.

    Raises
    ------
    ValueError
        Fewer than two targets -- a single target's share is one by
        construction and says nothing but which target was analysed -- a
        share outside ``[0, 1]``, shares that do not add to one, or this
        divertor already carrying fractions at this instant. That last one
        is checked on the divertor rather than on the targets named here:
        shares close to one over one *write*, so a second write at the same
        instant sums past one even when it names only targets nobody has
        written yet.
    """
    if len(fractions) < 2:
        raise ValueError(
            f"a share needs something to be a share of; {len(fractions)} "
            "target(s) were given. power_incident_fraction is each target's "
            "part of the total over all of them, so with one target it is 1 "
            "by construction and records only what was analysed."
        )
    values = {str(key): float(value) for key, value in fractions.items()}
    for key, value in values.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"target {key!r} has a fraction of {value!r}, not in [0, 1]")
    total = sum(values.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"the fractions add to {total!r}, not 1. They are shares of one "
            "total, so a set that does not close is a set of different totals."
        )

    root = f"divertors.divertor.{divertor}"
    identifiers = _target_identifiers(ods, root)
    # A collision is found before the time base moves or a target is added,
    # so a refused write leaves the ODS exactly as it found it. Only an
    # instant that is already in the base can collide: a new one lands past
    # the end of every target's data.
    #
    # The check is on the *divertor*, not on the targets this call names.
    # Shares close to one over the targets of one write, so a second write
    # at the same instant -- even naming targets nobody has written yet --
    # leaves the divertor summing to two, and each write looks correct on
    # its own. One write per divertor per instant is the whole contract.
    existing = _time_index(ods, "divertors", time)
    if existing is not None:
        for identifier, position in identifiers.items():
            data = np.atleast_1d(np.asarray(
                path_value(ods, f"{root}.target.{position}.power_incident_fraction.data",
                           []), dtype=float))
            if existing < data.size and np.isfinite(data[existing]):
                named = "it" if identifier in values else f"its target {identifier!r}"
                raise ValueError(
                    f"divertor {divertor} already carries fractions at "
                    f"t = {float(time)!r} s ({named} reads "
                    f"{float(data[existing])!r}). They are shares of one total over "
                    "every target of this divertor, so a second write at the "
                    "same instant makes them sum past one however plausible "
                    "each write looks alone; pass every target in one call."
                )

    index = _time_base(ods, "divertors", time)
    length = int(np.asarray(ods["divertors.time"]).size)
    if divertor_name:
        ods[f"{root}.name"] = divertor_name

    for identifier, fraction in values.items():
        position = identifiers.get(identifier)
        if position is None:
            position = len(identifiers)
            identifiers[identifier] = position
            ods[f"{root}.target.{position}.identifier"] = identifier
        path = f"{root}.target.{position}.power_incident_fraction.data"
        data = np.atleast_1d(np.asarray(path_value(ods, path, []), dtype=float))
        if data.size < length:
            data = np.concatenate([data, np.full(length - data.size, np.nan)])
        data[index] = fraction
        ods[path] = data

    # Every target of this divertor shares the time base, so one that was
    # written at an earlier instant is padded rather than left short.
    for position in identifiers.values():
        path = f"{root}.target.{position}.power_incident_fraction.data"
        data = np.atleast_1d(np.asarray(path_value(ods, path, []), dtype=float))
        if data.size < length:
            ods[path] = np.concatenate([data, np.full(length - data.size, np.nan)])

    _append_code_parameters(
        ods, "divertors",
        f'<footprint_incident_fraction divertor="{divertor}"'
        f' time="{float(time)!r}" model={quoteattr(provenance)}'
        ' quantity="a relative field-line footprint proxy integrated over each'
        ' target and divided by the total over all of them; it carries no'
        ' heat-flux units, and power_flux_peak, wetted_area and tilt_angle_pol'
        ' are deliberately not written"/>',
        code_name=code_name, repository=code_repository,
    )


def _target_identifiers(ods: ODS, root: str) -> dict[str, int]:
    """``{identifier: position}`` for the targets already under ``root``."""
    found: dict[str, int] = {}
    for position in range(path_count(ods, f"{root}.target")):
        identifier = path_value(ods, f"{root}.target.{position}.identifier", None)
        if identifier:
            found[str(identifier)] = position
    return found


def divertor_incident_fractions_from_flare_footprints(
    ods: ODS,
    proxies: Mapping[str, Any],
    *,
    time: float,
    divertor: int = 0,
    divertor_name: str = "",
    code_name: str = "FLARE",
    code_repository: str = "",
) -> dict[str, float]:
    """Integrate one proxy per target, and write their shares.

    ``proxies`` maps a target identifier to a
    :class:`~vaft.process.field_line_topology.FootprintProxy`. Returns the
    fractions it wrote.

    Raises
    ------
    ValueError
        The proxies were not all built from the same model, or the same
        incidence choice -- their integrals would then be in different
        scales, and a share between them would compare two different
        quantities.
    """
    from vaft.process.field_line_topology import (
        footprint_incident_total,
        target_incident_fractions,
    )

    models = {proxy.model.describe() for proxy in proxies.values()}
    incidence = {bool(proxy.incidence_applied) for proxy in proxies.values()}
    if len(models) > 1 or len(incidence) > 1:
        raise ValueError(
            "every target's proxy must come from one model and one incidence "
            f"choice; got {sorted(models)} with incidence applied "
            f"{sorted(incidence)}. A share between different models compares "
            "two different quantities."
        )
    totals = {
        str(key): footprint_incident_total(proxy) for key, proxy in proxies.items()
    }
    fractions = target_incident_fractions(totals)
    sample = next(iter(proxies.values()))
    write_divertor_incident_fractions(
        ods, fractions=fractions, time=time, divertor=divertor,
        divertor_name=divertor_name, provenance=sample.describe(),
        code_name=code_name, code_repository=code_repository,
    )
    return fractions

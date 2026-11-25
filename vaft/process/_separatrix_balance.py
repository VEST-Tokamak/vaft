"""Signed dRsep and divertor topology across an equilibrium history.

The single-slice analysis is :func:`derive_boundary_representation`; this
module walks it over every slice of an equilibrium IDS and returns one
:class:`~vaft.data.equilibrium.SeparatrixBalanceSeries`, the record plots and
comparisons read.  Nothing here detects a saddle or inverts a flux map itself.

Native EFIT ``DRSEP`` is read from the a-file scalars VAFT stores under
``equilibrium.code.parameters.time_slice.<i>.aeqdsk`` and aligned to the VAFT
convention before any comparison.  Its semantics, from EFIT's
``shapesurf.F90``: centimetres; the 40 cm default whenever EFIT calls the
boundary limited (its LCFS within 10 cm of a wall region); signed by the
X-point's side for ``SNT``/``SNB`` (positive top); for ``DN`` (both offsets
below 2.5 mm) the larger magnitude signed by which X-point EFIT found first;
for ``MAR`` an unsigned magnitude.  Only ``SNT``, ``SNB`` and ``DN`` carry the
DIII-D sign this module compares against.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from vaft.data.equilibrium import SeparatrixBalanceSeries, Topology

#: EFIT's DRSEP default for a configuration it does not treat as diverted [cm].
EFIT_DRSEP_SENTINEL_CM = 40.0
#: The EFIT ``limloc`` values whose DRSEP carries the DIII-D sign.
EFIT_SIGNED_CONFIGURATIONS = ("SNT", "SNB", "DN")


def _read(node: Any, path: str) -> Any:
    """The value at *path* or ``None``, without creating it (OMAS reads vivify)."""
    from vaft.ods_access import path_value

    try:
        return path_value(node, path, None)
    except Exception:
        return None


def _scalar(node: Any, path: str) -> float:
    value = _read(node, path)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def native_efit_d_r_sep(drsep_cm: float, limloc: str | None) -> tuple[float, str | None]:
    """Convert an EFIT a-file ``DRSEP`` to the VAFT dRsep convention.

    Parameters
    ----------
    drsep_cm : float
        The a-file ``DRSEP`` value [cm].
    limloc : str or None
        The a-file ``LIMLOC`` configuration code (``SNT``, ``SNB``, ``DN``,
        ``MAR``, ``IN``, ``OUT``, ``TOP``, ``BOT``) [-].

    Returns
    -------
    tuple of (float, str or None)
        dRsep in metres, NaN when EFIT's value is not a signed separatrix
        separation, and the reason it is not, ``None`` otherwise [m].

    Convention
    ----------
    EFIT writes centimetres with the DIII-D sign (positive top single null), so
    only the unit changes.  The 40 cm default, a limited ``LIMLOC`` and the
    unsigned ``MAR`` magnitude are reported as NaN with a reason rather than
    converted.

    Applicability
    -------------
    Machine-independent.  Reads EFIT a-file values as parsed by
    :func:`vaft.data.aeqdsk.read_aeqdsk`.

    Limitations
    -----------
    EFIT decides "limited" from a fixed 10 cm wall distance, so on a small
    machine nearly every slice carries the sentinel whatever its topology.

    Provenance
    ----------
    .. [1] EFIT ``shapesurf.F90`` (gap and DRSEP calculation) and the a-file
       table at https://efit-ai.gitlab.io/efit/files.html.
    """
    code = (limloc or "").strip().upper() or None
    try:
        value = float(drsep_cm)
    except (TypeError, ValueError):
        return float("nan"), "DRSEP is not a number"
    if not np.isfinite(value):
        return float("nan"), "DRSEP is not finite"
    if value == EFIT_DRSEP_SENTINEL_CM:
        return float("nan"), f"EFIT wrote its {EFIT_DRSEP_SENTINEL_CM:g} cm default (LIMLOC={code})"
    if code not in EFIT_SIGNED_CONFIGURATIONS:
        return float("nan"), f"LIMLOC={code} does not carry a signed separatrix separation"
    return value / 100.0, None


def _no_plasma(slice_node: Any) -> bool:
    psi_axis = _scalar(slice_node, "global_quantities.psi_axis")
    psi_boundary = _scalar(slice_node, "global_quantities.psi_boundary")
    ip = _scalar(slice_node, "global_quantities.ip")
    degenerate = np.isfinite(psi_axis) and np.isfinite(psi_boundary) and psi_axis == psi_boundary
    return bool(degenerate or (np.isfinite(ip) and ip == 0.0))


def separatrix_balance_series(
    source: Any, *, time_indices: Sequence[int] | None = None, representations: Sequence[Any] | None = None,
) -> SeparatrixBalanceSeries:
    """Signed dRsep, topology state and native comparison for every equilibrium slice.

    Parameters
    ----------
    source : ODS or mapping
        An object holding an ``equilibrium`` IDS with ``time_slice`` entries,
        each adaptable by :func:`as_equilibrium` [-].
    time_indices : sequence of int, optional
        The slices to analyse, in order; every slice by default [-].
    representations : sequence or mapping of BoundaryRepresentation, optional
        Already-derived representations to reuse instead of deriving them
        again: a sequence aligned with *time_indices*, or a mapping from slice
        index (as :func:`vaft.omas.update.update_equilibrium_boundary_topology`
        returns), in which slices absent from the mapping are derived [-].

    Returns
    -------
    SeparatrixBalanceSeries
        Time, signed dRsep (NaN unless a diverted balance sample), per-slice
        state, topology and reason, the grid's dRsep resolution, the declared
        double-null half-width eps_DN, the geometric separation of limited or
        ambiguous slices, and the aligned native EFIT value [m].

    Convention
    ----------
    dRsep is ``R_out(psi_X,lower) - R_out(psi_X,upper)`` at the magnetic-axis
    height (negative lower single null).  A slice counts as ``no_plasma`` when
    its plasma current is zero or its axis and boundary fluxes coincide,
    before any flux-map analysis.  The input is never modified.

    Applicability
    -------------
    Machine-independent.  Any source :func:`as_equilibrium` reads; the
    native comparison needs the EFIT a-file cache, which a local product keeps
    and an Access Layer round trip discards.

    Limitations
    -----------
    One full boundary analysis per slice, so it costs as much as
    :func:`derive_boundary_representation` times the number of slices; pass
    *representations* when they already exist.

    Provenance
    ----------
    .. [1] T. W. Petrie et al., GA-A24338 (2003): dRsep = R_L - R_U at the
       outer midplane.
    """
    from vaft.process._equilibrium_parametric import as_equilibrium, derive_boundary_representation

    root = source["equilibrium"] if "equilibrium" in source else source
    slices = root["time_slice"]
    count = len(slices)
    indices = list(range(count)) if time_indices is None else [int(i) for i in time_indices]
    if representations is not None and not hasattr(representations, "keys"):
        if len(representations) != len(indices):
            raise ValueError("a representations sequence must have one entry per analysed slice")
        representations = dict(zip(indices, representations))

    times, values, geometric, resolution, native, tolerance = [], [], [], [], [], []
    states, topologies, reasons, configurations = [], [], [], []
    for index in indices:
        node = slices[index]
        time = _scalar(node, "time")
        if not np.isfinite(time):
            # path_value does not index into an array leaf, so read it whole.
            history = _read(root, "time")
            try:
                time = float(np.asarray(history, dtype=float).reshape(-1)[index])
            except (TypeError, ValueError, IndexError):
                time = float("nan")
        times.append(time)
        # The a-file cache stops at the local product, so after an Access Layer
        # round trip the native value is simply absent (NaN), never guessed.
        parameters = f"equilibrium.code.parameters.time_slice.{index}.aeqdsk"
        drsep = _scalar(source, f"{parameters}.drsep")
        limloc = _read(source, f"{parameters}.limloc")
        native_value, _ = native_efit_d_r_sep(drsep, None if limloc is None else str(limloc))
        native.append(native_value)
        configurations.append(None if limloc is None else str(limloc).strip())

        value = geometric_value = resolution_value = tolerance_value = float("nan")
        topology_name: str | None = None
        if _no_plasma(node):
            state, reason = "no_plasma", "zero plasma current or degenerate flux"
        else:
            try:
                representation = (
                    representations[index] if representations is not None and index in representations
                    else derive_boundary_representation(as_equilibrium(source, time_index=index))
                )
            except Exception as exc:  # noqa: BLE001 -- one bad slice must not end the series
                state, reason = "failed", f"boundary analysis failed: {exc}"
            else:
                topology = representation.topology
                topology_name = topology.value
                derived = representation.d_r_sep
                resolution_value = float(derived.quality.get("resolution_m", float("nan")))
                tolerance_value = float(representation.topology_evidence.get("double_null_tolerance_m", float("nan")))
                reason = derived.reason or representation.reason
                if derived.available:
                    state, value = "valid", float(derived.value)
                else:
                    if derived.value is not None:
                        geometric_value = float(derived.value)
                    state = (
                        "limited" if topology is Topology.LIMITED
                        else "ambiguous" if topology is Topology.AMBIGUOUS
                        else "unavailable"
                    )
        states.append(state)
        topologies.append(topology_name)
        reasons.append(reason)
        values.append(value)
        geometric.append(geometric_value)
        resolution.append(resolution_value)
        tolerance.append(tolerance_value)

    return SeparatrixBalanceSeries(
        time=np.asarray(times, dtype=float),
        d_r_sep=np.asarray(values, dtype=float),
        state=tuple(states),
        topology=tuple(topologies),
        reason=tuple(reasons),
        resolution=np.asarray(resolution, dtype=float),
        geometric_d_r_sep=np.asarray(geometric, dtype=float),
        native_d_r_sep=np.asarray(native, dtype=float),
        native_configuration=tuple(configurations),
        double_null_tolerance=np.asarray(tolerance, dtype=float),
    )


__all__ = ["EFIT_DRSEP_SENTINEL_CM", "EFIT_SIGNED_CONFIGURATIONS", "native_efit_d_r_sep", "separatrix_balance_series"]

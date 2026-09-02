"""Adapter between validation evidence and native IMAS/OMAS status fields (issue #253).

The Data Dictionary already encodes several *different* kinds of status, and
VAFT reuses them rather than inventing a competing convention.  They are not
interchangeable, and this module deliberately gives each its own reader so that
nothing can flatten them into one generic flag:

===========================  =========================================
``*.validity``               usability of the datum itself
``*.validity_timed``         the same, resolved per time sample
``<ids>.code.output_flag``   external-code execution status
``convergence.result``       solver convergence state
constraint ``chi_squared``   a fit metric, not a status at all
===========================  =========================================

This module owns the *schema coupling* -- it is the only place in the
validation layer that knows an IDS path.  Validation results themselves never
carry one (#253 §9), so a Data Dictionary change lands here instead of leaking
through every result object.

Path access here goes through :mod:`vaft.ods_access` (issue #118) rather than a
bare read: probing a channel that carries no ``validity`` would otherwise
materialize an integer leaf holding NaN, which fails the next
consistency-checked load.  That is not hypothetical -- it is how this module
first broke the EFIT constraint tests.

Two rules govern writing.

**Missing is not zero.**  ``validity == 0`` is a positive statement that
automated processing found the datum valid.  An ODS that carries no validity
node has said nothing.  :func:`read_validity` returns ``None`` for the second
case, and every consumer helper takes a ``default`` so an ODS produced before
the quality layer existed behaves exactly as it did before.

**Only the datum's own assessment may write its validity.**  Native validity
may be set by validation whose subject is exactly the datum that field
describes -- magnetics signal quality writing ``magnetics.*.field.validity``,
for instance.  Cross-diagnostic and model-agreement results (measured magnetics
versus a synthetic equilibrium response, EFIT pressure versus Thomson) must
never invalidate source data through this module, however poor the agreement.

Interpretation of the non-validity fields stays with its canonical owner:
:func:`vaft.omas.efit_quality.convergence_metrics` for EFIT convergence, and
:mod:`vaft.machine_mapping.mhd_linear` for the ``output_flag`` the GPEC suite
writes.  The readers here extract; they do not judge.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from vaft.ods_access import path_value as _lookup

__all__ = [
    "VALIDITY_CERTIFIED",
    "VALIDITY_INVALID",
    "VALIDITY_MEANINGS",
    "VALIDITY_SUSPECT",
    "VALIDITY_VALID",
    "aggregate_validity",
    "read_convergence_result",
    "read_output_flag",
    "read_validity",
    "read_validity_timed",
    "resolve_signal_time",
    "valid_fraction",
    "validity_codes",
    "validity_mask",
    "write_validity",
]

#: Data Dictionary validity codes, in descending order of confidence.  The
#: codes are severity-monotone, which is what makes ``min()`` the natural
#: "worst state wins" aggregation in :func:`aggregate_validity`.
VALIDITY_CERTIFIED = 1
VALIDITY_VALID = 0
VALIDITY_SUSPECT = -1
VALIDITY_INVALID = -2

#: Human-readable meanings, for manifests and error messages.  Codes below -2
#: are code-specific by the DD's own definition and are reported verbatim.
VALIDITY_MEANINGS: Mapping[int, str] = {
    VALIDITY_CERTIFIED: "valid and certified",
    VALIDITY_VALID: "valid from automated processing",
    VALIDITY_SUSPECT: "problem identified in processing; verification requested",
    VALIDITY_INVALID: "invalid; should not be used",
}


def _array(value: Any, dtype: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype).reshape(-1)
    return None if array.size == 0 else array


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def resolve_signal_time(source: Any, base: str) -> np.ndarray | None:
    """The time coordinate of the signal node at ``base``.

    ``base`` names a signal structure such as
    ``magnetics.b_field_pol_probe.3.field``.  The Data Dictionary coordinates
    both ``data`` and ``validity_timed`` on that node's own ``time``, but an ODS
    written with ``homogeneous_time = 1`` stores only the IDS root's time and
    the per-node coordinate is lost on round-trip -- the packaged VEST samples
    are exactly that shape.  So the node's own ``time`` wins when present, and
    the IDS root's is the fallback.

    ``None`` when neither exists.
    """
    explicit = _array(_lookup(source, f"{base}.time"), float)
    if explicit is not None:
        return explicit
    ids = base.split(".", 1)[0]
    return _array(_lookup(source, f"{ids}.time"), float)


def read_validity(source: Any, base: str) -> int | None:
    """``<base>.validity``, or ``None`` when the node carries none.

    The ``None`` is the point: it distinguishes "nothing has assessed this
    datum" from ``0``, "automated processing found it valid".
    """
    value = _lookup(source, f"{base}.validity")
    if value is None:
        return None
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def read_validity_timed(source: Any, base: str) -> np.ndarray | None:
    """``<base>.validity_timed`` as an integer array, or ``None`` when absent."""
    return _array(_lookup(source, f"{base}.validity_timed"), int)


def aggregate_validity(timed: Iterable[int]) -> int:
    """The scalar validity summarizing a time-resolved one: **worst state wins**.

    The Data Dictionary codes descend monotonically in confidence
    (1 > 0 > -1 > -2), so the minimum is the most severe state the channel
    reached.  A channel that saturates halfway through therefore reads
    ``-2`` at the scalar level even though it was good earlier -- which is why
    ``validity_timed`` stays authoritative for consumers that work slice by
    slice, and why this scalar must never be used to discard a whole channel
    when the timed field is available.

    An empty input aggregates to :data:`VALIDITY_INVALID`: a waveform with no
    samples carries no usable datum.  That is distinct from an *absent*
    ``validity_timed``, which :func:`read_validity_timed` reports as ``None``.
    """
    values = np.asarray(list(timed), dtype=int).reshape(-1)
    if values.size == 0:
        return VALIDITY_INVALID
    return int(values.min())


def validity_codes(
    source: Any,
    base: str,
    *,
    times: np.ndarray | None = None,
) -> np.ndarray | None:
    """Validity codes for the node at ``base``, per sample.

    Without ``times`` the codes come back on the node's own time grid.  With
    ``times`` they are resampled to those instants by nearest sample, which is
    what a consumer working at reconstruction time slices needs: validity is a
    per-sample state, not something to interpolate between.

    ``validity_timed`` wins wherever it exists; otherwise the scalar
    ``validity`` is broadcast.  ``None`` when the node declares neither.
    """
    timed = read_validity_timed(source, base)
    if timed is None:
        scalar = read_validity(source, base)
        if scalar is None:
            return None
        if times is None:
            grid = resolve_signal_time(source, base)
            size = 1 if grid is None else grid.size
            return np.full(size, scalar, dtype=int)
        return np.full(np.asarray(times, dtype=float).reshape(-1).size, scalar, dtype=int)

    if times is None:
        return timed
    grid = resolve_signal_time(source, base)
    query = np.asarray(times, dtype=float).reshape(-1)
    if grid is None or grid.size != timed.size:
        # Without a usable coordinate the timed field cannot be placed in time.
        # Falling back to its own aggregate is honest: it says what the channel
        # was at its worst rather than pretending to a per-slice answer.
        return np.full(query.size, aggregate_validity(timed), dtype=int)
    index = np.clip(np.searchsorted(grid, query), 0, grid.size - 1)
    left = np.clip(index - 1, 0, grid.size - 1)
    take_left = np.abs(grid[left] - query) <= np.abs(grid[index] - query)
    return timed[np.where(take_left, left, index)]


def validity_mask(
    source: Any,
    base: str,
    *,
    times: np.ndarray | None = None,
    min_validity: int = VALIDITY_VALID,
    default: bool = True,
) -> np.ndarray:
    """Which samples of ``base`` a consumer accepting ``min_validity`` may use.

    ``min_validity`` is a floor on the DD code, so the default accepts
    ``valid`` (0) and ``certified`` (1) while rejecting ``suspect`` (-1) and
    ``invalid`` (-2).

    ``default`` decides what an ODS carrying *no* validity information means.
    It is ``True`` deliberately: absence of an assessment is not a rejection,
    and it keeps every consumer byte-identical on data produced before the
    quality layer existed.
    """
    codes = validity_codes(source, base, times=times)
    if codes is None:
        if times is not None:
            size = np.asarray(times, dtype=float).reshape(-1).size
        else:
            grid = resolve_signal_time(source, base)
            size = 1 if grid is None else grid.size
        return np.full(size, bool(default))
    return codes >= int(min_validity)


def valid_fraction(
    source: Any,
    base: str,
    *,
    times: np.ndarray | None = None,
    window: np.ndarray | None = None,
    min_validity: int = VALIDITY_VALID,
    default: bool = True,
) -> float:
    """Fraction of the samples in ``window`` that ``base`` declares usable.

    A metric, not a verdict: it says how much of the record survives
    ``min_validity``, and the consumer decides what fraction it requires.  That
    threshold is policy and belongs to the consumer, not here (#253 §7).

    ``nan`` when the window selects no samples.
    """
    mask = validity_mask(
        source, base, times=times, min_validity=min_validity, default=default
    )
    if window is not None:
        selector = np.asarray(window, dtype=bool).reshape(-1)
        if selector.size != mask.size:
            raise ValueError(
                f"window has {selector.size} samples but {base!r} has {mask.size}"
            )
        mask = mask[selector]
    if mask.size == 0:
        return float("nan")
    return float(mask.mean())


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_validity(source: Any, base: str, timed: Iterable[int]) -> int:
    """Project a per-sample assessment into ``base``'s native validity nodes.

    Writes ``validity_timed`` and the :func:`aggregate_validity` scalar that
    summarizes it, and returns that scalar.

    The node's time coordinate is *not* written: ``validity_timed`` is
    coordinated on ``<base>.time``, which an ODS may legitimately leave to the
    IDS root (see :func:`resolve_signal_time`).  Adding one here would change
    the ODS's time-homogeneity for a reason unrelated to quality.  Instead the
    coordinate must already resolve, and to the right length, or the projection
    would be unreadable -- so that is checked rather than assumed.

    Only call this for an assessment whose subject is the datum at ``base``
    itself.  Model-agreement failures must not reach here.
    """
    values = np.asarray(list(timed), dtype=int).reshape(-1)
    grid = resolve_signal_time(source, base)
    if grid is None:
        raise ValueError(
            f"{base!r} has no time coordinate, so validity_timed would have "
            "nothing to be resolved against"
        )
    if grid.size != values.size:
        raise ValueError(
            f"validity_timed for {base!r} has {values.size} samples but its "
            f"time coordinate has {grid.size}"
        )
    scalar = aggregate_validity(values)
    if isinstance(source, dict):
        raise TypeError(
            "write_validity needs an ODS; a plain mapping has no IDS structure "
            "to write validity into"
        )
    source[f"{base}.validity_timed"] = values
    source[f"{base}.validity"] = scalar
    return scalar


# ---------------------------------------------------------------------------
# The status fields that are *not* validity
# ---------------------------------------------------------------------------

def read_output_flag(
    source: Any, ids: str = "equilibrium", *, time_slice: int | None = None
) -> Any:
    """``<ids>.code.output_flag`` -- an external code's execution status.

    Not a validity: it describes whether the *run* succeeded, which says
    nothing about whether the measurements it consumed were usable.  0 means
    success and negative values mean the result must not be used; other values
    are code-specific, so nothing is interpreted here.

    Returns the whole ``INT_1D`` array, or one slice's flag when ``time_slice``
    is given.  ``None`` when the IDS carries no flag.
    """
    flags = _array(_lookup(source, f"{ids}.code.output_flag"), int)
    if flags is None or time_slice is None:
        return flags
    if not 0 <= int(time_slice) < flags.size:
        return None
    return int(flags[int(time_slice)])


def read_convergence_result(source: Any, *, time_slice: int) -> dict[str, Any] | None:
    """``equilibrium.time_slice[i].convergence.result`` -- a solver's own verdict.

    A DD identifier structure (``name``/``index``/``description``), returned as
    a plain mapping of whatever it carries.  Distinct again from both validity
    and ``output_flag``: a converged solution can rest on invalid channels, and
    a successful run can be unconverged.  The numerical interpretation of
    convergence lives in :func:`vaft.omas.efit_quality.convergence_metrics`;
    this only extracts the declared result.
    """
    base = f"equilibrium.time_slice.{int(time_slice)}.convergence.result"
    entry: dict[str, Any] = {}
    for key in ("name", "index", "description"):
        value = _lookup(source, f"{base}.{key}")
        if value is not None:
            entry[key] = int(value) if key == "index" else str(value)
    return entry or None

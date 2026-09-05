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
through every result object.  What the nodes *mean* is not decided here
either: :func:`read_validity_record` extracts them into a
:class:`~vaft.validation.validity.ValidityRecord`, and every reader below is a
one-liner over :mod:`vaft.validation.validity`, the access-free interpretation
layer (#424).  A native-IMAS accessor builds the same record from an IDS
handle and cannot disagree with this one about what it means.

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

from vaft.validation.validity import (
    VALIDITY_CERTIFIED,
    VALIDITY_INVALID,
    VALIDITY_MEANINGS,
    VALIDITY_SUSPECT,
    VALIDITY_VALID,
    ValidityRecord,
    aggregate_validity,
    codes_at,
    is_condemned,
    is_usable,
    usable_fraction,
    usable_mask,
)

__all__ = [
    "VALIDITY_CERTIFIED",
    "VALIDITY_INVALID",
    "VALIDITY_MEANINGS",
    "VALIDITY_SUSPECT",
    "VALIDITY_VALID",
    "aggregate_validity",
    "is_condemned_channel",
    "is_usable_channel",
    "read_convergence_result",
    "read_output_flag",
    "read_validity",
    "read_validity_record",
    "read_validity_timed",
    "resolve_signal_time",
    "resolve_signal_waveform",
    "signal_label",
    "valid_fraction",
    "validity_codes",
    "validity_mask",
    "write_validity",
]


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


def resolve_signal_waveform(source: Any, base: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``(time, data)`` of the signal node at ``base``, or ``None``.

    The one rule for "a channel carries a usable waveform": ``data`` is
    present and numeric, has at least two samples, and its resolved time
    axis (:func:`resolve_signal_time`) has the same length.  Finiteness is
    not judged here -- a reader that needs finite samples checks the array.
    """
    raw = _lookup(source, f"{base}.data")
    if raw is None:
        return None
    try:
        values = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 2:
        return None
    time = resolve_signal_time(source, base)
    if time is None or time.size != values.size:
        return None
    return time, values


def signal_label(source: Any, base: str, fallback: str) -> str:
    """The channel's ``name``, else its ``identifier``, else ``fallback``."""
    for key in ("name", "identifier"):
        value = _lookup(source, f"{base}.{key}")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


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


def read_validity_record(source: Any, base: str) -> ValidityRecord:
    """The validity nodes of ``base``, extracted and uninterpreted.

    The one accessor the interpretation layer needs: ``validity``,
    ``validity_timed`` and the grid the latter is coordinated on
    (:func:`resolve_signal_time`).  The grid is resolved even when the timed
    node is absent, because a scalar broadcast over the record must know how
    long the record is.
    """
    return ValidityRecord(
        scalar=read_validity(source, base),
        timed=read_validity_timed(source, base),
        time=resolve_signal_time(source, base),
    )


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
    See :func:`vaft.validation.validity.codes_at`.
    """
    return codes_at(read_validity_record(source, base), times=times)


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
    quality layer existed.  See :func:`vaft.validation.validity.usable_mask`.
    """
    return usable_mask(
        read_validity_record(source, base),
        times=times,
        min_validity=min_validity,
        default=default,
    )


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

    ``nan`` when the window selects no samples.  See
    :func:`vaft.validation.validity.usable_fraction`.
    """
    try:
        return usable_fraction(
            read_validity_record(source, base),
            times=times,
            window=window,
            min_validity=min_validity,
            default=default,
        )
    except ValueError as exc:
        raise ValueError(str(exc).replace("the record", repr(base))) from None


def is_usable_channel(
    source: Any,
    base: str,
    *,
    times: np.ndarray | None = None,
    min_validity: int = VALIDITY_VALID,
    default: bool = True,
) -> bool:
    """Whether ``base`` has any usable sample at all, or any of ``times``.

    The timed field wins; a scalar alone is broadcast; an unassessed node is
    ``default``.  See :func:`vaft.validation.validity.is_usable`.
    """
    return is_usable(
        read_validity_record(source, base),
        times=times,
        min_validity=min_validity,
        default=default,
    )


def is_condemned_channel(
    source: Any, base: str, *, min_validity: int = VALIDITY_VALID
) -> bool:
    """Whether ``base`` was assessed and has no usable sample anywhere.

    The sanctioned whole-channel reading: never the scalar below the floor,
    which would condemn every channel that was good early and bad late.
    ``False`` for a node nothing has assessed.  See
    :func:`vaft.validation.validity.is_condemned`.
    """
    return is_condemned(read_validity_record(source, base), min_validity=min_validity)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_validity(
    source: Any, base: str, timed: Iterable[int], *, scalar: int | None = None
) -> int:
    """Project a per-sample assessment into ``base``'s native validity nodes.

    Writes ``validity_timed`` and the scalar that summarizes it, and returns
    that scalar.  ``scalar`` overrides :func:`aggregate_validity` for callers
    whose vocabulary the plain aggregate cannot express -- a layer that uses
    ``1`` to mean "flagged" cannot let a wholly flagged channel aggregate to
    ``1``, which the Data Dictionary reads as *certified*, and better than a
    clean one.

    The node's time coordinate is *not* written: ``validity_timed`` is
    coordinated on ``<base>.time``, which an ODS may legitimately leave to the
    IDS root (see :func:`resolve_signal_time`).  Adding one here would change
    the ODS's time-homogeneity for a reason unrelated to quality.  Instead the
    coordinate must already resolve, and to the right length, or the projection
    would be unreadable -- so that is checked rather than assumed.

    Only call this for an assessment whose subject is the datum at ``base``
    itself.  Model-agreement failures must not reach here.

    The scalar written here is a summary, not a verdict on the channel.  The
    one sanctioned whole-channel reading of it is
    :func:`is_condemned_channel`, which judges the timed field it summarizes.
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
    summary = aggregate_validity(values) if scalar is None else int(scalar)
    if isinstance(source, dict):
        raise TypeError(
            "write_validity needs an ODS; a plain mapping has no IDS structure "
            "to write validity into"
        )
    source[f"{base}.validity_timed"] = values
    source[f"{base}.validity"] = summary
    return summary


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

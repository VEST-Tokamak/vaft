"""Which poloidal-flux convention an equilibrium stores, read through the accessor.

The IMAS Data Dictionary keeps ``equilibrium.*.psi`` in full weber (COCOS
11-18); legacy VAFT artifacts hold the g-file's weber per radian (COCOS 1-8).
The plotting layer labels psi from this probe (issue #478): a declared COCOS
index settles it, otherwise ``vaft.data.eqdsk``'s ladder decides.

**The ladder itself is not reimplemented here.** It used to be, with two of its
three probes, and a label two pi away from the value the same file computed to
was the result on an ODS carrying neither ``profiles_1d.phi`` nor a boundary
outline. What this module supplies is the *reader*: a
:data:`~vaft.data.eqdsk.SliceReader` per time slice built on
:mod:`vaft.plot.backend.access`, so an OMAS ODS and an IMAS entry answer alike
without converting one into the other, and so a probe added to the ladder
reaches the plotting layer on the same commit.

This module knows no data model: it imports numpy, the accessor and, lazily,
``vaft.data.eqdsk``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .access import count, get

#: The two storage families, spelled as the canonical display-unit tokens.
FLUX_CONVENTIONS = ("Wb", "Wb/rad")

_COCOS_TAG = re.compile(r"<cocos>\s*(\d+)\s*</cocos>")


def declared_cocos(obj: Any) -> int | None:
    """The COCOS index the equilibrium declares, or ``None``.

    DD 4's ``ids_properties.cocos`` is read first, then VAFT's
    ``code.parameters.cocos``.  On an IMAS entry ``code.parameters`` is the
    XML string the DD defines, so the tag is parsed from it.
    """
    for value in (
        get(obj, "equilibrium.ids_properties.cocos"),
        get(obj, "equilibrium.code.parameters.cocos"),
    ):
        index = _index(value)
        if index is not None:
            return index
    parameters = get(obj, "equilibrium.code.parameters")
    if isinstance(parameters, (str, bytes)):
        text = parameters.decode() if isinstance(parameters, bytes) else parameters
        match = _COCOS_TAG.search(text)
        return _index(match.group(1)) if match else None
    if hasattr(parameters, "get"):
        try:
            return _index(parameters.get("cocos"))
        except Exception:
            return None
    return None


def _index(value: Any) -> int | None:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    try:
        index = int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return index if 1 <= index <= 8 or 11 <= index <= 18 else None


def psi_convention(obj: Any, time_slice: int = 0) -> str:
    """``"Wb"`` or ``"Wb/rad"``: the family ``obj``'s equilibrium psi is stored in.

    A declared index wins.  Otherwise the requested slice is probed first and
    the others after it, since the convention is a property of the file and a
    degenerate slice must not force the default.  When nothing can answer the
    DD convention (``"Wb"``) is assumed, as :func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor` does.
    """
    index = declared_cocos(obj)
    if index is not None:
        return "Wb/rad" if index < 10 else "Wb"
    exponent = flux_exponent(obj, time_slice)
    if exponent is None:
        return "Wb"
    return "Wb/rad" if exponent == 0 else "Wb"


def slice_reader(obj: Any, time_slice: int):
    """A :data:`~vaft.data.eqdsk.SliceReader` over one slice, through the accessor."""
    base = f"equilibrium.time_slice.{time_slice}"
    return lambda path: get(obj, f"{base}.{path}")


def flux_exponent(obj: Any, time_slice: int = 0) -> int | None:
    """``e_Bp`` (0 per radian, 1 weber) from the data, else ``None``.

    The requested slice is consulted first and the others after it, since the
    convention is a property of the file and a degenerate slice must not force
    the default.  Precedence between the probes runs ahead of precedence
    between slices; :func:`vaft.data.eqdsk.flux_exponent_tier` owns both.
    """
    from vaft.data.eqdsk import flux_exponent_tier

    total = count(obj, "equilibrium.time_slice")
    order = [time_slice] + [i for i in range(total) if i != time_slice] if total else [time_slice]
    _tier, decided = flux_exponent_tier([slice_reader(obj, i) for i in order])
    return decided[min(decided)] if decided else None



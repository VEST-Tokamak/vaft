"""Which poloidal-flux convention an equilibrium stores, read through the accessor.

The IMAS Data Dictionary keeps ``equilibrium.*.psi`` in full weber (COCOS
11-18); legacy VAFT artifacts hold the g-file's weber per radian (COCOS 1-8).
The plotting layer labels psi from this probe (issue #478): a declared COCOS
index settles it, otherwise the same two data probes ``vaft.data.eqdsk``
uses -- the dphi/dpsi-vs-q slope, then Ampere's law round the LCFS -- run on
arrays read through :mod:`vaft.plot.backend.access`, so an OMAS ODS and an
IMAS entry answer alike without converting one into the other.

This module knows no data model: it imports numpy, the accessor and, lazily,
``vaft.process.cocos``.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import numpy as np

from .access import array, count, get

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
    total = count(obj, "equilibrium.time_slice")
    order = [time_slice] + [i for i in range(total) if i != time_slice] if total else [time_slice]
    for index_ in order:
        exponent = slice_flux_exponent(obj, index_)
        if exponent is not None:
            return "Wb/rad" if exponent == 0 else "Wb"
    return "Wb"


def slice_flux_exponent(obj: Any, time_slice: int) -> int | None:
    """``e_Bp`` (0 per radian, 1 weber) from one slice's own data, else ``None``."""
    for probe in (_slope_exponent, _ampere_exponent):
        exponent = probe(obj, time_slice)
        if exponent is not None:
            return exponent
    return None


def _slope_exponent(obj: Any, time_slice: int) -> int | None:
    base = f"equilibrium.time_slice.{time_slice}.profiles_1d"
    phi = array(obj, f"{base}.phi")
    q = array(obj, f"{base}.q")
    psi = array(obj, f"{base}.psi")
    if phi is None or q is None or psi is None:
        return None
    phi, q, psi = phi.reshape(-1), q.reshape(-1), psi.reshape(-1)
    if phi.size < 3 or phi.size != q.size or phi.size != psi.size or not np.all(np.isfinite(phi)):
        return None
    dpsi = np.diff(psi)
    good = np.abs(dpsi) > 0
    if np.count_nonzero(good) < 2:
        return None
    slope = np.diff(phi)[good] / dpsi[good]
    q_mid = 0.5 * (q[1:] + q[:-1])[good]
    finite = np.isfinite(slope) & np.isfinite(q_mid) & (np.abs(q_mid) > 1e-12)
    if np.count_nonzero(finite) < 2:
        return None
    ratio = float(np.nanmedian(np.abs(slope[finite] / q_mid[finite])))
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    from vaft.process.cocos import FLUX_EXPONENT_TOLERANCE

    for exponent, expected in ((1, 1.0), (0, 2.0 * np.pi)):
        if abs(ratio - expected) <= FLUX_EXPONENT_TOLERANCE * expected:
            return exponent
    return None


def _ampere_exponent(obj: Any, time_slice: int) -> int | None:
    base = f"equilibrium.time_slice.{time_slice}"
    r = array(obj, f"{base}.profiles_2d.0.grid.dim1")
    z = array(obj, f"{base}.profiles_2d.0.grid.dim2")
    psi = array(obj, f"{base}.profiles_2d.0.psi")
    boundary_r = array(obj, f"{base}.boundary.outline.r")
    boundary_z = array(obj, f"{base}.boundary.outline.z")
    ip = get(obj, f"{base}.global_quantities.ip")
    if r is None or z is None or psi is None or boundary_r is None or boundary_z is None or ip is None:
        return None
    r, z = r.reshape(-1), z.reshape(-1)
    if psi.shape == (z.size, r.size) and psi.shape != (r.size, z.size):
        psi = psi.T
    if psi.shape != (r.size, z.size):
        return None
    try:
        ip = float(np.asarray(ip, dtype=float).reshape(-1)[0])
    except (IndexError, TypeError, ValueError):
        return None
    if not np.isfinite(ip) or ip == 0.0:
        return None
    from vaft.process.cocos import identify_flux_exponent

    equilibrium = SimpleNamespace(
        r=r, z=z, psi=psi, ip=ip,
        lcfs=SimpleNamespace(r=boundary_r.reshape(-1), z=boundary_z.reshape(-1)),
    )
    try:
        exponent, _ratio = identify_flux_exponent(equilibrium)
    except Exception:
        return None
    return exponent

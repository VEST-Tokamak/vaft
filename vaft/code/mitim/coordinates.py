"""The radial coordinate bridge between VAFT and MITIM (#1588 stage A2).

VAFT's TGLF/NEO adapters address a surface by ``r/a`` (``rmin/rmin[-1]``), the
coordinate GACODE's local codes take. MITIM's ``TGLF(rhos=...)``/``NEO(rhos=...)``
address it by ``rho_tor_norm`` and convert with ``r_is_rho=True``: on 48224,
``rho_tor_norm`` 0.5 and 0.7 ran at ``r/a`` 0.575 and 0.789. These two maps are
the only place the conversion is made, both from the one profile that defines
both coordinates, and a value outside that profile is refused, never extrapolated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["r_over_a_at", "rho_tor_norm_at"]


def _pair(profile: Any) -> tuple[np.ndarray, np.ndarray]:
    rmin = np.asarray(getattr(profile, "rmin"), dtype=float)
    rho = np.asarray(getattr(profile, "rho"), dtype=float)
    if rmin.size < 2 or rmin.size != rho.size or not (rmin[-1] > 0):
        raise ValueError("the profile carries no usable rmin/rho pair")
    roa = rmin / rmin[-1]
    if not (np.all(np.diff(roa) > 0) and np.all(np.diff(rho) > 0)):
        raise ValueError("rmin and rho must both increase along the profile")
    return roa, rho


def _map(source: np.ndarray, target: np.ndarray, values: Any, name: str) -> np.ndarray:
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if np.any(values < source[0]) or np.any(values > source[-1]):
        raise ValueError(f"{name} {values.tolist()} lies outside the profile's "
                         f"[{source[0]:.4g}, {source[-1]:.4g}]")
    # anti-alias: not a time series and not a downsample; a monotone radial
    # coordinate map of one profile, evaluated at a few surfaces.
    return np.interp(values, source, target)


def rho_tor_norm_at(profile: Any, r_over_a: Any) -> np.ndarray:
    """``rho_tor_norm`` of the surfaces at ``r/a`` (``rmin/rmin[-1]``) of ``profile`` [-]."""
    roa, rho = _pair(profile)
    return _map(roa, rho, r_over_a, "r/a")


def r_over_a_at(profile: Any, rho_tor_norm: Any) -> np.ndarray:
    """``r/a`` of the surfaces at ``rho_tor_norm`` of ``profile`` [-]."""
    roa, rho = _pair(profile)
    return _map(rho, roa, rho_tor_norm, "rho_tor_norm")

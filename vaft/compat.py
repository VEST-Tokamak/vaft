"""Runtime compatibility helpers for dependency and platform differences."""

from __future__ import annotations

import os
from pathlib import Path

IS_WINDOWS = os.name == "nt"

_RUNTIME_PATCH_APPLIED = False
__all__ = [
    "IS_WINDOWS",
    "trapz_compat",
    "cumtrapz_compat",
    "apply_runtime_compat_patches",
    "ensure_home_environment",
]


def trapz_compat(y, x=None, dx=1.0, axis: int = -1):
    """Evaluate trapezoidal integration across NumPy versions."""
    import numpy as np

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x=x, dx=dx, axis=axis)

    legacy_trapz = getattr(np, "trapz", None)
    if legacy_trapz is not None:
        return legacy_trapz(y, x=x, dx=dx, axis=axis)

    from scipy import integrate

    scipy_trapezoid = getattr(integrate, "trapezoid", None)
    if scipy_trapezoid is not None:
        return scipy_trapezoid(y, x=x, dx=dx, axis=axis)

    raise AttributeError("No trapezoidal integration function is available")


def cumtrapz_compat(y, x=None, dx=1.0, axis: int = -1, initial=0.0):
    """Evaluate cumulative trapezoidal integration across SciPy versions."""
    from scipy import integrate

    cumulative = getattr(integrate, "cumulative_trapezoid", None)
    if cumulative is None:
        cumulative = getattr(integrate, "cumtrapz", None)
    if cumulative is None:
        raise AttributeError(
            "Neither scipy.integrate.cumulative_trapezoid nor scipy.integrate.cumtrapz is available"
        )
    return cumulative(y, x=x, dx=dx, axis=axis, initial=initial)


def ensure_home_environment() -> str | None:
    """Set ``HOME`` from the platform home on native Windows when absent."""
    if "HOME" in os.environ:
        return os.environ["HOME"]
    if not IS_WINDOWS:
        return None
    home = str(Path.home())
    os.environ["HOME"] = home
    return home


def apply_runtime_compat_patches() -> None:
    """Apply broad NumPy/SciPy shims that are safe for all runtime paths."""
    global _RUNTIME_PATCH_APPLIED
    if _RUNTIME_PATCH_APPLIED:
        return

    # OMAS reads HOME while some of its modules are imported. Native Windows
    # normally provides USERPROFILE instead, so establish HOME before any
    # optional dependency imports below can reach OMAS.
    ensure_home_environment()

    try:
        import numpy as np
        from scipy import integrate
    except Exception:
        # Optional dependencies are unavailable; skip without breaking importers.
        return

    # NumPy 2.0 removed `np.NaN`.
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    if not hasattr(np, "RankWarning"):
        try:
            from numpy.exceptions import RankWarning as _RankWarning
        except Exception:
            try:
                from numpy.polynomial.polyutils import RankWarning as _RankWarning
            except Exception:
                _RankWarning = RuntimeWarning
        np.RankWarning = _RankWarning

    # SciPy removed `integrate.cumtrapz` in favor of cumulative_trapezoid.
    if not hasattr(integrate, "cumtrapz") and hasattr(integrate, "cumulative_trapezoid"):
        integrate.cumtrapz = integrate.cumulative_trapezoid

    _RUNTIME_PATCH_APPLIED = True

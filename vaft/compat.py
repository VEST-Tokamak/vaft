"""Runtime compatibility helpers for external dependency API changes."""

from __future__ import annotations

_RUNTIME_PATCH_APPLIED = False
__all__ = [
    "trapz_compat",
    "cumtrapz_compat",
    "apply_runtime_compat_patches",
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


def apply_runtime_compat_patches() -> None:
    """Apply broad NumPy/SciPy shims that are safe for all runtime paths."""
    global _RUNTIME_PATCH_APPLIED
    if _RUNTIME_PATCH_APPLIED:
        return

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

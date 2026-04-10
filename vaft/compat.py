"""Runtime compatibility helpers for external dependency API changes."""

from __future__ import annotations

from functools import wraps
from typing import Any
import sys

_OMFIT_PATCH_APPLIED = False
_RUNTIME_PATCH_APPLIED = False
__all__ = [
    "trapz_compat",
    "cumtrapz_compat",
    "apply_runtime_compat_patches",
    "apply_omfit_compat_patches",
]


def _build_interp2d_compat(interpolate_module: Any):
    """Create a lightweight interp2d-compatible wrapper using RGI."""
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np

    class _Interp2DCompat:
        def __init__(
            self,
            x,
            y,
            z,
            kind: str = "linear",
            bounds_error: bool = False,
            fill_value=None,
        ):
            x_arr = np.asarray(x, dtype=float).reshape(-1)
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            z_arr = np.asarray(z, dtype=float)

            if z_arr.shape != (y_arr.size, x_arr.size):
                raise ValueError(
                    "interp2d compatibility wrapper expects z.shape == (len(y), len(x))"
                )

            method = "linear"
            if str(kind).lower() == "nearest":
                method = "nearest"
            self._rgi = RegularGridInterpolator(
                (y_arr, x_arr),
                z_arr,
                method=method,
                bounds_error=bounds_error,
                fill_value=fill_value,
            )

        def __call__(self, x_new, y_new):
            x_scalar = np.isscalar(x_new)
            y_scalar = np.isscalar(y_new)
            x_arr = np.asarray(x_new, dtype=float).reshape(-1)
            y_arr = np.asarray(y_new, dtype=float).reshape(-1)
            xx, yy = np.meshgrid(x_arr, y_arr, indexing="xy")
            points = np.column_stack([yy.reshape(-1), xx.reshape(-1)])
            values = self._rgi(points).reshape(y_arr.size, x_arr.size)
            if x_scalar and y_scalar:
                # Legacy interp2d scalar query is consumed as `[0]` in OMFIT code.
                return np.asarray([float(values[0, 0])], dtype=float)
            if x_scalar:
                return values[:, 0]
            if y_scalar:
                return values[0, :]
            return values

    def _interp2d(x, y, z, kind="linear", copy=True, bounds_error=False, fill_value=None):
        _ = copy  # kept for signature compatibility
        return _Interp2DCompat(
            x=x,
            y=y,
            z=z,
            kind=kind,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

    return _interp2d


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


def apply_omfit_compat_patches() -> None:
    """Apply additional NumPy/SciPy compatibility shims required by omfit_classes."""
    global _OMFIT_PATCH_APPLIED
    if _OMFIT_PATCH_APPLIED:
        return

    apply_runtime_compat_patches()

    try:
        import numpy as np
        from scipy import interpolate
    except Exception:
        return

    # NumPy 2.0 `errstate` objects are not re-entrant, but OMFIT uses
    # decorator-style errstate wrappers that expect re-entrancy.
    original_errstate = np.errstate

    class _ReentrantErrState:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self._stack = []

        def __enter__(self):
            cm = original_errstate(**self._kwargs)
            self._stack.append(cm)
            return cm.__enter__()

        def __exit__(self, exc_type, exc, exc_tb):
            cm = self._stack.pop()
            return cm.__exit__(exc_type, exc, exc_tb)

        def __call__(self, func):
            @wraps(func)
            def _wrapped(*args, **kwargs):
                with _ReentrantErrState(**self._kwargs):
                    return func(*args, **kwargs)

            return _wrapped

    def _reentrant_errstate(**kwargs):
        return _ReentrantErrState(**kwargs)

    # Patch only once per process.
    if getattr(np.errstate, "__name__", "") != "_reentrant_errstate":
        np.errstate = _reentrant_errstate

    # SciPy >=1.14 removed interp2d implementation (kept as raising stub).
    needs_interp2d_patch = False
    if hasattr(interpolate, "interp2d"):
        try:
            _ = interpolate.interp2d([0.0, 1.0], [0.0, 1.0], [[0.0, 0.0], [0.0, 0.0]])
        except NotImplementedError:
            needs_interp2d_patch = True
        except Exception:
            # Any other error implies symbol exists and is callable enough for our purpose.
            needs_interp2d_patch = False
    else:
        needs_interp2d_patch = True

    if needs_interp2d_patch:
        interpolate.interp2d = _build_interp2d_compat(interpolate)

    # If omfit_classes was imported before this patch, refresh its numpy-error
    # decorators so they no longer close over a non-reentrant errstate context.
    _retrofit_omfit_utils_math_np_errors()

    _OMFIT_PATCH_APPLIED = True


def _retrofit_omfit_utils_math_np_errors() -> None:
    mod = sys.modules.get("omfit_classes.utils_math")
    if mod is None:
        return

    np_errors = getattr(mod, "np_errors", None)
    if np_errors is None:
        return

    # Idempotency guard: skip when already retrofitted by this module.
    if getattr(np_errors, "__module__", "") == __name__ and getattr(np_errors, "__name__", "") == "_np_errors_reentrant":
        return

    import functools
    import numpy as np

    def _np_errors_reentrant(**kw):
        def decorator(f):
            @functools.wraps(f)
            def decorated(*args, **kwargs):
                with np.errstate(**kw):
                    return f(*args, **kwargs)

            return decorated

        return decorator

    mod.np_errors = _np_errors_reentrant
    mod.np_ignored = _np_errors_reentrant(all="ignore")
    mod.np_raised = _np_errors_reentrant(all="raise")
    mod.np_printed = _np_errors_reentrant(all="print")
    mod.np_warned = _np_errors_reentrant(all="warn")


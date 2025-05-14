# Top-level package initialization
from .version import __version__


# ────────────────────────────────────────────────────────
### alias for cumtrapz, and interp2d in old SciPy which is not available in latest scipy version
# it is used for using omfit_classes.omfit_eqdsk.py script without version conflict
import importlib
import numpy as np


def _shim_interp2d(x, y, z, kind="linear", copy=True,
                   bounds_error=False, fill_value=None, **kw):
    """
    Drop‑in replacement for deprecated scipy.interpolate.interp2d.
    Supports both regular-grid vectors and meshgrid arrays.
    """
    from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

    # --- 1. 입력을 1‑D 벡터로 정규화 ---------------------------------
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    # meshgrid → 벡터 추출
    if x.ndim == 2 and y.ndim == 2:
        # assume `z.shape == x.shape == y.shape == (ny, nx)`
        x = x[0, :]
        y = y[:, 0]
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x, y must be 1‑D vectors or 2‑D meshgrid arrays")

    if z.shape != (y.size, x.size):
        raise ValueError(
            f"z shape {z.shape} is incompatible with x({x.size}) / y({y.size})"
        )

    # --- 2. 내부 보간기 준비 -----------------------------------------
    kind = kind.lower()
    if kind in ("linear", "cubic"):
        if kind == "linear":
            rgi = RegularGridInterpolator(
                (y, x), z,
                bounds_error=bounds_error,
                fill_value=fill_value,
                method="linear",
            )

            def _call(xx, yy, **ckw):
                xx = np.atleast_1d(xx)
                yy = np.atleast_1d(yy)
                XY = np.stack(np.meshgrid(yy, xx, indexing="ij"), -1).reshape(-1, 2)
                zz = rgi(XY).reshape(yy.size, xx.size)
                return zz
        else:  # cubic
            spline = RectBivariateSpline(
                y, x, z, kx=3, ky=3
            )  # RectBivariateSpline는 bounds_error / fill_value 옵션 없음

            def _call(xx, yy, **ckw):
                xx = np.atleast_1d(xx)
                yy = np.atleast_1d(yy)
                return spline(yy, xx, grid=True)
    else:
        raise ValueError("kind must be 'linear' or 'cubic'")

    return _call


try:
    integ = importlib.import_module("scipy.integrate")
    if not hasattr(integ, "cumtrapz"):
        from scipy.integrate import cumulative_trapezoid
        integ.cumtrapz = cumulative_trapezoid          # alias 주입

    ip = importlib.import_module("scipy.interpolate")
    ip.interp2d = _shim_interp2d

except ModuleNotFoundError:
    print("Requirements.txt is not installed in the system. Please install dependencies by 'pip install . -r requirements.txt' at vaft repository directory.")


# ────────────────────────────────────────────────────────


__all__ = ['process', 'formula', 'machine_mapping', 'plot', 'omas', 'code', 'database']

from . import process
from . import formula
from . import machine_mapping
from . import plot
from . import omas
from . import code
from . import database

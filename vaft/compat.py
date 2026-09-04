"""Runtime compatibility helpers for dependency and platform differences.

Two kinds of difference are absorbed here, and nowhere else in VAFT:

* **Dependency API drift** -- NumPy/SciPy renames that would otherwise force
  every call site to branch on a version.
* **Platform differences** -- the handful of places where POSIX and Windows do
  not behave the same. Windows is a first-class VAFT platform, so these are
  resolved once, in named helpers that document *why* the platforms differ,
  rather than by scattering ``os.name`` checks through the codebase.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator
import warnings

IS_WINDOWS = os.name == "nt"

_RUNTIME_PATCH_APPLIED = False
__all__ = [
    "IS_WINDOWS",
    "trapz_compat",
    "cumtrapz_compat",
    "apply_runtime_compat_patches",
    "ensure_home_environment",
    "user_home",
    "reopenable_temporary_file",
    "remove_directory",
    "temporary_directory",
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


def user_home() -> Path:
    """Return the user's home directory on any platform.

    ``Path.home()`` already consults ``USERPROFILE`` on Windows, so this is a
    thin wrapper. It exists so callers have one obvious answer instead of
    reaching for ``os.environ["HOME"]``, which does not exist on a stock
    Windows install.
    """
    return Path.home()


def ensure_home_environment() -> str | None:
    """Set ``HOME`` from the platform home on native Windows when absent.

    Several dependencies read ``os.environ["HOME"]`` unguarded -- notably
    ``omas.omas_imas``, which evaluates it in a *default argument* and so
    raises ``KeyError: 'HOME'`` at import time, before any VAFT code runs and
    with a traceback that says nothing about the real problem.
    """
    if "HOME" in os.environ:
        return os.environ["HOME"]
    if not IS_WINDOWS:
        # A POSIX system without $HOME is unusual enough that guessing would
        # hide a real problem; report nothing and let the caller fail loudly.
        return None
    home = str(user_home())
    os.environ["HOME"] = home
    return home


@contextmanager
def reopenable_temporary_file(
    *, suffix: str = "", prefix: str = "vaft-"
) -> Iterator[Path]:
    """Yield a path to a temporary file another library may open by name.

    ``tempfile.NamedTemporaryFile`` cannot be used for this on Windows: the
    handle it holds is exclusive, so any library that takes a *path* and opens
    it itself -- ``omas.ODS.load`` and ``ODS.save`` among them -- fails with
    ``PermissionError: [Errno 13]``. POSIX allows the second open, which is
    why this only ever showed up on Windows.

    The file is created inside a private temporary directory and left closed,
    so the caller writes to it and hands the path on freely. Cleanup goes
    through :func:`temporary_directory` rather than ``TemporaryDirectory``
    directly: the consumer that opened the payload by name may still be
    holding it -- ``h5py`` and ``imas_core`` both do -- and a raw cleanup
    would then raise the very sharing violation this helper exists to avoid.
    """
    with temporary_directory(prefix=prefix) as scratch:
        yield scratch / f"payload{suffix}"


def remove_directory(path: str | Path, *, missing_ok: bool = True) -> bool:
    """Delete a directory tree; return whether it is now gone.

    POSIX unlinks a file that is still open, so a tree can always be removed.
    Windows refuses with ``WinError 32`` while any handle remains, and not
    every handle is VAFT's to close: ``imas_core``'s HDF5 backend keeps large
    IDS files open even after ``DBEntry.close()`` returns.

    Rather than crash a caller that has already finished its work, report the
    failure through the return value and leave the remains to the operating
    system's temporary-file reaper. Callers that genuinely require the removal
    can act on ``False``; most only need not to raise.
    """
    target = Path(path)
    if not target.exists():
        return bool(missing_ok)
    try:
        shutil.rmtree(target)
        return True
    except FileNotFoundError:
        # Something else removed it between the check above and the call --
        # a sibling stage, or the system's temp reaper. The tree is gone,
        # which is all this function promises.
        return True
    except OSError:
        if not IS_WINDOWS:
            # On POSIX this is a real bug, not a platform quirk. Silencing it
            # would hide it.
            raise
        shutil.rmtree(target, ignore_errors=True)
        return not target.exists()


@contextmanager
def temporary_directory(*, prefix: str = "vaft-") -> Iterator[Path]:
    """Like ``tempfile.TemporaryDirectory``, but never raises on cleanup.

    ``TemporaryDirectory.cleanup`` propagates the Windows sharing violation
    described in :func:`remove_directory`, which turns a foreign library's
    leaked handle into a failure of otherwise-complete VAFT work.

    Cleanup runs in a ``finally``, so anything raised there would *replace* an
    exception raised by the body and hide the real cause. A POSIX failure is
    still a genuine bug, so it is reported as a warning rather than swallowed
    outright -- visible, but never at the cost of the caller's own error.
    """
    scratch = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield scratch
    finally:
        try:
            remove_directory(scratch)
        except OSError as error:
            warnings.warn(
                f"Could not remove the temporary directory {scratch}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )


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

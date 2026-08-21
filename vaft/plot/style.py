"""Figure and axes plumbing shared by every canonical renderer.

The renderer contract from issue #62 is:

* ``ax=None`` -- when omitted a figure is created, otherwise the caller's axes
  are used and no new figure appears;
* ``show=False`` -- rendering never displays implicitly;
* the return value is ``(Figure, Axes)`` or ``(Figure, ndarray[Axes])``.

:func:`resolve_axes` and :func:`finalize` implement that contract once so the
renderers themselves only describe what to draw.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "axis_label",
    "finalize",
    "resolve_axes",
    "save_figure",
]


def axis_label(label: str, unit: str = "") -> str:
    """Compose ``"Label [unit]"``, omitting the bracket when there is no unit."""
    label = str(label or "")
    unit = str(unit or "")
    if not unit:
        return label
    if not label:
        return f"[{unit}]"
    return f"{label} [{unit}]"


def resolve_axes(
    ax: Axes | Sequence[Axes] | np.ndarray | None,
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
) -> tuple[Figure, Any]:
    """Return ``(figure, axes)`` for a renderer.

    When ``ax`` is ``None`` a new figure is created.  When ``ax`` is supplied it
    is authoritative: no figure is created, and its owning figure is returned.
    A single axes is accepted for a multi-panel request only when one panel was
    requested; otherwise the number of supplied axes must match.
    """
    needed = int(nrows) * int(ncols)
    if ax is None:
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            squeeze=squeeze and needed == 1,
        )
        return figure, axes

    if isinstance(ax, Axes):
        if needed != 1:
            raise ValueError(
                f"this renderer draws {needed} panels; pass a sequence of "
                f"{needed} axes instead of a single Axes"
            )
        return ax.figure, ax

    axes = np.asarray(ax, dtype=object)
    flat = axes.ravel()
    if flat.size != needed:
        raise ValueError(
            f"this renderer draws {needed} panels but received {flat.size} axes"
        )
    for item in flat:
        if not isinstance(item, Axes):
            raise TypeError(f"ax entries must be matplotlib Axes; got {type(item).__name__}")
    return flat[0].figure, axes


def finalize(
    figure: Figure,
    axes: Any,
    *,
    show: bool = False,
    tight_layout: bool = True,
) -> tuple[Figure, Any]:
    """Apply the shared closing steps and honor the ``show`` contract."""
    if tight_layout:
        # Dense panel grids can be impossible to lay out tightly; that is a
        # cosmetic outcome, not a failure, so do not surface it to callers.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*[Tt]ight layout.*")
            try:
                figure.tight_layout()
            except Exception:  # pragma: no cover - layout engines can refuse
                pass
    if show:
        plt.show()
    return figure, axes


def save_figure(figure: Figure, path: Any, *, close: bool = True, **savefig_kwargs: Any):
    """Write ``figure`` to ``path`` and release it.

    Callers outside :mod:`vaft.plot` use this instead of importing pyplot just to
    close a figure, which keeps rendering confined to this package.
    """
    savefig_kwargs.setdefault("dpi", 300)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    figure.savefig(path, **savefig_kwargs)
    if close:
        plt.close(figure)
    return path

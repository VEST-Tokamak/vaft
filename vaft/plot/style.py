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
    "draw_series",
    "UNCERTAINTY_MODES",
    "VALIDITY_MODES",
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


#: How stored uncertainty is drawn.  ``auto`` picks a shaded band for a
#: continuous trace and error bars for a scatter-like one.
UNCERTAINTY_MODES = ("auto", "band", "errorbar", "none")

#: What to do with data the source flagged invalid.  ``show`` -- the default --
#: never hides it: an invalid channel is demoted rather than dropped, so a
#: reader sees that it exists and that it is not to be trusted.
VALIDITY_MODES = ("show", "mask", "ignore")

#: Colour used to demote invalid data.
INVALID_COLOR = "0.65"


def _scatter_like(series: Any, options: dict) -> bool:
    """Whether a trace reads as discrete points rather than a continuous line.

    A marker alone does not make a scatter: Matplotlib still joins the points
    with a solid line unless the line is explicitly switched off, so only an
    explicit ``linestyle`` of "none" counts.
    """
    marker = options.get("marker") or series.style.get("marker")
    linestyle = options.get("linestyle", series.style.get("linestyle"))
    return bool(marker) and linestyle in ("none", "None", "")


def draw_series(
    axes: Any,
    series: Any,
    *,
    uncertainty: str = "auto",
    validity: str = "show",
    **options: Any,
) -> bool:
    """Draw one :class:`~vaft.plot.models.Series`, honouring validity and error.

    Returns whether the trace contributed a legend entry.  Four states are kept
    distinct, and none of them is silently hidden by default (issue #256):

    ``valid``
        drawn normally;
    ``invalid channel``
        drawn in grey and dashed, its label marked, so the reader sees the
        channel exists and is untrustworthy;
    ``invalid interval``
        the samples are demoted and the span is shaded, so a partial dropout
        does not masquerade as a gap in the physics;
    ``missing``
        never reaches here -- a trace with no data is not built at all.

    ``validity="mask"`` removes what is flagged, and ``"ignore"`` renders as
    though no validity metadata existed.
    """
    if uncertainty not in UNCERTAINTY_MODES:
        raise ValueError(
            f"uncertainty must be one of {', '.join(UNCERTAINTY_MODES)}; "
            f"got {uncertainty!r}"
        )
    if validity not in VALIDITY_MODES:
        raise ValueError(
            f"validity must be one of {', '.join(VALIDITY_MODES)}; got {validity!r}"
        )

    x, y, yerr = series.x, series.y, series.yerr
    invalid_channel = series.is_invalid_channel and validity != "ignore"
    mask = None if validity == "ignore" else series.valid_mask

    if validity == "mask":
        if invalid_channel:
            return False
        if mask is not None:
            if not mask.any():
                return False
            x, y = x[mask], y[mask]
            if yerr is not None:
                yerr = yerr[..., mask]
            mask = None

    if invalid_channel:
        options.setdefault("color", INVALID_COLOR)
        options.setdefault("linestyle", "--")
        if options.get("label"):
            options["label"] = f"{options['label']} (invalid)"

    labelled = bool(options.get("label"))
    draw_errorbars = yerr is not None and uncertainty in ("errorbar", "auto") and (
        uncertainty == "errorbar" or _scatter_like(series, options)
    )
    if yerr is not None and uncertainty == "none":
        yerr = None

    if draw_errorbars:
        axes.errorbar(x, y, yerr=yerr, **options)
    else:
        lines = axes.plot(x, y, **options)
        if yerr is not None and uncertainty in ("band", "auto"):
            spread = np.asarray(yerr, dtype=float)
            low, high = (spread[0], spread[1]) if spread.ndim == 2 else (spread, spread)
            axes.fill_between(
                x, y - low, y + high,
                color=lines[0].get_color() if lines else None,
                alpha=0.2, linewidth=0,
            )

    if mask is not None and not mask.all() and validity == "show":
        for start, end in _invalid_runs(mask):
            span = _run_extent(x, start, end)
            if span is not None:
                axes.axvspan(*span, color=INVALID_COLOR, alpha=0.25, linewidth=0)
    return labelled


def _invalid_runs(mask: Any):
    """Yield ``(start, end)`` index pairs for each contiguous invalid run."""
    invalid = ~np.asarray(mask, dtype=bool)
    if not invalid.any():
        return
    padded = np.concatenate([[False], invalid, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for start, end in zip(edges[0::2], edges[1::2]):
        yield int(start), int(end)


def _run_extent(x: Any, start: int, end: int):
    """The x-range to shade for the invalid samples ``x[start:end]``.

    A run covers half the gap to the neighbouring valid sample on each side, so
    a single bad sample still shades a visible width.  Taking ``x[start]`` to
    ``x[end - 1]`` instead would give a one-sample dropout zero width and hide
    exactly the case this is meant to reveal.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return None
    left = x[start] if start == 0 else 0.5 * (x[start - 1] + x[start])
    right = x[end - 1] if end >= x.size else 0.5 * (x[end - 1] + x[end])
    if right == left:
        return None
    return left, right

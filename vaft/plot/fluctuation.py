"""Plots of fluctuation-analysis results that are not a single IDS read (issue #1005).

Two figures a fluctuation study needs that the recipe registry does not
describe on its own:

* :func:`plot_cross_spectrum` -- coherence with its 95 % significance line
  and the relative phase, from a :class:`vaft.process.fluctuation.CrossSpectrum`.
  :func:`cross_spectrum_model` builds the :class:`~vaft.plot.models.Panels`
  it draws, and the canonical ``diagnostics_spectrum_coherence`` plot reuses
  the same model for two channels read from an ODS.
* :func:`plot_fluctuation_frequency_coverage` -- each diagnostic's Nyquist
  frequency on a log-frequency axis beside the typical frequency bands of
  common tokamak perturbations.  The bands are order-of-magnitude guidance,
  and the figure says so on its face: a frequency is not a mode
  identification.

``vaft.process`` is imported inside the functions, so importing
``vaft.plot`` loads no physics package.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .intent import palette, resolve_color
from .models import LineSeries, Panels, Series

__all__ = [
    "COVERAGE_CAPTION",
    "TYPICAL_PHENOMENON_BANDS",
    "cross_spectrum_model",
    "plot_cross_spectrum",
    "plot_fluctuation_frequency_coverage",
]

#: Typical observed frequency bands of common tokamak perturbations [Hz].
#: Order of magnitude only (issue #1005 section 4): the numbers depend on the
#: machine, the rotation and the mode, and a feature inside a band is not
#: thereby that phenomenon.  ``None`` marks a phenomenon with no single band.
TYPICAL_PHENOMENON_BANDS: dict[str, tuple[float, float] | None] = {
    "locked / slowly rotating MHD": (1.0, 3e3),
    "tearing mode / NTM": (1e3, 5e4),
    "IRE precursor (low-n MHD)": (1e3, 5e4),
    "fishbone / energetic-particle MHD": (1e4, 1e5),
    "Alfven eigenmodes": (3e4, 1e6),
    "IRE / disruption (broadband transient)": None,
}

#: The sentence the coverage figure carries under its axes.
COVERAGE_CAPTION = (
    "Phenomenon bands: typical order of magnitude, not an identification rule. "
    "Diagnostic bars end at the Nyquist frequency; the usable band is lower "
    "(sensor response, filtering, exposure, SNR, line integration)."
)


def cross_spectrum_model(
    result: Any,
    *,
    x_label: str = "x",
    y_label: str = "y",
    title: str | None = None,
    max_frequency: float | None = None,
) -> Panels:
    """Coherence (with the 95 % line) above phase, both against frequency.

    ``result`` is a :class:`vaft.process.fluctuation.CrossSpectrum`.  The phase
    is drawn in degrees, faint everywhere and marked where the coherence clears
    its 95 % level, because a phase is only meaningful where the two records are
    coherent.  ``max_frequency`` crops the drawn band; nothing is recomputed.
    """
    frequency = np.asarray(result.frequency, dtype=float)
    coherence = np.asarray(result.coherence, dtype=float)
    phase = np.degrees(np.asarray(result.phase, dtype=float))
    keep = np.ones(frequency.size, dtype=bool) if max_frequency is None else frequency <= float(max_frequency)
    frequency, coherence, phase = frequency[keep], coherence[keep], phase[keep]
    kilohertz = frequency / 1e3
    significance = float(result.significance_95)
    limits = (float(kilohertz[0]), float(kilohertz[-1])) if kilohertz.size > 1 else None

    coherence_series = [
        Series(x=kilohertz, y=coherence, label=f"{y_label} vs {x_label}",
               style={"color": palette(0), "linewidth": 1.2}),
    ]
    if np.isfinite(significance):
        coherence_series.append(Series(
            x=np.array([kilohertz[0], kilohertz[-1]]), y=np.array([significance, significance]),
            label=f"95 % significance ({result.n_segments} segments)",
            style={"color": "role:reference", "linestyle": "--", "linewidth": 1.0},
        ))
    coherent = coherence > significance if np.isfinite(significance) else np.zeros_like(coherence, dtype=bool)
    phase_x, phase_y = _phase_with_breaks(kilohertz, phase)
    phase_series = [
        Series(x=phase_x, y=phase_y, label="all frequencies",
               style={"color": "emphasis:faint", "linewidth": 0.8}),
    ]
    if coherent.any():
        phase_series.append(Series(
            x=kilohertz[coherent], y=phase[coherent], label="coherent above 95 %",
            style={"color": palette(0), "marker": "o", "markersize": 3, "linestyle": "none"},
        ))
    heading = title if title is not None else f"{y_label} relative to {x_label}"
    return Panels(
        models=(
            LineSeries(
                series=tuple(coherence_series), y_label="Coherence", y_unit="-",
                x_label="Frequency [kHz]", title=heading, x_limits=limits, y_limits=(0.0, 1.02),
            ),
            LineSeries(
                series=tuple(phase_series), y_label="Phase (y rel. x)", y_unit="deg",
                x_label="Frequency [kHz]", x_limits=limits, y_limits=(-185.0, 185.0),
            ),
        ),
        ncols=1,
        share_x=True,
    )


def _phase_with_breaks(x: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert a NaN where the wrapped phase jumps by more than 180 degrees, so no line crosses the plot."""
    if x.size < 2:
        return x, phase
    jumps = np.nonzero(np.abs(np.diff(phase)) > 180.0)[0] + 1
    return (
        np.insert(x.astype(float), jumps, np.nan),
        np.insert(phase.astype(float), jumps, np.nan),
    )


def plot_cross_spectrum(
    result: Any,
    *,
    x_label: str = "x",
    y_label: str = "y",
    title: str | None = None,
    max_frequency: float | None = None,
    ax: Any = None,
    show: bool = False,
    **style: Any,
) -> tuple[Any, Any]:
    """Draw a :class:`~vaft.process.fluctuation.CrossSpectrum`; returns ``(Figure, axes)``.

    Two panels: magnitude-squared coherence with the 95 % significance line, and
    the phase of ``y`` relative to ``x``.  ``ax`` may supply the two axes.
    """
    from .renderers.panels import render_panels

    model = cross_spectrum_model(
        result, x_label=x_label, y_label=y_label, title=title, max_frequency=max_frequency
    )
    return render_panels(model, ax=ax, show=show, **style)


def _nyquist(value: Any) -> float:
    if isinstance(value, (tuple, list, np.ndarray)):
        return float(value[1]) if len(value) > 1 else float(value[0]) / 2.0
    return float(value)


def plot_fluctuation_frequency_coverage(
    diagnostics: Mapping[str, Any] | None = None,
    phenomena: Mapping[str, tuple[float, float] | None] | None = None,
    *,
    f_min: float = 1.0,
    f_max: float = 2e6,
    caption: str | None = COVERAGE_CAPTION,
    ax: Any = None,
    show: bool = False,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Diagnostics' Nyquist frequencies against typical phenomenon bands, on log frequency.

    ``diagnostics`` maps a label to ``(sample_rate, nyquist)`` -- what
    :func:`vaft.omas.fluctuation_bandwidths` returns -- or to a Nyquist frequency
    alone, in hertz.  Each is drawn as a bar from ``f_min`` to its Nyquist
    frequency with the value written at the end.  ``phenomena`` maps a label to a
    ``(f_low, f_high)`` band in hertz, or ``None`` for a phenomenon with no single
    band (drawn hatched across the whole axis); the default is
    :data:`TYPICAL_PHENOMENON_BANDS`.  ``caption`` is printed under the axes; pass
    ``None`` to omit it -- but the bands are order-of-magnitude guidance and the
    figure should not be shown without that said somewhere.
    """
    from .style import finalize, resolve_axes

    diagnostics = dict(diagnostics or {})
    phenomena = dict(TYPICAL_PHENOMENON_BANDS if phenomena is None else phenomena)
    rows = len(diagnostics) + len(phenomena)
    if rows == 0:
        raise ValueError("nothing to draw: pass diagnostics= and/or phenomena=")
    height = 1.2 + 0.38 * rows + (0.5 if caption else 0.0)
    figure, axes = resolve_axes(ax, figsize=figsize or (8.5, height))

    labels: list[str] = []
    position = rows - 1
    phenomenon_colour = resolve_color("emphasis:low")
    for label, band in phenomena.items():
        if band is None:
            axes.barh(position, np.log10(f_max) - np.log10(f_min), left=np.log10(f_min), height=0.6,
                      color="none", edgecolor=phenomenon_colour, hatch="//", linewidth=0.8)
        else:
            low, high = float(band[0]), float(band[1])
            axes.barh(position, np.log10(high) - np.log10(low), left=np.log10(low), height=0.6,
                      color=phenomenon_colour, alpha=0.55, edgecolor=phenomenon_colour)
        labels.append(label)
        position -= 1
    if phenomena and diagnostics:
        axes.axhline(position + 0.5, color=resolve_color("emphasis:faint"), linewidth=0.8)
    for index, (label, value) in enumerate(diagnostics.items()):
        nyquist = _nyquist(value)
        colour = resolve_color(palette(index))
        axes.barh(position, np.log10(nyquist) - np.log10(f_min), left=np.log10(f_min), height=0.6,
                  color=colour, alpha=0.85)
        text = f"f_N = {nyquist / 1e3:.4g} kHz" if nyquist < 1e6 else f"f_N = {nyquist / 1e6:.4g} MHz"
        axes.text(np.log10(nyquist), position, f"  {text}", va="center", ha="left", fontsize=8)
        labels.append(label)
        position -= 1

    axes.set_yticks(np.arange(rows)[::-1])
    axes.set_yticklabels(labels)
    decades = np.arange(np.floor(np.log10(f_min)), np.ceil(np.log10(f_max)) + 1)
    axes.set_xticks(decades)
    axes.set_xticklabels([_decade_label(d) for d in decades])
    axes.set_xlim(np.log10(f_min), np.log10(f_max) + 0.9)
    axes.set_ylim(-0.7, rows - 0.3)
    # The caption rides on the axis label so a tight layout keeps room for it.
    xlabel = "Frequency"
    if caption:
        import textwrap

        xlabel += "\n\n" + "\n".join(textwrap.wrap(caption, 70))
    axes.set_xlabel(xlabel, fontsize=9 if caption else None)
    axes.grid(axis="x", alpha=0.3)
    axes.set_title("Fluctuation frequency coverage")
    return finalize(figure, axes, show=show, tight_layout=ax is None)


def _decade_label(exponent: float) -> str:
    value = 10.0 ** exponent
    if value >= 1e6:
        return f"{value / 1e6:g} MHz"
    if value >= 1e3:
        return f"{value / 1e3:g} kHz"
    return f"{value:g} Hz"

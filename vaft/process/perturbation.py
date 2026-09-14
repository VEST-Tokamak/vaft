"""Resonant response at the rational surfaces: windows, reductions, ratios.

A 3-D perturbation is felt at the surfaces where it resonates, and a code
like GPEC reports one row per rational surface: the resonant flux, the
shielding current, the island it would open, the Chirikov parameter. Reading
those rows is :mod:`vaft.code.gpec`'s job and it is done; this module is what
turns a table of them into the few numbers a study actually compares.

Nothing here is specific to a machine or to a code. The input is a radial
coordinate and a column of values; :func:`resonant_metrics` takes the mapping
:meth:`~vaft.code.gpec.GpecProfileOutput.resonant_table` returns because that
is the shape the data already has, not because the module knows about GPEC.

Two things this module is careful about, because the code it replaces was
not.

**Where a region starts and ends is not a constant.** The legacy metrics
hard-code the core at psi_n <= 0.8 and the edge at 0.8 to 0.95, and seven
different window sets are scattered across the notebooks that used them
(convention C-16). Decision D-05 settles it: the boundary comes from the
pedestal, through :func:`vaft.process.profile.pedestal_top`, and every
reduction records which window it used and where that window came from. The
legacy numbers survive only as :data:`LEGACY_WINDOWS`, so an old result can
still be reproduced deliberately.

**A reduction is not a verdict.** ``vaft.process`` computes; it does not
decide. The thresholds the legacy compared against -- a Chirikov parameter of
1, a penetration ratio of 1, an edge overlap of 7.4e-4 -- are arguments here
and defaults nowhere, and nothing in this module returns "stable". A study
that wants a verdict registers one in :mod:`vaft.validation`.

Units are SI throughout: the resonant flux is in tesla, as GPEC writes it,
and is *not* multiplied by 1e4. The legacy reported gauss, silently, inside
the reduction; a factor of ten thousand belongs to a display layer that says
which unit it is showing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "LEGACY_WINDOWS",
    "RESONANT_RESPONSE_COLUMNS",
    "RESONANT_STATISTICS",
    "ResonantWindow",
    "amplification_ratio",
    "resonant_metrics",
    "resonant_windows",
    "reduce_resonant",
    "rms_resonant_field",
]

#: The statistics a resonant column may be reduced with. ``rms`` is the
#: legacy's choice for the resonant field, ``max`` for the island and
#: overlap columns, where the worst surface is the one that matters, and
#: ``min`` for the critical width that is a ratio's denominator.
RESONANT_STATISTICS: tuple[str, ...] = ("rms", "max", "min", "mean", "sum")

#: The columns of a resonant table that are a *response* -- what the
#: perturbation did at each surface -- as opposed to where the surface is or
#: what the equilibrium was like there.
#:
#: :func:`resonant_metrics` reduces these and nothing else unless told
#: otherwise, and the distinction is not pedantry. A real GPEC table carries
#: twenty-five columns, and among them are ``rho_rational`` and
#: ``q1_rational``, which are coordinates, and ``T_e_rational`` and
#: ``n_e_rational``, which are the equilibrium profiles sampled at the
#: surfaces. Reducing those gives a number -- an RMS of a radial coordinate,
#: or an edge temperature of zero on a run that never filled the column --
#: that reads exactly like a resonant metric and is not one.
RESONANT_RESPONSE_COLUMNS: tuple[str, ...] = (
    "Phi_res", "Phi_res_v", "Phi_res_crit",
    "Delta", "B_pen", "I_res",
    "w_isl", "w_isl_v", "w_isl_v_crit",
    "K_isl", "K_isl_v",
)

#: The windows the legacy metrics hard-coded, kept so that an old number can
#: be reproduced on purpose. They are *not* a default: the boundary between
#: core and edge is a property of the plasma, and D-05 takes it from the
#: pedestal fit instead.
LEGACY_WINDOWS: Mapping[str, tuple[float, float]] = {
    "core": (0.0, 0.8),
    "edge": (0.8, 0.95),
    "total": (0.0, 1.0),
}


@dataclass(frozen=True)
class ResonantWindow:
    """A radial interval a reduction was taken over, and where it came from.

    ``source`` is the point: a window from a resolved pedestal fit and one
    from the 0.85 fallback are different claims about the same plasma, and a
    number reduced over either looks identical without it.
    """

    name: str
    low: float
    high: float
    source: str

    def mask(self, psi_norm) -> np.ndarray:
        """Which samples of ``psi_norm`` fall inside, endpoints included."""
        psi_norm = np.asarray(psi_norm, dtype=float)
        return (psi_norm >= self.low) & (psi_norm <= self.high)


def resonant_windows(pedestal=None, *, legacy: bool = False) -> dict[str, ResonantWindow]:
    """The core, edge and total radial windows a reduction is taken over.

    The core/edge boundary comes from the pedestal rather than from a
    constant. With a ``PedestalTop`` that resolved a fit, the edge begins at
    the pedestal's inner knee; with one that fell back, it begins at the
    fallback position, and the window says so.

    Parameters
    ----------
    pedestal : PedestalTop, optional
        The pedestal boundary, from :func:`vaft.process.profile.pedestal_top`.
        Omit it only together with ``legacy=True`` [-].
    legacy : bool, optional
        Return :data:`LEGACY_WINDOWS` instead, for reproducing a published
        number that used them [-].

    Returns
    -------
    dict of str to ResonantWindow
        Keyed ``core``, ``edge``, ``total``, each carrying its own bounds and
        the source they were derived from [-].

    Raises
    ------
    ValueError
        Neither a pedestal nor ``legacy=True`` was given, or the pedestal's
        coordinate is not ``psi_norm``.

    Defaults
    --------
    The 0.8 and 0.95 boundaries of :data:`LEGACY_WINDOWS` are a legacy
    compatibility value, not a physical one: they are what the metrics this
    module replaces hard-coded.

    Convention
    ----------
    Bounds are normalized poloidal flux, endpoints included at both ends, so
    a rational surface exactly at the boundary is counted in both neighbours
    rather than dropped by one.

    Applicability
    -------------
    Machine-independent. Any radial coordinate expressed as normalized
    poloidal flux.

    Provenance
    ----------
    .. [D-05] Migration decision D-05: the pedestal top is determined from an
       EPED-style profile fit, and core, edge and pedestal regions are derived
       from that result rather than fixed.
    .. [C-16] Conventions register C-16: seven different core/edge/pedestal
       window sets were in use across the code this replaces.
    """
    if legacy:
        return {
            name: ResonantWindow(name, low, high, "legacy fixed window")
            for name, (low, high) in LEGACY_WINDOWS.items()
        }
    if pedestal is None:
        raise ValueError(
            "resonant_windows needs a PedestalTop, or legacy=True to use the fixed "
            "0.8/0.95 windows the code this replaces hard-coded"
        )
    coordinate = getattr(pedestal, "coordinate", None)
    if coordinate is None:
        # Defaulting to psi_norm would turn "this object never said which
        # coordinate it is on" into "it is on psi_norm", which is the
        # conflation pedestal_top raises CoordinateUnavailableError to stop.
        raise ValueError(
            f"{type(pedestal).__name__} declares no radial coordinate; these "
            "windows are normalized poloidal flux and will not assume that of "
            "an object that did not say so"
        )
    if coordinate != "psi_norm":
        raise ValueError(
            f"the pedestal boundary is in {coordinate!r}; these windows are "
            "normalized poloidal flux, and converting between the two needs an "
            "equilibrium this layer does not have"
        )
    boundary = pedestal.inner_edge
    if boundary is None:
        boundary = float(pedestal.position)
        source = f"pedestal_top position ({pedestal.method})"
    else:
        source = f"pedestal_top inner edge ({pedestal.method})"
    if not np.isfinite(boundary) or not 0.0 <= boundary <= 1.0:
        # Out of range the windows still *build*, and one of them silently
        # becomes the whole plasma under another name: a boundary of 1.3 makes
        # "core" cover everything and "edge" cover nothing, and the caller
        # gets a finite, plausible number labelled edge.
        raise ValueError(
            f"the pedestal boundary is at {boundary!r}, outside the normalized "
            "flux range [0, 1]; a window built from it would relabel the whole "
            "plasma as one region rather than divide it"
        )
    if pedestal.reason:
        source += f": {pedestal.reason}"
    return {
        "core": ResonantWindow("core", 0.0, float(boundary), source),
        "edge": ResonantWindow("edge", float(boundary), 1.0, source),
        "total": ResonantWindow("total", 0.0, 1.0, "the whole plasma"),
    }


def reduce_resonant(psi_norm, values, *, window: ResonantWindow, statistic: str = "rms") -> float:
    """Reduce one resonant column over one radial window.

    Complex columns -- the resonant flux, the shielding current, the
    resonance parameter -- are reduced on their magnitude: a complex mean
    depends on a gauge the file does not fix, so averaging the phase would
    give an answer that changes with a convention rather than with the
    plasma. A **real** column keeps its sign, so ``mean`` of a rotation
    profile that changes sign is that mean and not its rectification; pass
    ``numpy.abs(values)`` to reduce a signed column on magnitude.

    Parameters
    ----------
    psi_norm : array_like
        Normalized poloidal flux of each rational surface [-].
    values : array_like
        One column of the resonant table, real or complex [any].
    window : ResonantWindow
        The interval to reduce over [-].
    statistic : str, optional
        One of :data:`RESONANT_STATISTICS` [n/a].

    Returns
    -------
    float
        The reduced value, in the column's own unit; ``nan`` when the window
        holds no surface [any].

    Raises
    ------
    ValueError
        ``psi_norm`` and ``values`` differ in length, or ``statistic`` is not
        one of :data:`RESONANT_STATISTICS`.

    Processing steps
    ----------------
    1. Take the magnitude of a complex column; leave a real one signed.
    2. Select the surfaces inside ``window``.
    3. Drop non-finite samples, then apply ``statistic``.

    Convention
    ----------
    A complex column reduces on its magnitude and a real one keeps its sign;
    the value comes back in the column's own unit, unscaled.

    Limitations
    -----------
    Returns ``nan`` rather than raising when the window is empty: a plasma
    with no rational surface in a region is a normal result, and a run with
    four rational surfaces has regions that are legitimately empty.

    Applicability
    -------------
    Machine-independent. Any per-rational-surface column.

    Provenance
    ----------
    .. [C-18] Conventions register C-18: quantities on the rational-surface
       index must not be averaged with quantities on the mode index, and a
       complex mean is gauge-dependent.
    """
    if statistic not in RESONANT_STATISTICS:
        raise ValueError(
            f"statistic must be one of {list(RESONANT_STATISTICS)}, not {statistic!r}"
        )
    psi_norm = np.asarray(psi_norm, dtype=float)
    column = np.asarray(values)
    # A complex column has no ordering, so it reduces on its magnitude. A real
    # one keeps its sign: rectifying it would turn the mean of a rotation
    # profile that changes sign into a number with no physical reading.
    magnitude = np.abs(column) if np.iscomplexobj(column) else column.astype(float)
    if magnitude.shape != psi_norm.shape:
        raise ValueError(
            f"{magnitude.shape} values against {psi_norm.shape} surfaces; a resonant "
            "column carries one entry per rational surface"
        )
    selected = magnitude[window.mask(psi_norm)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return float("nan")
    if statistic == "rms":
        return float(np.sqrt(np.mean(selected**2)))
    return float(getattr(np, statistic)(selected))


def rms_resonant_field(psi_norm, phi_res, *, window: ResonantWindow) -> float:
    """Root-mean-square resonant field over a radial window.

    Parameters
    ----------
    psi_norm : array_like
        Normalized poloidal flux of each rational surface [-].
    phi_res : array_like
        Resonant flux per surface, complex as GPEC writes it [T].
    window : ResonantWindow
        The interval to reduce over [-].

    Returns
    -------
    float
        RMS magnitude, in tesla; ``nan`` when the window holds no surface [T].

    Convention
    ----------
    Tesla, not gauss. GPEC normalizes this flux by the surface area, so it
    carries field units already; the code this replaces multiplied by 1e4
    inside the reduction, which put a unit conversion where nothing said so.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``gpec_analysis_metrics.compute_rms_resonant_field``, whose
       ``to_gauss=True`` default is the factor not applied here.
    """
    return reduce_resonant(psi_norm, phi_res, window=window, statistic="rms")


def resonant_metrics(
    table: Mapping[str, np.ndarray],
    *,
    windows: Mapping[str, ResonantWindow],
    columns: Sequence[str] | None = None,
    statistic: str = "rms",
) -> dict[tuple[str, str], float]:
    """Reduce every named column of a resonant table over every window.

    Parameters
    ----------
    table : mapping of str to ndarray
        A resonant table, as
        :meth:`vaft.code.gpec.GpecProfileOutput.resonant_table` returns it;
        it must carry ``psi_n_rational`` [any].
    windows : mapping of str to ResonantWindow
        The windows to reduce over, from :func:`resonant_windows` [-].
    columns : sequence of str, optional
        Which columns to reduce; those of :data:`RESONANT_RESPONSE_COLUMNS`
        the table carries, by default. Anything else -- a coordinate, or an
        equilibrium profile sampled at the surfaces -- has to be named, so
        that reducing it is a decision rather than a side effect [n/a].
    statistic : str, optional
        One of :data:`RESONANT_STATISTICS` [n/a].

    Returns
    -------
    dict
        Keyed ``(column, window)``, in the column's own unit [any].

    Raises
    ------
    KeyError
        The table carries no ``psi_n_rational``, no response column at all, or
        not one that ``columns`` named.

    Convention
    ----------
    Inherits :func:`reduce_resonant`'s -- magnitudes for complex columns,
    the column's own unit -- and the window convention of whatever
    :func:`resonant_windows` produced.

    Limitations
    -----------
    A reconstruction run carries rational surfaces and no perturbed response,
    and a kinetic run may carry no rational-surface block at all -- its table
    comes back empty. Both are refused here rather than reduced into something
    that looks like an answer.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``gpec_analysis_metrics.compute_metric_table``, which carried
       the same reductions with the windows and the gauss factor built in.
    """
    if "psi_n_rational" not in table:
        raise KeyError(
            "the table carries no 'psi_n_rational'; a resonant reduction needs the "
            f"coordinate its rows sit on, and this one has {sorted(table)}"
        )
    psi_norm = np.asarray(table["psi_n_rational"], dtype=float)
    if columns is None:
        columns = [name for name in RESONANT_RESPONSE_COLUMNS if name in table]
        if not columns:
            raise KeyError(
                "the table carries none of the resonant response columns "
                f"{list(RESONANT_RESPONSE_COLUMNS)}; it has {sorted(table)}, which "
                "are coordinates and equilibrium samples. A reconstruction run "
                "carries rational surfaces without a response, and reducing what it "
                "does have would return a plausible table holding no resonant physics"
            )
    missing = [name for name in columns if name not in table]
    if missing:
        raise KeyError(f"the table carries no {missing}; it has {sorted(table)}")
    return {
        (name, window_name): reduce_resonant(
            psi_norm, table[name], window=window, statistic=statistic
        )
        for name in columns
        for window_name, window in windows.items()
    }


def amplification_ratio(
    psi_norm,
    values,
    reference_psi_norm,
    reference_values,
    *,
    window: ResonantWindow,
    statistic: str = "rms",
) -> float:
    """How much larger one run's resonant response is than another's.

    Parameters
    ----------
    psi_norm : array_like
        Rational-surface coordinate of the case [-].
    values : array_like
        The case's resonant column [any].
    reference_psi_norm : array_like
        Rational-surface coordinate of the reference, usually the vacuum run
        [-].
    reference_values : array_like
        The reference's resonant column, in the same unit as ``values`` [any].
    window : ResonantWindow
        The interval both are reduced over [-].
    statistic : str, optional
        One of :data:`RESONANT_STATISTICS` [n/a].

    Returns
    -------
    float
        Case over reference; ``nan`` when either window is empty or the
        reference reduces to zero [-].

    Processing steps
    ----------------
    1. Reduce each run over ``window`` independently, so the two need not
       share a rational-surface set.
    2. Divide.

    Limitations
    -----------
    Both runs are reduced on their *own* surfaces. Two equilibria resonate at
    different places, and interpolating one onto the other's surfaces would
    invent a resonance where there is none.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``gpec_analysis.calculate_amplification_ratio``, which fell
       back to the vacuum column silently when the plasma one was absent --
       returning a ratio of one and calling it no amplification. Not carried
       over: an absent column raises here.
    """
    case = reduce_resonant(psi_norm, values, window=window, statistic=statistic)
    reference = reduce_resonant(
        reference_psi_norm, reference_values, window=window, statistic=statistic
    )
    if not np.isfinite(case) or not np.isfinite(reference) or reference == 0.0:
        return float("nan")
    return float(case / reference)

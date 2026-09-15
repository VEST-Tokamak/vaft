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

**An island width is not a threshold.** GPEC reports the island each
resonant surface would open, and whether those islands overlap decides
whether the region goes stochastic. The comparison it is made against --
the distance to the nearest neighbouring surface -- is pure geometry, and
GPEC leaves its own ``w_isl_v_crit`` at zero on an ideal run, so it has to be
computed. It is, here, and it reproduces GPEC's own Chirikov parameter
exactly: ``w_isl / d_nn`` equals the ``K_isl`` in the file to the last bit on
every surface of every run checked, which is what makes the derived
quantities trustworthy rather than merely plausible.

**A coupling matrix and a field have to be on one basis.** A GPEC run
reports five couplings -- flux, current, island width, penetrated flux and
Delta -- and writes them as ``C_f_x_out`` and its four siblings on the
*output* harmonic basis ``m_out``, which in the DIII-D example has 129
entries. The external field ``Phi_xe`` is on ``m``, which has 34. The one
array that pairs with it is ``C_xe``, the flux coupling on the field's own
basis -- its singular values agree with ``C_f_xe_out_singval`` to 1.4e-5.
:func:`edge_overlap_metric` refuses a mismatch rather than broadcasting one,
and nothing here maps a family name to a variable, because four of the five
have no counterpart on the basis the field lives on.

**Several mode numbers meet only at a rational surface.** A harmonic
``(m, n)`` varies as ``exp(i (m theta - n zeta))``, so on the surface
``q = m / n`` its phase is ``n (q theta - zeta)`` -- ``n`` times one angle,
whatever ``n`` is. Every mode resonating at one ``q`` is therefore a
harmonic of a single Fourier series, and :func:`composite_drive_at_q` sums
them as one. Away from such a surface they are unlike things and nothing
here adds them; Chirikov parameters in particular are not additive across
``n`` at all, which the legacy acknowledged by naming the column that added
them ``sum_K_isl_wrong``.

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
from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "align_surfaces_by_q",
    "AlignedSurface",
    "amplification_ratio",
    "chirikov",
    "COINCIDENT_POLICIES",
    "composite_drive_at_q",
    "CompositeDrive",
    "critical_island_width",
    "DELTA_E_COMBINATIONS",
    "edge_overlap_metric",
    "EdgeOverlap",
    "energy_norm_matrix",
    "group_coincident_islands",
    "helical_phase_sweep",
    "island_overlap_width",
    "island_pairs",
    "IslandOverlap",
    "IslandPairs",
    "LEGACY_WINDOWS",
    "nearest_surface_spacing",
    "NEGATIVE_EIGENVALUE_POLICIES",
    "penetration_ratio",
    "q_composite_table",
    "reduce_delta_e",
    "reduce_resonant",
    "resonant_metrics",
    "RESONANT_RESPONSE_COLUMNS",
    "RESONANT_STATISTICS",
    "resonant_windows",
    "ResonantWindow",
    "rms_resonant_field",
    "SURFACE_MEMBERSHIPS",
    "SurfaceMember",
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


# --- island geometry ----------------------------------------------------------

#: How to give one width to several islands that sit on the same surface.
#: ``envelope`` takes the widest, ``rss`` adds them in quadrature; neither is
#: derivable from the others, so the choice is the caller's and is recorded.
COINCIDENT_POLICIES: tuple[str, ...] = ("envelope", "rss")

#: How large a residual gap still counts as separatrices in contact, as a
#: fraction of the spacing between the two surfaces.
_CONTACT_TOLERANCE = 1.0e-12


@dataclass(frozen=True, eq=False)
class IslandPairs:
    """Adjacent island pairs, **sorted outward**, and whether each touches.

    Sorted, not in the caller's order: a pair is defined by two neighbours,
    and which two are neighbours is a property of the sorted sequence. The
    index arrays map each pair back to the rows it came from, so a caller
    with unsorted input can still say which helicities a pair joins.

    ``eq`` is off because the fields are arrays: a generated ``__eq__`` would
    raise rather than answer.
    """

    #: Index into the caller's arrays of the inner and outer member [-].
    inner_index: np.ndarray
    outer_index: np.ndarray
    #: Where the two islands sit [-].
    inner_psi_norm: np.ndarray
    outer_psi_norm: np.ndarray
    #: Distance between their centres [-].
    spacing: np.ndarray
    #: The separatrices that face each other [-].
    inner_right: np.ndarray
    outer_left: np.ndarray
    #: Distance between those separatrices; negative once they overlap [-].
    gap: np.ndarray
    #: Half-widths summed over the spacing; one at contact [-].
    chirikov: np.ndarray
    #: Whether the two islands touch or overlap [-].
    overlaps: np.ndarray

    def __len__(self) -> int:
        return int(self.spacing.size)


@dataclass(frozen=True)
class IslandOverlap:
    """How far in from the boundary the islands are continuously overlapping."""

    #: Radial extent of the region, inward from the separatrix; zero when the
    #: islands do not reach it [-].
    width: float
    #: The gap that ended the chain, and the two separatrices bounding it;
    #: ``None`` when no gap was reached [-].
    first_gap_psi_norm: float | None
    first_gap_inner: float | None
    first_gap_outer: float | None
    #: Whether the outermost island reaches the separatrix at all [-].
    edge_connected: bool
    #: Factor the field would have to grow by for the outermost pair to touch,
    #: on square-root scaling; ``None`` when that pair already touches or has
    #: no width [-].
    onset_scale: float | None
    #: The largest value of each Chirikov definition over the islands [-].
    max_surface_chirikov: float | None
    max_pair_chirikov: float | None
    #: What the result was measured against, and how stacks were combined [-].
    separatrix_psi_norm: float
    policy: str


def _island_arrays(psi_norm, width) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(sorted psi, sorted width, the order that sorted them)``, refusing ties."""
    psi_norm = np.asarray(psi_norm, dtype=float)
    width = np.asarray(width, dtype=float)
    if psi_norm.ndim != 1 or width.shape != psi_norm.shape:
        raise ValueError(
            f"{width.shape} widths against {psi_norm.shape} surfaces; islands are "
            "one per rational surface"
        )
    if not np.all(np.isfinite(psi_norm)):
        raise ValueError(
            "a rational surface is at a non-finite position; it would defeat the "
            "tie check and place an island nowhere"
        )
    if not np.all(np.isfinite(width)):
        # A NaN width silently breaks an overlap chain -- its gap is NaN, so
        # the pair reads as separated and the region ends there.
        raise ValueError("an island has a non-finite width")
    if np.any(width < 0.0):
        raise ValueError("an island has a negative width")
    order = np.argsort(psi_norm)
    ordered = psi_norm[order]
    if ordered.size > 1 and np.any(np.diff(ordered) <= 0.0):
        raise ValueError(
            "two rational surfaces share a position; islands of different helicity "
            "can sit on the same surface, and group_coincident_islands has to give "
            "them one width before the spacing between surfaces means anything"
        )
    return ordered, width[order], order


def nearest_surface_spacing(psi_norm) -> np.ndarray:
    """Distance from each rational surface to its nearest neighbour.

    Parameters
    ----------
    psi_norm : array_like
        Rational-surface positions, in any order [-].

    Returns
    -------
    ndarray
        One distance per surface, in the input's order; ``nan`` for a single
        surface, which has no neighbour [-].

    Raises
    ------
    ValueError
        Two surfaces share a position.

    Processing steps
    ----------------
    1. Sort the surfaces outward.
    2. Take the smaller of the two gaps either side; an end surface has one.
    3. Put the result back in the caller's order.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``gpec_analysis_metrics._nearest_neighbour_spacing`` and
       ``gpec_multimode_metrics.compute_surface_chirikov``, which computed the
       same spacing twice and disagreed about repeated positions: one
       tolerated them, the other refused. This refuses.
    """
    ordered, _, order = _island_arrays(psi_norm, np.zeros_like(np.asarray(psi_norm, float)))
    result = np.full(ordered.size, np.nan)
    if ordered.size < 2:
        return result
    gaps = np.diff(ordered)
    result[0], result[-1] = gaps[0], gaps[-1]
    if ordered.size > 2:
        result[1:-1] = np.minimum(gaps[:-1], gaps[1:])
    unsorted = np.empty_like(result)
    unsorted[order] = result
    return unsorted


def critical_island_width(psi_norm) -> np.ndarray:
    """The island width at which a surface's islands reach its neighbour's.

    Purely geometric: an island overlaps its neighbour when it is as wide as
    the distance to that neighbour, so the critical width *is* the
    nearest-neighbour spacing.

    Parameters
    ----------
    psi_norm : array_like
        Rational-surface positions [-].

    Returns
    -------
    ndarray
        Critical width per surface, in the same units as ``psi_norm`` [-].

    Convention
    ----------
    Widths are full widths, in normalized poloidal flux, matching what GPEC
    writes as ``w_isl``. The identity this rests on -- ``w_isl / w_crit``
    equalling GPEC's own ``K_isl`` -- holds because GPEC forms its Chirikov
    parameter from a half width over a half distance, and the halves cancel.

    Limitations
    -----------
    Undefined for a single surface, which has no neighbour: ``nan``.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [gpec] Reproduces GPEC's own ``K_isl`` exactly when divided into
       ``w_isl``: zero relative difference on every surface of the DIII-D
       n=1, n=2 and n=3 ideal runs.
    .. [ideal] GPEC leaves its own ``w_isl_v_crit`` at zero on an ideal run,
       which is why this is computed rather than read.
    """
    return nearest_surface_spacing(psi_norm)


def penetration_ratio(psi_norm, width) -> np.ndarray:
    """Island width as a fraction of the width at which islands would overlap.

    Parameters
    ----------
    psi_norm : array_like
        Rational-surface positions [-].
    width : array_like
        Full island width per surface, in the same units [-].

    Returns
    -------
    ndarray
        ``width / critical_island_width``; one at overlap [-].

    Raises
    ------
    ValueError
        ``width`` is not one per surface, two surfaces share a position, or a
        position or width is not finite or a width is negative.

    Convention
    ----------
    A ratio, not a verdict: the value one is where islands touch, and
    comparing against it is the caller's to do.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``gpec_analysis_metrics.compute_critical_island_width``.
    """
    _island_arrays(psi_norm, width)  # refuses a tie or a shape mismatch
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(width, dtype=float) / critical_island_width(psi_norm)


def chirikov(psi_norm, width, *, definition: str = "surface") -> np.ndarray:
    """The island-overlap parameter, by either of the two definitions in use.

    Parameters
    ----------
    psi_norm : array_like
        Rational-surface positions [-].
    width : array_like
        Full island width per surface [-].
    definition : str, optional
        ``"surface"`` centres it on each surface, ``width / d_nn``, and
        returns one value per surface. ``"pair"`` takes each adjacent pair,
        ``(w_inner + w_outer) / 2 / spacing``, and returns one value per
        *pair* -- one fewer than the surfaces [n/a].

    Returns
    -------
    ndarray
        For ``"surface"``, one value per surface in the caller's order. For
        ``"pair"``, one per adjacent pair **sorted outward** -- which two
        islands are neighbours is a property of the sorted sequence, and
        :class:`IslandPairs` carries the indices that map each pair back [-].

    Raises
    ------
    ValueError
        ``definition`` is neither, or two surfaces share a position.

    Defaults
    --------
    ``definition="surface"`` is a legacy compatibility value: it is the form
    GPEC itself reports as ``K_isl``, so it is what a caller comparing with a
    file will want. It also returns a different array length from ``"pair"``.

    Convention
    ----------
    The two definitions answer different questions and disagree by design:
    the surface form asks whether one island reaches its nearest neighbour,
    the pair form whether two particular islands reach each other. GPEC's own
    ``K_isl`` is the surface form.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics.compute_surface_chirikov`` and
       ``compute_pair_chirikov``.
    """
    if definition not in ("surface", "pair"):
        raise ValueError(f"definition must be 'surface' or 'pair', not {definition!r}")
    if definition == "surface":
        return penetration_ratio(psi_norm, width)
    return island_pairs(psi_norm, width).chirikov


def island_pairs(psi_norm, width) -> IslandPairs:
    """Every adjacent pair of islands, with the gap between their separatrices.

    Parameters
    ----------
    psi_norm : array_like
        Rational-surface positions [-].
    width : array_like
        Full island width per surface [-].

    Returns
    -------
    IslandPairs
        One entry per adjacent pair, sorted outward; empty for fewer than two
        surfaces [-].

    Raises
    ------
    ValueError
        ``width`` is not one per surface, two surfaces share a position, or a
        position or width is not finite or a width is negative.

    Processing steps
    ----------------
    1. Sort outward.
    2. For each pair, the gap is the spacing less the two half widths.
    3. A gap at or below zero -- within rounding -- is contact.

    Defaults
    --------
    The contact tolerance is a numerical convenience, 1e-12 of the spacing.

    Convention
    ----------
    Residual gaps below that fraction of the spacing count as contact rather
    than separation, so two separatrices that meet exactly are not reported
    as apart -- and "round-off" means the same thing at every radius, which a
    fixed absolute tolerance does not.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics.compute_pair_chirikov``.
    """
    ordered, ordered_width, order = _island_arrays(psi_norm, width)
    if ordered.size < 2:
        empty = np.empty(0, dtype=float)
        index = np.empty(0, dtype=int)
        return IslandPairs(index, index, empty, empty, empty, empty, empty,
                           empty, empty, np.empty(0, dtype=bool))
    spacing = np.diff(ordered)
    half_sum = 0.5 * (ordered_width[:-1] + ordered_width[1:])
    gap = spacing - half_sum
    return IslandPairs(
        inner_index=order[:-1],
        outer_index=order[1:],
        inner_psi_norm=ordered[:-1],
        outer_psi_norm=ordered[1:],
        spacing=spacing,
        inner_right=ordered[:-1] + 0.5 * ordered_width[:-1],
        outer_left=ordered[1:] - 0.5 * ordered_width[1:],
        gap=gap,
        chirikov=half_sum / spacing,
        # Scaled to the spacing, so "round-off" means the same thing whether
        # the surfaces are a thousandth apart or a whole unit. A fixed atol --
        # what the code this replaces used, with an rtol against zero that
        # could never contribute -- is a different criterion at every radius.
        overlaps=gap <= _CONTACT_TOLERANCE * spacing,
    )


def group_coincident_islands(
    psi_norm,
    width,
    *,
    policy: str = "envelope",
    psi_tol: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    """Collapse islands that sit on the same surface into one.

    Several toroidal modes resonate at the same q, so a multi-mode study has
    islands stacked on one surface. The spacing between *surfaces* means
    nothing until those are one island, and the width that one island should
    have is not derivable -- hence ``policy``.

    Parameters
    ----------
    psi_norm : array_like
        Island positions, possibly with repeats [-].
    width : array_like
        Full width of each [-].
    policy : str, optional
        One of :data:`COINCIDENT_POLICIES`: ``envelope`` takes the widest of a
        stack, ``rss`` adds them in quadrature [n/a].
    psi_tol : float, optional
        How close two positions must be to count as the same surface.
        Clustering is single-linkage: each island is compared with the last
        one added, so a chain of islands each within ``psi_tol`` of the
        previous collapses together however far the chain reaches [-].

    Returns
    -------
    tuple
        ``(psi_norm, width, members)`` -- one entry per distinct surface,
        sorted outward, with ``members`` giving the input indices that went
        into each [-].

    Raises
    ------
    ValueError
        ``policy`` is not one of :data:`COINCIDENT_POLICIES`, ``psi_tol`` is
        negative, or ``width`` is not one per surface.

    Defaults
    --------
    ``policy="envelope"`` is a numerical convenience -- the conservative of
    the two -- and ``psi_tol=1e-8`` is an empirical estimate: it separates
    the genuinely distinct surfaces of the runs looked at, whose closest
    non-coincident pair is 7.9e-13 apart, from the exactly coincident ones.
    Neither is supplied by the physics.

    Convention
    ----------
    ``envelope`` is the conservative reading -- a stack is as wide as its
    widest member -- and ``rss`` treats the modes as independent
    contributions. They differ by up to the square root of the stack size,
    and neither is a default the physics supplies, so the choice travels with
    the result.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics._group_coincident_islands``.
    """
    if policy not in COINCIDENT_POLICIES:
        raise ValueError(f"policy must be one of {list(COINCIDENT_POLICIES)}, not {policy!r}")
    if psi_tol < 0.0:
        raise ValueError(f"psi_tol must be non-negative, not {psi_tol!r}")
    psi_norm = np.asarray(psi_norm, dtype=float)
    width = np.asarray(width, dtype=float)
    if psi_norm.ndim != 1 or width.shape != psi_norm.shape:
        raise ValueError(f"{width.shape} widths against {psi_norm.shape} surfaces")
    if psi_norm.size == 0:
        return np.empty(0), np.empty(0), ()

    order = np.argsort(psi_norm)
    clusters: list[list[int]] = []
    for index in order:
        if clusters and abs(psi_norm[index] - psi_norm[clusters[-1][-1]]) <= psi_tol:
            clusters[-1].append(int(index))
        else:
            clusters.append([int(index)])

    centres, widths, members = [], [], []
    for cluster in clusters:
        stack = width[cluster]
        widths.append(float(np.max(stack)) if policy == "envelope"
                      else float(np.sqrt(np.sum(stack**2))))
        centres.append(float(np.mean(psi_norm[cluster])))
        members.append(tuple(cluster))
    return np.asarray(centres), np.asarray(widths), tuple(members)


def island_overlap_width(
    psi_norm,
    width,
    *,
    separatrix: float = 1.0,
    policy: str = "envelope",
    psi_tol: float = 1.0e-8,
) -> IslandOverlap:
    """How far in from the boundary the islands overlap without a break.

    Two conditions, and the first is what makes the quantity mean anything:
    the outermost island must actually reach ``separatrix``, and from there
    the chain is followed inward while adjacent pairs touch. The first gap is
    the inner edge of the region. Islands overlapping deep in the core are
    not an edge-connected layer however much they overlap, and a region of
    zero extent is not connected to anything.

    **This differs from the code it replaces**, which decided the question
    from whether the outermost *pair* overlapped and never looked at where
    the outermost island ended -- so a stochastic patch at psi_n = 0.2 came
    back as an edge layer 0.83 wide.

    Parameters
    ----------
    psi_norm : array_like
        Island positions [-].
    width : array_like
        Full island width per surface [-].
    separatrix : float, optional
        Where the boundary is, which the width is measured in from [-].
    policy : str, optional
        Passed to :func:`group_coincident_islands` [n/a].
    psi_tol : float, optional
        Passed to :func:`group_coincident_islands` [-].

    Returns
    -------
    IslandOverlap
        The width, the first gap that broke the chain, whether the chain
        reached the boundary at all, and the two Chirikov maxima [-].

    Raises
    ------
    ValueError
        ``separatrix`` is not finite, ``policy``/``psi_tol`` are invalid, or
        the islands do not pass :func:`island_pairs`' checks.

    Processing steps
    ----------------
    1. Collapse coincident surfaces under ``policy``.
    2. Form the adjacent pairs and their gaps.
    3. If the outermost island does not reach ``separatrix``, the width is
       zero and the region is not edge-connected.
    4. If the outermost pair does not touch, likewise.
    5. Otherwise walk inward to the first pair that does not touch; its gap
       midpoint is the inner boundary. With no such gap, the innermost
       island's inner separatrix is.

    Convention
    ----------
    Measured inward from ``separatrix`` and clipped to it: the result is a
    width measured from the boundary, so it cannot exceed the boundary's own
    position. A fixed upper bound of one -- what the code this replaces used
    -- reports a width larger than the plasma whenever the innermost island
    extends past the axis.

    Defaults
    --------
    ``separatrix = 1.0`` is a numerical convenience: the last closed flux
    surface in normalized poloidal flux. ``policy`` and ``psi_tol`` are
    passed through and documented on :func:`group_coincident_islands`.

    Limitations
    -----------
    ``onset_scale`` is the factor the perturbation field would have to be
    multiplied by for the outermost pair to reach contact, on the assumption
    that island width scales as the square root of the field. That assumption
    is fixed-equilibrium linear response and is recorded, not checked.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics.compute_island_overlap_width``.
    """
    if not np.isfinite(separatrix):
        raise ValueError(f"separatrix must be finite, not {separatrix!r}")
    centres, widths, _ = group_coincident_islands(
        psi_norm, width, policy=policy, psi_tol=psi_tol
    )
    surface = penetration_ratio(centres, widths) if centres.size else np.empty(0)
    pairs = island_pairs(centres, widths)

    def finite_max(values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(finite.max()) if finite.size else None

    max_surface = finite_max(surface)
    max_pair = finite_max(pairs.chirikov)
    separatrix = float(separatrix)

    def result(width_value, gap_psi, gap_inner, gap_outer, connected, onset):
        # A region of zero extent is not connected to anything, whatever the
        # geometry says about reaching: a separatrix that sits inside the
        # island stack would otherwise be reported as reached and empty.
        connected = bool(connected and width_value > 0.0)
        return IslandOverlap(width_value, gap_psi, gap_inner, gap_outer, connected,
                             onset, max_surface, max_pair, separatrix, policy)

    if centres.size == 0:
        return result(0.0, None, None, None, False, None)

    outer_edge = float(centres[-1] + 0.5 * widths[-1])
    # The question is whether the stochastic region touches the wall, so the
    # outermost island has to reach the separatrix. Deciding it from whether
    # the outermost *pair* overlap -- which is what the code this replaces did,
    # and what an earlier draft of this one did -- reports a stochastic patch
    # deep in the core as an edge-connected layer.
    reach = _CONTACT_TOLERANCE * max(abs(separatrix), 1.0)
    if outer_edge < separatrix - reach:
        return result(0.0, None, None, None, False, None)

    if len(pairs) == 0:
        # One island, and it reaches the wall: the region is that island.
        return result(
            float(np.clip(separatrix - (centres[0] - 0.5 * widths[0]), 0.0, separatrix)),
            None, None, None, True, None,
        )

    outer_kappa = float(pairs.chirikov[-1])
    # Island width goes as the square root of the field, so the field has to
    # grow by 1/kappa^2 for the outermost pair to reach contact.
    onset = float(1.0 / outer_kappa**2) if outer_kappa > 0.0 else None
    if not bool(pairs.overlaps[-1]):
        inner, outer = float(pairs.inner_right[-1]), float(pairs.outer_left[-1])
        return result(0.0, 0.5 * (inner + outer), inner, outer, False, onset)

    broken = np.nonzero(~pairs.overlaps)[0]
    if broken.size:
        index = int(broken[-1])
        inner, outer = float(pairs.inner_right[index]), float(pairs.outer_left[index])
        gap_psi = 0.5 * (inner + outer)
        boundary = gap_psi
    else:
        inner = outer = gap_psi = None
        boundary = float(centres[0] - 0.5 * widths[0])
    # Clipped to the separatrix, not to one: the result is a width measured
    # inward from it, so it cannot exceed it, and a fixed upper bound of one
    # would report a width larger than the plasma it was measured in.
    return result(
        float(np.clip(separatrix - boundary, 0.0, abs(separatrix))),
        gap_psi, inner, outer, True, onset,
    )


# --- singular coupling and the edge overlap -----------------------------------

#: What to do with a negative eigenvalue when building an energy norm. The
#: code this replaces had three constructions with three different answers
#: and no way to tell which had been used.
NEGATIVE_EIGENVALUE_POLICIES: tuple[str, ...] = ("raise", "abs", "drop")


@dataclass(frozen=True, eq=False)
class EdgeOverlap:
    """How much of an external field drives the dominant coupling mode."""

    #: The metric: the dominant mode's share of the field, per unit field [-].
    delta_e: float
    #: How much field each singular mode carries, as a magnitude [T].
    #: Magnitudes, not complex amplitudes: the phase of a projection onto a
    #: singular vector is a gauge, and returning it invites a caller to use
    #: a number that changes with the decomposition rather than the plasma.
    projection: np.ndarray
    #: The coupling matrix's singular values, strongest first [-].
    singular_values: np.ndarray
    #: Index of the mode carrying the most field, into ``projection`` [-].
    dominant_mode: int
    #: What the projection was divided by [T].
    b_t0: float


def energy_norm_matrix(
    eigenvalues,
    eigenvectors,
    *,
    negative_eigenvalues: str,
) -> np.ndarray:
    """The inverse square root of an energy matrix, from its eigendecomposition.

    Parameters
    ----------
    eigenvalues : array_like
        Eigenvalues of the energy matrix [-].
    eigenvectors : array_like
        Its eigenvectors, one per row, matching ``eigenvalues`` [-].
    negative_eigenvalues : str
        What to do with an eigenvalue at or below zero, one of
        :data:`NEGATIVE_EIGENVALUE_POLICIES`. Required: the three
        constructions this replaces answered it three ways, and the answer
        changes the metric [n/a].

    Returns
    -------
    ndarray
        ``W^{-1/2}``, square and Hermitian [-].

    Raises
    ------
    ValueError
        ``negative_eigenvalues`` is not one of the three; the shapes disagree
        or ``eigenvectors`` is not square; an eigenvalue is complex or not
        finite; ``"raise"`` was chosen and an eigenvalue is not positive; or
        the policy left nothing.

    Processing steps
    ----------------
    1. Apply the policy to the eigenvalues.
    2. Form ``V^H diag(lambda^{-1/2}) V`` over what survives.

    Convention
    ----------
    Rows of ``eigenvectors`` are the eigenvectors, so the reconstruction is
    ``V^H diag(lambda) V``. That is this function's contract, not a claim
    about any file: see the Limitations.

    Limitations
    -----------
    **No run in reach satisfies this contract**, so it is unvalidated against
    real data. The ideal examples carry ``W_xe`` as identically zero, and
    every policy then refuses. The kinetic example carries a non-zero one,
    but that matrix is not Hermitian at all -- ``max|W - W^H|`` is 1.8 times
    ``max|W|`` -- and its own eigenvector array reconstructs it under none of
    the four orientations, the closest being ``V^T diag V*`` against the
    Hermitian part at a relative error of 1.0. Feeding GPEC's own pair in
    gives a matrix ``N`` with ``max|N W N - I|`` of order ten. Whatever
    ``W_xe`` and ``W_xe_eigenvector`` are to each other, it is not an
    eigendecomposition in the sense used here. Establish that before norming
    a coupling matrix with a file's own arrays.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [C-11] Conventions register C-11: three different ``W^{-1/2}``
       constructions, with different negative-eigenvalue policies, fed the
       same metric.
    """
    if negative_eigenvalues not in NEGATIVE_EIGENVALUE_POLICIES:
        raise ValueError(
            f"negative_eigenvalues must be one of "
            f"{list(NEGATIVE_EIGENVALUE_POLICIES)}, not {negative_eigenvalues!r}"
        )
    values = np.asarray(eigenvalues)
    if np.iscomplexobj(values):
        # float(complex) discards the imaginary part with a warning, which
        # would answer for an input this cannot honour. numpy.linalg.eig
        # returns a complex dtype even for a real spectrum, so a caller can
        # reach here with one by accident.
        if not np.allclose(values.imag, 0.0):
            raise ValueError(
                "the eigenvalues are complex; an energy matrix with a complex "
                "spectrum has no real inverse square root"
            )
        values = values.real
    values = values.astype(float)
    vectors = np.asarray(eigenvectors)
    if vectors.ndim != 2 or vectors.shape[0] != values.size:
        raise ValueError(
            f"{vectors.shape} eigenvectors against {values.size} eigenvalues; one "
            "row per eigenvalue"
        )
    if vectors.shape[0] != vectors.shape[1]:
        raise ValueError(
            f"{vectors.shape} eigenvectors; a complete set is square, and an "
            "incomplete one gives a norm of a dimension the caller did not ask for"
        )
    if not np.all(np.isfinite(values)):
        # A NaN is not <= 0, so it would slip past every policy and give an
        # all-NaN norm; an inf gives lambda**-0.5 = 0, dropping a mode in
        # silence.
        raise ValueError("an eigenvalue is not finite")
    keep = np.ones(values.size, dtype=bool)
    if np.any(values <= 0.0):
        if negative_eigenvalues == "raise":
            raise ValueError(
                f"{int(np.count_nonzero(values <= 0.0))} of {values.size} eigenvalues "
                "are not positive, so the matrix has no real inverse square root"
            )
        if negative_eigenvalues == "abs":
            values = np.abs(values)
            keep = values > 0.0
        else:
            keep = values > 0.0
    if not np.any(keep):
        raise ValueError("no positive eigenvalue survives; there is nothing to norm with")
    vectors, values = vectors[keep], values[keep]
    return vectors.conj().T @ np.diag(values ** -0.5) @ vectors


def edge_overlap_metric(
    coupling,
    external_field,
    *,
    b_t0: float,
    norm=None,
) -> EdgeOverlap:
    """How much of an external field lands on the dominant coupling mode.

    The coupling matrix is decomposed, the field is projected onto its
    singular modes, and the dominant mode's share is reported per unit field.

    Parameters
    ----------
    coupling : array_like
        Coupling matrix, modes by harmonics, complex [-].
    external_field : array_like
        The external field on the same harmonic basis, complex [T].
    b_t0 : float
        Vacuum toroidal field on axis, which the projection is divided by.
        Required: the code this replaces defaulted it to one, which silently
        reports a field in tesla as a dimensionless metric. A GPEC run
        carries it as a global attribute [T].
    norm : array_like, optional
        An energy norm to apply to ``coupling`` first, from
        :func:`energy_norm_matrix`. Omit it when the matrix is already
        normed -- GPEC's own ``C_xe`` is [-].

    Returns
    -------
    EdgeOverlap
        The metric, the per-mode projection and the singular values [-].

    Raises
    ------
    ValueError
        The shapes disagree, the coupling matrix or the field holds a
        non-finite entry, the coupling matrix is empty or identically zero,
        or ``b_t0`` is not a positive finite real number.

    Processing steps
    ----------------
    1. Apply ``norm`` to the coupling matrix when one is given.
    2. Take its singular value decomposition.
    3. Project the field onto the conjugated right singular vectors.
    4. Divide the largest projection by ``b_t0``.

    Convention
    ----------
    The projection is reported as a magnitude. A right singular vector is
    fixed only up to a phase, so the complex overlap is a gauge -- GPEC's own
    vectors and a fresh decomposition of the same matrix differ by exactly
    that phase -- and a caller handed the complex number would be reading the
    decomposition rather than the plasma. The magnitude is not affected. The
    dominant mode is the one carrying the most field, which need not be the
    one with the largest singular value.

    Limitations
    -----------
    A coupling matrix that a run never filled decomposes to zero singular
    values and a projection that is just the field's own components, which
    would report a confident metric for a run that computed nothing. An
    all-zero matrix is refused for that reason; a merely tiny one is not,
    and a run whose external field is at the level of floating-point residue
    -- the n=2 ideal example has a field norm of 1e-19 -- gives a ratio of
    two residues.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [C-25] Conventions register C-25: the dominant singular vector needs
       explicit gauge fixing or the complex overlap phase is arbitrary.
    .. [gpec] Reproduces GPEC's own ``O_CPhi_xe`` to 3e-16 and its
       ``Phi_overlap_norm`` to 4e-5 on the DIII-D n=1, n=2 and n=3 ideal runs.
    """
    matrix = np.asarray(coupling)
    field = np.asarray(external_field)
    if matrix.ndim != 2:
        raise ValueError(f"the coupling matrix is {matrix.ndim}-dimensional")
    if norm is not None:
        norm = np.asarray(norm)
        if norm.shape != (matrix.shape[1], matrix.shape[1]):
            raise ValueError(
                f"the norm is {norm.shape} against a coupling matrix with "
                f"{matrix.shape[1]} harmonics"
            )
        matrix = matrix @ norm
    if field.shape != (matrix.shape[1],):
        raise ValueError(
            f"the field has {field.shape} entries against the coupling matrix's "
            f"{matrix.shape[1]} harmonics; they must be on one basis"
        )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(field)):
        # numpy's SVD raises LinAlgError on a NaN, which is not the ValueError
        # this documents, and an inf decomposes to a confident wrong answer.
        raise ValueError("the coupling matrix or the field holds a non-finite entry")
    if matrix.size == 0 or not np.any(matrix):
        # The zero matrix decomposes with V = I, so the projection is just the
        # field's own components and delta_e comes out plausible for a run
        # that computed no coupling at all.
        raise ValueError(
            "the coupling matrix is empty or identically zero, so it has no "
            "singular directions; a run that never filled it cannot be given an "
            "overlap metric"
        )
    if not isinstance(b_t0, (int, float, np.integer, np.floating)):
        raise ValueError(f"b_t0 must be a real number, not {type(b_t0).__name__}")
    if not np.isfinite(b_t0) or b_t0 <= 0.0:
        raise ValueError(f"b_t0 must be positive and finite, not {b_t0!r}")

    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    # numpy returns V^H, so row i is already the conjugate transpose of the
    # right singular vector: the projection onto it is that row times the
    # field. Conjugating again gives v^T f, a bilinear form and not a
    # projection onto anything -- and it disagrees with GPEC by up to 56%.
    projection = np.abs(right @ field)
    dominant = int(np.argmax(projection))
    return EdgeOverlap(
        delta_e=float(projection[dominant] / b_t0),
        projection=projection,
        singular_values=singular_values,
        dominant_mode=dominant,
        b_t0=float(b_t0),
    )


# --------------------------------------------------------------------------
# Several toroidal mode numbers at once
# --------------------------------------------------------------------------

SURFACE_MEMBERSHIPS: tuple[str, ...] = ("resonant", "non_resonant", "absent")
"""Why a mode does or does not appear at an aligned surface.

``resonant``
    The run reports a surface there.
``non_resonant``
    ``n * q`` is not an integer, so this mode has no rational surface at that
    ``q`` and cannot have one. Decidable from arithmetic alone.
``absent``
    ``n * q`` *is* an integer, so the surface exists, but the run does not
    report it -- it fell outside what that run resolved. GPEC truncates at
    its own ``qlim``, which differs between modes of one equilibrium: 5.2,
    5.1 and 5.066667 for n = 1, 2 and 3 of the DIII-D reference. Those three
    limits happen to produce no ``absent`` row on that reference, because
    every surface they exclude sits above the highest ``q`` any of the three
    reaches; the case is real but the reference does not exhibit it.
"""

DELTA_E_COMBINATIONS: tuple[str, ...] = ("quadrature", "linear")
"""How :func:`reduce_delta_e` may combine one metric across mode numbers."""


@dataclass(frozen=True)
class SurfaceMember:
    """One toroidal mode's standing at one aligned rational surface."""

    n: int
    membership: str
    m: int | None = None
    index: int | None = None
    psi_norm: float | None = None


@dataclass(frozen=True)
class AlignedSurface:
    """One rational ``q``, and where each mode stands on it."""

    q: float
    members: tuple[SurfaceMember, ...]

    @property
    def resonant_modes(self) -> tuple[int, ...]:
        """The mode numbers that actually have a surface here."""
        return tuple(x.n for x in self.members if x.membership == "resonant")


@dataclass(frozen=True)
class CompositeDrive:
    """A drive summed over the modes that share one rational surface."""

    q: float
    modes: tuple[int, ...]
    contributions: tuple[complex, ...]
    peak: float
    peak_angle: float
    quadrature: float
    linear_bound: float


def _contributions_at_q(
    tables: Mapping[int, Mapping[str, object]],
    q: float,
    column: str,
    *,
    weights: Mapping[int, complex] | None,
    integrality_tolerance: float,
) -> tuple[tuple[int, ...], np.ndarray]:
    """The weighted contributions of every mode resonating at ``q``.

    Shared by :func:`composite_drive_at_q` and :func:`helical_phase_sweep` so
    that the alignment is done once for both.
    """
    if weights:
        unknown = sorted(set(weights) - set(tables))
        if unknown:
            raise ValueError(f"weights name modes that are not in tables: {unknown}")
    if not np.isfinite(q):
        raise ValueError(f"q must be finite, not {q!r}")
    aligned = align_surfaces_by_q(tables, integrality_tolerance=integrality_tolerance)
    # Match on the rational the caller's q rounds to, not on float equality:
    # a q taken straight out of one of these tables can sit an ulp away from
    # the value align_surfaces_by_q reports back.
    wanted = {
        _rational_q(mode, float(q), tolerance=integrality_tolerance) for mode in tables
    } - {None}
    match = next((s for s in aligned if Fraction(s.q).limit_denominator(10**9) in wanted), None)
    if match is None:
        match = next((s for s in aligned if s.q == float(q)), None)
    modes = () if match is None else match.resonant_modes
    if not modes:
        raise ValueError(
            f"no mode in {sorted(tables)} resonates at q = {q!r}; the surfaces "
            f"present are {[s.q for s in aligned]}"
        )
    contributions = []
    for member in match.members:
        if member.membership != "resonant":
            continue
        table = tables[member.n]
        if column not in table:
            raise ValueError(f"n = {member.n} resonates at q = {q!r} but has no {column!r} column")
        values = np.asarray(table[column])
        if values.ndim != 1:
            raise ValueError(
                f"n = {member.n} gives {column!r} with shape {values.shape}; one "
                "value per surface is expected, so a netCDF (2, N) real/imaginary "
                "pair has to be assembled into a complex column first"
            )
        if values.size != np.atleast_1d(np.asarray(table["q_rational"])).size:
            raise ValueError(
                f"n = {member.n} gives {values.size} values of {column!r} against "
                f"{np.atleast_1d(np.asarray(table['q_rational'])).size} surfaces"
            )
        value = complex(values[member.index])
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"n = {member.n} has a non-finite {column!r} at q = {q!r}: {value!r}")
        weight = complex(1.0) if not weights else complex(weights.get(member.n, 1.0))
        if not np.isfinite(weight.real) or not np.isfinite(weight.imag):
            raise ValueError(f"the weight for n = {member.n} is not finite: {weight!r}")
        contributions.append(value * weight)
    return tuple(modes), np.asarray(contributions)


def _rational_q(n: int, q: float, *, tolerance: float) -> Fraction | None:
    """``q`` as the fraction ``m / n``, or ``None`` when ``n * q`` is not an
    integer to within ``tolerance``."""
    product = n * q
    nearest = round(product)
    if abs(product - nearest) > tolerance:
        return None
    return Fraction(int(nearest), int(n))


def align_surfaces_by_q(
    tables: Mapping[int, Mapping[str, object]],
    *,
    integrality_tolerance: float = 1e-6,
) -> tuple[AlignedSurface, ...]:
    """Match rational surfaces across toroidal mode numbers by their ``q``.

    Parameters
    ----------
    tables : mapping of int to mapping
        One resonant table per toroidal mode number, each carrying at least a
        ``q_rational`` column -- the shape
        :meth:`~vaft.code.gpec.GpecProfileOutput.resonant_table` returns.
        ``psi_n_rational`` is recorded when present [n/a].
    integrality_tolerance : float, optional
        How far ``n * q`` may sit from an integer and still count as a
        rational surface of that mode [-].

    Returns
    -------
    tuple of AlignedSurface
        Every ``q`` at which any mode resonates, ascending, each carrying one
        :class:`SurfaceMember` per mode in the input [n/a].

    Raises
    ------
    ValueError
        A mode number is not a positive integer, a table has no
        ``q_rational``, a ``q`` is not finite, or ``n * q`` is not an integer
        for a surface the run itself reports.

    Applicability
    -------------
    Machine-independent. Nothing here reads a machine, a code or a file; the
    input is a column of ``q`` per mode number.

    Convention
    ----------
    Surfaces are matched on the **exact rational** ``m / n``, not on a
    tolerance around a float. On the DIII-D reference ``q_rational`` is
    exactly ``m / n`` in double precision for every surface of every mode,
    and IEEE division is correctly rounded, so two fractions that reduce to
    the same rational give the same double -- ``4/3`` and ``8/6`` compare
    equal. The positions do not: at ``q = 2`` the three runs put
    ``psi_n_rational`` at 0.59364383816324717, 0.59364383816324717 and
    0.59364383816403732, first differing in the twelfth significant digit.
    ``q`` is the key because it is exact; ``psi`` is the root finder's answer
    and is not.

    Defaults
    --------
    ``integrality_tolerance = 1e-6`` is a numerical convenience: loose
    enough for a ``q`` that has been through a file format. Rounding to the
    nearest integer is what decides which rational a value belongs to, so
    two rationals cannot be confused at any mode number whatever the
    tolerance is; it only decides how far from *any* of them a value may sit
    before it is refused, and being absolute on ``n * q`` it tightens as
    ``1e-6 / n`` on ``q`` itself. The match is exact -- this is not a
    matching tolerance.

    Processing steps
    ----------------
    1. Turn each reported ``q`` into the fraction ``m / n`` and refuse a
       surface whose ``n * q`` is not integral -- a run reporting one is
       reporting something that is not a rational surface.
    2. Collect the union of those fractions across modes, ascending.
    3. For every mode at every ``q``, record ``resonant`` when the run
       reports a surface, ``non_resonant`` when ``n * q`` is not an integer,
       and ``absent`` when it is but the run reports nothing.

    Limitations
    -----------
    ``absent`` and ``resonant`` are statements about one run's radial extent,
    not about the plasma: a surface the run truncated away is real. Nothing
    here reads ``qlim``, so ``absent`` cannot distinguish a truncated run
    from a failed root find.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics.align_surfaces_by_q``, which matched
       the single nearest surface within ``q_tol = 0.05`` of four hard-coded
       targets ``(2, 3, 4, 5)``. Three departures. The window is absolute, so
       it admits more than one rational once the spacing ``1 / n`` falls
       below it -- at n = 20 exactly, where it silently keeps the nearest and
       drops the rest. A mode out of tolerance was dropped from the group
       with no record, which is indistinguishable here from a mode that
       cannot resonate at that ``q`` at all. And its tolerance test was
       ``abs(q - target) > q_tol``, which is ``False`` for a ``nan``, so a
       ``nan`` ``q`` was returned as a match.
    """
    if not tables:
        return ()
    per_mode: dict[int, dict[Fraction, tuple[int, float | None]]] = {}
    for mode, table in tables.items():
        if not isinstance(mode, (int, np.integer)) or isinstance(mode, bool) or int(mode) <= 0:
            raise ValueError(f"toroidal mode numbers are positive integers, not {mode!r}")
        mode = int(mode)
        if "q_rational" not in table:
            raise ValueError(f"the table for n = {mode} has no q_rational column")
        q_values = np.atleast_1d(np.asarray(table["q_rational"], dtype=float))
        psi_values = None
        if "psi_n_rational" in table:
            psi_values = np.atleast_1d(np.asarray(table["psi_n_rational"], dtype=float))
            if psi_values.size != q_values.size:
                raise ValueError(
                    f"n = {mode} reports {q_values.size} values of q against "
                    f"{psi_values.size} positions"
                )
        found: dict[Fraction, tuple[int, float | None]] = {}
        for index, q in enumerate(q_values):
            if not np.isfinite(q):
                raise ValueError(f"n = {mode} surface {index} has q = {q}")
            if q <= 0.0:
                raise ValueError(
                    f"n = {mode} surface {index} has q = {q!r}; a safety factor "
                    "is positive, and q = 0 is integral for every mode number, "
                    "so admitting it would mark every mode resonant there"
                )
            fraction = _rational_q(mode, float(q), tolerance=integrality_tolerance)
            if fraction is None:
                raise ValueError(
                    f"n = {mode} reports a surface at q = {q!r}, where n * q = "
                    f"{mode * q!r} is not an integer, so it is not a rational surface"
                )
            if fraction in found:
                raise ValueError(
                    f"n = {mode} reports two surfaces at q = {q!r}, at indices "
                    f"{found[fraction][0]} and {index}. A reversed-shear q "
                    "profile really does resonate twice at one q, and one "
                    "SurfaceMember per mode cannot hold both; aligning such a "
                    "run needs a model this does not have"
                )
            found[fraction] = (index, None if psi_values is None else float(psi_values[index]))
        per_mode[mode] = found

    surfaces = []
    for fraction in sorted(set().union(*(set(f) for f in per_mode.values()))):
        members = []
        for mode in sorted(per_mode):
            m_times_n = fraction * mode
            if m_times_n.denominator != 1:
                members.append(SurfaceMember(n=mode, membership="non_resonant"))
                continue
            hit = per_mode[mode].get(fraction)
            if hit is None:
                members.append(SurfaceMember(n=mode, membership="absent", m=int(m_times_n)))
                continue
            index, psi = hit
            members.append(
                SurfaceMember(
                    n=mode, membership="resonant", m=int(m_times_n), index=index, psi_norm=psi
                )
            )
        surfaces.append(AlignedSurface(q=float(fraction), members=tuple(members)))
    return tuple(surfaces)


def composite_drive_at_q(
    tables: Mapping[int, Mapping[str, object]],
    q: float,
    column: str,
    *,
    weights: Mapping[int, complex] | None = None,
    integrality_tolerance: float = 1e-6,
    angle_points: int | None = None,
) -> CompositeDrive:
    """Sum one resonant column over every mode that resonates at a given ``q``.

    Parameters
    ----------
    tables : mapping of int to mapping
        One resonant table per toroidal mode number, as
        :func:`align_surfaces_by_q` takes [n/a].
    q : float
        The rational surface to sum at [-].
    column : str
        The column to sum. Complex is the point -- a magnitude carries no
        phase to interfere with [n/a].
    weights : mapping of int to complex, optional
        Per-mode complex weight, applied before the sum. A unit-modulus
        weight rotates a mode's phase; a real one rescales its amplitude
        [n/a].
    integrality_tolerance : float, optional
        Passed to :func:`align_surfaces_by_q` [-].
    angle_points : int, optional
        How finely the helical angle is sampled before the peak is refined.
        Left unset it is sized from the highest resonating mode number;
        given, it must be at least ``16 * n_max + 1`` [-].

    Returns
    -------
    CompositeDrive
        ``peak`` and ``peak_angle`` are the largest real amplitude over the
        helical angle and where it occurs; ``quadrature`` is
        ``sqrt(sum |z|**2)``; ``linear_bound`` is ``sum |z|``, the value the
        peak would reach if every mode aligned. ``peak``, ``quadrature`` and
        ``linear_bound`` carry ``column``'s own units and ``peak_angle`` is
        an angle [rad].

    Raises
    ------
    ValueError
        No mode resonates at ``q``, the column is missing from a resonating
        mode's table, a contribution is not finite, ``angle_points`` is less
        than 2, or a weight names a mode that is not in ``tables``.

    Applicability
    -------------
    Machine-independent. The helical angle exists wherever ``q = m / n``
    does, which is every tokamak.

    Convention
    ----------
    **The modes that share a rational surface also share a helical angle,
    and that is what makes this sum well defined.** A harmonic ``(m, n)``
    varies as ``exp(i (m theta - n zeta))``, and on the surface ``q = m / n``
    that phase is ``n (q theta - zeta)`` -- ``n`` times the single angle
    ``u = q theta - zeta``, whatever ``n`` is. So every mode resonating at
    one ``q`` is a harmonic of one Fourier series in ``u``, the ``n``-th, and
    summing them is ordinary Fourier synthesis rather than an addition of
    unlike things. Away from a rational surface it would be neither.

    The origin of ``u`` is a gauge: shifting it by ``d`` takes ``z_n`` to
    ``z_n exp(i n d)``, so the amplitude at a *particular* angle means
    nothing unless the caller fixed that origin -- through the coil geometry,
    say. ``peak`` is invariant under the shift and is the number to quote;
    ``peak_angle`` is reported relative to the input's own origin and moves
    with it. ``quadrature`` and ``linear_bound`` are invariant too.

    The decomposition assumed is ``exp(i (m theta - n phi))``, which is
    GPEC's, and its spectral ``Phi_res`` is written without the
    ``-helicity`` flip its real-space outputs carry, so no helicity factor
    enters the sum. **What helicity does reach is** ``peak_angle``: GPEC's
    ``phi`` runs counter-clockwise for a left-handed plasma and clockwise
    for a right-handed one, so on a ``helicity = +1`` run the same spectral
    convention makes ``u = q theta + phi`` in machine coordinates and the
    reported angle runs the other way. ``peak``, ``quadrature`` and
    ``linear_bound`` are untouched -- conjugating every ``z_n`` mirrors
    ``u -> -u``, and a mirrored curve has the same height.

    The amplitude summed is ``Re(z exp(i n u))`` and not twice it. A real
    field reconstructed from positive ``n`` alone carries the conjugate
    harmonics too, so a comparison against a real-space output will differ
    by a factor of two; this follows the convention the legacy's own
    artefact records rather than that one.

    Defaults
    --------
    ``angle_points`` unset is a numerical convenience: ``16 * n_max + 1``
    points, sixteen per period of the top harmonic, but never fewer than
    721. The size has to follow ``n_max`` because the sum is a trigonometric
    polynomial of that degree and has up to ``2 n_max`` maxima. A fixed
    721-point grid was measured against a two-million-point scan on 400
    random cases per row: it already missed the peak in 2 of 400 at
    ``n_max = 20`` -- worst deficit 1.5e-3, on the two-mode case (14, 19) --
    and in 281 of 400 at ``n_max = 400``, worst deficit 0.20. Sized from
    ``n_max`` and refined from every sampled local maximum it misses none.

    Processing steps
    ----------------
    1. Align the surfaces and keep the modes that resonate at ``q``.
    2. Weight each mode's contribution.
    3. Evaluate ``sum_n Re(z_n exp(i n u))`` over ``u`` in ``[0, 2 pi)``,
       take the largest, and refine it by golden-section search.
    4. Report the quadrature sum and the aligned-phase bound alongside.

    Limitations
    -----------
    The peak is over the helical angle at one surface; it is not an island
    width, and the modes do not combine into one. Chirikov parameters in
    particular are **not** additive across ``n`` -- the legacy summed them
    into a column it named ``sum_K_isl_wrong``, and nothing here reproduces
    that sum under a better name.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics.composite_drive_at_q``, which
       returned ``abs(sum(z))`` at the caller's fixed phases and
       ``sum(abs(z))`` beside it, with a third key duplicating the second.
       The first is this function's drive at whatever angle the weights
       happen to encode, which is gauge-dependent and was reported as though
       it were not; the second is ``linear_bound``. The angle the legacy
       swept was never identified as the helical one.
    .. [convention] The legacy's own multi-n artefact records the phase
       convention it combined under: ``sample/gpec_optimization_examples/``
       ``q2p0_harmonic/gpec_multimode_summary_n12.json`` carries
       ``phase_convention = "real(B_n * exp(i * n * phi))"``.
    """
    modes, values = _contributions_at_q(
        tables, q, column, weights=weights, integrality_tolerance=integrality_tolerance
    )
    orders = np.asarray(modes, dtype=float)
    highest = int(max(modes))
    # The sum is a trigonometric polynomial of degree `highest`, so it has up
    # to 2*highest maxima and a grid that does not resolve them can bracket
    # the search around the wrong one. Sample 16 points per period of the top
    # harmonic.
    required = 16 * highest + 1
    if angle_points is None:
        points = max(721, required)
    else:
        points = int(angle_points)
        if points != angle_points or points < required:
            raise ValueError(
                f"angle_points must be a whole number and at least {required} "
                f"for a top harmonic of n = {highest}, not {angle_points!r}; "
                "leave it unset to have it sized automatically"
            )

    def amplitude(u: float) -> float:
        return float(np.sum((values * np.exp(1j * orders * u)).real))

    grid = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    sampled = (values[None, :] * np.exp(1j * np.outer(grid, orders))).real.sum(axis=1)
    step = 2.0 * np.pi / points
    # Refine from every sampled local maximum, not only the largest: with a
    # resolved grid each one brackets a true maximum, and which sample happens
    # to be highest does not decide which bracket holds the global peak.
    interior = (sampled >= np.roll(sampled, 1)) & (sampled >= np.roll(sampled, -1))
    candidates = np.flatnonzero(interior)
    if candidates.size == 0:
        candidates = np.array([int(np.argmax(sampled))])
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    peak = float(sampled.max())
    peak_angle = float(grid[int(np.argmax(sampled))])
    for index in candidates:
        low, high = grid[index] - step, grid[index] + step
        for _ in range(60):
            left, right = high - ratio * (high - low), low + ratio * (high - low)
            if amplitude(left) > amplitude(right):
                high = right
            else:
                low = left
        angle = 0.5 * (low + high)
        height = amplitude(angle)
        if height > peak:
            peak, peak_angle = height, angle
    magnitudes = np.abs(values)
    return CompositeDrive(
        q=float(q),
        modes=tuple(modes),
        contributions=tuple(complex(v) for v in values),
        peak=peak,
        peak_angle=float(np.mod(peak_angle, 2.0 * np.pi)),
        quadrature=float(np.sqrt(np.sum(magnitudes**2))),
        linear_bound=float(np.sum(magnitudes)),
    )


def helical_phase_sweep(
    tables: Mapping[int, Mapping[str, object]],
    q: float,
    column: str,
    *,
    angles,
    weights: Mapping[int, complex] | None = None,
    integrality_tolerance: float = 1e-6,
) -> np.ndarray:
    """The composite drive at one surface, as a function of the helical angle.

    Parameters
    ----------
    tables : mapping of int to mapping
        One resonant table per toroidal mode number [n/a].
    q : float
        The rational surface to sweep at [-].
    column : str
        The column to sum [n/a].
    angles : array_like
        Helical angles to evaluate at [rad].
    weights : mapping of int to complex, optional
        Per-mode complex weight, applied before the sum [n/a].
    integrality_tolerance : float, optional
        Passed to :func:`align_surfaces_by_q` [-].

    Returns
    -------
    ndarray
        The real amplitude at each angle, in the order given. Units follow
        ``column`` [n/a].

    Raises
    ------
    ValueError
        As :func:`composite_drive_at_q`, or an angle is not finite.

    Applicability
    -------------
    Machine-independent.

    Convention
    ----------
    The angle is the helical one, ``u = q theta - zeta``, in which the mode
    ``n`` advances as ``exp(i n u)``; :func:`composite_drive_at_q` derives
    it. Its origin is the input's, and shifting that origin translates the
    whole curve rather than changing its shape.

    Processing steps
    ----------------
    1. Align, weight and collect the contributions, as
       :func:`composite_drive_at_q` does.
    2. Evaluate ``sum_n Re(z_n exp(i n u))`` at each requested angle.

    Limitations
    -----------
    A sweep resolves a series whose highest harmonic is the largest
    resonating ``n``; sampling it more coarsely than a few points per period
    of that harmonic will miss the peak, and nothing here checks the caller's
    spacing against it.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics.phase_sweep_composite``, which
       rotated one nominated mode through a default three angles -- 0, 90 and
       180 degrees, so a sweep could not tell ``+90`` from ``-90`` -- and
       returned ``nan`` rows rather than an error when no surface matched.
       Rotating one mode and sweeping the shared angle differ once more than
       two modes resonate.
    """
    requested = np.atleast_1d(np.asarray(angles, dtype=float))
    if not np.all(np.isfinite(requested)):
        raise ValueError("every angle must be finite")
    modes, values = _contributions_at_q(
        tables, q, column, weights=weights, integrality_tolerance=integrality_tolerance
    )
    orders = np.asarray(modes, dtype=float)
    return (values[None, :] * np.exp(1j * np.outer(requested, orders))).real.sum(axis=1)


def reduce_delta_e(
    values: Mapping[int, float],
    *,
    combination: str = "quadrature",
    weights: Mapping[int, float] | None = None,
) -> float:
    """Combine a per-mode edge-overlap metric into one number.

    Parameters
    ----------
    values : mapping of int to float
        One :attr:`EdgeOverlap.delta_e` per toroidal mode number [-].
    combination : {'quadrature', 'linear'}, optional
        ``quadrature`` is ``sqrt(sum (w d)**2)``; ``linear`` is
        ``sum w d`` [n/a].
    weights : mapping of int to float, optional
        Per-mode weight; every mode present in ``values`` needs one if any
        does [-].

    Returns
    -------
    float
        The combined metric, in the same units as the inputs [-].

    Raises
    ------
    ValueError
        ``values`` is empty, a value or weight is not finite or not real, a
        value is negative, the combination is not one of the two, or
        ``weights`` names a mode that ``values`` does not.

    Applicability
    -------------
    Machine-independent.

    Convention
    ----------
    Both combinations are homogeneous of degree one in the weights, so
    rescaling every weight by the same non-negative factor moves either
    result by that factor; what differs is how modes combine with each
    other, and quadrature is the default because ``delta_e`` is a magnitude
    with no phase left to interfere. Negative weights are refused rather
    than given a meaning: they are homogeneous of degree one only for
    ``linear`` -- quadrature squares them away -- and they reintroduce the
    cancellation the negative-value check exists to forbid.

    Defaults
    --------
    ``combination = "quadrature"`` is a conventional choice that follows
    from what the input is: ``delta_e`` is the magnitude of a projection, so
    the modes carry no relative phase and adding them linearly would assume
    one they do not have.

    Processing steps
    ----------------
    1. Weight each mode's value.
    2. Sum the weighted values, in quadrature or linearly.

    Limitations
    -----------
    Neither combination is a physical amplitude. ``delta_e`` measures how
    much of one mode's external field lands on that mode's own most-coupled
    direction, and those directions belong to different singular value
    problems, so no combination of them is the overlap of anything.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics.delta_e_weighted_norm``, whose
       ``use_complex_sum=True`` branch took ``abs(sum(w * complex(v)))``.
       Its inputs are real by construction -- ``delta_e_vector`` reads
       ``projected_overlap``, a magnitude -- so that branch equalled the
       linear sum and the "coherent" name promised a phase that had already
       been discarded upstream. It is ``linear`` here, under a name that
       claims no more than it does. Its companion ``delta_e_vector`` was a
       dictionary comprehension over an attribute name, with an unvalidated
       region returning ``None`` for every mode; assembling that mapping is
       the caller's job and needs no function.
    """
    if combination not in DELTA_E_COMBINATIONS:
        raise ValueError(
            f"combination must be one of {DELTA_E_COMBINATIONS}, not {combination!r}"
        )
    if not values:
        raise ValueError("no modes were given, so there is nothing to combine")
    if weights is not None:
        missing = sorted(set(values) - set(weights))
        if missing:
            raise ValueError(f"weights are given but modes {missing} have none")
        unknown = sorted(set(weights) - set(values))
        if unknown:
            raise ValueError(f"weights name modes with no value: {unknown}")
    total_square = 0.0
    total_linear = 0.0
    for mode in sorted(values):
        value = values[mode]
        if isinstance(value, complex) or not np.isreal(value):
            raise ValueError(f"delta_e for n = {mode} is not real: {value!r}")
        value = float(np.real(value))
        if not np.isfinite(value):
            raise ValueError(f"delta_e for n = {mode} is not finite: {value!r}")
        if value < 0.0:
            raise ValueError(
                f"delta_e for n = {mode} is negative ({value!r}); it is the "
                "magnitude of a projection and cannot be"
            )
        weight = 1.0 if weights is None else float(weights[mode])
        if not np.isfinite(weight):
            raise ValueError(f"the weight for n = {mode} is not finite: {weight!r}")
        if weight < 0.0:
            raise ValueError(
                f"the weight for n = {mode} is negative ({weight!r}); these are "
                "magnitudes, so a negative weight would cancel one mode against "
                "another exactly as a negative value would, and the quadrature "
                "combination would not even notice"
            )
        total_square += (weight * value) ** 2
        total_linear += weight * value
    return float(np.sqrt(total_square)) if combination == "quadrature" else float(total_linear)


def q_composite_table(
    tables: Mapping[int, Mapping[str, object]],
    column: str,
    *,
    weights: Mapping[int, complex] | None = None,
    integrality_tolerance: float = 1e-6,
    shared_only: bool = False,
) -> dict[str, np.ndarray]:
    """One row per rational surface, summed over the modes that reach it.

    Parameters
    ----------
    tables : mapping of int to mapping
        One resonant table per toroidal mode number [n/a].
    column : str
        The column to sum [n/a].
    weights : mapping of int to complex, optional
        Per-mode complex weight [n/a].
    integrality_tolerance : float, optional
        Passed to :func:`align_surfaces_by_q` [-].
    shared_only : bool, optional
        Keep only the surfaces every mode resonates at [n/a].

    Returns
    -------
    dict of str to ndarray
        Columns ``q``, ``n_modes``, ``peak``, ``peak_angle``, ``quadrature``
        and ``linear_bound``, plus ``psi_norm_<n>`` and ``m_<n>`` per mode
        (``nan`` and ``-1`` where that mode does not resonate). Rows ascend
        in ``q``, and every aligned surface gets one [n/a].

    Raises
    ------
    ValueError
        As :func:`composite_drive_at_q`, or ``shared_only`` leaves nothing.

    Applicability
    -------------
    Machine-independent.

    Convention
    ----------
    A surface a mode cannot resonate at is not a gap in the data, so its
    ``m_<n>`` is ``-1`` rather than missing, and every row has every column.

    Processing steps
    ----------------
    1. Align the surfaces across the modes.
    2. Compute the composite drive at each.
    3. Lay the per-mode positions and poloidal mode numbers out beside it.

    Limitations
    -----------
    Every row is one surface's own Fourier series; nothing sums down a
    column. A radial reduction over these rows would mix surfaces whose
    helical angles are different angles, which is what
    :func:`reduce_resonant` refuses to do across modes.

    Provenance
    ----------
    .. [legacy] ``gpec_multimode_metrics.compute_q_composite_table``, which
       built rows only for four hard-coded ``q`` values, dropped any row
       where no mode matched so the row count depended on the data, and
       emitted per-mode columns only where a mode was present, leaving a
       frame whose columns varied by row. Two of its columns are not ported:
       ``K_proxy_sqrt``, the square root of a flux magnitude standing in for
       a Chirikov parameter, which is not dimensionally a Chirikov parameter
       of anything; and ``sum_K_isl_wrong``, whose name was its own warning.
    """
    aligned = align_surfaces_by_q(tables, integrality_tolerance=integrality_tolerance)
    modes = sorted(tables)
    if shared_only:
        aligned = tuple(s for s in aligned if len(s.resonant_modes) == len(modes))
        if not aligned:
            raise ValueError(
                f"no surface is resonant on every one of n = {modes}; drop "
                "shared_only to keep the surfaces that some of them reach"
            )
    rows: dict[str, list] = {
        "q": [], "n_modes": [], "peak": [], "peak_angle": [],
        "quadrature": [], "linear_bound": [],
    }
    for mode in modes:
        rows[f"psi_norm_{mode}"] = []
        rows[f"m_{mode}"] = []
    for surface in aligned:
        drive = composite_drive_at_q(
            tables, surface.q, column, weights=weights,
            integrality_tolerance=integrality_tolerance,
        )
        rows["q"].append(surface.q)
        rows["n_modes"].append(len(drive.modes))
        rows["peak"].append(drive.peak)
        rows["peak_angle"].append(drive.peak_angle)
        rows["quadrature"].append(drive.quadrature)
        rows["linear_bound"].append(drive.linear_bound)
        by_mode = {x.n: x for x in surface.members}
        for mode in modes:
            member = by_mode[mode]
            rows[f"psi_norm_{mode}"].append(
                np.nan if member.psi_norm is None else member.psi_norm
            )
            rows[f"m_{mode}"].append(-1 if member.membership != "resonant" else member.m)
    return {
        key: np.asarray(value, dtype=int if key.startswith("m_") or key == "n_modes" else float)
        for key, value in rows.items()
    }

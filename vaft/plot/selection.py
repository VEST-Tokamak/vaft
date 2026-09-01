"""Physical-region classification and the channel-selection vocabulary (#259).

Many VAFT diagnostics carry dozens of channels under one physical subject, and
asking for them by raw index cannot express what a physicist actually wants --
*the inboard flux loops*, *the probe nearest the outboard midplane*.  This
module supplies the vocabulary for those requests and the rule that decides
which channels answer them.

The rule is deliberately **device-independent and family-relative**.  It reads
nothing but the radial positions of one diagnostic family and infers that
family's own inboard/outboard divider, so flux loops are classified only
against other flux loops and B-pol probes only against other B-pol probes.  No
machine-specific radius is hard-coded here, and no global population is formed
across families.

The divider is the midpoint of the **widest gap** in the family's sorted radii,
not the extent midpoint and not the mean:

* the mean moves when one side is more densely instrumented, which would make
  the classification depend on channel count rather than geometry;
* the extent midpoint moves when a family contains an outlying channel.  On
  VEST the ``b_field_pol_probe`` array holds an IMPA Hall-probe scan reaching
  R = 1.26 m, well outside the vessel, which drags the extent midpoint from
  0.44 m to 0.67 m.  The widest gap ignores it.

Where a family has no meaningful gap -- every fluctuation Mirnov sits at the
same major radius -- the answer is *no split*, not an arbitrary one.

Everything here works on plain numbers so that ``vaft.plot`` stays free of OMAS:
the ``vaft.omas`` layer reads positions out of an ODS and calls in.  The policy
these rules serve is documented in
``notebooks/plotting_sample_using_vaft_plot_module.ipynb``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "INBOARD",
    "OUTBOARD",
    "PRESETS",
    "REGIONS",
    "UNCLASSIFIED",
    "RadialSplit",
    "classify_regions",
    "radial_divider",
    "representative_index",
]

INBOARD = "inboard"
OUTBOARD = "outboard"
#: A channel too close to the divider to be assigned a side honestly.
UNCLASSIFIED = "unclassified"

#: The physical regions this phase classifies.  Upper/lower is deliberately not
#: part of it (issue #259 keeps the generic vocabulary small).
REGIONS = (INBOARD, OUTBOARD, UNCLASSIFIED)

#: The generic public selection presets.  ``*_mid`` name one representative
#: channel rather than a set; see :func:`representative_index`.
PRESETS = (INBOARD, OUTBOARD, "inboard_mid", "outboard_mid")

#: The widest gap must be at least this many times the typical spacing between
#: neighbouring channels before it counts as the divide between two sides.
#: Without it a family spread evenly across one wall would be cut arbitrarily in
#: half: its widest gap is no more meaningful than any of its others.
MIN_GAP_DOMINANCE = 3.0

#: A gap smaller than this fraction of the family's radial scale is numerical
#: noise in the stored geometry rather than a real separation.
NOISE_FRACTION = 1e-6

#: Half-width of the band around the divider in which a channel is not assigned
#: a side, as a fraction of the gap.  It absorbs floating-point noise in stored
#: geometry; a channel genuinely sitting mid-gap is a third cluster, not a
#: borderline case, and changes the family's divider instead.
TOLERANCE_FRACTION = 1e-3


class RadialSplit:
    """The inboard/outboard divider inferred for one diagnostic family."""

    __slots__ = ("divider", "tolerance")

    def __init__(self, divider: float | None, tolerance: float = 0.0) -> None:
        self.divider = divider
        self.tolerance = tolerance

    def __bool__(self) -> bool:
        return self.divider is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.divider is None:
            return "RadialSplit(none)"
        return f"RadialSplit(divider={self.divider:.4f}, tolerance={self.tolerance:.4g})"


def radial_divider(r_values: Sequence[float] | Any) -> RadialSplit:
    """Infer one family's inboard/outboard divider from its own radii.

    Returns a :class:`RadialSplit` that is falsy when the family shows no
    meaningful gap -- a single cluster of channels has no inboard and no
    outboard, and inventing a divider through the middle of it would classify
    neighbouring sensors as opposites.
    """
    radii = np.asarray(r_values, dtype=float).ravel()
    radii = radii[np.isfinite(radii)]
    if radii.size < 2:
        return RadialSplit(None)

    ordered = np.unique(radii)
    if ordered.size < 2:
        return RadialSplit(None)
    extent = float(ordered[-1] - ordered[0])
    if extent <= 0.0:
        return RadialSplit(None)

    gaps = np.diff(ordered)
    widest = int(np.argmax(gaps))
    gap = float(gaps[widest])
    scale = float(np.max(np.abs(ordered)))
    if scale > 0.0 and gap < NOISE_FRACTION * scale:
        # Two radii a nanometre apart are one position recorded twice, not two
        # sides of a machine.  Without this a two-channel family would always
        # split, however tightly clustered.
        return RadialSplit(None)
    others = np.delete(gaps, widest)
    if others.size:
        typical = float(np.median(others))
        if typical > 0.0 and gap < MIN_GAP_DOMINANCE * typical:
            # The widest gap is no more meaningful than the rest: one cluster.
            return RadialSplit(None)

    divider = float(0.5 * (ordered[widest] + ordered[widest + 1]))
    return RadialSplit(divider, tolerance=TOLERANCE_FRACTION * gap)


def classify_regions(
    r_values: Sequence[float] | Any, split: "RadialSplit | None" = None
) -> list[str]:
    """Classify each channel of one family as inboard, outboard or neither.

    ``r_values`` are the major radii of a single diagnostic family, in the
    caller's own order; the returned list matches that order.  Every channel is
    :data:`UNCLASSIFIED` when the family has no meaningful radial split.  Pass
    ``split`` to classify a subset against the whole family's divider rather
    than re-inferring one from the subset.
    """
    radii = np.asarray(r_values, dtype=float).ravel()
    if split is None:
        split = radial_divider(radii)
    if not split:
        return [UNCLASSIFIED] * radii.size

    regions = []
    for radius in radii:
        if not np.isfinite(radius) or abs(radius - split.divider) <= split.tolerance:
            regions.append(UNCLASSIFIED)
        elif radius < split.divider:
            regions.append(INBOARD)
        else:
            regions.append(OUTBOARD)
    return regions


def representative_index(
    z_values: Sequence[float] | Any, candidates: Sequence[int]
) -> int | None:
    """The channel of ``candidates`` that best represents the midplane.

    The midplane channel is the one nearest ``Z = 0``; ties are broken by the
    lowest index so the same ODS always yields the same answer.  This resolves
    against the channels actually present, never a remembered index -- a
    representative is a real measurement, not a position in an array.
    """
    z = np.asarray(z_values, dtype=float).ravel()
    best: tuple[float, int] | None = None
    for index in candidates:
        if index >= z.size:
            continue
        height = abs(float(z[index]))
        if not np.isfinite(height):
            continue
        if best is None or height < best[0]:
            best = (height, index)
    return None if best is None else best[1]

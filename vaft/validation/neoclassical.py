"""Compare the analytic neoclassical models against a drift-kinetic solve.

Level 4 of the verification hierarchy in issue #550 section 9. The levels below it are
already established and are *tests*, not studies: the formula kernels reproduce published
Sauter values, the NEO adapter reproduces GACODE's own ``reg18`` regression case, and
VAFT's Sauter and Redl agree with NEO's own implementations of them to 1e-8. This module
asks the different question those cannot: **how far apart are the models themselves**.

That distinction is the whole point. A gap between a fitted analytic model and a
drift-kinetic solve is a property of the models, not a defect in either, and #550 is
explicit that validation must not classify it as a solver failure. So nothing here
returns a status. :func:`model_comparison` returns numbers; :func:`model_agreement` is
the only function that applies a tolerance, and the tolerances are the caller's, passed
in and echoed back -- the arrangement ``vaft.validation.wall_reduction`` uses for the
same reason.

Metrics are normalised by the profile's own scale, never pointwise. The bootstrap current
changes sign near the axis, so a pointwise ratio explodes at the crossing: on the packaged
VEST 48224 state the Sauter-Redl difference is 4.9 percent integrated and 4.1 percent as
an RMS against the peak, but 487 percent as a worst-case pointwise ratio. The same
reasoning rules out a log ratio on the profile, which is undefined where the current is
negative.

The physical content is the trend rather than any single number. On 48224 the analytic
models part company as the trapped fraction rises, which is what makes Redl the
defensible choice for a spherical tokamak, and Redl sits closer to NEO than Sauter does
on that state by both the integrated and the RMS measure.

**This is an end-to-end comparison, not a like-for-like one.** The analytic profiles are
evaluated here from the ODS's own fitted kinetic profiles and equilibrium, while NEO ran
on the GACODE conversion of the same state -- so the two sides do not share a trapped
fraction, a collisionality or a radial grid, and some of any difference is the input path
rather than the model. The like-for-like check exists separately and is tight: NEO writes
its own Sauter and Redl columns beside its solve, and ``test_formula_neoclassical.py``
holds VAFT's kernels against them to 1e-7 in two regimes. Read a gap here as a question
to investigate, not as a measured model error.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "METRICS",
    "bootstrap_models",
    "model_agreement",
    "model_comparison",
]

#: The comparison's metric names, so a caller can validate a tolerance mapping against
#: them rather than discovering a typo as a silently ignored key.
#: The trend metric is relative, so it is evaluated only where the reference carries at
#: least this fraction of its own peak -- below it, a ratio is dominated by the profile's
#: zero crossing rather than by the models.
_TREND_FLOOR = 0.1

METRICS: tuple[str, ...] = (
    "integrated_relative_difference",
    "rms_over_peak",
    "peak_relative_difference",
)


def bootstrap_models(
    ods: Any,
    *,
    models: Sequence[str] = ("sauter", "redl"),
    include_stored: bool = True,
    time_slice: Optional[int] = None,
    **options: Any,
) -> dict[str, Any]:
    """Evaluate each analytic model on *ods*, alongside whatever it already stores.

    The analytic profiles come from :func:`vaft.omas.neoclassical.compute_bootstrap_current`;
    the solver's comes from ``core_profiles.j_bootstrap``, which is where
    :mod:`vaft.machine_mapping.neoclassical` puts a NEO result. Reading it rather than
    re-running anything is what keeps this layer composing providers instead of becoming
    one.

    Parameters
    ----------
    ods
        Carrying an ``equilibrium`` and a ``core_profiles`` slice.
    models
        Which analytic formulations to evaluate.
    include_stored
        Whether to pick up an already-written ``j_bootstrap``. Its series is named after
        ``core_profiles.code.name`` when the ODS says who wrote it, else ``"stored"``.
    time_slice
        The ``core_profiles`` slice; the equilibrium is matched to its time.
    **options
        Passed to the provider -- ``z_eff``, ``rho_range``, ``ion_index``, ``b0``.

    Returns
    -------
    dict
        ``{"rho_tor_norm", "series", "area", "provenance"}``. Each entry of ``series`` is
        ``{"j_bootstrap", "source", ...}``; the analytic ones also carry ``f_trap``,
        ``nu_e_star`` and the coefficients.
    """
    from vaft.omas.neoclassical import compute_bootstrap_current

    series: dict[str, dict[str, Any]] = {}
    grid = None
    provenance: dict[str, Any] = {"models": list(models)}

    for name in models:
        result = compute_bootstrap_current(ods, model=name, time_slice=time_slice, **options)
        grid = result.rho_tor_norm if grid is None else grid
        series[name] = {
            "j_bootstrap": np.asarray(result.j_bootstrap, dtype=float),
            "source": "analytic",
            "f_trap": np.asarray(result.trapped_fraction, dtype=float),
            "nu_e_star": np.asarray(result.nu_e_star, dtype=float),
            "nu_i_star": np.asarray(result.nu_i_star, dtype=float),
            "coefficients": result.coefficients,
        }
        provenance.setdefault("provider", dict(result.provenance))
        provenance.setdefault("b0", result.b0)
        provenance.setdefault("time", result.time)

    if include_stored:
        stored, label = _stored_bootstrap(ods, provenance.get("provider", {}))
        if stored is not None and grid is not None and stored.size == grid.size:
            series[label] = {"j_bootstrap": stored, "source": "stored"}
            provenance["stored_series"] = label

    if grid is None:
        raise ValueError("no model was evaluated; `models` was empty")

    index = provenance.get("provider", {}).get("equilibrium_index", 0)
    area = _array(ods, f"equilibrium.time_slice.{index}.profiles_1d.area")
    return {
        "rho_tor_norm": grid,
        "series": series,
        "area": area,
        "provenance": provenance,
    }


def model_comparison(models: Mapping[str, Any], *, reference: str = "neo") -> dict[str, Any]:
    """Measure how far each series sits from *reference*. Numbers only, no status.

    Every metric is normalised by the reference profile's own scale. Points where either
    profile is not finite are excluded and counted: a NEO run solves a handful of
    surfaces, so its mapped profile is deliberately NaN outside them.

    Returns
    -------
    dict
        ``{"reference", "overlap", "models": {name: {...metrics..., "trend": ...}},
        "effect_size": {...}}``. A model with no overlap against the reference reports
        ``None`` metrics and a ``reason``, rather than a number computed from nothing.
    """
    grid = np.asarray(models["rho_tor_norm"], dtype=float)
    series = models["series"]
    area = models.get("area")

    if reference not in series:
        raise ValueError(
            f"reference {reference!r} is not among the evaluated series "
            f"({', '.join(sorted(series))}); pass reference= to name one that is"
        )
    baseline = np.asarray(series[reference]["j_bootstrap"], dtype=float)

    compared: dict[str, Any] = {}
    for name, entry in series.items():
        if name == reference:
            continue
        values = np.asarray(entry["j_bootstrap"], dtype=float)
        overlap = np.isfinite(values) & np.isfinite(baseline)
        if int(np.count_nonzero(overlap)) < 2:
            compared[name] = {
                "reason": "fewer than two radii where both profiles are finite",
                **{metric: None for metric in METRICS},
            }
            continue
        subset = None if area is None else np.asarray(area, dtype=float)[overlap]
        compared[name] = {
            **_metrics(values[overlap], baseline[overlap], grid[overlap], subset),
            "points": int(np.count_nonzero(overlap)),
            "trend": _trend(entry, values, baseline, overlap),
        }

    return {
        "reference": reference,
        "overlap": int(np.count_nonzero(np.isfinite(baseline))),
        "models": compared,
        "effect_size": _effect_size(baseline, grid, area),
        "provenance": dict(models.get("provenance", {})),
    }


def model_agreement(
    comparison: Mapping[str, Any], tolerances: Mapping[str, float]
) -> dict[str, Any]:
    """Apply the caller's tolerances to a comparison, and echo them back.

    The only place in this module where a number becomes a judgement. The tolerances are
    the study's, not the library's: what counts as agreement between a fitted model and a
    drift-kinetic solve depends entirely on what the answer is for, and #550 warns against
    turning a real model difference into a failure.
    """
    unknown = set(tolerances) - set(METRICS)
    if unknown:
        raise ValueError(
            f"unknown metric(s) {sorted(unknown)}; expected a subset of {list(METRICS)}"
        )
    if not tolerances:
        raise ValueError("no tolerances given; there is nothing to decide")

    within: dict[str, Any] = {}
    for name, metrics in comparison["models"].items():
        verdicts = {}
        for metric, bound in tolerances.items():
            value = metrics.get(metric)
            verdicts[metric] = None if value is None else bool(abs(value) <= float(bound))
        decided = [answer for answer in verdicts.values() if answer is not None]
        within[name] = {
            "by_metric": verdicts,
            "joint": bool(decided) and all(decided),
            "evaluated": len(decided),
        }
    return {
        "reference": comparison["reference"],
        "tolerances": dict(tolerances),
        "models": within,
    }


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _array(ods: Any, path: str) -> Optional[np.ndarray]:
    try:
        if path not in ods:
            return None
        values = np.atleast_1d(np.asarray(ods[path], dtype=float))
    except (KeyError, ValueError, IndexError, TypeError):
        return None
    return values if values.size else None


def _stored_bootstrap(ods: Any, provider: Mapping[str, Any]) -> tuple[Optional[np.ndarray], str]:
    """An already-written j_bootstrap and the name of whatever wrote it."""
    index = provider.get("core_profiles_index", 0)
    values = _array(ods, f"core_profiles.profiles_1d.{index}.j_bootstrap")
    label = "stored"
    try:
        if "core_profiles.code.name" in ods:
            written_by = str(ods["core_profiles.code.name"]).strip()
            if written_by:
                label = written_by.lower()
    except (KeyError, ValueError, TypeError):
        pass
    return values, label


def _integrate(values: np.ndarray, grid: np.ndarray, area: Optional[np.ndarray]) -> float:
    """Integrate a current density over the cross-section, or over rho as a fallback."""
    values = np.asarray(values, dtype=float)
    if area is not None:
        area = np.asarray(area, dtype=float)
        if area.size == values.size:
            return float(np.trapezoid(values, area))
    return float(np.trapezoid(values, np.asarray(grid, dtype=float)))


def _metrics(
    values: np.ndarray, baseline: np.ndarray, grid: np.ndarray, area: Optional[np.ndarray]
) -> dict[str, float]:
    scale = float(np.max(np.abs(baseline)))
    reference_total = _integrate(baseline, grid, area)
    total = _integrate(values, grid, area)
    difference = values - baseline
    worst = int(np.argmax(np.abs(difference)))
    return {
        "integrated_relative_difference": (
            float(total / reference_total - 1.0) if reference_total != 0.0 else float("nan")
        ),
        "rms_over_peak": float(np.sqrt(np.mean(difference**2)) / scale) if scale > 0 else float("nan"),
        "peak_relative_difference": (
            float(np.max(np.abs(values)) / np.max(np.abs(baseline)) - 1.0) if scale > 0 else float("nan")
        ),
        "max_discrepancy": float(difference[worst]),
        "max_discrepancy_rho": float(grid[worst]),
        "integrated": total,
        "reference_integrated": reference_total,
    }


def _trend(
    entry: Mapping[str, Any], values: np.ndarray, baseline: np.ndarray, overlap: np.ndarray
) -> Optional[dict[str, Any]]:
    """How the disagreement varies with the trapped fraction.

    The comparison's physical content: a fit built at conventional aspect ratio should
    part company with a drift-kinetic solve as the trapped fraction rises.

    This one metric is *relative*, because that is the claim -- a fractional disagreement
    that grows -- and an absolute difference cannot express it: towards the edge both
    profiles fall away, so the absolute gap shrinks even as the fractional one widens. To
    keep the relative form usable it is evaluated only where the reference is a
    meaningful fraction of its own peak, which is what excludes the sign change that
    makes a pointwise ratio explode.
    """
    trapped = entry.get("f_trap")
    if trapped is None:
        return None
    trapped = np.asarray(trapped, dtype=float)[overlap]
    reference = baseline[overlap]
    if trapped.size < 6 or not np.all(np.isfinite(trapped)):
        return None
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    if scale <= 0.0:
        return None

    significant = np.abs(reference) >= _TREND_FLOOR * scale
    if int(np.count_nonzero(significant)) < 6:
        return None
    trapped = trapped[significant]
    difference = np.abs(
        values[overlap][significant] / reference[significant] - 1.0
    )
    order = np.argsort(trapped)
    third = max(1, trapped.size // 3)
    low, high = order[:third], order[-third:]
    return {
        "f_trap_low": float(np.mean(trapped[low])),
        "f_trap_high": float(np.mean(trapped[high])),
        "difference_low": float(np.mean(difference[low])),
        "difference_high": float(np.mean(difference[high])),
        "grows_with_trapping": bool(np.mean(difference[high]) > np.mean(difference[low])),
        "evaluated_above": _TREND_FLOOR,
    }


def _effect_size(
    baseline: np.ndarray, grid: np.ndarray, area: Optional[np.ndarray]
) -> dict[str, Any]:
    """The size of the thing being compared, against doing nothing at all.

    ``wall_reduction`` carries a ``no_wall`` reference for the same reason: without it a
    reader cannot tell a 5 percent disagreement about a large effect from a 5 percent
    disagreement about a negligible one. On VEST 48224 the bootstrap current is under two
    percent of the plasma current, which is the first thing to know about any comparison
    of it.
    """
    usable = np.isfinite(baseline)
    if int(np.count_nonzero(usable)) < 2:
        return {"integrated_current": None, "reason": "the reference profile has no usable span"}
    subset = None if area is None else np.asarray(area, dtype=float)[usable]
    return {
        "integrated_current": _integrate(baseline[usable], grid[usable], subset),
        "peak": float(np.max(np.abs(baseline[usable]))),
        "peak_rho": float(grid[usable][int(np.argmax(np.abs(baseline[usable])))]),
    }

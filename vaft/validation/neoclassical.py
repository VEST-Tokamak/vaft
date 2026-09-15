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
    "REQUIREMENTS",
    "bootstrap_models",
    "input_readiness",
    "model_agreement",
    "model_comparison",
]

#: The comparison's metric names, so a caller can validate a tolerance mapping against
#: them rather than discovering a typo as a silently ignored key.
#: The trend metric is relative, so it is evaluated only where the reference carries at
#: least this fraction of its own peak -- below it, a ratio is dominated by the profile's
#: zero crossing rather than by the models.
_TREND_FLOOR = 0.1

#: How many points the trend needs at all, and how many bins it splits them into.
#: Five bins of at least two points is enough to see a reversal without inventing
#: structure from noise.
_TREND_MIN_POINTS = 6
_TREND_BINS = 5

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
    solver_charge: Optional[float] = None,
    solver_ion_fraction: Optional[float] = None,
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
        stored, label, stored_grid = _stored_bootstrap(ods, provenance.get("provider", {}))
        admitted, reason = _admissible(stored, stored_grid, grid)
        if admitted:
            series[label] = {"j_bootstrap": stored, "source": "stored"}
            provenance["stored_series"] = label
        elif reason is not None:
            # Say why rather than leaving the series quietly absent: "NEO is missing
            # from this comparison" and "NEO was dropped" look identical otherwise.
            provenance["stored_series_rejected"] = {"label": label, "reason": reason}

    if grid is None:
        raise ValueError("no model was evaluated; `models` was empty")

    if solver_charge is not None:
        # The charge the solver's own species list implies, from
        # NeoOutputs.effective_charge. Comparing against a run that used a different
        # one compares two plasmas: on VEST that inverted which model looked closer
        # (#803), so it is refused rather than recorded and hoped for.
        provider = provenance.get("provider", {})
        analytic = provider.get("z_eff_value")
        if analytic is not None and abs(float(analytic) - float(solver_charge)) > 1e-6:
            raise ValueError(
                f"the analytic models were evaluated at Z_eff = {float(analytic):g} and "
                f"the solver ran at {float(solver_charge):g}; those are different "
                "plasmas. Pass z_eff= (and impurity=) matching the run, or omit "
                "solver_charge= to compare anyway and own the mismatch."
            )
        # Agreeing on Z_eff is not agreeing on the plasma: the same charge can be
        # carried by carbon or by oxygen, and n_i/n_e differs (0.833 against 0.875).
        # Above all, an analytic side with no impurity at all reaches this point with
        # n_i = n_e and would pass a charge-only check -- which is the other half of
        # the mismatch #803 is about.
        fraction = provider.get("ion_density_over_electron")
        if solver_ion_fraction is not None and fraction is not None:
            if abs(float(fraction) - float(solver_ion_fraction)) > 1e-6:
                raise ValueError(
                    f"the analytic models used n_i/n_e = {float(fraction):.4g} and the "
                    f"solver's species list gives {float(solver_ion_fraction):.4g}; the "
                    "two carry the same Z_eff in different species. Pass the impurity= "
                    "the run was built with."
                )
            provenance["solver_ion_fraction"] = float(solver_ion_fraction)
        elif solver_ion_fraction is None and fraction is not None and fraction == 1.0:
            # Nothing to compare against, so say what was assumed rather than let a
            # hydrogenic analytic side look like it was checked.
            provenance["ion_fraction_unchecked"] = (
                "the analytic models assumed n_i = n_e; pass solver_ion_fraction= "
                "(NeoOutputs.ion_density_fraction) to have that checked against the run"
            )
        provenance["solver_charge"] = float(solver_charge)

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


#: What a state must carry before NEO can be run on it, in the order
#: :func:`input_readiness` reports them. Named rather than implied so a caller can
#: act on one requirement -- supply a Z_eff, truncate the grid, choose another time
#: -- instead of reading a sentence.
#: Where the grid is truncated unless a caller decides otherwise. Named so the
#: recorded assumption can say which of the two it was.
_DEFAULT_RHO_MAX = 0.95

REQUIREMENTS: tuple[str, ...] = (
    "convertible",
    "ion_temperature",
    "effective_charge",
    "radial_extent",
)


def input_readiness(
    ods: Any,
    *,
    time: Optional[float] = None,
    time_index: Optional[int] = None,
    rho_max: Optional[float] = _DEFAULT_RHO_MAX,
    z_eff: Optional[float] = None,
    ion_index: int = 0,
    minimum_points: int = 8,
) -> dict[str, Any]:
    """Whether this ODS is a kinetic state NEO can be run on, and what is missing.

    #550 defers routine production to a *qualified* kinetic state rather than to a
    shot: a usable equilibrium, electron density and temperature, an ion temperature
    and species, a Z_eff or an explicit assumption in its place, and a time alignment
    that pairs the equilibrium with the profiles. This function is that rule, written
    as something a pipeline can evaluate from an ODS -- never from a list of shot
    numbers, which would go stale the moment a reprocessing changed what a shot holds.

    It reports a status, unlike the rest of this module, and the distinction matters:
    a model difference is a property of the models and must not be graded, but whether
    an input carries the quantities a solver needs is a fact about the data. Nothing
    here judges a *result*.

    The rule is evaluated by asking the converter, not by restating its rules:
    :func:`vaft.code.gacode.inputs.prepare_gacode_profile` already refuses a
    non-positive density or temperature, a ``sqrt(psi_N)`` proxy standing in for
    ``rho_tor_norm``, and slices that cannot be paired. A reimplementation here would
    drift from it, and the drift would show up as a shot that qualifies and then fails
    to convert.

    Parameters
    ----------
    ods
        A state carrying ``equilibrium`` and ``core_profiles``.
    time, time_index
        Which slice to qualify; the same arguments the converter takes.
    rho_max
        Where the caller intends to truncate. Part of the rule because a state that
        qualifies only to 0.8 is a different answer from one that qualifies to 0.95.
    z_eff
        An effective charge to use if the state carries none. Supplying it makes the
        state qualify *with a recorded assumption*, not silently.
    ion_index
        Which ion species carries the temperature the models will use.
    minimum_points
        How many radial points must survive the truncation. A three-point profile
        converts and is useless.

    Returns
    -------
    dict
        ``{"qualified", "requirements", "unmet", "assumptions", "provenance"}``.
        ``requirements`` is one entry per name in :data:`REQUIREMENTS`, each with
        ``satisfied`` and a ``detail`` saying why -- present for the satisfied ones
        too, so a passing state records what it passed on.

        A refusal from the converter stops the rest of the rule, and the reply then
        carries ``not_evaluated`` naming what was never reached. ``unmet`` lists
        only checks that actually ran and failed, so every name in it can be looked
        up in ``requirements``.
    """
    from vaft.code.gacode.inputs import ProfileConversionError, prepare_gacode_profile

    checks: list[dict[str, Any]] = []
    assumptions: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"rho_max": rho_max, "ion_index": int(ion_index)}

    try:
        profile = prepare_gacode_profile(
            ods, time=time, time_index=time_index, rho_max=rho_max, z_eff=z_eff
        )
    except ProfileConversionError as refusal:
        # The converter's refusal is the answer, quoted rather than paraphrased.
        checks.append(
            {"name": "convertible", "satisfied": False, "detail": str(refusal)}
        )
        # Only what was actually evaluated. Padding this with the requirements the
        # refusal stopped us reaching would report a missing ion temperature for a
        # state that has one, and name entries `requirements` does not carry.
        return {
            "qualified": False,
            "requirements": tuple(checks),
            "unmet": ("convertible",),
            "not_evaluated": REQUIREMENTS[1:],
            "assumptions": (),
            "provenance": provenance,
        }

    checks.append(
        {
            "name": "convertible",
            "satisfied": True,
            "detail": (
                f"the equilibrium and core_profiles slices pair and project onto "
                f"{profile.n_exp} GACODE radial points"
            ),
        }
    )
    provenance["n_exp"] = profile.n_exp
    for key in ("time", "equilibrium_index", "core_profiles_index"):
        if key in profile.provenance:
            provenance[key] = profile.provenance[key]

    # An ion temperature, which is what separates a state NEO can be run on from an
    # electron-only fit: the ion collisionality is half the physics.
    ion_temperature = None if profile.ti is None else np.asarray(profile.ti, dtype=float)
    index = int(ion_index)
    has_ion = (
        ion_temperature is not None
        and ion_temperature.ndim == 2
        and index < ion_temperature.shape[0]
        and np.all(np.isfinite(ion_temperature[index]))
        and np.all(ion_temperature[index] > 0.0)
    )
    checks.append(
        {
            "name": "ion_temperature",
            "satisfied": bool(has_ion),
            "detail": (
                f"ion {index} ({profile.name[index] if index < len(profile.name) else '?'}) "
                f"carries a positive temperature on every retained point -- present, "
                f"which is not the same as measured: an electron-only fit with an "
                f"assumed Ti/Te ratio satisfies this too, and the ODS does not record "
                f"which it is"
                if has_ion
                else f"no positive temperature for ion {index}; NEO needs the ion "
                "collisionality, so an electron-only fit does not qualify"
            ),
        }
    )

    # Z_eff may be measured, derived from the species mix, or supplied -- all three
    # qualify, but only the first is a measurement, and the state must say which.
    record = dict(profile.provenance.get("z_eff", {}))
    kind = str(record.get("kind", "")) or "absent"
    checks.append(
        {
            "name": "effective_charge",
            "satisfied": kind != "absent",
            "detail": f"z_eff is {kind}" if kind != "absent" else (
                "no zeff profile, no impurity species to derive one from, and none "
                "supplied; pass z_eff= to qualify the state with a recorded assumption"
            ),
        }
    )
    if kind in {"caller_supplied", "policy_assumption"}:
        assumptions.append({"quantity": "z_eff", **record})
    if rho_max is not None:
        # A default is not a decision. The converter's vocabulary distinguishes them
        # so an assumption can be audited, and this one travels into the published
        # manifest -- a reader has to be able to tell 0.95 chosen for this shot from
        # 0.95 inherited from the signature.
        assumptions.append(
            {
                "quantity": "rho_max",
                "kind": "caller_supplied" if rho_max != _DEFAULT_RHO_MAX else "default",
                "value": float(rho_max),
            }
        )

    enough = profile.n_exp >= int(minimum_points)
    checks.append(
        {
            "name": "radial_extent",
            "satisfied": bool(enough),
            "detail": (
                f"{profile.n_exp} points survive the truncation at rho_max={rho_max}"
                if enough
                else f"only {profile.n_exp} points survive the truncation at "
                f"rho_max={rho_max}; fewer than {minimum_points} is a profile in name only"
            ),
        }
    )

    unmet = tuple(check["name"] for check in checks if not check["satisfied"])
    return {
        "qualified": not unmet,
        "requirements": tuple(checks),
        "unmet": unmet,
        "assumptions": tuple(assumptions),
        "provenance": provenance,
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


def _admissible(
    values: Optional[np.ndarray], source: Optional[np.ndarray], grid: Optional[np.ndarray]
) -> tuple[bool, Optional[str]]:
    """Whether a stored profile may join a comparison on *grid*.

    Equal length is not equal radii. The stored profile lives on the ``core_profiles``
    grid and the analytic ones on the ``equilibrium`` grid; those are different objects
    that often have the same number of points, so admitting on length alone would
    compare two models at different radii and report full agreement about it.
    """
    if values is None or grid is None:
        return False, None
    if source is None:
        return False, "core_profiles carries no grid.rho_tor_norm to place it on"
    if values.size != source.size:
        return False, (
            f"it has {values.size} points against a {source.size}-point core_profiles grid"
        )
    if source.size != grid.size or not np.allclose(source, grid, rtol=0.0, atol=1e-9):
        return False, (
            "it is on the core_profiles radial grid, which is not the equilibrium grid "
            "the analytic models were evaluated on; comparing them would pair different radii"
        )
    return True, None


def _stored_bootstrap(
    ods: Any, provider: Mapping[str, Any]
) -> tuple[Optional[np.ndarray], str, Optional[np.ndarray]]:
    """An already-written j_bootstrap, who wrote it, and the grid it sits on."""
    index = provider.get("core_profiles_index", 0)
    values = _array(ods, f"core_profiles.profiles_1d.{index}.j_bootstrap")
    source = _array(ods, f"core_profiles.profiles_1d.{index}.grid.rho_tor_norm")
    label = "stored"
    try:
        if "core_profiles.code.name" in ods:
            written_by = str(ods["core_profiles.code.name"]).strip()
            if written_by:
                label = written_by.lower()
    except (KeyError, ValueError, TypeError):
        pass
    return values, label, source


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
    """How the disagreement varies with the trapped fraction, as a shape and not a verdict.

    The claim this exists to test is the one that justifies Redl: a fit built at
    conventional aspect ratio should part company with a drift-kinetic solve as the
    trapped fraction rises. On VEST 48224 the relationship is **not monotonic**, so
    that question has no single answer here, and the two ways of forcing one both give
    wrong answers (issue #808):

    * comparing the mean of the lowest third of ``f_trap`` with the highest third --
      what this function used to do -- reported "grows" for the treatment published in
      #776 and "shrinks" for two others that differ from it by less than a percentage
      point in the integrated comparison;
    * a straight-line fit reports a *negative* slope in all three, because it is
      dominated by a handful of near-axis points where the bootstrap current crosses
      zero and a relative metric is meaningless however the floor is set.

    Binned, the shape is the same in every treatment: a large, badly scattered value in
    the innermost bin, a minimum around ``f_trap`` 0.75-0.8, and a clean monotonic rise
    to the edge. So the metric reports the bins with their scatter, states whether the
    sequence is monotonic within that scatter, and answers ``grows_with_trapping`` only
    when it is. ``None`` means this state cannot say -- which is the honest answer, and
    the one a reader can act on.
    """
    trapped = entry.get("f_trap")
    if trapped is None:
        return None
    trapped = np.asarray(trapped, dtype=float)[overlap]
    reference = baseline[overlap]
    if trapped.size < _TREND_MIN_POINTS or not np.all(np.isfinite(trapped)):
        return None
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    if scale <= 0.0:
        return None

    significant = np.abs(reference) >= _TREND_FLOOR * scale
    if int(np.count_nonzero(significant)) < _TREND_MIN_POINTS:
        return None
    trapped = trapped[significant]
    difference = np.abs(values[overlap][significant] / reference[significant] - 1.0)

    order = np.argsort(trapped)
    trapped, difference = trapped[order], difference[order]
    # Equal-count bins rather than equal-width: f_trap is far denser near the edge, and
    # equal-width bins leave the innermost one holding one or two points.
    groups = [g for g in np.array_split(np.arange(trapped.size), _TREND_BINS) if g.size >= 2]
    if len(groups) < 3:
        return None
    bins = [
        {
            "f_trap": float(np.mean(trapped[g])),
            "difference": float(np.mean(difference[g])),
            "stderr": float(np.std(difference[g], ddof=1) / np.sqrt(g.size)),
            "points": int(g.size),
        }
        for g in groups
    ]

    direction, reason = _monotonic_direction(bins)
    grows = None if direction is None else direction > 0
    summary = {
        "bins": bins,
        "points": int(trapped.size),
        "evaluated_above": _TREND_FLOOR,
        "monotonic": None if direction is None else ("increasing" if direction > 0 else "decreasing"),
        "grows_with_trapping": grows,
        # Descriptive, and only that: the first and last bin. They are what the old
        # two-point verdict was built on, kept so a reader can see why it was unstable.
        "f_trap_low": bins[0]["f_trap"],
        "f_trap_high": bins[-1]["f_trap"],
        "difference_low": bins[0]["difference"],
        "difference_high": bins[-1]["difference"],
    }
    if reason is not None:
        summary["reason"] = reason
    return summary


def _monotonic_direction(bins: Sequence[Mapping[str, Any]]) -> tuple[Optional[int], Optional[str]]:
    """``(+1, None)``, ``(-1, None)`` or ``(None, why not)`` for a binned sequence.

    A step smaller than the two bins' combined standard error is flat, not a reversal:
    the question is whether the sequence moves one way, not whether every adjacent pair
    is strictly ordered. A step that reverses by *more* than that error is a real
    reversal and the sequence has no direction.
    """
    ups: list[int] = []
    downs: list[int] = []
    for before, after in zip(bins[:-1], bins[1:]):
        step = after["difference"] - before["difference"]
        noise = float(np.hypot(before["stderr"], after["stderr"]))
        if abs(step) <= noise:
            continue
        (ups if step > 0 else downs).append(1)
    if ups and downs:
        return None, (
            f"the gap falls and rises again across the band ({len(downs)} falling and "
            f"{len(ups)} rising steps beyond their own scatter), so it has no single "
            "direction against the trapped fraction on this state"
        )
    if not ups and not downs:
        return None, "every step is within its own scatter, so no direction is resolved"
    return (1 if ups else -1), None


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

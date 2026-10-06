"""Equilibrium-quality cohorts: why a reconstruction is good, admissible or unreconstructible (#1644).

This module **composes** existing layers; it defines no rule of its own.

* the study policy -- ``workflow/efit_uncertainty_calibration/criteria.py``
  (#891/#1331, criteria v2): ``evaluate`` per (shot, time, setting) and
  ``slice_labels`` across settings.  Loaded by path (it is a workflow module,
  not library code), never copied;
* the EFIT evidence -- :mod:`vaft.omas.efit_quality` (``fit_quality_metrics``,
  ``convergence_metrics``);
* the generic scientific report -- :func:`vaft.validation.equilibrium.validate_equilibrium`.

:func:`equilibrium_quality_table` turns study records into one row per
``(shot, time, setting)`` with three kinds of column kept apart:

* **raw metrics** -- the numbers the rules read (``beta_p``, ``probe_reduced_chi2`` …);
* **verdicts** -- each study criterion's status (``*_status``) and each
  sub-rule it is made of (``rule_*``), read back from the criterion's own
  verdict, never re-thresholded here;
* **classification** -- the slice-level label (``good`` / ``admissible`` /
  ``unreconstructible``) and this setting's ``good`` / ``physically_consistent``.

Thomson consistency stays a separate column: criteria v2 never lets it decide
``good``, and nothing here folds it back in.  The optional evidence join adds
the :mod:`vaft.omas.efit_quality` and generic-validation columns for rows whose
equilibrium product the caller can supply.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

#: Where the study policy lives in a source checkout.
CRITERIA_PATH = Path(__file__).resolve().parents[2] / "workflow" / "efit_uncertainty_calibration" / "criteria.py"

#: Cohorts in their report order.  ``admissible`` means admissible-only (no
#: setting good there); ``unreconstructible`` means no tested setting gave an
#: admissible reconstruction -- a statement across settings, not a bad fit.
COHORTS = ("good", "admissible", "unreconstructible")

#: Admissibility sub-rules, recognised by the reason prefix ``criteria.admissible``
#: writes for each.  The thresholds stay in ``criteria.CRITERIA``.
ADMISSIBILITY_RULES = (
    ("rule_convergence", "not converged"),
    ("rule_pressure_nonnegative", "pressure_min"),
    ("rule_beta_p_positive", "betap"),
    ("rule_w_positive", "wmhd"),
    ("rule_q95", "q95"),
    ("rule_ip_ratio", "ip ratio"),
)
#: Measurement sub-rules: the families ``criteria.measurement`` grades.
MEASUREMENT_RULES = (
    ("rule_probe_fit", "probe"),
    ("rule_loop_fit", "loop"),
    ("rule_ip_fit", "ip"),
    ("rule_dia_fit", "dia"),
)
#: The criterion-level verdict columns, in criteria order.
CRITERION_COLUMNS = ("admissible_status", "measurement_status", "virial_status",
                     "grad_shafranov_status", "thomson_status")
#: Every verdict column the census reports, in the order of the issue's matrix.
FIT_QUALITY_RULES = (tuple(name for name, _ in ADMISSIBILITY_RULES) + tuple(name for name, _ in MEASUREMENT_RULES)
                     + ("virial_status", "grad_shafranov_status"))
#: The census matrix also shows Thomson, as its own dimension; the failure
#: Pareto (what keeps a slice from ``good``) does not, because it never does.
RULE_COLUMNS = FIT_QUALITY_RULES + ("thomson_status",)
#: The production configuration's rows are reported, never counted towards a
#: cohort (``criteria.slice_labels`` excludes them the same way).
ROUTINE_SETTING = "routine"
STATUSES = ("pass", "fail", "indeterminate", "not_available")


def load_study_criteria(path: str | Path | None = None) -> ModuleType:
    """The #891/#1331 study criteria module, loaded from ``path`` (default: this checkout's)."""
    path = (Path(path) if path is not None else CRITERIA_PATH).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"study criteria not found at {path}: the classification policy is a workflow module "
            "(workflow/efit_uncertainty_calibration/criteria.py); pass its path from a source checkout"
        )
    name = f"_vaft_study_criteria_{abs(hash(str(path)))}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling resolve the module by name
    spec.loader.exec_module(module)
    return module


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _rule_status(verdict: Mapping[str, Any], prefix: str) -> str:
    """A sub-rule's status read back from its criterion's verdict."""
    status = verdict.get("status")
    if status in ("not_available", "indeterminate"):
        return status
    failed = any(str(reason).startswith(prefix) for reason in verdict.get("reasons") or ())
    return "fail" if failed else "pass"


def _family_status(measurement: Mapping[str, Any], family: str) -> str:
    entry = (measurement.get("families") or {}).get(family) or {}
    if not entry.get("graded"):
        return "not_available"
    return _rule_status(measurement, f"{family} ")


def _row(record: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    verdicts = evaluation["verdicts"]
    scalars = record.get("scalars") or {}
    fit = record.get("fit") or {}
    measurement = verdicts["measurement"]
    families = measurement.get("families") or {}
    thomson = (record.get("thomson") or {}).get("log_ratio")
    ip_measured = _finite(record.get("ip_measured"))
    reasons = [f"{name}: {reason}" for name in verdicts for reason in verdicts[name].get("reasons") or ()
               if verdicts[name].get("status") == "fail"]
    row = {
        "shot": int(record["shot"]),
        "time_s": round(int(record["time_ms"]) * 1e-3, 4),
        "setting": record.get("setting"),
        # verdicts
        "admissible_status": verdicts["admissible"]["status"],
        "measurement_status": measurement["status"],
        "virial_status": verdicts["virial"]["status"],
        "grad_shafranov_status": verdicts["grad_shafranov"]["status"],
        "thomson_status": verdicts["thomson"]["status"],
        **{name: _rule_status(verdicts["admissible"], prefix) for name, prefix in ADMISSIBILITY_RULES},
        **{name: _family_status(measurement, family) for name, family in MEASUREMENT_RULES},
        # this setting's classification
        "setting_good": bool(evaluation["good"]),
        "physically_consistent": evaluation["physically_consistent"],
        "criteria_version": evaluation.get("criteria_version"),
        # raw metrics
        "converged": bool(record.get("converged")),
        "pressure_min": _finite(record.get("pressure_min")),
        "beta_p": _finite(scalars.get("betap")),
        "w_mhd_j": _finite(scalars.get("wmhd")),
        "q95": _finite(scalars.get("q95")),
        "ip_reconstructed_over_measured": (_finite(scalars.get("ipmhd")) / ip_measured
                                           if ip_measured else math.nan),
        "probe_reduced_chi2": _finite(fit.get("probe_reduced_chi2")),
        "loop_reduced_chi2": _finite(fit.get("loop_reduced_chi2")),
        "ip_z": _finite((families.get("ip") or {}).get("z")),
        "dia_z": _finite((families.get("dia") or {}).get("z")),
        "gs_residual": _finite((record.get("gs") or {}).get("whole")),
        "virial_log_ratio": _finite(verdicts["virial"].get("log_ratio")),
        "p_over_p_e_points": math.exp(-thomson) if thomson is not None and math.isfinite(_finite(thomson)) else math.nan,
        "failure_reasons": "; ".join(reasons),
    }
    return row


def equilibrium_quality_table(records: Iterable[Mapping[str, Any]], *, criteria: ModuleType | None = None,
                              evidence: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None):
    """One row per ``(shot, time, setting)``: raw metrics, verdicts and the study classification.

    ``records`` are study slice records (``weight_scan.py`` / the #1331 Tier A
    analysis); records carrying ``"error"`` are attempts that produced nothing
    and appear only through the slice label.  ``evidence`` -- optional --
    maps a row (the dict before it becomes a frame) to extra columns, e.g.
    :func:`efit_evidence_columns` on that slice's equilibrium product.
    Returns a :class:`pandas.DataFrame`.
    """
    import pandas as pd

    criteria = criteria or load_study_criteria()
    records = list(records)
    rows = []
    for record in records:
        if "error" in record:
            continue
        row = _row(record, criteria.evaluate(record))
        if evidence is not None:
            row.update(evidence(row) or {})
        rows.append(row)
    labels = {(int(entry["shot"]), int(entry["time_ms"])): entry for entry in criteria.slice_labels(records)}
    for row in rows:
        entry = labels.get((row["shot"], round(row["time_s"] * 1e3)))
        row["quality_label"] = entry["label"] if entry else None
        for key in ("good", "admissible", "consistent", "inconsistent"):
            row[f"{key}_settings"] = ",".join(entry.get(key) or ()) if entry else ""
    columns = ["shot", "time_s", "setting", "quality_label", "setting_good", "physically_consistent"]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    # Three-valued (True / False / None): kept as Python objects, so an
    # all-judged column does not become numpy booleans that `is True` misses.
    frame["physically_consistent"] = pd.Series([row["physically_consistent"] for row in rows], dtype=object)
    return frame[columns + [c for c in frame.columns if c not in columns]]


def slice_cohorts(table) -> Any:
    """One row per slice: its label and, for the census, the row of its *best* setting.

    The best setting is the first good one, else the first admissible one, else
    the first attempted (setting name order), so a slice is counted once with
    the rule outcomes that earned -- or failed to earn -- its label.  The
    ``routine`` setting's rows are left out, as the label leaves them out.
    """
    if table.empty:
        return table
    study = table[table["setting"] != ROUTINE_SETTING]
    rank = study.assign(_rank=(~study["setting_good"].astype(bool)).astype(int) * 2
                        + (study["admissible_status"] != "pass").astype(int))
    # Whole rows: groupby().first() would take each column's first non-null
    # value and stitch cells of different settings into one row.
    best = rank.sort_values(["shot", "time_s", "_rank", "setting"]).drop_duplicates(["shot", "time_s"])
    return best.drop(columns="_rank").reset_index(drop=True)


def equilibrium_quality_summary(table) -> dict[str, Any]:
    """Slice counts per cohort, and Thomson consistency within each.

    For ``good`` slices Thomson is judged over all good settings (any
    consistent one makes the slice consistent); for the others, on the best
    setting's row.
    """
    slices = slice_cohorts(table)
    out: dict[str, Any] = {"slices": int(len(slices)), "rows": int(len(table)), "cohorts": {}}
    for cohort in COHORTS:
        group = slices[slices["quality_label"] == cohort] if len(slices) else slices
        if cohort == "good":
            # Over all good settings, as criteria.slice_labels reports it: a
            # slice is consistent when any good setting is.
            verdicts = [True if c else False if i else None
                        for c, i in zip(group.get("consistent_settings", []), group.get("inconsistent_settings", []))]
        else:
            verdicts = list(group["physically_consistent"]) if len(group) else []
        out["cohorts"][cohort] = {
            "slices": int(len(group)),
            "thomson_consistent": int(sum(value is True for value in verdicts)),
            "thomson_inconsistent": int(sum(value is False for value in verdicts)),
            "thomson_not_judged": int(sum(value is None or value != value for value in verdicts)),
        }
    return out


def equilibrium_quality_failure_census(table) -> dict[str, Any]:
    """The rule × cohort matrix (status fractions) and a failure Pareto per cohort.

    Each slice contributes its best setting's row (:func:`slice_cohorts`).
    Fractions are over all of a cohort's slices, ``not_available`` included.
    The Pareto counts, among slices that are not good, how often each
    fit-quality rule fails; Thomson is in the matrix but not in the Pareto.
    """
    slices = slice_cohorts(table)
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for rule in RULE_COLUMNS:
        matrix[rule] = {}
        for cohort in COHORTS:
            group = slices[slices["quality_label"] == cohort][rule] if len(slices) else []
            n = len(group)
            matrix[rule][cohort] = {"n": n, **{status: (float((group == status).sum()) / n if n else math.nan)
                                               for status in STATUSES}}
    pareto = {}
    for cohort in ("admissible", "unreconstructible"):
        group = slices[slices["quality_label"] == cohort] if len(slices) else slices
        counts = {rule: int((group[rule] == "fail").sum()) for rule in FIT_QUALITY_RULES if len(group)}
        pareto[cohort] = sorted(((rule, n) for rule, n in counts.items() if n), key=lambda item: -item[1])
    return {"matrix": matrix, "pareto": pareto, "slices": int(len(slices))}


#: Generic-validation key ↔ study column pairs the crosswalk compares.  Not a
#: one-to-one mapping: each pair answers related but different questions, and
#: the crosswalk reports agreement, not equivalence.
CROSSWALK = (
    ("verification.convergence", "rule_convergence"),
    ("diagnostic_fit.global", "measurement_status"),
    ("diagnostic_fit.bpol_probe", "rule_probe_fit"),
    ("diagnostic_fit.flux_loop", "rule_loop_fit"),
    ("diagnostic_fit.ip", "rule_ip_fit"),
    ("diagnostic_fit.diamagnetic_flux", "rule_dia_fit"),
    ("physical_validity.virial_pair_consistency", "virial_status"),
    ("physical_validity.pressure_profile", "rule_pressure_nonnegative"),
    ("independent_validation.thomson_pressure", "thomson_status"),
)


def equilibrium_quality_crosswalk(table) -> list[dict[str, Any]]:
    """Generic validation status × study verdict, per paired concept, where both exist.

    Rows need ``validation.<category>.<check>`` columns (from
    :func:`efit_evidence_columns`); pairs without them are reported as absent.
    """
    out = []
    for generic, study in CROSSWALK:
        column = f"validation.{generic}"
        if column not in table.columns or study not in table.columns:
            out.append({"generic": generic, "study": study, "available": False})
            continue
        pairs = table[[column, study]].dropna()
        counts: dict[str, int] = {}
        for g, s in zip(pairs[column], pairs[study]):
            counts[f"{g}|{s}"] = counts.get(f"{g}|{s}", 0) + 1
        decided = [(g, s) for g, s in zip(pairs[column], pairs[study])
                   if g in ("pass", "fail", "warn") and s in ("pass", "fail")]
        # ``warn`` has no study counterpart: agreement is reported both ways.
        agree = sum((g == "fail") == (s == "fail") for g, s in decided)
        agree_strict = sum((g in ("fail", "warn")) == (s == "fail") for g, s in decided)
        out.append({"generic": generic, "study": study, "available": True, "rows": int(len(pairs)),
                    "decided": len(decided),
                    "agreement": (agree / len(decided)) if decided else math.nan,
                    "agreement_warn_as_fail": (agree_strict / len(decided)) if decided else math.nan,
                    "counts": counts})
    return out


def efit_evidence_columns(ods: Any, time_slice: int, *, diagnostics: Any = None,
                          kinetic_profiles: Any = None) -> dict[str, Any]:
    """The :mod:`vaft.omas.efit_quality` and generic-validation columns of one equilibrium slice.

    Raw evidence: per-family ``z_rms`` / ``z_bias`` / ``z_abs_max`` / outlier
    fraction, the global reduced chi-square and EFIT's own acceptance.  Verdicts:
    every ``validate_equilibrium`` check's status for this slice, as
    ``validation.<category>.<check>``.  Definitions are those modules'; nothing
    is recomputed here.
    """
    from vaft.omas.efit_quality import convergence_metrics, fit_quality_metrics
    from vaft.validation.equilibrium import validate_equilibrium

    out: dict[str, Any] = {}
    fit = fit_quality_metrics(ods, time_slice=time_slice)
    # Evidence columns carry an ``evidence.`` prefix so they never overwrite
    # a study record's raw metric of the same name (e.g. ``ip_z``).
    out["evidence.global_reduced_chi2"] = _finite(fit.get("chi_squared_reduced"))
    out["evidence.uncertainty_model"] = fit.get("uncertainty_model")
    for family, entry in (fit.get("families") or {}).items():
        for key in ("z_rms", "z_bias", "z_abs_max", "residual_rms_display"):
            if key in entry:
                out[f"evidence.{family}_{key}"] = _finite(entry[key])
        fractions = entry.get("outlier_fraction") or {}
        if "gt_3sigma" in fractions:
            out[f"evidence.{family}_outlier_fraction_3sigma"] = _finite(fractions["gt_3sigma"])
    for family, entry in (fit.get("scalars") or {}).items():
        out[f"evidence.{family}_z"] = _finite(entry.get("z"))
    convergence = convergence_metrics(ods, time_slice=time_slice)
    out["evidence.efit_accepted"] = (convergence.get("verdict") or {}).get("accepted")
    out["evidence.efit_hit_iteration_cap"] = (convergence.get("iterations") or {}).get("hit_cap")
    report = validate_equilibrium(ods, diagnostics=diagnostics, kinetic_profiles=kinetic_profiles,
                                  time_slice=time_slice)
    for category, value in report.items():
        if not isinstance(value, Mapping) or category in ("summary", "provenance"):
            continue
        for check, result in value.items():
            if isinstance(result, Mapping) and "status" in result:
                slices = result.get("slices") or []
                status = slices[0].get("status") if len(slices) == 1 else result["status"]
                out[f"validation.{category}.{check}"] = status
    return out


def equilibrium_quality_constraint_points(ods: Any, time_slice: int) -> list[dict[str, Any]]:
    """Every constraint channel of one slice: measured, reconstructed and normalised residual.

    Read through :func:`vaft.omas.efit_quality.constraint_table`, with ``z``
    from its ``normalized_residuals`` at the family's own units-of-fit factor
    and ``fitted`` from ``fitted_mask`` -- the definitions the fit-quality
    metrics use, so a population plot and the per-slice metrics agree.
    Array families come in their display units; Ip [kA] and the diamagnetic
    flux [mWb] as one point each, with ``z`` from ``fit_quality_metrics``.
    """
    from vaft.omas import efit_quality as q

    points: list[dict[str, Any]] = []
    for family, _title, unit, scale, is_array in q.FAMILIES:
        table = q.constraint_table(ods, time_slice=time_slice, family=family, is_array=is_array, scale=scale)
        if not len(table.index):
            continue
        k, _spread = q.sigma_unit_factor(table)
        z = q.normalized_residuals(table, k) if math.isfinite(k) and k > 0 else np.full(len(table.index), math.nan)
        fitted = q.fitted_mask(table)
        for i in range(len(table.index)):
            points.append({"family": family, "channel": table.source[i], "unit": unit,
                           "measured": float(table.measured[i]), "reconstructed": float(table.reconstructed[i]),
                           "uncertainty": float(table.uncertainty[i]), "state": table.state[i],
                           "fitted": bool(fitted[i]), "z": float(z[i])})
    scalars = (q.fit_quality_metrics(ods, time_slice=time_slice).get("scalars") or {})
    for family, unit, scale in (("ip", "kA", 1e-3), ("diamagnetic_flux", "mWb", 1e3)):
        entry = scalars.get(family) or {}
        measured, reconstructed = _finite(entry.get("measured")), _finite(entry.get("reconstructed"))
        if math.isfinite(measured) or math.isfinite(reconstructed):
            points.append({"family": family, "channel": family, "unit": unit, "measured": measured * scale,
                           "reconstructed": reconstructed * scale, "uncertainty": math.nan, "state": "enabled",
                           "fitted": math.isfinite(_finite(entry.get("z"))), "z": _finite(entry.get("z"))})
    return points


#: The four representative cases of #1644 §6, in report order.
REPRESENTATIVE_CASES = ("good_consistent", "good_inconsistent", "admissible_only", "unreconstructible")
#: The metrics a "typical" good slice is judged typical on (#1644 §7): distance
#: to the cohort median in each, scaled by the cohort's own spread.
TYPICALITY_METRICS = ("probe_reduced_chi2", "loop_reduced_chi2", "gs_residual", "virial_log_ratio")


def _typicality(group, metrics: Sequence[str]):
    """Scaled distance of each row to the group median over ``metrics`` (log for chi-square)."""
    import pandas as pd

    distance = pd.Series(0.0, index=group.index)
    for metric in metrics:
        if metric not in group:
            continue
        values = pd.to_numeric(group[metric], errors="coerce")
        if metric.endswith("reduced_chi2"):
            values = np.log(values.where(values > 0))
        elif metric == "virial_log_ratio":
            values = values.abs()
        centre = values.median()
        spread = (values - centre).abs().median() or values.std() or 1.0
        if not math.isfinite(centre):
            continue
        distance = distance.add(((values - centre) / spread).abs().fillna(10.0), fill_value=0.0)
    return distance


def select_representative_cases(table) -> list[dict[str, Any]]:
    """Deterministic representative slices for the four #1644 cases, with why each was chosen.

    * ``good_consistent`` / ``good_inconsistent`` -- among good slices whose
      best setting is Thomson-consistent / inconsistent, the one closest to the
      cohort median in :data:`TYPICALITY_METRICS`;
    * ``admissible_only`` -- among admissible-only slices failing the cohort's
      dominant blocking rule (the census Pareto), the one closest to the median
      of that rule's metric;
    * ``unreconstructible`` -- among unreconstructible slices failing the
      cohort's dominant rule, preferring rows that still carry product evidence
      (a failed attempt to inspect), the earliest (shot, time).

    Ties break by (shot, time); nothing is hand-picked.  A case with no
    candidate is returned with ``shot`` ``None`` and the reason.
    """
    slices = slice_cohorts(table)
    census = equilibrium_quality_failure_census(table)
    out: list[dict[str, Any]] = []

    def pick(case: str, group, distance, reason: str) -> None:
        if group is None or not len(group):
            out.append({"case": case, "shot": None, "time_s": None, "setting": None, "reason": reason + " -- no candidate"})
            return
        order = group.assign(_d=distance).sort_values(["_d", "shot", "time_s"])
        row = order.iloc[0]
        out.append({"case": case, "shot": int(row["shot"]), "time_s": float(row["time_s"]), "setting": row["setting"],
                    "candidates": int(len(group)), "reason": reason, "failure_reasons": row.get("failure_reasons", "")})

    good = slices[slices["quality_label"] == "good"] if len(slices) else slices
    for case, flag in (("good_consistent", True), ("good_inconsistent", False)):
        group = good[good["physically_consistent"].map(lambda v: v is flag)] if len(good) else good
        pick(case, group, _typicality(group, TYPICALITY_METRICS) if len(group) else None,
             f"good, Thomson {'consistent' if flag else 'inconsistent'}; closest to the cohort median in "
             + ", ".join(TYPICALITY_METRICS))

    metric_of = {"rule_probe_fit": "probe_reduced_chi2", "rule_loop_fit": "loop_reduced_chi2",
                 "rule_ip_fit": "ip_z", "rule_dia_fit": "dia_z", "virial_status": "virial_log_ratio",
                 "grad_shafranov_status": "gs_residual"}
    admissible = slices[slices["quality_label"] == "admissible"] if len(slices) else slices
    pareto = census["pareto"].get("admissible") or []
    if pareto:
        rule = pareto[0][0]
        group = admissible[admissible[rule] == "fail"]
        metric = metric_of.get(rule)
        pick("admissible_only", group, _typicality(group, [metric]) if metric and len(group) else
             group["shot"] * 0.0, f"admissible-only, failing the dominant blocking rule {rule}"
             + (f"; closest to the median {metric} of those" if metric else ""))
    else:
        pick("admissible_only", None, None, "admissible-only")

    unrec = slices[slices["quality_label"] == "unreconstructible"] if len(slices) else slices
    pareto = census["pareto"].get("unreconstructible") or []
    if pareto:
        rule = pareto[0][0]
        group = unrec[unrec[rule] == "fail"]
        has_evidence = (group["evidence_status"] == "ok") if "evidence_status" in group else group["shot"] * 0 == 0
        pick("unreconstructible", group, (~has_evidence).astype(float),
             f"unreconstructible, failing the dominant rule {rule}; a slice with a stored failed attempt first, "
             "then the earliest (shot, time)")
    else:
        pick("unreconstructible", None, None, "unreconstructible")
    return out


__all__: Sequence[str] = (
    "ADMISSIBILITY_RULES", "COHORTS", "CRITERIA_PATH", "CROSSWALK", "MEASUREMENT_RULES", "RULE_COLUMNS",
    "FIT_QUALITY_RULES", "REPRESENTATIVE_CASES", "ROUTINE_SETTING", "TYPICALITY_METRICS", "STATUSES", "efit_evidence_columns", "equilibrium_quality_constraint_points", "equilibrium_quality_crosswalk", "equilibrium_quality_failure_census",
    "equilibrium_quality_summary", "equilibrium_quality_table", "load_study_criteria", "select_representative_cases", "slice_cohorts",
)

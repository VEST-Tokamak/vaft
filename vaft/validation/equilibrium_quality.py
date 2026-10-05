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
RULE_COLUMNS = (tuple(name for name, _ in ADMISSIBILITY_RULES) + tuple(name for name, _ in MEASUREMENT_RULES)
                + ("virial_status", "grad_shafranov_status", "thomson_status"))
STATUSES = ("pass", "fail", "indeterminate", "not_available")


def load_study_criteria(path: str | Path | None = None) -> ModuleType:
    """The #891/#1331 study criteria module, loaded from ``path`` (default: this checkout's)."""
    path = Path(path) if path is not None else CRITERIA_PATH
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
    the rule outcomes that earned -- or failed to earn -- its label.
    """
    if table.empty:
        return table
    rank = table.assign(_rank=(~table["setting_good"]).astype(int) * 2
                        + (table["admissible_status"] != "pass").astype(int))
    best = rank.sort_values(["shot", "time_s", "_rank", "setting"]).groupby(["shot", "time_s"], as_index=False).first()
    return best.drop(columns="_rank")


def equilibrium_quality_summary(table) -> dict[str, Any]:
    """Slice counts per cohort, and Thomson consistency within each."""
    slices = slice_cohorts(table)
    out: dict[str, Any] = {"slices": int(len(slices)), "rows": int(len(table)), "cohorts": {}}
    for cohort in COHORTS:
        group = slices[slices["quality_label"] == cohort] if len(slices) else slices
        consistent = group["physically_consistent"] if len(group) else []
        out["cohorts"][cohort] = {
            "slices": int(len(group)),
            "thomson_consistent": int(sum(value is True for value in consistent)),
            "thomson_inconsistent": int(sum(value is False for value in consistent)),
            "thomson_not_judged": int(sum(value is None or value != value for value in consistent)),
        }
    return out


def equilibrium_quality_failure_census(table) -> dict[str, Any]:
    """The rule × cohort matrix (status fractions) and a failure Pareto per cohort.

    Each slice contributes its best setting's row (:func:`slice_cohorts`).  The
    Pareto counts, among slices that are not good, how often each rule fails.
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
        counts = {rule: int((group[rule] == "fail").sum()) for rule in RULE_COLUMNS if len(group)}
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
    out["global_reduced_chi2"] = _finite(fit.get("chi_squared_reduced"))
    out["uncertainty_model"] = fit.get("uncertainty_model")
    for family, entry in (fit.get("families") or {}).items():
        for key in ("z_rms", "z_bias", "z_abs_max", "residual_rms_display"):
            if key in entry:
                out[f"{family}_{key}"] = _finite(entry[key])
        fractions = entry.get("outlier_fraction") or {}
        if "gt_3sigma" in fractions:
            out[f"{family}_outlier_fraction_3sigma"] = _finite(fractions["gt_3sigma"])
    for family, entry in (fit.get("scalars") or {}).items():
        out[f"{family}_z"] = _finite(entry.get("z"))
    convergence = convergence_metrics(ods, time_slice=time_slice)
    out["efit_accepted"] = (convergence.get("verdict") or {}).get("accepted")
    out["efit_hit_iteration_cap"] = (convergence.get("iterations") or {}).get("hit_cap")
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


__all__: Sequence[str] = (
    "ADMISSIBILITY_RULES", "COHORTS", "CRITERIA_PATH", "CROSSWALK", "MEASUREMENT_RULES", "RULE_COLUMNS",
    "STATUSES", "efit_evidence_columns", "equilibrium_quality_crosswalk", "equilibrium_quality_failure_census",
    "equilibrium_quality_summary", "equilibrium_quality_table", "load_study_criteria", "slice_cohorts",
)

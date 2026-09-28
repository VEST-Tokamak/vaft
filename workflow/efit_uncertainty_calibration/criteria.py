"""What makes a scanned EFIT setting good, per slice (#891).

The four criteria agreed on #891 after the diamagnetic sign was corrected
(#1196), behind a physical-admissibility veto:

* **admissible** -- converged, positive pressure everywhere, beta_p and W > 0,
  q95 above a floor, Ip within a bound of the measurement.  A slice that fails
  is *not reconstructed*, whatever its chi-square.
* **measurement** -- every fitted family's reduced chi-square against the sigma
  EFIT fitted with (probes, loops, Ip, PF currents, diamagnetic flux) in a band.
* **virial** -- the reconstruction's own beta_p (pressure volume integral)
  against the RT-free ``pair_13`` virial closure.  The closure is inverted
  only where its denominator is not near singular; elsewhere the slice is
  *indeterminate* rather than scored (#649: the other closures are
  ill-conditioned at VEST's aspect ratio).
* **grad_shafranov** -- the relative Grad-Shafranov residual of the g-file.
* **thomson** -- only where Thomson samples lie in the slice's window:
  ``p_e <= p_recon <= 3 p_e`` at the sampled points, i.e.
  ``-ln 3 <= ln(sum p_e / sum p_recon) <= 0``.  Thomson measures electrons, so
  ``p_e`` is a lower bound; three times it bounds any credible ion and
  impurity share.

Every function here is pure: a slice record in, a verdict out.  Verdicts are
``"pass"``, ``"fail"`` or ``"indeterminate"``; a criterion that cannot be
evaluated on a slice (no Thomson there) is ``"not_available"`` and is never
counted against it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PASS, FAIL, INDETERMINATE, NOT_AVAILABLE = "pass", "fail", "indeterminate", "not_available"

#: The fitted families and the band their reduced chi-square must fall in.
FAMILIES = ("probe", "loop", "ip", "pf", "dia")

CRITERIA: dict[str, Any] = {
    "q95_min": 2.0,
    "ip_relative_max": 0.03,
    "reduced_chi2_band": (0.5, 2.0),
    # Graded families.  The PF currents are coil currents fitted against a
    # 1e-4 relative sigma: their chi-square is reported, not graded.
    "graded_families": ("probe", "loop", "ip", "dia"),
    "virial_log_ratio_max": math.log(2.0),
    "virial_denominator_min": 0.1,
    # Provisional: the routine reconstructions sit near 0.5 %; revisit once
    # stage 1 shows the spread across settings.
    "grad_shafranov_max": 0.05,
    "thomson_log_ratio_band": (-math.log(3.0), 0.0),
}

ORDER = ("admissible", "measurement", "virial", "grad_shafranov", "thomson")


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def admissible(record: Mapping[str, Any], criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    """The veto: a slice that fails this is not a reconstruction."""
    reasons = []
    if not record.get("converged"):
        reasons.append("not converged")
    scalars = record.get("scalars") or {}
    if not (_finite(record.get("pressure_min")) >= 0.0):
        reasons.append(f"pressure_min {record.get('pressure_min')}")
    for key in ("betap", "wmhd"):
        if not (_finite(scalars.get(key)) > 0.0):
            reasons.append(f"{key} {scalars.get(key)}")
    if not (_finite(scalars.get("q95")) > criteria["q95_min"]):
        reasons.append(f"q95 {scalars.get('q95')}")
    ip_measured, ip_mhd = _finite(record.get("ip_measured")), _finite(scalars.get("ipmhd"))
    if not (abs(ip_mhd - ip_measured) <= criteria["ip_relative_max"] * abs(ip_measured)):
        reasons.append(f"ip {ip_mhd} vs {ip_measured}")
    return {"status": FAIL if reasons else PASS, "reasons": reasons}


def measurement(record: Mapping[str, Any], criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    """Every graded family's reduced chi-square inside the band."""
    fit = record.get("fit") or {}
    low, high = criteria["reduced_chi2_band"]
    families, reasons = {}, []
    for family in FAMILIES:
        n = int(fit.get(f"{family}_n") or 0)
        value = _finite(fit.get(f"{family}_reduced_chi2"))
        # A family the setting held inactive (a sigma scaled out of the fit)
        # still appears in the m-file with a positive processed weight.
        active = family not in (record.get("inactive_families") or ())
        families[family] = {"n": n, "reduced_chi2": value,
                            "graded": family in criteria["graded_families"] and n > 0 and active}
        if families[family]["graded"] and not (low <= value <= high):
            reasons.append(f"{family} reduced chi2 {value:.3g}")
    if not any(v["graded"] for v in families.values()):
        return {"status": NOT_AVAILABLE, "families": families, "reasons": ["no fitted family"]}
    return {"status": FAIL if reasons else PASS, "families": families, "reasons": reasons}


def virial(record: Mapping[str, Any], criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    """beta_p from the pressure integral against the pair_13 virial closure."""
    v = record.get("virial") or {}
    integral, closure = _finite(v.get("beta_p_integral")), _finite(v.get("beta_p_pair_13"))
    denominator = _finite(v.get("denominator_pair_13"))
    if not math.isfinite(denominator) or abs(denominator) < criteria["virial_denominator_min"]:
        return {"status": INDETERMINATE, "reasons": [f"pair_13 denominator {denominator}"]}
    if not (integral > 0.0 and closure > 0.0):
        return {"status": INDETERMINATE, "reasons": [f"beta_p {integral} vs {closure}"]}
    ratio = math.log(integral / closure)
    ok = abs(ratio) <= criteria["virial_log_ratio_max"]
    return {"status": PASS if ok else FAIL, "log_ratio": ratio,
            "reasons": [] if ok else [f"ln(beta_p ratio) {ratio:.3g}"]}


def grad_shafranov(record: Mapping[str, Any], criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    whole = _finite((record.get("gs") or {}).get("whole"))
    if not math.isfinite(whole):
        return {"status": INDETERMINATE, "reasons": ["no residual"]}
    ok = whole <= criteria["grad_shafranov_max"]
    return {"status": PASS if ok else FAIL, "whole": whole,
            "reasons": [] if ok else [f"GS residual {whole:.3g}"]}


def thomson(record: Mapping[str, Any], criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    ts = record.get("thomson")
    if not ts:
        return {"status": NOT_AVAILABLE, "reasons": []}
    ratio = _finite(ts.get("log_ratio"))
    if not math.isfinite(ratio):
        # No positive reconstructed pressure at the sampled points: the
        # admissibility veto already says why.
        return {"status": INDETERMINATE, "reasons": [str(ts.get("reason"))]}
    low, high = criteria["thomson_log_ratio_band"]
    ok = low <= ratio <= high
    return {"status": PASS if ok else FAIL, "log_ratio": ratio,
            "reasons": [] if ok else [f"ln(sum p_e / sum p_recon) {ratio:.3g}"]}


CHECKS = {
    "admissible": admissible,
    "measurement": measurement,
    "virial": virial,
    "grad_shafranov": grad_shafranov,
    "thomson": thomson,
}


def evaluate(record: Mapping[str, Any], criteria: Mapping[str, Any] = CRITERIA) -> dict[str, Any]:
    """Every criterion on one slice, plus whether the slice is good.

    ``good`` needs admissible and every other criterion either passing or not
    available; an indeterminate virial or Thomson verdict does not count
    against a slice but is reported.
    """
    verdicts = {name: CHECKS[name](record, criteria) for name in ORDER}
    good = verdicts["admissible"]["status"] == PASS and all(
        verdicts[name]["status"] != FAIL for name in ORDER[1:]
    )
    return {"verdicts": verdicts, "good": bool(good)}


def summarize(records: Sequence[Mapping[str, Any]], criteria: Mapping[str, Any] = CRITERIA) -> list[dict[str, Any]]:
    """Per setting: how many slices each criterion passes, and how many are good.

    Counts are over the slices the setting was run on; a criterion's
    denominator excludes its not-available slices, so Thomson is scored only
    where Thomson exists.
    """
    by_setting: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_setting.setdefault(record["setting"], []).append(record)
    rows = []
    for setting, group in by_setting.items():
        evaluated = [evaluate(r, criteria) for r in group]
        row: dict[str, Any] = {"setting": setting, "slices": len(group),
                               "good": sum(e["good"] for e in evaluated)}
        for name in ORDER:
            statuses = [e["verdicts"][name]["status"] for e in evaluated]
            row[name] = {
                status: statuses.count(status)
                for status in (PASS, FAIL, INDETERMINATE, NOT_AVAILABLE)
                if statuses.count(status)
            }
        for family in FAMILIES:
            values = [
                e["verdicts"]["measurement"]["families"][family]["reduced_chi2"]
                for e in evaluated
                if e["verdicts"]["admissible"]["status"] == PASS
                and e["verdicts"]["measurement"].get("families", {}).get(family, {}).get("n")
            ]
            finite = [v for v in values if math.isfinite(v)]
            row[f"{family}_reduced_chi2_median"] = float(np.median(finite)) if finite else float("nan")
        rows.append(row)
    return rows


def good_fraction(row: Mapping[str, Any]) -> float:
    return row["good"] / row["slices"] if row["slices"] else float("nan")


__all__: Iterable[str] = (
    "CRITERIA", "FAMILIES", "ORDER", "PASS", "FAIL", "INDETERMINATE", "NOT_AVAILABLE",
    "admissible", "measurement", "virial", "grad_shafranov", "thomson", "evaluate",
    "summarize", "good_fraction",
)

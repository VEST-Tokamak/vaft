#!/usr/bin/env python3
"""Summarize a ``weight_scan.py --stage 6`` ensemble (#579) on three separate axes.

    python workflow/efit_uncertainty_calibration/sensitivity_report.py \\
        --tables pilot/table_42962.json pilot/v2_table_39915.json ... --output report.json

The axes are kept apart, as #579 asks.  None of them selects a configuration.

1. **Residual credibility** -- fit quality (``criteria.evaluate``'s ``good``:
   admissible, measurement, virial, Grad-Shafranov) and each family's reduced
   chi-square.
2. **Physical consistency** -- Thomson ``1 <= p/p_e <= 2``
   (``physically_consistent``), reported where Thomson exists.
3. **Robustness** -- per slice, the spread of W, beta_p, l_i and q95 across
   the configurations that pass fit quality: the model-form uncertainty
   sigma_EFIT,model.  Thomson never enters this set.  The same spread is also
   given over the *admissible* members: the reduced-chi-square band is not
   invariant along the sigma axis (halving a family's sigma quadruples its
   chi2r), so ``good`` alone would confound that axis with the band.

The marginal effect of each axis of the ensemble (diamagnetic sigma, profile
basis, probe/loop sigma) is reported as the median, over slices, of each
quantity relative to the working setting on the same slice.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
QUANTITIES = ("wmhd", "betap", "li", "q95")
#: The working setting (#891 stage 4, the library default since #1440).
WORKING = {"basis": (2, 1), "dia": 16.0, "probe": 3.62, "loop": 2.15}
NAME = re.compile(r"^p(?P<p>\d)f(?P<f>\d)_probe_x(?P<probe>[\d.]+)_loop_x(?P<loop>[\d.]+)_dia_(?P<dia>off|x[\d.]+)")


def _criteria():
    spec = importlib.util.spec_from_file_location("sensitivity_criteria", HERE / "criteria.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def axes(setting: str) -> dict[str, Any]:
    """The ensemble coordinates encoded in a stage-6 setting name."""
    match = NAME.match(setting)
    if not match:
        return {}
    dia = match["dia"]
    return {"basis": (int(match["p"]), int(match["f"])), "dia": None if dia == "off" else float(dia[1:]),
            "probe": float(match["probe"]), "loop": float(match["loop"])}


def is_working(coords: Mapping[str, Any]) -> bool:
    return bool(coords) and all(coords.get(k) == v for k, v in WORKING.items())


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _spread(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if not finite.size:
        return {"n": 0}
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    out = {"n": int(finite.size), "median": float(median), "q1": float(q1), "q3": float(q3),
           "min": float(finite.min()), "max": float(finite.max())}
    # Relative model-form spread: half the interquartile range over the median.
    out["relative_half_iqr"] = float((q3 - q1) / 2 / abs(median)) if median else float("nan")
    return out


def load(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records.extend(r for r in payload["records"] if "error" not in r)
    return records


def report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    criteria = _criteria()
    rows = []
    for record in records:
        evaluation = criteria.evaluate(record)
        scalars = record.get("scalars") or {}
        fit = record.get("fit") or {}
        thomson = evaluation["verdicts"]["thomson"]
        rows.append({
            "shot": int(record["shot"]), "time_ms": int(record["time_ms"]), "setting": record["setting"],
            "axes": axes(record["setting"]), "converged": bool(record.get("converged")),
            "admissible": evaluation["verdicts"]["admissible"]["status"] == criteria.PASS,
            "good": evaluation["good"], "consistent": evaluation["physically_consistent"],
            "p_over_p_e": (math.exp(-thomson["log_ratio"]) if thomson.get("log_ratio") is not None else float("nan")),
            "probe_chi2r": _finite(fit.get("probe_reduced_chi2")), "loop_chi2r": _finite(fit.get("loop_reduced_chi2")),
            **{q: _finite(scalars.get(q)) for q in QUANTITIES},
        })

    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_slice: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[row["setting"]].append(row)
        by_slice[(row["shot"], row["time_ms"])].append(row)

    configurations = []
    for setting, group in sorted(by_setting.items()):
        judged = [r for r in group if r["consistent"] is not None]
        configurations.append({
            "setting": setting, "axes": {k: v for k, v in group[0]["axes"].items()},
            "slices": len(group), "converged": sum(r["converged"] for r in group),
            "admissible": sum(r["admissible"] for r in group), "good": sum(r["good"] for r in group),
            "thomson_judged": len(judged), "consistent": sum(r["consistent"] is True for r in judged),
            "good_and_consistent": sum(r["good"] and r["consistent"] is True for r in group),
            "probe_chi2r_median": float(np.nanmedian([r["probe_chi2r"] for r in group])) if group else float("nan"),
            "loop_chi2r_median": float(np.nanmedian([r["loop_chi2r"] for r in group])) if group else float("nan"),
        })

    slices = []
    for (shot, time_ms), group in sorted(by_slice.items()):
        good = [r for r in group if r["good"]]
        working = next((r for r in group if is_working(r["axes"])), None)
        entry = {"shot": shot, "time_ms": time_ms, "configurations": len(group), "good": len(good),
                 "good_consistent": sum(r["consistent"] is True for r in good),
                 "good_inconsistent": sum(r["consistent"] is False for r in good),
                 "working": None if working is None else {
                     "good": working["good"], "consistent": working["consistent"],
                     "p_over_p_e": working["p_over_p_e"], **{q: working[q] for q in QUANTITIES}},
                 "model_form": {q: _spread([r[q] for r in good]) for q in QUANTITIES},
                 "admissible": sum(r["admissible"] for r in group),
                 "model_form_admissible": {q: _spread([r[q] for r in group if r["admissible"]]) for q in QUANTITIES},
                 "p_over_p_e_over_good": _spread([r["p_over_p_e"] for r in good])}
        slices.append(entry)

    # Marginal effect of each axis: quantity / working-setting quantity on the
    # same slice, median over slices, over the configurations that are good.
    marginal: dict[str, dict[str, Any]] = {}
    for axis in ("basis", "dia", "probe", "loop"):
        values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for (shot, time_ms), group in by_slice.items():
            working = next((r for r in group if is_working(r["axes"])), None)
            if working is None:
                continue
            for r in group:
                if not r["good"] or not r["axes"]:
                    continue
                others = {k: v for k, v in r["axes"].items() if k != axis}
                if any(others[k] != WORKING[k] for k in others if k in WORKING):
                    continue  # one axis at a time, the others at the working setting
                key = str(r["axes"][axis])
                for q in QUANTITIES:
                    if math.isfinite(r[q]) and math.isfinite(working[q]) and working[q]:
                        values[key][q].append(r[q] / working[q])
                values[key]["p_over_p_e"].append(r["p_over_p_e"])
        marginal[axis] = {key: {q: _spread(v) for q, v in qs.items()} for key, qs in sorted(values.items())}

    return {"rows": len(rows), "configurations": configurations, "slices": slices, "marginal": marginal}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report(load(args.tables))
    args.output.write_text(json.dumps(result, indent=1, default=str, allow_nan=True) + "\n", encoding="utf-8")
    print(f"{result['rows']} records, {len(result['configurations'])} configurations, {len(result['slices'])} slices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

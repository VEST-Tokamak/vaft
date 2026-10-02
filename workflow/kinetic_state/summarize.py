"""R_p summary and the #1430 figure candidate from an atlas state table.

Reads ``state.csv`` (and the analysis JSONs, for the #1414 cross-check) and
writes ``summary.json`` plus ``figures/pressure_consistency.{png,pdf}``.

Every statistic is split by EFIT quality.  Since criteria version 2
(2026-10-02) ``good`` is fit quality only and is no longer gated on Thomson,
so ``R_sum`` is free on both; the physical band ``1 <= p/p_e <= 2`` (no fast
ions, ``T_i <= T_e``, ``n_i <= n_e``) is shown and counted, never selected on.
Atlases built under version 1 gated ``good`` rows on ``[1, 3]``.

Usage::

    python3 workflow/kinetic_state/summarize.py --atlas ~/runs/campaign/atlas/v1 \\
        --analysis ~/runs/campaign/tierA_analysis.json \\
        --analysis-tite017 ~/runs/campaign/tierA_analysis_tite017.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _stats(values: Iterable[float]) -> dict[str, Any]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if not finite.size:
        return {"n": 0}
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    return {"n": int(finite.size), "median": float(median), "q1": float(q1), "q3": float(q3),
            "below_1": int(np.sum(finite < 1.0)), "above_2": int(np.sum(finite > 2.0)),
            "above_3": int(np.sum(finite > 3.0))}


def _kinetic_reference(path: Path | None) -> dict[str, Any] | None:
    """The #1414 numbers: median exp(-ts_log_ratio) per lineage over the kinetic rows."""
    if path is None or not path.is_file():
        return None
    rows = json.loads(path.read_text())["kinetic"]
    ratio = lambda key: [math.exp(-r[key]) for r in rows if r.get(key) is not None]  # noqa: E731
    return {"rows": len(rows), "kinetic": _stats(ratio("ts_log_ratio_kin")),
            "magnetics": _stats(ratio("ts_log_ratio_mag"))}


def summarize(rows: Sequence[dict[str, str]], references: dict[str, Path | None]) -> dict[str, Any]:
    matched = [r for r in rows if r["ts_status"] == "matched"]
    summary: dict[str, Any] = {"rows": len(rows), "matched": len(matched), "by": {}}
    for lineage in ("magnetics", "electron_kinetic"):
        for quality in ("good", "admissible", "all"):
            group = [r for r in matched if r["efit_lineage"] == lineage and quality in ("all", r["efit_quality"])]
            summary["by"][f"{lineage}/{quality}"] = {
                "r_sum": _stats(_f(r["r_sum"]) for r in group),
                "r_w": _stats(_f(r["r_w"]) for r in group),
                "r_w_full": _stats(_f(r["r_w_full"]) for r in group),
                "shots": len({r["shot"] for r in group}),
            }
    # the paired set: magnetics and electron-kinetic at the same (shot, time)
    kinetic = {(r["shot"], r["time_efit_s"]): r for r in matched if r["efit_lineage"] == "electron_kinetic"}
    paired = [(m, kinetic[(m["shot"], m["time_efit_s"])]) for m in matched
              if m["efit_lineage"] == "magnetics" and (m["shot"], m["time_efit_s"]) in kinetic]
    summary["paired"] = {
        "n": len(paired),
        "r_sum_magnetics": _stats(_f(m["r_sum"]) for m, _ in paired),
        "r_sum_kinetic": _stats(_f(k["r_sum"]) for _, k in paired),
        "delta_r_w": _stats(_f(k["delta_r_w"]) for _, k in paired),
        "c_p": _stats(_f(k["c_p"]) for _, k in paired),
        "kinetic_admissible": {s: sum(1 for _, k in paired if k["kinetic_admissible"] == s) for s in ("pass", "fail")},
    }
    summary["reference_1414"] = {name: _kinetic_reference(path) for name, path in references.items()}
    summary["paired_kin_status"] = {
        s: sum(1 for r in rows if r["efit_lineage"] == "magnetics" and r["paired_kin_status"] == s)
        for s in ("valid", "failed", "unavailable", "not_attempted")
    }
    return summary


def figure(rows: Sequence[dict[str, str]], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched = [r for r in rows if r["ts_status"] == "matched"]
    colours = {"good": "#2166ac", "admissible": "#e08214"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    left.axhspan(1.0, 2.0, color="0.92", zorder=0, label="physical band: 1 \u2264 p/p$_e$ \u2264 2")
    left.axhline(1.0, color="0.3", lw=0.8, ls="--")
    for quality in ("admissible", "good"):
        for lineage, marker, face in (("magnetics", "o", None), ("electron_kinetic", "s", "none")):
            group = [r for r in matched if r["efit_lineage"] == lineage and r["efit_quality"] == quality]
            if not group:
                continue
            x = [_f(r["ip_measured_a"]) / 1e3 for r in group]
            y = [_f(r["r_sum"]) for r in group]
            left.scatter(x, y, s=22, marker=marker, edgecolors=colours[quality],
                         facecolors=colours[quality] if face is None else face,
                         label=f"{'magnetics-only' if lineage == 'magnetics' else 'electron-kinetic'} · {quality}")
    left.set_xlabel("measured $I_p$ [kA]")
    left.set_ylabel(r"$R_\mathrm{sum}=\sum p_\mathrm{EFIT}/\sum p_{e,\mathrm{TS}}$")
    left.set_yscale("log")
    left.set_title("Pressure consistency at Thomson channels")
    left.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

    kinetic = {(r["shot"], r["time_efit_s"]): r for r in matched if r["efit_lineage"] == "electron_kinetic"}
    pairs = [(m, kinetic[(m["shot"], m["time_efit_s"])]) for m in matched
             if m["efit_lineage"] == "magnetics" and (m["shot"], m["time_efit_s"]) in kinetic]
    xs = [_f(m["r_sum"]) for m, _ in pairs]
    ys = [_f(k["r_sum"]) for _, k in pairs]
    finite = [v for v in xs + ys if math.isfinite(v) and v > 0]
    if finite:
        lo, hi = min(finite) / 1.3, max(finite) * 1.3
        right.plot([lo, hi], [lo, hi], color="0.3", lw=0.8, ls="--", label="$y=x$")
        right.set_xlim(lo, hi)
        right.set_ylim(lo, hi)
    for (m, k), x, y in zip(pairs, xs, ys):
        right.scatter([x], [y], s=26, color=colours[m["efit_quality"]],
                      marker="s" if k["kinetic_admissible"] == "pass" else "x")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel(r"$R_\mathrm{sum}$, magnetics-only (held out)")
    right.set_ylabel(r"$R_\mathrm{sum}$, electron-kinetic ($T_i=T_e$, fit to TS)")
    right.set_title(f"Reconstruction response, {len(pairs)} paired slices")
    right.legend(fontsize=7, loc="upper left")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    ticks = [0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10]
    for axis in (left.yaxis, right.xaxis, right.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(ScalarFormatter())
    fig.text(0.01, -0.10, "#1331 Tier A, statistical_891. Quality is fit quality only (criteria v2); "
             "the shaded band is the physical-consistency check, not a selection. Kinetic markers: "
             "square = own admissibility passes, x = fails.", fontsize=7)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "pressure_consistency.png", out / "pressure_consistency.pdf"]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--analysis-tite017", type=Path)
    args = parser.parse_args(argv)
    atlas = args.atlas.expanduser()
    with open(atlas / "state.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize(rows, {"ti_te_1.0": args.analysis and args.analysis.expanduser(),
                               "ti_te_0.17": args.analysis_tite017 and args.analysis_tite017.expanduser()})
    (atlas / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    figure(rows, atlas / "figures")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

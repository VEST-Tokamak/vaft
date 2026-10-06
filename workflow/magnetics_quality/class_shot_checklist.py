"""Can this shot's diagnostics be trusted?  One row per shot (Lane U, #1543).

The #58 worker turns every new shot into a diagnostics product within minutes.
This script answers, for a list of such shots, the questions a class
instructor or the next pipeline stage asks before using one:

* **Plasma current** -- peak, when, and what is left at the end of the
  analysis window (a Rogowski drift or a baseline extrapolated across the
  discharge shows up there first, #1373).
* **Rogowski sensors** -- the plasma-current and diamagnetic (TF) Rogowski
  validity, and whether a diamagnetic flux was produced at all (#993).
* **Equilibrium magnetics** -- how many probes and flux loops the signal-quality
  layer condemns, per family, and *why*, grouped by detector (#189).
* **PF currents** -- with ``--raw``, the drift of every coil's acquisition
  between its pre-shot baseline and the end of the record, in amperes (#1424).
* **Filterscope** -- how much of the fast filterscope's tail is a clamped copy
  of its last raw sample rather than a measurement (#430).

Everything here is report-only.  The ``flags`` column names conditions worth a
look; the thresholds behind it are provisional until the population scan sets
them (#189 forbids unjustified thresholds becoming gates), so nothing in the
pipeline reads this table.

    PYTHONPATH=$PWD python workflow/magnetics_quality/class_shot_checklist.py \\
        --shots 48930-48940 --filedb /srv/vest.filedb --raw \\
        --table /tmp/checklist.json --markdown /tmp/checklist.md

The FileDB is only read.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
import shutil
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vaft.ods_access import path_value as _get

LOGGER = logging.getLogger("class_shot_checklist")

SCHEMA = 1
#: Report-only, provisional: what the ``flags`` column calls worth a look.
PROVISIONAL = {
    # Record peak on VEST is ~290 kA; #1373 lists 398-1674 kA artefacts.
    "ip_peak_ceiling_kA": 350.0,
    # |Ip| left at the end of the window, as a fraction of the peak.
    "ip_tail_fraction": 0.10,
    # |Ip| in the first 5 ms of the window, as a fraction of the peak.
    "ip_head_fraction": 0.10,
    # Any PF coil whose acquisition moved this much between pre-shot and end.
    "pf_drift_A": 50.0,
    # Fast filterscope tail that is a clamped copy, in milliseconds.
    "filterscope_clamp_ms": 1.0,
}
#: The detectors whose whole-record verdicts are sensor faults rather than a
#: disagreement with neighbours.
HARD_REASONS = ("known_fault", "implausible_magnitude", "flatline", "saturation", "dropout", "unavailable")


def parse_shots(text: str) -> list[int]:
    shots: list[int] = []
    for piece in str(text).replace(" ", "").split(","):
        if not piece:
            continue
        if "-" in piece[1:]:
            first, _, last = piece.partition("-")
            shots.extend(range(int(first), int(last) + 1))
        else:
            shots.append(int(piece))
    return list(dict.fromkeys(shots))


def product_path(filedb: Path, shot: int) -> Path:
    return filedb / "omas" / "diagnostics" / str(shot) / "output" / "diagnostics.json.gz"


def load_product(path: Path):
    """The diagnostics ODS, read from a copy so the FileDB is never opened for writing."""
    from omas import load_omas_json

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as staged:
        with gzip.open(path, "rb") as handle:
            shutil.copyfileobj(handle, staged)
    try:
        return load_omas_json(staged.name, consistency_check=False)
    finally:
        Path(staged.name).unlink(missing_ok=True)


def plasma_current_row(ods) -> dict[str, Any]:
    data = _get(ods, "magnetics.ip.0.data")
    time = _get(ods, "magnetics.ip.0.time")
    if time is None:
        time = _get(ods, "magnetics.time")
    if data is None or time is None or np.size(data) == 0:
        return {"available": False}
    ip = np.asarray(data, dtype=float)
    t = np.asarray(time, dtype=float)
    k = int(np.nanargmax(ip))
    head = t <= t[0] + 0.005
    tail = t >= t[-1] - 0.005
    return {
        "available": True,
        "peak_kA": round(float(ip[k]) / 1e3, 2),
        "t_peak_s": round(float(t[k]), 5),
        "min_kA": round(float(np.nanmin(ip)) / 1e3, 2),
        "head_kA": round(float(np.nanmedian(ip[head])) / 1e3, 2),
        "tail_kA": round(float(np.nanmedian(ip[tail])) / 1e3, 2),
        "window_s": [round(float(t[0]), 4), round(float(t[-1]), 4)],
    }


def rogowski_row(ods) -> dict[str, Any]:
    def validity(index: int):
        value = _get(ods, f"magnetics.rogowski_coil.{index}.current.validity")
        return None if value is None else int(value)

    flux = _get(ods, "magnetics.diamagnetic_flux.0.data")
    return {
        "plasma_rogowski_validity": validity(0),
        "tf_rogowski_validity": validity(1),
        "diamagnetic_flux": bool(flux is not None and np.size(flux)),
    }


def magnetics_row(ods) -> dict[str, Any]:
    from vaft.validation.magnetics import magnetics_quality_metrics, validate_magnetics_signals

    report = validate_magnetics_signals(ods)
    metrics = magnetics_quality_metrics(ods, report)
    families: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"declared": 0, "condemned": 0})
    condemned: list[dict[str, Any]] = []
    reasons: collections.Counter[str] = collections.Counter()
    for channel in metrics["channels"]:
        family = str(channel.get("family") or channel["kind"])
        if family == "not_in_efit":
            continue
        families[family]["declared"] += 1
        fraction = channel.get("valid_fraction")
        if fraction is None or not np.isfinite(fraction) or fraction > 0.0:
            continue
        families[family]["condemned"] += 1
        kinds = sorted({event["reason"] for event in channel.get("events", ())}) or ["unavailable"]
        primary = next((kind for kind in HARD_REASONS if kind in kinds), kinds[0])
        reasons[primary] += 1
        condemned.append({"kind": channel["kind"], "index": channel["index"], "name": channel["name"], "reason": primary})
    return {
        "families": {name: dict(counts) for name, counts in sorted(families.items())},
        "condemned": condemned,
        "condemned_by_reason": dict(reasons.most_common()),
    }


def filterscope_row(ods) -> dict[str, Any]:
    """Length of the bit-identical run that ends each fast filterscope line."""
    from vaft.validation.magnetics import constant_runs

    time = _get(ods, "spectrometer_uv.time")
    if time is None:
        return {"available": False}
    t = np.asarray(time, dtype=float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    worst = 0.0
    line = 0
    while True:
        data = _get(ods, f"spectrometer_uv.channel.2.processed_line.{line}.intensity.data")
        if data is None:
            break
        values = np.asarray(data, dtype=float)
        runs = [run for run in constant_runs(values, 16) if run[1] == values.size]
        if runs:
            worst = max(worst, (runs[-1][1] - runs[-1][0]) * dt)
        line += 1
    return {"available": line > 0, "lines": line, "clamped_tail_ms": round(worst * 1e3, 2)}


def pf_drift_row(shot: int) -> dict[str, Any]:
    """Each PF coil's acquisition at the end of the record against its baseline, in A.

    The mapper removes only the mean of the leading ``baseline_samples``; a
    drift that is still there when every coil is off is what #1424 asks about.
    """
    from vaft.database import raw as raw_db
    from vaft.machine_mapping import pf_active as pf
    from vaft.machine_mapping.utils import resolve_vest_diagnostic

    processing = resolve_vest_diagnostic(shot, "pf_active")["processing"]
    baseline = int(processing["baseline_samples"])
    gains = pf._coil_gain_by_index(shot)
    drift: dict[str, float] = {}
    for index, field in sorted(pf.coil_field_code_by_index(shot).items()):
        loaded = raw_db.vest_load(shot, field)
        if loaded is None or np.size(loaded[0]) < 2 * baseline:
            continue
        time, values = (np.asarray(item, dtype=float) for item in loaded)
        pre = float(np.mean(values[:baseline]))
        post = float(np.median(values[time >= time[-1] - 0.05]))
        drift[f"PF{index + 1}"] = round((post - pre) * gains.get(index, 0.0), 1)
    worst = max(drift.items(), key=lambda item: abs(item[1])) if drift else (None, 0.0)
    return {"drift_A": drift, "worst": {"coil": worst[0], "drift_A": worst[1]}}


def recorded_faults_row(shot: int) -> list[dict[str, Any]]:
    """The vest.yaml ``diagnostic_faults`` records in force on *shot* (#1543)."""
    from vaft.machine_mapping.diagnostic_faults import known_diagnostic_faults

    return [
        {"ids": f["ids"], "label": f["label"], "kind": f["kind"]} for f in known_diagnostic_faults(shot)
    ]


def tf_repair_row(shot: int) -> dict[str, Any]:
    """TF-current excursions the mapper repairs on *shot*, from raw (#1543)."""
    from vaft.machine_mapping.tf import vfit_tf_current_detailed

    _time, _current, intervals = vfit_tf_current_detailed(shot)
    return {
        "repaired_ms": round(sum(end - start for start, end in intervals) * 1e3, 1),
        "intervals": [[round(start, 4), round(end, 4)] for start, end in intervals],
    }


def flags_for(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    ip = row.get("ip", {})
    if not ip.get("available"):
        flags.append("no_ip")
    else:
        peak = max(ip["peak_kA"], 1e-6)
        if ip["peak_kA"] > PROVISIONAL["ip_peak_ceiling_kA"]:
            flags.append("ip_peak_implausible")
        if abs(ip["tail_kA"]) > PROVISIONAL["ip_tail_fraction"] * peak:
            flags.append("ip_tail_residual")
        if abs(ip["head_kA"]) > PROVISIONAL["ip_head_fraction"] * peak:
            flags.append("ip_head_offset")
    rog = row.get("rogowski", {})
    if rog.get("plasma_rogowski_validity") is not None and rog["plasma_rogowski_validity"] < 0:
        flags.append("plasma_rogowski_invalid")
    if not rog.get("diamagnetic_flux"):
        flags.append("no_diamagnetic_flux")
    for family, counts in row.get("magnetics", {}).get("families", {}).items():
        if counts["declared"] and counts["condemned"] * 2 >= counts["declared"]:
            flags.append(f"{family}_half_condemned")
    pf = row.get("pf")
    if pf and abs(pf["worst"]["drift_A"]) > PROVISIONAL["pf_drift_A"]:
        flags.append(f"pf_drift_{pf['worst']['coil']}")
    fs = row.get("filterscope", {})
    if fs.get("clamped_tail_ms", 0.0) > PROVISIONAL["filterscope_clamp_ms"]:
        flags.append("filterscope_tail_clamped")
    if row.get("tf", {}).get("repaired_ms", 0.0) > 0.0:
        flags.append("tf_excursion_repaired")
    for fault in row.get("recorded_faults", ()):
        flags.append(f"recorded:{fault['ids']}:{fault['label']}:{fault['kind']}")
    return flags


def check_shot(shot: int, *, filedb: Path, raw: bool) -> dict[str, Any]:
    path = product_path(filedb, shot)
    if not path.is_file():
        return {"shot": shot, "status": "absent"}
    try:
        ods = load_product(path)
        row: dict[str, Any] = {
            "shot": shot,
            "status": "assessed",
            "ip": plasma_current_row(ods),
            "rogowski": rogowski_row(ods),
            "magnetics": magnetics_row(ods),
            "filterscope": filterscope_row(ods),
            "recorded_faults": recorded_faults_row(shot),
        }
        if raw:
            row["pf"] = pf_drift_row(shot)
            row["tf"] = tf_repair_row(shot)
    except Exception as error:  # a shot that cannot be judged is part of the answer
        LOGGER.exception("shot %s", shot)
        return {"shot": shot, "status": "error", "error": f"{type(error).__name__}: {error}"[:300]}
    row["flags"] = flags_for(row)
    return row


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| shot | Ip peak kA (t) | Ip tail kA | Rogowski p/TF | dia | condemned probes in/out/side | loops | top reasons | PF worst drift | fs clamp ms | flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.get("status") != "assessed":
            lines.append(f"| {row['shot']} | {row.get('status')} {row.get('error', '')} |||||||||| ")
            continue
        ip, rog, mag = row["ip"], row["rogowski"], row["magnetics"]
        fam = mag["families"]

        def c(name: str) -> str:
            item = fam.get(name)
            return f"{item['condemned']}/{item['declared']}" if item else "-"

        loops = sum(v["condemned"] for k, v in fam.items() if "loop" in k)
        loops_declared = sum(v["declared"] for k, v in fam.items() if "loop" in k)
        reasons = ", ".join(f"{k} {v}" for k, v in list(mag["condemned_by_reason"].items())[:3])
        pf = row.get("pf", {}).get("worst", {})
        pf_text = f"{pf.get('coil')} {pf.get('drift_A')}" if pf else "-"
        peak = f"{ip['peak_kA']} ({ip['t_peak_s']})" if ip.get("available") else "-"
        lines.append(
            f"| {row['shot']} | {peak} | {ip.get('tail_kA', '-')} | {rog['plasma_rogowski_validity']}/{rog['tf_rogowski_validity']} "
            f"| {'y' if rog['diamagnetic_flux'] else 'n'} | {c('inboard')} / {c('outboard')} / {c('side')} | {loops}/{loops_declared} "
            f"| {reasons} | {pf_text} | {row['filterscope'].get('clamped_tail_ms', '-')} | {' '.join(row['flags'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shots", required=True, help="e.g. '48930-48940,48224'")
    parser.add_argument("--filedb", default="/srv/vest.filedb", type=Path)
    parser.add_argument("--raw", action="store_true", help="also read raw PF channels from the VEST database")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    warnings.filterwarnings("ignore")

    rows = []
    for shot in parse_shots(args.shots):
        rows.append(check_shot(shot, filedb=args.filedb, raw=args.raw))
        LOGGER.info("shot %s: %s %s", shot, rows[-1]["status"], rows[-1].get("flags", ""))
    table = {
        "schema_version": SCHEMA,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filedb": str(args.filedb),
        "provisional_thresholds": PROVISIONAL,
        "rows": rows,
    }
    args.table.write_text(json.dumps(table, indent=1, default=float))
    if args.markdown:
        write_markdown(rows, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

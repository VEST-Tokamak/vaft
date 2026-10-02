#!/usr/bin/env python3
"""Check vest.yaml's probe wiring and recorded faults against the raw signals (issue #956).

Read-only. For each shot it answers two questions from the raw archive alone:

1. **Which layout feeds the outboard probes?** The two disputed positions,
   Z = +0.06 m and Z = -0.42 m in the R = 0.796 m column, are scored under
   every assignment of fields 170 and 225 with either calibration sign: the
   2409 layout, the pre-39438 layout, and the sign-flipped pair VFIT's 2306 and
   2310 files carry. Each candidate signal is compared with the linear
   prediction from its nearest untouched neighbours in the same column; the
   score is the mean relative residual, and the margin is the runner-up's
   score over the winner's. The winner is compared with the layout
   :func:`vaft.machine_mapping.magnetics.magnetics_wiring_for_shot` resolves.
2. **Is the Z = +0.06 channel working?** Its mean raw voltage, and its gain
   and residual against the mean of its two neighbours, compared with
   :func:`vaft.machine_mapping.magnetics.known_magnetics_faults`.

Usage (on a host with the FileDB raw archive)::

    PYTHONPATH=. python workflow/magnetics_geometry/verify_wiring.py 36480 36481 39437 39438
    PYTHONPATH=. python workflow/magnetics_geometry/verify_wiring.py --range 39430 39450

Exit status 1 when a confident winner (margin above ``--confident``)
disagrees with vest.yaml, so the tool can guard a future edit of the table.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable
import warnings

import numpy as np

from vaft.machine_mapping.magnetics import (
    _load_static_channels,
    known_magnetics_faults,
    magnetics_wiring_for_shot,
)
from vaft.process.magnetics import vest_b_field_pol_probe_legacy, vest_magnetics_time_window

R_OUTBOARD = 0.796
DISPUTED = {35: 0.06, 47: -0.42}
_PULSE = re.compile(r'"pulse_datetime"\s*:\s*"([^"]+)"')


def raw_path(root: Path, shot: int) -> Path:
    return root / str(shot) / f"vest_{shot}_daq_raw.json.gz"


def load_raw(root: Path, shot: int) -> tuple[dict[str, Any], str] | None:
    path = raw_path(root, shot)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    match = _PULSE.search(text)
    return json.loads(text)["fields"], (match.group(1) if match else "?")


def processed(fields: dict[str, Any], code: int, calibration: float, shot: int, grid: np.ndarray):
    entry = fields.get(str(code))
    if not entry or not entry.get("data"):
        return None, None
    raw = np.asarray(entry["data"], dtype=float)
    time = float(entry["t0"]) + float(entry["dt"]) * np.arange(raw.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = vest_b_field_pol_probe_legacy(time, raw, calibration, shot=shot, allow_zero_fallback=True)
    return np.interp(grid, time, field), raw


def outboard_column() -> dict[int, float]:
    """``{index: z}`` for every probe in the R = 0.796 m column."""
    return {
        index: float(row["z"])
        for index, row in enumerate(_load_static_channels())
        if row["kind"] == "b_field_pol_probe" and abs(float(row["r"]) - R_OUTBOARD) < 1e-9
    }


def neighbour_prediction(z0: float, signals: dict[float, np.ndarray]) -> np.ndarray:
    zs = np.array(sorted(signals))
    below, above = zs[zs < z0], zs[zs > z0]
    if below.size and above.size:
        z1, z2 = below[-1], above[0]
    elif above.size >= 2:
        z1, z2 = above[0], above[1]
    else:
        z1, z2 = below[-2], below[-1]
    weight = (z0 - z1) / (z2 - z1)
    return (1 - weight) * signals[z1] + weight * signals[z2]


def candidates(magnitudes: dict[int, float]) -> dict[str, dict[int, tuple[int, float]]]:
    """Every assignment of fields 170/225 to the two positions, both signs."""
    out = {}
    for label, (plus_field, minus_field) in {"2409": (225, 170), "pre-39438": (170, 225)}.items():
        for flipped in (False, True):
            sign = -1.0 if flipped else 1.0
            out[label + (" sign-flipped" if flipped else "")] = {
                35: (plus_field, sign * magnitudes[35]),
                47: (minus_field, -sign * magnitudes[47]),
            }
    return out


def score_shot(shot: int, fields: dict[str, Any]) -> dict[str, Any]:
    wiring = magnetics_wiring_for_shot(shot)
    column = outboard_column()
    grid = vest_magnetics_time_window(shot)
    signals = {}
    for index, z in column.items():
        if index in DISPUTED:
            continue
        channel = wiring.channels[index]
        signal, _raw = processed(fields, int(channel["field_code"]), float(channel["calibration"]), shot, grid)
        if signal is not None:
            signals[z] = signal
    if len(signals) < 10 or np.sqrt(np.mean(np.vstack(list(signals.values())) ** 2)) < 1e-5:
        return {"status": "outboard column missing or flat"}
    magnitudes = {index: abs(float(wiring.channels[index]["calibration"])) for index in DISPUTED}
    # A +0.06 channel recorded as broken carries no layout information; the
    # -0.42 position then decides alone.
    minus_only = ("b_field_pol_probe", 35) in known_magnetics_faults(shot)
    scores = {}
    for label, assignment in candidates(magnitudes).items():
        residuals = []
        for index, (code, calibration) in assignment.items():
            signal, _raw = processed(fields, code, calibration, shot, grid)
            if signal is None:
                residuals.append(float("inf"))
                continue
            prediction = neighbour_prediction(DISPUTED[index], signals)
            residuals.append(
                float(np.sqrt(np.mean((signal - prediction) ** 2)) / max(np.sqrt(np.mean(prediction**2)), 1e-12))
            )
        scores[label] = residuals[1] if minus_only else float(np.mean(residuals))
    layouts = list(scores)
    ordered = sorted(layouts, key=lambda label: scores[label])
    best, second = ordered[0], ordered[1]
    margin = scores[second] / max(scores[best], 1e-12)
    health = z006_health(shot, fields, wiring, signals, grid)
    return {
        "status": "scored",
        "resolved": wiring.layout,
        "best": best,
        "margin": margin,
        "scores": scores,
        "decided_by": "-0.42 only (+0.06 recorded broken)" if minus_only else "both positions",
        "z006": health,
    }


def z006_health(shot, fields, wiring, signals, grid) -> dict[str, Any]:
    channel = wiring.channels[35]
    signal, raw = processed(fields, int(channel["field_code"]), float(channel["calibration"]), shot, grid)
    recorded = ("b_field_pol_probe", 35) in known_magnetics_faults(shot)
    if signal is None:
        return {"field": int(channel["field_code"]), "recorded_fault": recorded, "status": "absent"}
    prediction = neighbour_prediction(0.06, signals)
    gain = float(np.dot(signal, prediction) / max(np.dot(prediction, prediction), 1e-30))
    residual = float(np.sqrt(np.mean((signal - prediction) ** 2)) / max(np.sqrt(np.mean(prediction**2)), 1e-12))
    return {
        "field": int(channel["field_code"]),
        "mean_voltage": float(np.mean(raw)),
        "gain": gain,
        "residual": residual,
        "recorded_fault": recorded,
    }


def _shots(args: argparse.Namespace) -> Iterable[int]:
    yield from (int(shot) for shot in args.shots)
    if args.range:
        yield from range(args.range[0], args.range[1] + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("shots", nargs="*", type=int)
    parser.add_argument("--range", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(os.environ.get("VAFT_FILEDB_DIR", "/srv/vest.filedb")) / "raw",
    )
    parser.add_argument("--confident", type=float, default=1.5, help="margin that counts as a verdict")
    parser.add_argument("--json", action="store_true", help="one JSON record per shot")
    args = parser.parse_args(argv)

    disagreements = 0
    for shot in _shots(args):
        loaded = load_raw(args.raw_root, shot)
        if loaded is None:
            continue
        fields, stamp = loaded
        result = {"shot": shot, "pulse_datetime": stamp, **score_shot(shot, fields)}
        if result["status"] == "scored":
            confident = result["margin"] >= args.confident
            base = result["best"].split(" ")[0]
            result["agrees"] = (not confident) or (base == result["resolved"] and "flipped" not in result["best"])
            disagreements += not result["agrees"]
        if args.json:
            print(json.dumps(result, sort_keys=True))
            continue
        if result["status"] != "scored":
            print(f"{shot} {stamp[:16]}  {result['status']}")
            continue
        health = result["z006"]
        z006 = (
            f"field {health['field']} absent"
            if health.get("status") == "absent"
            else f"field {health['field']} mean {health['mean_voltage'] * 1e3:+7.1f} mV gain {health['gain']:+6.2f} res {health['residual']:5.2f}"
        )
        print(
            f"{shot} {stamp[:16]}  best {result['best']:<22s} x{result['margin']:4.1f}  "
            f"vest.yaml {result['resolved']:<10s} {'ok' if result['agrees'] else 'DISAGREES'}  |  "
            f"+0.06 {z006}  recorded fault: {'yes' if health['recorded_fault'] else 'no'}"
        )
    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())

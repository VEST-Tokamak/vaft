#!/usr/bin/env python3
"""Generate EFIT constraints OMAS ODS from eddy-current diagnostics ODS."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
from omas import load_omas_json

from vaft.code.efit import correct_flux_loop, generate_constraints_ods as build_constraints


LOGGER = logging.getLogger("vaft.generate_constraints_ods")
DEFAULT_UNCERTAINTY = [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2]
DEFAULT_WEIGHTING = [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01]


def _csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "on"}


def _table_dir(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("/") else text + "/"


def _select_times(ods, timeset: str, tstep: float, tstart: float | None, tend: float | None) -> np.ndarray:
    ip_time = np.asarray(ods["magnetics.ip.0.time"], dtype=float)
    ip_data = np.asarray(ods["magnetics.ip.0.data"], dtype=float)
    if ip_time.size == 0:
        raise ValueError("magnetics.ip.0.time is empty")

    if timeset == "manual":
        if tstart is None or tend is None:
            raise ValueError("manual timeset requires --tstart and --tend")
        return np.arange(tstart, tend, tstep, dtype=float)

    base_start = max(0.28, float(ip_time[0]))
    base_end = min(0.38, float(ip_time[-1]))
    base_index = (ip_time >= base_start) & (ip_time <= base_end)
    valid_index = base_index & (ip_data > 20e3)
    selected = ip_time[valid_index] if np.any(valid_index) else ip_time[base_index]
    if selected.size == 0:
        selected = ip_time

    start = float(selected[0]) if tstart is None else max(float(selected[0]), tstart)
    end = float(selected[-1]) if tend is None else min(float(selected[-1]), tend)
    if timeset == "auto":
        start = round(start / tstep) * tstep
        end = round(end / tstep) * tstep
    if end <= start:
        return np.array([start], dtype=float)
    return np.arange(start, end + 0.5 * tstep, tstep, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--eddy-ods", "--input", required=True, type=Path, help="Input eddy ODS JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output constraints ODS JSON.")
    parser.add_argument("--efit-table-dir", default="", help="EFIT table/input directory written into kfiles.")
    parser.add_argument("--timeset", default="auto", choices=["auto", "manual"], help="EFIT constraint time selection mode.")
    parser.add_argument("--tstep", default=0.001, type=float, help="EFIT time step in seconds.")
    parser.add_argument("--tstart", default=None, type=float, help="Manual lower time bound.")
    parser.add_argument("--tend", default=None, type=float, help="Manual upper time bound.")
    parser.add_argument("--uncertainty", default=",".join(str(v) for v in DEFAULT_UNCERTAINTY))
    parser.add_argument("--weighting", default=",".join(str(v) for v in DEFAULT_WEIGHTING))
    parser.add_argument("--broken", default="", help="Comma-separated one-based broken diagnostic indices.")
    parser.add_argument("--detect-broken", default="false", help="Reserved for future automatic broken-channel detection.")
    parser.add_argument("--fl-correct-option", default=0, type=int, help="Reserved for future flux-loop correction.")
    parser.add_argument("--gaussian-fit-option", default=1, type=int, help="Gaussian fit option forwarded to EFIT constraints.")
    parser.add_argument("--npprime", default=2, type=int, help="EFIT KPPCUR value.")
    parser.add_argument("--nffprime", default=2, type=int, help="EFIT KFFCUR value.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if _bool(args.detect_broken):
        LOGGER.warning("Automatic broken-channel detection is not enabled in the offline VAFT workflow; using --broken only.")
    ods = load_omas_json(str(args.eddy_ods), consistency_check=False)
    times = _select_times(ods, args.timeset, args.tstep, args.tstart, args.tend)
    if times.size == 0:
        raise ValueError("No EFIT constraint times selected")
    ods["equilibrium.time"] = times
    fl_correct_coeff = correct_flux_loop(ods) if args.fl_correct_option else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Generating constraints for shot %s at %d time slices", args.shot, len(times))
    build_constraints(
        ods,
        args.shot,
        str(args.output.parent),
        _table_dir(args.efit_table_dir),
        times,
        _csv_floats(args.uncertainty),
        _csv_floats(args.weighting),
        broken=_csv_ints(args.broken),
        fit=args.gaussian_fit_option,
        fl_correct_coeff=fl_correct_coeff,
        FFCUR=args.nffprime,
        PPCUR=args.npprime,
    )

    produced = args.output.parent / f"{args.shot}_constraints.json"
    if produced != args.output and produced.exists():
        shutil.move(str(produced), str(args.output))
    if not args.output.exists():
        raise FileNotFoundError(f"Expected constraints output was not created: {args.output}")
    LOGGER.info("Constraints ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

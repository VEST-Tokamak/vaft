"""Full-discharge EFIT profile-model uncertainty study (issue #579).

The first pass is deliberately a profile-model experiment, not a numerical
tuning experiment.  It pins the qualified seed at ``RELIP=0.32 m``, writes
``ICINIT=2`` explicitly, states the executable's current termination defaults
explicitly, and varies only ``KPPCUR``, ``KFFCUR``, ``PCURBD``, ``FCURBD`` and
``FWTBP``.

Example::

    PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install \
      python workflow/efit_profile_models/profile_model_study.py \
        --output /scratch/efit-profile-models --shots 41672

The default pilot includes the low-order model that VFIT found informative,
the routine-like (2,2) baseline, edge freedom, and the FWTBP interaction.  Use
``--full-matrix`` to add (2,3), (3,2), and (3,3) order cases.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time as _clock
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA = 3
REPOSITORY = Path(__file__).resolve().parents[2]
SEED_STUDY = REPOSITORY / "workflow" / "efit_numerics" / "seed_basin.py"
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"

_OUTPUT_TIME = re.compile(r"\.(\d{5})(?:_(\d{3}))?$")


@dataclass(frozen=True)
class Model:
    name: str
    kppcur: int
    kffcur: int
    pcurbd: int
    fcurbd: int
    fwtbp: int
    purpose: str


PILOT_MODELS = (
    Model("p11_zero", 1, 1, 1, 1, 0, "minimum-order zero-edge comparison"),
    Model("p22_zero", 2, 2, 1, 1, 0, "routine-like baseline"),
    Model("p22_free", 2, 2, 0, 0, 0, "edge-current freedom"),
    Model("p22_zero_fwtbp", 2, 2, 1, 1, 1, "P-prime/FF-prime regularization"),
    Model("p22_free_fwtbp", 2, 2, 0, 0, 1, "edge x regularization interaction"),
)

ORDER_MODELS = (
    Model("p23_zero", 2, 3, 1, 1, 0, "asymmetric FF-prime order"),
    Model("p32_zero", 3, 2, 1, 1, 0, "asymmetric P-prime order"),
    Model("p33_zero", 3, 3, 1, 1, 0, "higher-order zero-edge comparison"),
)

BASELINE_MODEL = "p22_zero"
PLASMA_PHASES = frozenset(("ramp_up", "quasi_stationary", "ramp_down"))


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_vest_tables(source: Path, destination: Path) -> tuple[Path, dict[str, Any]]:
    """Clone the validated 129x129 tables and install VEST's acceptance envelope.

    The response tables are copied byte-for-byte.  Only ``mhdin.dat`` is
    regenerated from canonical geometry with the machine-derived geometric
    bounds and the ill-conditioned virial gates disabled (#646/#649).
    """
    from vaft.code.efit.efund import EFUNDConfig, write_mhdin
    from vaft.machine_mapping.efund_geometry import (
        efund_geometry_from_static,
        vest_acceptance_envelope,
    )
    from vaft.omas.vest_upstream import build_static_ods

    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)
    static, static_manifest = build_static_ods("vest-pre-43017-pf1906")
    geometry = efund_geometry_from_static(static, manifest=static_manifest)
    envelope = vest_acceptance_envelope(static)
    write_mhdin(
        geometry,
        EFUNDConfig(workdir=destination),
        destination / "mhdin.dat",
        envelope=envelope,
    )
    consumed = ("mhdin.dat", "ec129129.ddd", "ep129129.ddd", "rfcoil.ddd", "rv129129.ddd")
    record = {
        "source": str(source),
        "destination": str(destination),
        "policy": "canonical VEST geometry; VEST bounds; virial gates disabled",
        "envelope": envelope.to_dict(),
        "files": {
            name: {"sha256": _sha256(destination / name), "bytes": (destination / name).stat().st_size}
            for name in consumed
        },
    }
    (destination / "profile_model_table.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination, record


def fixed_scientific_config():
    """The frozen non-profile baseline for #579.

    Optional termination values are stated even when they equal the executable
    defaults.  This makes the study reproducible against a future EFIT build.
    """
    from vaft.code.efit.config import (
        EFITConstraintConfig,
        EFITInitializationConfig,
        EFITNumericsConfig,
        EFITProfileConfig,
        EFITScientificConfig,
    )

    return EFITScientificConfig(
        profile=EFITProfileConfig(),
        initialization=EFITInitializationConfig(ellipse_rzero=0.32, icinit=2),
        numerics=EFITNumericsConfig(
            relaxation=1.0,
            error_tolerance=1.0e-5,
            measurement_error_floor=5.0e-4,
            max_iterations=100,
            error_minimum=1.0e-2,
            chi_squared_target=80.0,
            convergence_mode=2,
            inner_iterations=1,
        ),
        constraints=EFITConstraintConfig(),
    )


def scientific_for(model: Model):
    from dataclasses import replace
    from vaft.code.efit.config import EFITProfileConfig

    return replace(
        fixed_scientific_config(),
        profile=EFITProfileConfig(
            kppcur=model.kppcur,
            kffcur=model.kffcur,
            pcurbd=model.pcurbd,
            fcurbd=model.fcurbd,
            fwtbp=model.fwtbp,
        ),
    )


def _output_time_ms(path: Path) -> float | None:
    match = _OUTPUT_TIME.search(path.name)
    if match is None:
        return None
    return float(int(match.group(1))) + float(int(match.group(2) or 0)) / 1000.0


def _phase_map(constraints: Any, times: np.ndarray, current_cut: float) -> tuple[dict[int, dict[str, Any]], float]:
    currents = []
    for index in range(times.size):
        path = f"equilibrium.time_slice.{index}.constraints.ip.measured"
        try:
            currents.append(float(constraints[path]))
        except Exception:
            currents.append(float("nan"))
    values = np.asarray(currents, dtype=float)
    finite = np.isfinite(values)
    derivative = np.full_like(values, np.nan)
    if finite.sum() >= 2:
        derivative[finite] = np.gradient(values[finite], times[finite])
    scale = float(np.nanmax(np.abs(derivative))) if np.isfinite(derivative).any() else 0.0
    threshold = 0.15 * scale
    result: dict[int, dict[str, Any]] = {}
    for time_s, current, slope in zip(times, values, derivative):
        if not np.isfinite(current):
            phase = "unknown"
        elif abs(current) < current_cut:
            phase = "vacuum"
        elif not np.isfinite(slope) or abs(slope) <= threshold:
            phase = "quasi_stationary"
        elif slope > 0:
            phase = "ramp_up"
        else:
            phase = "ramp_down"
        result[int(round(float(time_s) * 1000.0))] = {
            "current": None if not np.isfinite(current) else float(current),
            "dcurrent_dt": None if not np.isfinite(slope) else float(slope),
            "phase": phase,
        }
    return result, threshold


def _afile_record(path: Path) -> dict[str, Any]:
    from vaft.data import read_aeqdsk

    item = read_aeqdsk(path)
    names = (
        "chisq", "terror", "condno", "rm", "zm", "rcntr", "zcntr",
        "aminor", "elong", "utri", "ltri", "area", "volume", "li",
        "betap", "q95", "qmin", "wmhd", "cdflux", "ipmhd", "cjor0", "cjor95",
        "cjor99", "cj1ave", "peak",
    )
    return {
        "path": path.name,
        "time_ms": float(item.time_ms),
        "jflag": int(item.jflag),
        "lflag": int(item.lflag),
        "accepted": bool(item.accepted),
        "scalars": {
            name: float(item.scalars.get(name, float("nan"))) for name in names
        },
    }


def _gfile_record(path: Path) -> dict[str, Any]:
    from vaft.data import read_geqdsk

    data = read_geqdsk(path).mapping
    pressure = np.asarray(data["PRES"], dtype=float)
    pprime = np.asarray(data["PPRIME"], dtype=float)
    ffprime = np.asarray(data["FFPRIM"], dtype=float)
    f = np.asarray(data["FPOL"], dtype=float)
    q = np.asarray(data["QPSI"], dtype=float)
    psi_n = np.linspace(0.0, 1.0, pprime.size)
    reference_r = float(data["RMAXIS"])
    mu0 = 4.0e-7 * np.pi
    if not math.isfinite(reference_r) or reference_r == 0.0:
        jphi = np.full_like(pprime, np.nan)
    else:
        jphi = reference_r * pprime + ffprime / (mu0 * reference_r)

    def turns(values: np.ndarray) -> int:
        delta = np.diff(np.asarray(values, dtype=float))
        finite = delta[np.isfinite(delta)]
        if finite.size < 2:
            return 0
        tolerance = max(float(np.max(np.abs(finite))) * 1.0e-8, np.finfo(float).tiny)
        signs = np.sign(finite[np.abs(finite) > tolerance])
        return int(np.count_nonzero(signs[1:] != signs[:-1]))

    peak_jphi = float(np.max(np.abs(jphi))) if np.isfinite(jphi).any() else float("nan")
    shell = jphi[psi_n >= 0.9]
    boundary_r = np.asarray(data["RBBBS"], dtype=float)
    boundary_z = np.asarray(data["ZBBBS"], dtype=float)
    return {
        "path": path.name,
        "time_ms": _output_time_ms(path),
        "axis": [float(data["RMAXIS"]), float(data["ZMAXIS"])],
        "boundary": {
            "r": boundary_r.tolist(),
            "z": boundary_z.tolist(),
            "r_min": float(np.min(boundary_r)) if boundary_r.size else None,
            "r_max": float(np.max(boundary_r)) if boundary_r.size else None,
            "z_min": float(np.min(boundary_z)) if boundary_z.size else None,
            "z_max": float(np.max(boundary_z)) if boundary_z.size else None,
        },
        "profiles": {
            "pressure": pressure.tolist(),
            "pprime": pprime.tolist(),
            "ffprime": ffprime.tolist(),
            "f": f.tolist(),
            "q": q.tolist(),
            "jphi_reference_r": jphi.tolist(),
        },
        "profile_diagnostics": {
            "jphi_reference_r_m": reference_r,
            "q_axis": float(q[0]),
            "jphi_psi_080": float(np.interp(0.80, psi_n, jphi)),
            "jphi_psi_090": float(np.interp(0.90, psi_n, jphi)),
            "jphi_psi_095": float(np.interp(0.95, psi_n, jphi)),
            "jphi_edge": float(jphi[-1]),
            "jphi_edge_shell_mean": float(np.mean(shell)),
            "jphi_edge_shell_to_peak": (
                None if not math.isfinite(peak_jphi) or peak_jphi == 0.0
                else float(np.mean(np.abs(shell))) / peak_jphi
            ),
            "pressure_negative_fraction": float(np.mean(pressure < 0.0)),
            "f_nonfinite_count": int(np.count_nonzero(~np.isfinite(f))),
            "pprime_turns": turns(pprime),
            "ffprime_turns": turns(ffprime),
            "q_turns": turns(q[7:]),
            "jphi_turns": turns(jphi),
        },
    }


def _mfile_record(path: Path) -> dict[str, Any]:
    """Keep EFIT's diagnostic-resolved fit measures from the NetCDF m-file.

    The a-file total is dominated by VEST's plasma-current term and can print
    identically for visibly different magnetic fits. Issue #579 therefore
    compares EFIT's own flux-loop and magnetic-probe contributions directly.
    """
    from vaft.data import read_meqdsk

    data = read_meqdsk(path)

    def values(name: str) -> np.ndarray:
        if name not in data:
            return np.asarray([], dtype=float)
        array = np.asarray(data[name].data, dtype=float).reshape(-1)
        return array[np.isfinite(array)]

    def total(name: str) -> float | None:
        array = values(name)
        return float(array.sum()) if array.size else None

    bpol = total("saimpi")
    flux = total("saisil")
    magnetic = None if bpol is None and flux is None else float((bpol or 0.0) + (flux or 0.0))
    active_bpol = int(np.count_nonzero(values("fwtmp2")))
    active_flux = int(np.count_nonzero(values("fwtsi")))
    active_magnetic = active_bpol + active_flux
    return {
        "path": path.name,
        "time_ms": _output_time_ms(path),
        "scalars": {
            "magnetics_chisq": magnetic,
            "magnetics_chisq_per_active_signal": (
                None if magnetic is None or not active_magnetic else magnetic / active_magnetic
            ),
            "bpol_probe_chisq": bpol,
            "flux_loop_chisq": flux,
            "plasma_current_chisq": total("chipasma"),
            "diamagnetic_flux_chisq": total("chidflux"),
            "pf_current_chisq": total("chifcc"),
            "total_chisq": total("chifin"),
            "inclusive_chisq": total("chitot"),
        },
        "active_signals": {
            "bpol_probe": active_bpol,
            "flux_loop": active_flux,
            "magnetics": active_magnetic,
        },
    }


def _blank_log(time_ms: int) -> dict[str, Any]:
    return {
        "time_ms": time_ms,
        "iterations_n": 0,
        "chi2_initial": None,
        "chi2_final": None,
        "gs_error": None,
        "collapsed": False,
        "accepted": False,
        "exit_path": "missing_log",
        "failures": [],
        "solver_errors": [],
    }


def run_model(
    constraints: Any,
    *,
    shot: int,
    times: np.ndarray,
    workdir: Path,
    executable: str,
    scientific: Any,
    baseline_module: Any,
    phase_by_time: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one model on every requested time and retain every outcome."""
    from vaft.code.efit.magnetic import (
        EFITConfig,
        prepare_efit_inputs,
        resolved_efit_configuration,
        run_efit,
    )

    cache = workdir / "model-result.json"
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if (
            payload.get("analysis_schema") == SCHEMA
            and payload.get("scientific_sha256") == scientific.sha256
        ):
            return payload

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    config = EFITConfig(
        executable=executable,
        workdir=workdir,
        shot=shot,
        times=times.tolist(),
        args=("129",),
        profile=scientific.profile,
        initialization=scientific.initialization,
        numerics=scientific.numerics,
        constraints=scientific.constraints,
        provenance={"study": 579, "frozen_non_profile_settings": True},
    )
    inputs = prepare_efit_inputs(copy.deepcopy(constraints), config)
    started = _clock.perf_counter()
    result = run_efit(inputs, config)
    seconds = _clock.perf_counter() - started

    log_path = workdir / "run_efit.out"
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else result.stdout
    log_by_time = {
        int(round(float(item["time_ms"]))): item
        for item in baseline_module.parse_slices(log_text)
    }
    afiles = {_output_time_ms(path): _afile_record(path) for path in sorted(workdir.glob(f"a0{shot}.*"))}
    gfiles = {_output_time_ms(path): _gfile_record(path) for path in sorted(workdir.glob(f"g0{shot}.*"))}
    mfiles = {_output_time_ms(path): _mfile_record(path) for path in sorted(workdir.glob(f"m0{shot}.*"))}

    slices = []
    for time_s in times:
        time_ms = int(round(float(time_s) * 1000.0))
        row = dict(log_by_time.get(time_ms, _blank_log(time_ms)))
        afile_key = min(afiles, key=lambda value: abs(float(value) - time_ms)) if afiles else None
        gfile_key = min(gfiles, key=lambda value: abs(float(value) - time_ms)) if gfiles else None
        mfile_key = min(mfiles, key=lambda value: abs(float(value) - time_ms)) if mfiles else None
        row["afile"] = afiles.get(afile_key) if afile_key is not None and abs(float(afile_key) - time_ms) < 0.51 else None
        row["gfile"] = gfiles.get(gfile_key) if gfile_key is not None and abs(float(gfile_key) - time_ms) < 0.51 else None
        row["mfile"] = mfiles.get(mfile_key) if mfile_key is not None and abs(float(mfile_key) - time_ms) < 0.51 else None
        row.update(phase_by_time.get(time_ms, {"current": None, "dcurrent_dt": None, "phase": "unknown"}))
        if row["collapsed"]:
            row["outcome"] = "collapsed"
        elif row["afile"] is None or row["gfile"] is None:
            row["outcome"] = "no_output"
        else:
            row["outcome"] = "accepted" if row["afile"]["accepted"] else "flagged"
        slices.append(row)

    payload = {
        "analysis_schema": SCHEMA,
        "scientific": scientific.to_dict(),
        "scientific_sha256": scientific.sha256,
        "resolved_configuration": resolved_efit_configuration(config),
        "seconds": seconds,
        "returncode": result.returncode,
        "status": result.status,
        "requested_slices": int(times.size),
        "slices": slices,
    }
    cache.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _numbers(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    return array[np.isfinite(array)]


def spread(values: Iterable[Any], *, absolute: bool = False) -> dict[str, Any]:
    array = _numbers(values)
    if absolute:
        array = np.abs(array)
    if not array.size:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _curve_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float] | None:
    a = np.column_stack((np.asarray(left["r"], float), np.asarray(left["z"], float)))
    b = np.column_stack((np.asarray(right["r"], float), np.asarray(right["z"], float)))
    if min(len(a), len(b)) < 3:
        return None
    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    symmetric = np.concatenate((distances.min(axis=1), distances.min(axis=0)))
    return {
        "mean_m": float(symmetric.mean()),
        "rms_m": float(np.sqrt(np.mean(symmetric**2))),
        "hausdorff_m": float(symmetric.max()),
    }


def _relative_rms(candidate: Sequence[float], baseline: Sequence[float], *, start: int = 0) -> float | None:
    left = np.asarray(candidate, dtype=float)[start:]
    right = np.asarray(baseline, dtype=float)[start:]
    count = min(left.size, right.size)
    if count == 0:
        return None
    left, right = left[:count], right[:count]
    denominator = float(np.sqrt(np.mean(right**2)))
    if not math.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        return None
    return float(np.sqrt(np.mean((left - right) ** 2)) / denominator)


def summarize_run(run: Mapping[str, Any]) -> dict[str, Any]:
    slices = run["slices"]
    counts = Counter(item["outcome"] for item in slices)
    produced = [item for item in slices if item["afile"] and item["gfile"]]
    plasma = [item for item in slices if item["phase"] in PLASMA_PHASES]
    plasma_counts = Counter(item["outcome"] for item in plasma)
    plasma_produced = [item for item in plasma if item["afile"] and item["gfile"]]
    fit_records = [
        item["mfile"]["scalars"]
        for item in plasma_produced
        if item.get("mfile")
    ]
    profile_records = [item["gfile"]["profile_diagnostics"] for item in plasma_produced]
    return {
        "requested": len(slices),
        "outcomes": dict(sorted(counts.items())),
        "produced": len(produced),
        "plasma_requested": len(plasma),
        "plasma_outcomes": dict(sorted(plasma_counts.items())),
        "plasma_produced": len(plasma_produced),
        "seconds": float(run["seconds"]),
        "condno": spread(
            item["afile"]["scalars"]["condno"] for item in plasma_produced
        ),
        "chisq": spread(
            item["afile"]["scalars"]["chisq"] for item in plasma_produced
        ),
        "magnetics_chisq": spread(item["magnetics_chisq"] for item in fit_records),
        "magnetics_chisq_per_active_signal": spread(
            item["magnetics_chisq_per_active_signal"] for item in fit_records
        ),
        "edge_current": {
            name: spread(item[name] for item in profile_records)
            for name in (
                "jphi_psi_080",
                "jphi_psi_090",
                "jphi_psi_095",
                "jphi_edge",
                "jphi_edge_shell_mean",
                "jphi_edge_shell_to_peak",
            )
        },
        "profile_pathology": {
            "negative_pressure_slices": sum(
                item["pressure_negative_fraction"] > 0.0 for item in profile_records
            ),
            "nonfinite_f_slices": sum(
                item["f_nonfinite_count"] > 0 for item in profile_records
            ),
            "pprime_turns": spread(item["pprime_turns"] for item in profile_records),
            "ffprime_turns": spread(item["ffprime_turns"] for item in profile_records),
            "q_turns": spread(item["q_turns"] for item in profile_records),
            "jphi_turns": spread(item["jphi_turns"] for item in profile_records),
        },
        "by_phase": {
            phase: dict(Counter(item["outcome"] for item in slices if item["phase"] == phase))
            for phase in ("ramp_up", "quasi_stationary", "ramp_down", "vacuum", "unknown")
            if any(item["phase"] == phase for item in slices)
        },
        "axis_step_m": spread(
            math.hypot(
                float(b["gfile"]["axis"][0]) - float(a["gfile"]["axis"][0]),
                float(b["gfile"]["axis"][1]) - float(a["gfile"]["axis"][1]),
            )
            for a, b in zip(plasma_produced, plasma_produced[1:])
            if b["time_ms"] - a["time_ms"] <= 1.1
        ),
    }


def compare_runs(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    by_time = {int(item["time_ms"]): item for item in baseline["slices"]}
    transitions: Counter[str] = Counter()
    all_transitions: Counter[str] = Counter()
    pairs = []
    for item in candidate["slices"]:
        reference = by_time.get(int(item["time_ms"]))
        if reference is None:
            continue
        all_transitions[f"{reference['outcome']}->{item['outcome']}"] += 1
        if item["phase"] not in PLASMA_PHASES:
            continue
        transitions[f"{reference['outcome']}->{item['outcome']}"] += 1
        if not (reference["afile"] and reference["gfile"] and item["afile"] and item["gfile"]):
            continue
        scalar_delta = {}
        for name in (
            "rm", "zm", "area", "volume", "li", "betap", "q95", "qmin", "wmhd",
            "cjor0", "cjor95", "cjor99", "cj1ave", "peak", "chisq", "condno",
        ):
            before = float(reference["afile"]["scalars"][name])
            after = float(item["afile"]["scalars"][name])
            scalar_delta[name] = after - before
            scalar_delta[f"{name}_relative"] = None if before == 0 else (after - before) / abs(before)
        boundary = _curve_distance(item["gfile"]["boundary"], reference["gfile"]["boundary"])
        profiles = {
            name: _relative_rms(
                item["gfile"]["profiles"][name],
                reference["gfile"]["profiles"][name],
                start=(7 if name == "q" else 0),  # exclude psi_N <~ 0.05 (#317)
            )
            for name in ("pressure", "pprime", "ffprime", "f", "q", "jphi_reference_r")
        }
        profile_diagnostic_delta = {}
        for name in (
            "jphi_psi_080",
            "jphi_psi_090",
            "jphi_psi_095",
            "jphi_edge",
            "jphi_edge_shell_mean",
            "jphi_edge_shell_to_peak",
        ):
            before = reference["gfile"]["profile_diagnostics"].get(name)
            after = item["gfile"]["profile_diagnostics"].get(name)
            if before is None or after is None:
                continue
            profile_diagnostic_delta[name] = float(after) - float(before)
            profile_diagnostic_delta[f"{name}_relative"] = (
                None
                if float(before) == 0.0
                else (float(after) - float(before)) / abs(float(before))
            )
        fit_delta = {}
        if reference.get("mfile") and item.get("mfile"):
            for name in (
                "magnetics_chisq",
                "magnetics_chisq_per_active_signal",
                "bpol_probe_chisq",
                "flux_loop_chisq",
                "plasma_current_chisq",
                "total_chisq",
            ):
                before = reference["mfile"]["scalars"].get(name)
                after = item["mfile"]["scalars"].get(name)
                if before is None or after is None:
                    continue
                fit_delta[name] = float(after) - float(before)
                fit_delta[f"{name}_relative"] = (
                    None
                    if float(before) == 0.0
                    else (float(after) - float(before)) / abs(float(before))
                )
        pairs.append(
            {
                "time_ms": item["time_ms"],
                "phase": item["phase"],
                "scalar_delta": scalar_delta,
                "boundary": boundary,
                "profile_relative_rms": profiles,
                "profile_diagnostic_delta": profile_diagnostic_delta,
                "fit_delta": fit_delta,
            }
        )
    phase_comparison = {}
    for phase in ("ramp_up", "quasi_stationary", "ramp_down"):
        selected = [pair for pair in pairs if pair["phase"] == phase]
        if not selected:
            continue
        phase_comparison[phase] = {
            "common_produced": len(selected),
            "lcfs_rms_mm": spread(
                pair["boundary"]["rms_m"] * 1000.0
                for pair in selected
                if pair["boundary"] is not None
            ),
            "absolute_relative_change": {
                name: spread(
                    (pair["scalar_delta"][f"{name}_relative"] for pair in selected),
                    absolute=True,
                )
                for name in ("area", "volume", "li", "betap", "q95", "condno")
            },
            "magnetics_chisq_absolute_relative_change": spread(
                (
                    pair["fit_delta"].get("magnetics_chisq_relative")
                    for pair in selected
                ),
                absolute=True,
            ),
            "jphi_edge_shell_to_peak_absolute_change": spread(
                (
                    pair["profile_diagnostic_delta"].get(
                        "jphi_edge_shell_to_peak"
                    )
                    for pair in selected
                ),
                absolute=True,
            ),
        }
    return {
        "common_produced": len(pairs),
        "outcome_transitions": dict(sorted(transitions.items())),
        "all_outcome_transitions": dict(sorted(all_transitions.items())),
        "by_phase": phase_comparison,
        "lcfs_rms_mm": spread(
            pair["boundary"]["rms_m"] * 1000.0
            for pair in pairs if pair["boundary"] is not None
        ),
        "lcfs_hausdorff_mm": spread(
            pair["boundary"]["hausdorff_m"] * 1000.0
            for pair in pairs if pair["boundary"] is not None
        ),
        "absolute_relative_change": {
            name: spread(
                (pair["scalar_delta"][f"{name}_relative"] for pair in pairs),
                absolute=True,
            )
            for name in (
                "area", "volume", "li", "betap", "q95", "qmin", "wmhd", "cjor0",
                "cjor95", "cjor99", "cj1ave", "peak", "chisq", "condno",
            )
        },
        "profile_relative_rms": {
            name: spread(pair["profile_relative_rms"][name] for pair in pairs)
            for name in ("pressure", "pprime", "ffprime", "f", "q", "jphi_reference_r")
        },
        "edge_current_absolute_relative_change": {
            name: spread(
                (
                    pair["profile_diagnostic_delta"].get(f"{name}_relative")
                    for pair in pairs
                ),
                absolute=True,
            )
            for name in (
                "jphi_psi_080",
                "jphi_psi_090",
                "jphi_psi_095",
                "jphi_edge",
                "jphi_edge_shell_mean",
                "jphi_edge_shell_to_peak",
            )
        },
        "fit_absolute_relative_change": {
            name: spread(
                (
                    pair["fit_delta"].get(f"{name}_relative")
                    for pair in pairs
                ),
                absolute=True,
            )
            for name in (
                "magnetics_chisq",
                "magnetics_chisq_per_active_signal",
                "bpol_probe_chisq",
                "flux_loop_chisq",
                "plasma_current_chisq",
                "total_chisq",
            )
        },
        "pairs": pairs,
    }


def summarize_ensemble(models: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize across-model spread at each common plasma time."""
    by_time: dict[int, list[Mapping[str, Any]]] = {}
    for model in models.values():
        for item in model["run"]["slices"]:
            if (
                item["phase"] in PLASMA_PHASES
                and item.get("afile")
                and item.get("gfile")
            ):
                by_time.setdefault(int(item["time_ms"]), []).append(item)

    records = []
    for time_ms, items in sorted(by_time.items()):
        if len(items) < 2:
            continue
        values = {
            name: np.asarray(
                [item["afile"]["scalars"][name] for item in items], dtype=float
            )
            for name in ("area", "volume", "li", "betap", "q95")
        }
        magnetic = np.asarray(
            [
                item["mfile"]["scalars"]["magnetics_chisq"]
                for item in items
                if item.get("mfile")
                and item["mfile"]["scalars"]["magnetics_chisq"] is not None
            ],
            dtype=float,
        )
        if magnetic.size:
            values["magnetics_chisq"] = magnetic
        record = {
            "time_ms": time_ms,
            "phase": items[0]["phase"],
            "models_n": len(items),
            "quantities": {},
        }
        for name, array in values.items():
            finite = array[np.isfinite(array)]
            if finite.size < 2:
                continue
            center = float(np.median(finite))
            sigma = float(np.std(finite))
            record["quantities"][name] = {
                "sigma": sigma,
                "relative_sigma": None if center == 0.0 else sigma / abs(center),
                "relative_span": (
                    None
                    if center == 0.0
                    else float(finite.max() - finite.min()) / abs(center)
                ),
            }
        records.append(record)

    names = ("area", "volume", "li", "betap", "q95", "magnetics_chisq")
    return {
        "times_n": len(records),
        "relative_sigma": {
            name: spread(
                record["quantities"].get(name, {}).get("relative_sigma")
                for record in records
            )
            for name in names
        },
        "relative_span": {
            name: spread(
                record["quantities"].get(name, {}).get("relative_span")
                for record in records
            )
            for name in names
        },
        "records": records,
    }


def markdown(payload: Mapping[str, Any]) -> str:
    def cell(value: Any) -> str:
        return "–" if value is None else f"{float(value):.6g}"

    lines = ["# EFIT profile-model uncertainty — first pass (#579)", ""]
    lines.append(
        "All models use `ellipse_rzero=0.32 m`, explicit `ICINIT=2`, the same constraints, "
        "grid, temporal sampling, executable, and VEST acceptance envelope."
    )
    lines.extend(
        [
            "",
            "The headline counts exclude vacuum slices; the JSON retains every requested outcome.",
            "",
            "| shot | model | plasma produced/requested | accepted | collapsed | no output | magnetic chi-square median | condno median | seconds |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        for name, run in block["models"].items():
            summary = run["summary"]
            outcomes = summary["plasma_outcomes"]
            lines.append(
                f"| {shot} | `{name}` | {summary['plasma_produced']}/{summary['plasma_requested']} "
                f"| {outcomes.get('accepted', 0)} | {outcomes.get('collapsed', 0)} "
                f"| {outcomes.get('no_output', 0)} | {cell(summary['magnetics_chisq']['median'])} "
                f"| {cell(summary['condno']['median'])} "
                f"| {summary['seconds']:.1f} |"
            )
    lines.extend(
        [
            "",
            f"Paired differences below use `{BASELINE_MODEL}` as the baseline and only plasma times where both models produced an equilibrium.",
            "",
            "| shot | candidate | common plasma slices | LCFS RMS median [mm] | |Δarea| median | |Δvolume| median | |Δ magnetic chi-square| median |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        for name, comparison in block["comparisons"].items():
            change = comparison["absolute_relative_change"]
            fit_change = comparison["fit_absolute_relative_change"]
            lines.append(
                f"| {shot} | `{name}` | {comparison['common_produced']} "
                f"| {cell(comparison['lcfs_rms_mm']['median'])} "
                f"| {cell(change['area']['median'])} | {cell(change['volume']['median'])} "
                f"| {cell(fit_change['magnetics_chisq']['median'])} |"
            )
    lines.extend(
        [
            "",
            "Phase-resolved paired medians:",
            "",
            "| shot | candidate | phase | common | LCFS RMS [mm] | |Δarea| | |Δvolume| | |Δ magnetic chi-square| |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        for name, comparison in block["comparisons"].items():
            for phase, phase_result in comparison["by_phase"].items():
                change = phase_result["absolute_relative_change"]
                lines.append(
                    f"| {shot} | `{name}` | {phase} | {phase_result['common_produced']} "
                    f"| {cell(phase_result['lcfs_rms_mm']['median'])} "
                    f"| {cell(change['area']['median'])} "
                    f"| {cell(change['volume']['median'])} "
                    f"| {cell(phase_result['magnetics_chisq_absolute_relative_change']['median'])} |"
                )
    lines.extend(
        [
            "",
            "Edge-current and profile-pathology diagnostics:",
            "",
            "| shot | model | edge-shell |j_phi| / peak median | negative-pressure slices | non-finite F slices | p-prime turns max | FF-prime turns max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        for name, run in block["models"].items():
            summary = run["summary"]
            pathology = summary["profile_pathology"]
            lines.append(
                f"| {shot} | `{name}` "
                f"| {cell(summary['edge_current']['jphi_edge_shell_to_peak']['median'])} "
                f"| {pathology['negative_pressure_slices']} "
                f"| {pathology['nonfinite_f_slices']} "
                f"| {cell(pathology['pprime_turns']['max'])} "
                f"| {cell(pathology['ffprime_turns']['max'])} |"
            )
    lines.extend([
        "",
        "Magnetic chi-square is `sum(saimpi) + sum(saisil)` from EFIT's m-file. The a-file total is retained in JSON but is dominated by the model-invariant plasma-current term and is not used to rank profile models.",
        "",
        "`beta_p` and absolute pressure are retained as sensitivity outputs, not treated as qualified absolute values: #386 and #659 remain open. The q-profile comparison excludes the first ~5% of normalized flux because of #317.",
    ])
    return "\n".join(lines) + "\n"


def _selected_models(names: str | None, full_matrix: bool) -> tuple[Model, ...]:
    all_models = PILOT_MODELS + (ORDER_MODELS if full_matrix else ())
    if names is None:
        return all_models
    requested = [name.strip() for name in names.split(",") if name.strip()]
    by_name = {model.name: model for model in PILOT_MODELS + ORDER_MODELS}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}")
    selected = tuple(by_name[name] for name in requested)
    if BASELINE_MODEL not in requested:
        selected = (by_name[BASELINE_MODEL],) + selected
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shots", default="41672", help="comma-separated; first pass defaults to 41672")
    parser.add_argument("--models", default=None, help="comma-separated model names")
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument("--tables", default=None, help="source 129x129 table directory")
    parser.add_argument("--packaged-envelope", action="store_true", help="do not install the VEST acceptance envelope")
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--tstep", type=float, default=0.001)
    parser.add_argument("--average-window", type=float, default=0.0005)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())

    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path

    resolved = resolve_toolchain()
    if resolved.get("efit") is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    source_tables = Path(args.tables).expanduser() if args.tables else Path(data_path("efit")).resolve()
    if args.packaged_envelope:
        tables, table_record = source_tables, {"source": str(source_tables), "policy": "packaged envelope"}
    else:
        tables, table_record = prepare_vest_tables(source_tables, output / "tables")
    table_dir = str(tables.resolve()) + "/"

    reference = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }
    shots = [int(value) for value in args.shots.split(",")]
    models = _selected_models(args.models, args.full_matrix)
    seed_study = _module(SEED_STUDY, "profile_model_seed_support")
    base = fixed_scientific_config()

    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "issue": 579,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved),
        "table": table_record,
        "fixed_scientific": base.to_dict(),
        "fixed_scientific_sha256": base.sha256,
        "models": [model.__dict__ for model in models],
        "phase_rule": "vacuum below CUTIP; otherwise sign(dIp/dt), quasi-stationary within 15% of max |dIp/dt|",
        "shots": {},
    }

    for shot in shots:
        if shot not in products:
            raise ValueError(f"{shot}: no packaged pre-EFIT reference product")
        print(f"{shot}: building one shared constraint set", flush=True)
        constraints, times, window, baseline_module = seed_study.prepare_shot(
            shot,
            Path(data_path(products[shot])),
            workdir=output / f"shot_{shot}" / "constraints",
            tables=table_dir,
            tstep=args.tstep,
            average_window=args.average_window,
        )
        phase_by_time, phase_threshold = _phase_map(
            constraints, times, base.initialization.current_threshold
        )
        shot_block: dict[str, Any] = {
            "window": {"start": float(window.start), "end": float(window.end), "requested": int(times.size)},
            "phase_dcurrent_dt_threshold": phase_threshold,
            "models": {},
            "comparisons": {},
        }
        for model in models:
            print(f"  {model.name}: {model.purpose}", flush=True)
            scientific = scientific_for(model)
            run = run_model(
                constraints,
                shot=shot,
                times=times,
                workdir=output / f"shot_{shot}" / model.name,
                executable=str(resolved["efit"]),
                scientific=scientific,
                baseline_module=baseline_module,
                phase_by_time=phase_by_time,
            )
            shot_block["models"][model.name] = {
                "specification": model.__dict__,
                "summary": summarize_run(run),
                "run": run,
            }
            summary = shot_block["models"][model.name]["summary"]
            print(
                f"    {summary['produced']}/{summary['requested']} produced; "
                f"{summary['outcomes'].get('accepted', 0)} accepted; {summary['seconds']:.1f} s",
                flush=True,
            )
        baseline = shot_block["models"][BASELINE_MODEL]["run"]
        for model in models:
            if model.name != BASELINE_MODEL:
                shot_block["comparisons"][model.name] = compare_runs(
                    shot_block["models"][model.name]["run"], baseline
                )
        shot_block["ensemble"] = summarize_ensemble(shot_block["models"])
        payload["shots"][str(shot)] = shot_block

        # Checkpoint after each discharge; the raw model outputs and per-model
        # caches already make each individual run resumable.
        target = args.table or output / "profile_model_study.json"
        target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    target = args.table or output / "profile_model_study.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    report = markdown(payload)
    report_path = args.markdown or output / "profile_model_study.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

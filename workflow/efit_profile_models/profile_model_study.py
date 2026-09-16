"""Full-discharge EFIT profile-model uncertainty study (issue #579).

The first pass is deliberately a profile-model experiment, not a numerical
tuning experiment.  It pins the qualified seed at ``RELIP=0.32 m``, writes
``ICINIT=2`` explicitly, states the executable's current termination defaults
explicitly, and varies only ``KPPCUR``, ``KFFCUR``, ``PCURBD``, ``FCURBD`` and
``FWTBP``.

Example::

    cd <checkout> && EFITHOME=~/git/efit/vaft-install python - <<'EOF'
    import os, runpy, sys
    root = os.getcwd()
    sys.path.insert(0, root)
    import vaft; assert vaft.__file__.startswith(root), vaft.__file__
    sys.argv = ["profile_model_study.py", "--output", "/scratch/efit-profile-models"]
    runpy.run_path(
        "workflow/efit_profile_models/profile_model_study.py", run_name="__main__"
    )
    EOF

Run it that way rather than as ``python workflow/.../profile_model_study.py``.
Executing the script by path puts its own directory at ``sys.path[0]``, so an
editable install elsewhere on the machine -- the main checkout, when this is a
worktree -- supplies ``vaft`` and the study silently measures the wrong tree.
``PYTHONPATH`` does not fix it: the editable install's path finder wins anyway,
which is why the launcher asserts on ``vaft.__file__`` instead of trusting it.

Shots default to the reference set's packaged discharges.  The default pilot
includes the low-order model that VFIT found informative, the routine-like
(2,2) baseline, edge freedom, and the FWTBP interaction.  Use
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

import warnings

import numpy as np

SCHEMA = 4
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

#: Two paired reconstructions count as unchanged on a quantity when their
#: relative difference is at or below this.  The a-file prints about seven
#: significant digits and the m-file's per-channel sums carry their own
#: accumulation noise, so a difference down here is the file format rather
#: than the profile model.
#:
#: It is deliberately *not* a "physically negligible" band.  A half-percent
#: change counts as a change, and how big a change is gets read off the signed
#: median, never off these counts.  The counts answer a different question --
#: which way, and on how many slices.
UNCHANGED_RELATIVE = 1.0e-6


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
    """Per-slice phase, keyed by integer millisecond, with its inputs kept.

    The rule moved to `vaft.validation.classify_equilibrium_regimes` (#76).
    It used to be a slope test local to this study -- flat where
    |dIp/dt| <= 0.15 * max|dIp/dt| -- and the study in
    `workflow/efit_numerics/` used a level test instead, so the same slice
    could land in different cohorts depending on which study asked.

    The library rule is the level one. Measured over the three reference
    discharges the two agree on 295 of 300 slices, and the level rule's
    threshold is scaled by peak current rather than by max|dIp/dt|, which is
    set by the termination and varies 3.4x across those discharges.

    `current_cut` is still accepted so the caller's signature does not change,
    but the vacuum cut is the policy's now; a value that disagrees is
    reported rather than silently applied. The returned threshold is the
    absolute current the flat/ramp boundary sits at for this discharge.
    """
    from vaft.validation import classify_equilibrium_regimes
    from vaft.machine_mapping.equilibrium_regime import vest_equilibrium_regime_policy

    policy = vest_equilibrium_regime_policy()
    if float(current_cut) != float(policy.vacuum_current_amperes):
        warnings.warn(
            f"_phase_map was given current_cut={current_cut} but the regime policy "
            f"cuts vacuum at {policy.vacuum_current_amperes} A; the policy is used. "
            "Change vest.yaml's equilibrium_regime.phase.vacuum_current_amperes to move it.",
            stacklevel=2,
        )

    labels = classify_equilibrium_regimes(constraints, policy=policy)
    result: dict[int, dict[str, Any]] = {}
    peak = 0.0
    for index, item in enumerate(labels):
        if index >= times.size:
            break
        if item.current is not None:
            peak = max(peak, abs(item.current))
        result[int(round(float(times[index]) * 1000.0))] = {
            "current": item.current,
            "dcurrent_dt": item.current_rate,
            # This study's stored tables name the flat cohort
            # `quasi_stationary`; the library calls it `flat`.
            "phase": "quasi_stationary" if item.phase == "flat" else item.phase,
        }
    return result, peak * policy.flat_fraction


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


def outcome_of(record: Mapping[str, Any], shared: Any) -> str:
    """What became of one slice, in the vocabulary `seed_basin` established.

    `shared` is `seed_basin.outcome`.  It is called rather than restated
    because a second definition of "accepted" is how two studies stop being
    comparable, and the numbers here have to sit beside the seed, domain and
    termination scans.

    One condition is added and it is stated rather than hidden: this study
    also needs the g-file.  Every comparison it makes is about `p'`, `FF'`,
    the current profile or the boundary, all of which live in the g-file, so a
    slice that wrote an a-file and no g-file contributes nothing here.  It is
    `no_output` in this study and `flagged` in `seed_basin`, and that is the
    only label on which the two disagree.
    """
    label = shared(record)
    if label in ("accepted", "flagged") and record["gfile"] is None:
        return "no_output"
    return label


def run_model(
    constraints: Any,
    *,
    shot: int,
    times: np.ndarray,
    workdir: Path,
    executable: str,
    scientific: Any,
    baseline_module: Any,
    seed_module: Any,
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
        row["outcome"] = outcome_of(row, seed_module.outcome)
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


def signed_change(
    values: Iterable[Any], *, tolerance: float = UNCHANGED_RELATIVE
) -> dict[str, Any]:
    """Which way a paired quantity moved, and on how many slices.

    `spread(..., absolute=True)` says how far a model moved something and
    never which way, so it cannot answer "does raising the order *lower*
    chi-square".  This is the same summary with the sign kept, plus the
    counts the termination scan reported for exactly this reason (#171):
    on the slices both configurations produced, how many went down, how many
    went up, and how many did not move.

    The key names are direction-neutral because the same helper summarizes
    `area` and `li`, where smaller is not better.  Calling a decrease an
    improvement is a judgement about chi-square specifically, and it belongs
    in the rendered table where it can be explained, not in a stored key.

    `identical` counts exact equality and is a subset of `unchanged`.  It is
    kept separate because "the two configurations produced the same number on
    every slice" is a stronger and more useful statement than "they agreed to
    within the file's precision", and #171 found configurations that did.
    """
    array = _numbers(values)
    record: dict[str, Any] = dict(spread(array))
    record["tolerance"] = float(tolerance)
    record["p05"] = float(np.percentile(array, 5)) if array.size else None
    record["decreased"] = int(np.count_nonzero(array < -tolerance))
    record["increased"] = int(np.count_nonzero(array > tolerance))
    record["unchanged"] = int(np.count_nonzero(np.abs(array) <= tolerance))
    record["identical"] = int(np.count_nonzero(array == 0.0))
    return record


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
            # A model can lower chi-square through the flat-top and raise it
            # through the ramp.  The aggregate signed median is precisely the
            # statistic that cancels those two and reports nothing, so the
            # direction has to be resolved per cohort as well.
            "signed_relative_change": {
                name: signed_change(
                    pair["scalar_delta"][f"{name}_relative"] for pair in selected
                )
                for name in ("area", "volume", "li", "betap", "q95", "condno")
            },
            "magnetics_chisq_signed_relative_change": signed_change(
                pair["fit_delta"].get("magnetics_chisq_relative")
                for pair in selected
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
        # The same three quantities with the sign kept.  `n` is not
        # `common_produced`: a scalar's relative change is withheld when the
        # baseline value is exactly zero, and a fit delta needs an m-file on
        # both sides, so the counts sum to `n` and a report that prints only
        # `common_produced` beside them will not add up.
        "signed_relative_change": {
            name: signed_change(
                pair["scalar_delta"][f"{name}_relative"] for pair in pairs
            )
            for name in (
                "area", "volume", "li", "betap", "q95", "qmin", "wmhd", "cjor0",
                "cjor95", "cjor99", "cj1ave", "peak", "chisq", "condno",
            )
        },
        # `PCURBD` and `FCURBD` are edge-current switches, so the sign of the
        # edge-current change is what they do.  An absolute median there loses
        # the thing the model was varied to find out.
        "edge_current_signed_relative_change": {
            name: signed_change(
                pair["profile_diagnostic_delta"].get(f"{name}_relative")
                for pair in pairs
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
        "fit_signed_relative_change": {
            name: signed_change(
                pair["fit_delta"].get(f"{name}_relative") for pair in pairs
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


ENSEMBLE_QUANTITIES = ("area", "volume", "li", "betap", "q95", "magnetics_chisq")


def summarize_ensemble(models: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Across-model spread at each plasma time, over a population it names.

    The spread is only a trajectory if the ensemble it is taken over is the
    same one at every time.  Changing a profile model changes which slices
    reconstruct at all, so a sigma taken over "whatever produced an
    equilibrium here" is a sigma over a different ensemble at every slice --
    the same selection effect that made `NXITER = 3` look like an improvement
    in #171 when it had simply failed on the badly-fitted slices.

    Both populations are therefore reported and named.  `complete` is the
    headline: the times where every model in the roster produced an
    equilibrium.  The unqualified `relative_sigma` and `relative_span` keep
    their original meaning, over whatever was available, and
    `models_n_per_time` says in one row whether the two can differ at all.

    When one model produces nothing the strict intersection empties.  That is
    not repaired by quietly taking the largest subset that survives -- a
    population chosen to fit the discharge is what `vest_acceptance_envelope`
    refused to do for bounds.  `complete_times_n_without` names the model
    responsible and leaves the choice to the reader.
    """
    roster = sorted(models)
    models_total = len(roster)
    by_time: dict[int, dict[str, Mapping[str, Any]]] = {}
    # Every plasma time the discharge asked for, produced or not.  A time
    # where fewer than two models reconstructed carries no spread and so
    # leaves `records` entirely; counting the population only over the times
    # that survived would be the selection effect this function exists to
    # report, one level up.
    plasma_times: set[int] = set()
    for name in roster:
        for item in models[name]["run"]["slices"]:
            if item["phase"] not in PLASMA_PHASES:
                continue
            plasma_times.add(int(item["time_ms"]))
            if item.get("afile") and item.get("gfile"):
                by_time.setdefault(int(item["time_ms"]), {})[name] = item

    records = []
    for time_ms, entries in sorted(by_time.items()):
        if len(entries) < 2:
            continue
        items = list(entries.values())
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
        missing = sorted(set(roster) - set(entries))
        record = {
            "time_ms": time_ms,
            "phase": items[0]["phase"],
            # Every model is handed the same `phase_by_time`, built once per
            # shot, so this should never be false.  It is recorded so that
            # guarantee is falsifiable rather than assumed: the day it is
            # false, the phase merge in `run_model` is broken, not the slice.
            "phase_agreement": len({item["phase"] for item in items}) == 1,
            "models_n": len(items),
            "models_total": models_total,
            "models_missing": missing,
            "complete": not missing,
            "quantities": {},
        }
        for name, array in values.items():
            finite = array[np.isfinite(array)]
            if finite.size < 2:
                continue
            center = float(np.median(finite))
            sigma = float(np.std(finite))
            record["quantities"][name] = {
                "n": int(finite.size),
                # The normalizer, stored so `relative_sigma` can be checked
                # against the table instead of recomputed from nothing.
                "center": center,
                "sigma": sigma,
                "relative_sigma": None if center == 0.0 else sigma / abs(center),
                "relative_span": (
                    None
                    if center == 0.0
                    else float(finite.max() - finite.min()) / abs(center)
                ),
            }
        records.append(record)

    names = ENSEMBLE_QUANTITIES

    def _complete(record: Mapping[str, Any], name: str) -> bool:
        quantity = record["quantities"].get(name)
        return bool(record["complete"] and quantity and quantity["n"] == models_total)

    def _aggregate(
        selected: Sequence[Mapping[str, Any]], key: str, *, strict: bool
    ) -> dict[str, Any]:
        return {
            name: spread(
                record["quantities"].get(name, {}).get(key)
                for record in selected
                if not strict or _complete(record, name)
            )
            for name in names
        }

    complete_records = [record for record in records if record["complete"]]
    # Leave-one-out, over every plasma time rather than over the times that
    # reached `records`.  A time only one model reconstructed carries no
    # spread, so it is absent from `records` -- but it is exactly the kind of
    # time a reader wants attributed, and counting only the survivors would
    # hide the model that caused the loss.
    without = {name: 0 for name in roster}
    for time_ms in plasma_times:
        missing = set(roster) - set(by_time.get(time_ms, {}))
        if not missing:
            for name in roster:
                without[name] += 1
        elif len(missing) == 1:
            without[missing.pop()] += 1

    by_phase = {}
    for phase in ("ramp_up", "quasi_stationary", "ramp_down"):
        selected = [record for record in records if record["phase"] == phase]
        if not selected:
            continue
        by_phase[phase] = {
            "times_n": len(selected),
            "complete_times_n": sum(1 for record in selected if record["complete"]),
            "models_n_per_time": spread(record["models_n"] for record in selected),
            "relative_sigma": _aggregate(selected, "relative_sigma", strict=False),
            "complete": {
                "times_n": sum(1 for record in selected if record["complete"]),
                "relative_sigma": _aggregate(
                    selected, "relative_sigma", strict=True
                ),
            },
        }

    return {
        "models": roster,
        "models_total": models_total,
        # What the unqualified keys below are taken over.
        "population": "available",
        "statistics": {
            "center": "median",
            "sigma": "population standard deviation (numpy std, ddof=0)",
            "relative_to": "center",
        },
        "plasma_times_n": len(plasma_times),
        "times_n": len(records),
        # The selection-effect measure.  The two populations coincide only
        # when every model produced at every plasma time -- that is
        # `complete_times_n == plasma_times_n`, not `== times_n`, because a
        # time with one lone model never reached `records` to be counted.
        "models_n_per_time": spread(record["models_n"] for record in records),
        "complete_times_n": len(complete_records),
        "complete_times_n_without": without,
        "relative_sigma": _aggregate(records, "relative_sigma", strict=False),
        "relative_span": _aggregate(records, "relative_span", strict=False),
        "complete": {
            "times_n": len(complete_records),
            "models_n": models_total,
            "relative_sigma": _aggregate(records, "relative_sigma", strict=True),
            "relative_span": _aggregate(records, "relative_span", strict=True),
        },
        "by_phase": by_phase,
        "records": records,
    }


def _ensemble_section(lines: list[str], payload: Mapping[str, Any], cell) -> None:
    """The model-induced uncertainty #579 asks for, over a population it names."""
    blocks = {
        shot: block["ensemble"]
        for shot, block in payload["shots"].items()
        if block.get("ensemble")
    }
    if not blocks:
        return

    lines.extend(
        [
            "",
            "## Model-induced uncertainty",
            "",
            "Each plasma time carries a spread across the models that reconstructed it; the "
            "tables summarize those over time. `sigma` is a population standard deviation "
            "(numpy `std`, `ddof=0`) divided by the models' median. Over a handful of models "
            "that is a spread indicator and not a Gaussian one-sigma — with two models it is "
            "exactly half the span by construction. Plasma area and volume come first because "
            "#579 names them the primary model-sensitivity quantities.",
        ]
    )

    for shot, ensemble in blocks.items():
        per_time = ensemble["models_n_per_time"]
        total = ensemble["models_total"]
        complete_n = ensemble["complete_times_n"]
        plasma_n = ensemble["plasma_times_n"]
        if complete_n == plasma_n:
            lines.extend(
                [
                    "",
                    f"**{shot}: all {total} models produced an equilibrium at every one of the "
                    f"{plasma_n} plasma times, so the two populations coincide** and nothing "
                    f"below can be a selection effect.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    f"**{shot}: the population is not the same at every time.** Only "
                    f"{complete_n} of {plasma_n} plasma times carry all {total} models; where a "
                    f"spread exists at all, between {int(per_time['min'])} and "
                    f"{int(per_time['max'])} models produced one. A sigma taken over whatever "
                    f"happened to reconstruct is a sigma over a different ensemble at every "
                    f"slice — #171's `NXITER = 3` is the standing example of what that does to "
                    f"a median. The table therefore reports the all-{total}-model times. The "
                    f"all-available figures are kept in the JSON under "
                    f"`shots.{shot}.ensemble.relative_sigma` and are not the headline.",
                ]
            )

    lines.extend(
        [
            "",
            "| shot | quantity | times (every model) | models | relative sigma median | p95 | max | relative span median |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for shot, ensemble in blocks.items():
        complete = ensemble["complete"]
        for name in ENSEMBLE_QUANTITIES:
            sigma = complete["relative_sigma"][name]
            span = complete["relative_span"][name]
            lines.append(
                f"| {shot} | {name} | {sigma['n']} | {ensemble['models_total']} "
                f"| {cell(sigma['median'])} | {cell(sigma['p95'])} | {cell(sigma['max'])} "
                f"| {cell(span['median'])} |"
            )

    phase_rows = [
        (shot, phase, phase_block)
        for shot, ensemble in blocks.items()
        for phase, phase_block in ensemble["by_phase"].items()
    ]
    if phase_rows:
        lines.extend(
            [
                "",
                "Per cohort, on the same every-model population:",
                "",
                "| shot | phase | times (every model) | area | volume | li | q95 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for shot, phase, phase_block in phase_rows:
            sigma = phase_block["complete"]["relative_sigma"]
            lines.append(
                f"| {shot} | {phase} | {phase_block['complete']['times_n']} "
                + "".join(
                    f"| {cell(sigma[name]['median'])} "
                    for name in ("area", "volume", "li", "q95")
                )
                + "|"
            )

    incomplete = {
        shot: ensemble
        for shot, ensemble in blocks.items()
        if ensemble["complete_times_n"] < ensemble["plasma_times_n"]
    }
    if incomplete:
        lines.extend(
            [
                "",
                "Which model empties the common population, one at a time. This is the caveat "
                "as a table rather than a choice made on the reader's behalf: the largest "
                "subset that happens to survive would be a population fitted to the discharge.",
                "",
                "| shot | model dropped | plasma times complete without it |",
                "|---|---|---:|",
            ]
        )
        for shot, ensemble in incomplete.items():
            without = ensemble["complete_times_n_without"]
            for name in sorted(without, key=lambda item: (-without[item], item)):
                lines.append(f"| {shot} | `{name}` | {without[name]} |")

    lines.extend(
        [
            "",
            "The per-time series itself — sigma_A(t), sigma_V(t), sigma_li(t), sigma_q95(t) — is "
            "in the JSON at `shots.<shot>.ensemble.records[].quantities.<name>.relative_sigma`, "
            "with `models_n`, `models_missing` and `complete` on every record so any time's "
            "population can be checked. These tables are summaries of it, not a substitute.",
        ]
    )


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
            "`coefficients` is `KPPCUR + KFFCUR`, the fitted profile freedom each row bought "
            "its fit with. The chi-square beside it is **not** reduced by it: EFIT does not "
            "report its total free-parameter count, and `magnetics_chisq_per_active_signal` "
            "in the JSON divides by the number of active signals, not by degrees of freedom.",
            "",
            "| shot | model | coefficients | plasma produced/requested | accepted | collapsed | no output | magnetic chi-square median | condno median | seconds |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        for name, run in block["models"].items():
            summary = run["summary"]
            outcomes = summary["plasma_outcomes"]
            specification = run["specification"]
            coefficients = int(specification["kppcur"]) + int(specification["kffcur"])
            lines.append(
                f"| {shot} | `{name}` | {coefficients} "
                f"| {summary['plasma_produced']}/{summary['plasma_requested']} "
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
            "## Which way the fit moved",
            "",
            f"An absolute median says how far a model moved the fit and never which way, so it "
            f"cannot answer whether more profile freedom *lowers* chi-square. These counts are "
            f"taken on the slices both models produced. `unchanged` means the two agreed to "
            f"within {UNCHANGED_RELATIVE:g} relative — the files' own precision, not a physical "
            f"tolerance, so a half-percent change counts as a change and its size is the signed "
            f"median rather than these counts. `identical` is exact equality, a subset of "
            f"`unchanged`.",
            "",
            "`compared` is below `slices in both` wherever a slice is missing an m-file: the "
            "three counts sum to `compared`.",
            "",
            "| shot | candidate | slices in both | compared | chi-square improved | worsened | unchanged | identical | signed median | p05 | p95 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        for name, comparison in block["comparisons"].items():
            signed = comparison["fit_signed_relative_change"]["magnetics_chisq"]
            lines.append(
                f"| {shot} | `{name}` | {comparison['common_produced']} "
                f"| {signed['n']} | {signed['decreased']} | {signed['increased']} "
                f"| {signed['unchanged']} | {signed['identical']} "
                f"| {cell(signed['median'])} | {cell(signed['p05'])} "
                f"| {cell(signed['p95'])} |"
            )
    lines.extend(
        [
            "",
            "The same question per cohort, because a model can lower chi-square through the "
            "flat-top and raise it through the ramp, and one signed median over the discharge "
            "cancels exactly that:",
            "",
            "| shot | candidate | phase | compared | improved | worsened | unchanged | signed median |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for shot, block in payload["shots"].items():
        for name, comparison in block["comparisons"].items():
            for phase, phase_result in comparison["by_phase"].items():
                signed = phase_result["magnetics_chisq_signed_relative_change"]
                lines.append(
                    f"| {shot} | `{name}` | {phase} | {signed['n']} "
                    f"| {signed['decreased']} | {signed['increased']} "
                    f"| {signed['unchanged']} | {cell(signed['median'])} |"
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
    _ensemble_section(lines, payload, cell)
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
    parser.add_argument(
        "--shots",
        default=None,
        help="comma-separated; default: every reference-set shot with a packaged product",
    )
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
    shots = [int(value) for value in args.shots.split(",")] if args.shots else sorted(products)
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
        # What actually labelled the slices, which is the library's level rule
        # since #76 -- not the slope rule this study used to carry locally.
        "phase_rule": (
            "vaft.validation.classify_equilibrium_regimes under "
            "vest.yaml equilibrium_regime: vacuum below the policy current cut; "
            "otherwise flat within flat_fraction of that discharge's peak |Ip|, "
            "else ramp_up or ramp_down by the sign of dIp/dt"
        ),
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
            # An absolute current, not a rate: the level rule's flat/ramp
            # boundary for this discharge, `flat_fraction * peak |Ip|`.
            "phase_flat_current_threshold": phase_threshold,
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
                seed_module=seed_study,
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

#!/usr/bin/env python3
"""Score the archived legacy pipeline and the current one with the same kernels, slice by slice.

Inputs
------
* ``--legacy-root``: the restored shot-first FileDB ``public/`` tree
  (``<shot>/efit/{gfile,mfile,afile}/``, ``<shot>/linear_stability/``).
* ``--current-filedb``: the current FileDB, read at ``efit/magnetic/<shot>/``.
  Only slices whose time is in HSDS ``main``'s equilibrium are kept, so
  "current" means what ``main`` publishes and not a rerun or a retired product.
* HSDS ``public`` and ``main``, for the diagnostics of the vacuum check and the
  Thomson reference. Read with ``cache="off"``, one shot at a time, with
  ``--hsds-sleep`` seconds between reads.
* ``--atlas-dir``: the stability atlas (``atlas_n.csv``, ``atlas_surfaces.csv``),
  copied into the output.

The two sides go through one reader per check: the m-file for measured against
reconstructed (``magnetic_fit`` of ``workflow/efit_numerics/convergence_study.py``),
the g-file for the virial pair and the Grad-Shafranov residual (the #891 study's
definitions), the g-file against HSDS ``public`` Thomson for the pressure check
(``thomson_pressure_check.check_gfile``), and the DCON/RDCON outputs of the
legacy ``linear_stability`` tree read through :mod:`vaft.code.gpec`.

Outputs (CSV, one row per side and slice unless noted) go to ``--output``:
``slices.csv``, ``fit.csv``, ``constraint_points.csv``, ``virial.csv``,
``thomson.csv``, ``vacuum.csv`` (one row per source, shot and channel),
``stability_legacy.csv`` / ``stability_legacy_surfaces.csv``, the atlas copies
and ``manifest.json``. The run is resumable: a shot already in a check's
``done_<check>.txt`` is skipped. Every failure is a row with a ``status``, never
a silent drop.

Usage on vestserver (isolated worktree, see ``env.sh``)::

    python workflow/pipeline_improvement/build_comparison_tables.py \\
        --legacy-root /mnt/backup/vaft/public_restore_20261021 \\
        --current-filedb /srv/vest.filedb --atlas-dir ~/runs/campaign/atlas/stability \\
        --output ~/runs/pipeline_improvement/tables --checks fit virial thomson stability
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
CHECKS = ("fit", "virial", "thomson", "vacuum", "onset", "stability")
#: The onset-aligned grid of the onset check, t - t_onset [ms].
ONSET_GRID_MS = np.round(np.arange(-20.0, 15.0 + 1e-9, 0.2), 1)
#: Instants before the onset at which every channel's measured and vacuum value is kept [ms].
ONSET_POINT_MS = (-12.0, -8.0, -4.0, -1.0)
#: The instant of the onset QA check, t - t_onset [ms].
ONSET_QA_MS = -1.0
#: Slices of one shot whose g-file time is within this of a main slice belong to main [s].
MAIN_TIME_TOLERANCE = 5.0e-4
GFILE = re.compile(r"^g0*(\d+)\.0*(\d+)$")
#: Per-channel constraint points are kept for this many paired slices (the snapshot stays small).
CONSTRAINT_POINT_SLICES = 400

_MODULES: dict[str, Any] = {}


def _module(path: Path, name: str):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def convergence_study():
    return _module(REPOSITORY / "workflow" / "efit_numerics" / "convergence_study.py", "convergence_study")


def thomson_check():
    return _module(REPOSITORY / "workflow" / "efit_diamagnetic_weight" / "thomson_pressure_check.py",
                   "thomson_pressure_check")


# --------------------------------------------------------------------------
# Locating the EFIT files of one side
# --------------------------------------------------------------------------

def efit_slices(efit_dir: Path) -> list[dict[str, Any]]:
    """Every g-file in ``efit_dir/gfile`` with the m- and a-file of the same time, in time order."""
    out = []
    gdir = efit_dir / "gfile"
    if not gdir.is_dir():
        return out
    for path in sorted(gdir.iterdir()):
        match = GFILE.match(path.name)
        if not match or not path.is_file():
            continue
        suffix = path.name[1:]
        mfile = efit_dir / "mfile" / f"m{suffix}"
        afile = efit_dir / "afile" / f"a{suffix}"
        out.append({"time_ms": int(match.group(2)), "time_s": int(match.group(2)) * 1e-3,
                    "gfile": path, "mfile": mfile if mfile.is_file() else None,
                    "afile": afile if afile.is_file() else None})
    out.sort(key=lambda s: s["time_ms"])
    return out


def main_equilibrium_times(shot: int) -> np.ndarray:
    """The equilibrium times HSDS ``main`` publishes for ``shot``.

    Any failure propagates. The shot list holds only shots whose ``main``
    equilibrium exists, so a failure here is a read error, and treating it as
    "no current slices" would score the shot legacy-only and mark it done.
    The driver records the error and leaves the shot for a resume.
    """
    import vaft

    ods = vaft.database.open(shot, source="main", paths=["equilibrium"])
    return np.asarray(ods["equilibrium.time"], dtype=float).reshape(-1)


def side_slices(shot: int, *, legacy_root: Path, current_filedb: Path,
                main_times: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    """Legacy and current EFIT slices of one shot; current only at times ``main`` publishes.

    The membership test is by time: a FileDB g-file at a ``main`` time is taken
    to be the product ``main`` replicated. A stale generation left at the same
    time would pass; the FileDB stale-g-file scan of 2026-10-12 found none.
    """
    legacy = efit_slices(legacy_root / str(shot) / "efit")
    current = efit_slices(current_filedb / "efit" / "magnetic" / str(shot))
    current = [s for s in current if np.any(np.abs(main_times - s["time_s"]) <= MAIN_TIME_TOLERANCE)]
    return {"legacy": legacy, "current": current}


# --------------------------------------------------------------------------
# Per-slice kernels (identical for both sides)
# --------------------------------------------------------------------------

def fit_row(entry: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measured against reconstructed from the m-file: relative RMS per family, and per-channel points."""
    from vaft.data import read_meqdsk

    if entry["mfile"] is None:
        return {"status": "no_mfile"}, [], {}
    meq = read_meqdsk(entry["mfile"])
    variables = meq  # MEQDSK maps names to variables, which is what magnetic_fit reads
    fit = convergence_study().magnetic_fit(variables)
    row = {
        "status": "ok",
        "probe_relative_rms": fit.get("probe"),
        "loop_relative_rms": fit.get("loop"),
        "ip_relative_error": fit.get("ip"),
        "dia_relative_error": fit.get("dia"),
        "probe_n": fit.get("probe_n"),
        "loop_n": fit.get("loop_n"),
        "dia_n": fit.get("dia_n"),
        "probe_median_relative": fit.get("probe_median_relative"),
        "loop_median_relative": fit.get("loop_median_relative"),
    }
    # The single-channel constraints as values, for a measured-against-
    # reconstructed view in physical units; the weight says whether the fit used it.
    for family in ("ip", "dia"):
        measured, reconstructed, weight, _sigma = convergence_study().MAGNETIC_FAMILIES[family]
        for key, name in (("measured", measured), ("reconstructed", reconstructed), ("weight", weight)):
            try:
                value = np.asarray(getattr(variables[name], "data", variables[name]), float).reshape(-1)
                row[f"{family}_{key}"] = float(value[0]) if value.size else float("nan")
            except KeyError:
                row[f"{family}_{key}"] = float("nan")
    points = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for family, (measured, reconstructed, weight, _sigma) in convergence_study().MAGNETIC_FAMILIES.items():
        if family not in ("probe", "loop"):
            continue
        try:
            m = np.asarray(getattr(variables[measured], "data", variables[measured]), float).reshape(-1)
            r = np.asarray(getattr(variables[reconstructed], "data", variables[reconstructed]), float).reshape(-1)
            w = np.asarray(getattr(variables[weight], "data", variables[weight]), float).reshape(-1)
        except KeyError:
            continue
        n = min(m.size, r.size, w.size)
        arrays[family] = (m[:n], r[:n], w[:n])
        for channel in range(n):
            if w[channel] > 0 and np.isfinite(m[channel]) and np.isfinite(r[channel]):
                points.append({"family": family, "channel": channel,
                               "measured": float(m[channel]), "reconstructed": float(r[channel])})
    return row, points, arrays


def common_channel_fit(legacy: Mapping[str, tuple], current: Mapping[str, tuple]) -> dict[str, dict[str, Any]]:
    """Relative RMS of measured minus reconstructed over the channels *both* fits used.

    Each fit's own relative RMS runs over its own fitted channels, so a
    pipeline that leaves a broken probe out of the fit scores better without
    fitting better. Here a channel counts only when both m-files give it a
    positive weight. The m-file arrays are EFIT's own channel order, the same
    VEST table on both sides (64 probes, 11 loops); unequal lengths are
    refused rather than aligned by position.
    """
    out: dict[str, dict[str, Any]] = {"legacy": {}, "current": {}}
    for family in ("probe", "loop"):
        if family not in legacy or family not in current:
            continue
        (ml, rl, wl), (mc, rc, wc) = legacy[family], current[family]
        if ml.size != mc.size:
            for side in out:
                out[side][f"{family}_common_status"] = f"layout {ml.size} vs {mc.size}"
            continue
        keep = (wl > 0) & (wc > 0) & np.isfinite(ml) & np.isfinite(rl) & np.isfinite(mc) & np.isfinite(rc)
        for side, (m, r) in (("legacy", (ml, rl)), ("current", (mc, rc))):
            scale = float(np.sqrt(np.mean(m[keep] ** 2))) if keep.any() else 0.0
            value = float(np.sqrt(np.mean((m[keep] - r[keep]) ** 2)) / scale) if scale else float("nan")
            out[side][f"{family}_relative_rms_common"] = value
            out[side][f"{family}_common_n"] = int(keep.sum())
    return out


def _equilibrium_ods(gfile: Path):
    from omas import ODS

    from vaft.data import read_geqdsk

    geq = read_geqdsk(gfile)
    ods = ODS(consistency_check=False)
    geq.to_omas(ods)
    return geq, ods


def virial_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    """beta_p and l_i from the volume integral against the pair_13 virial closure, plus the GS residual.

    The #891 study's definitions (``weight_scan._virial``): the integral is
    ``beta_p_boundary_average`` of :func:`derive_global_descriptors`, the closure
    is ``compute_virial_equilibrium_quantities_ods``'s ``pair_13``.
    """
    from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    geq, ods = _equilibrium_ods(entry["gfile"])
    row: dict[str, Any] = {"status": "ok"}
    try:
        v = compute_virial_equilibrium_quantities_ods(copy.deepcopy(ods), time_slice=0)[0]
        descriptor = derive_global_descriptors(as_equilibrium(ods, time_index=0)).values.get("beta_p_boundary_average")
        integral = float(descriptor.value) if descriptor is not None and descriptor.available else float("nan")
        closure, li_closure = float(v["pair_13"]["beta_p"]), float(v["pair_13"]["li"])
        li_volume = float(v["li"])
        row.update({
            "beta_p_integral": integral,
            "beta_p_pair_13": closure,
            "denominator_pair_13": float(v["conditioning"]["denominators"]["pair_13"]),
            "li_volume": li_volume,
            "li_pair_13": li_closure,
            "virial_log_ratio": math.log(integral / closure) if integral > 0 and closure > 0 else float("nan"),
            "li_log_ratio": math.log(li_volume / li_closure) if li_volume > 0 and li_closure > 0 else float("nan"),
        })
    except Exception as error:  # a degenerate equilibrium is a result, not a crash
        row.update({"status": "virial_error", "error": repr(error)[:300]})
    try:
        row["gs_residual"] = float(convergence_study().grad_shafranov_metrics(geq.mapping)["whole"])
    except Exception as error:
        row["gs_error"] = repr(error)[:300]
    try:
        from vaft.data import read_aeqdsk

        if entry["afile"] is not None:
            scalars = read_aeqdsk(entry["afile"]).scalars
            for key in ("betap", "li", "li3", "wmhd", "q95", "ipmhd", "chisq", "terror"):
                if key in scalars:
                    row[f"afile_{key}"] = float(scalars[key])
    except Exception as error:
        row["afile_error"] = repr(error)[:300]
    return row


def thomson_row(entry: Mapping[str, Any], diagnostics: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The summary verdict of ``check_gfile`` and the per-channel samples behind it.

    The channels come from :func:`vaft.validation.equilibrium.thomson_pressure_samples`,
    the sampler the verdict itself is computed from, on the same g-file.
    """
    from vaft.validation.equilibrium import thomson_pressure_samples

    result = thomson_check().check_gfile(entry["gfile"], diagnostics)
    row = {"status": result.get("status"), "thomson_log_ratio": result.get("log_ratio"),
           "thomson_sum_ratio": result.get("sum_ratio"), "thomson_points": result.get("points"),
           "thomson_time_offset_s": result.get("time_offset_s"), "reason": result.get("reason")}
    _, ods = _equilibrium_ods(entry["gfile"])
    sampled = thomson_pressure_samples(ods, 0, diagnostics)
    channels = []
    if sampled.get("available"):
        for sample in sampled["channels"]:
            channels.append({key: sample.get(key) for key in
                             ("channel", "r", "z", "psi_norm", "n_e", "t_e", "n_e_error", "t_e_error", "p_e", "p_recon")})
            channels[-1]["ts_time"] = sampled.get("ts_time")
    return row, channels


#: Wall models of the vacuum check. ``stored`` keeps each product's own
#: ``pf_passive`` currents. ``common_2303`` re-solves the wall from the PF
#: currents alone with one resistance vintage on both sides
#: (:func:`vaft.validation.vacuum_benchmark.benchmark_wall_currents`), so what
#: is left differs only by the measured magnetics and PF currents themselves.
WALL_MODELS = ("stored", "common_2303")


def common_wall_ods(measured: Any, wall: Any) -> Any:
    """``measured``'s magnetics and PF coil currents on ``wall``'s coil and passive geometry.

    The common wall model needs one PF/passive geometry and one coupling for
    both sides, and the legacy product's own coil geometry does not match the
    coupling version VAFT selects for the shot. What the check compares is the
    measurement: the whole ``magnetics`` IDS and each PF coil's current waveform
    (matched by coil name) come from ``measured``; the geometry and the passive
    structure come from ``wall`` (the current product).
    """
    from vaft.omas.general import signal_time

    out = copy.deepcopy(wall)
    out["magnetics"] = copy.deepcopy(measured["magnetics"])
    labels = [str(measured[f"pf_active.coil.{i}.name"]) for i in range(len(measured["pf_active.coil"]))]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate PF coil names in the legacy product: {labels}")
    names = {name: i for i, name in enumerate(labels)}
    time_base = None
    for j in range(len(out["pf_active.coil"])):
        name = str(out[f"pf_active.coil.{j}.name"])
        if name not in names:
            raise KeyError(f"coil {name} has no measured current in the legacy product")
        i = names[name]
        data = np.asarray(measured[f"pf_active.coil.{i}.current.data"], dtype=float)
        t = np.asarray(signal_time(measured, f"pf_active.coil.{i}.current.data"), dtype=float)
        if time_base is None:
            time_base = t
        elif t.shape != time_base.shape or not np.allclose(t, time_base):
            raise ValueError("legacy PF currents do not share one time base")
        out[f"pf_active.coil.{j}.current.data"] = data
        if f"pf_active.coil.{j}.current.time" in out:
            out[f"pf_active.coil.{j}.current.time"] = t
    out["pf_active.time"] = time_base
    return out


def vacuum_rows(ods: Any, wall_model: str = "stored") -> list[dict[str, Any]]:
    """Per-channel vacuum residual over the plasma-free stretch before the onset."""
    from vaft.omas.general import signal_time
    from vaft.omas.vacuum_magnetics import (
        plasma_free_boundary,
        plasma_window,
        synthetic_vacuum_magnetics,
        vacuum_residual_metrics,
    )

    if wall_model == "common_2303":
        from vaft.machine_mapping.wall_resistance import LEGACY_CALIBRATIONS
        from vaft.validation.vacuum_benchmark import benchmark_wall_currents

        ods = benchmark_wall_currents(ods, calibration=LEGACY_CALIBRATIONS["2303"])
    elif wall_model != "stored":
        raise ValueError(f"unknown wall model {wall_model!r}")
    onset, source = plasma_free_boundary(plasma_window(ods))
    start = float(np.asarray(signal_time(ods, "pf_active.coil.0.current.data"), dtype=float)[0])
    window = (start, float(onset))
    channels = synthetic_vacuum_magnetics(ods, per_family=None, window=window)
    metrics = vacuum_residual_metrics(channels, window=window)
    keys = ("name", "kind", "family", "status", "reason", "samples", "measured_rms", "measured_dynamic_range",
            "residual_rms_coil", "residual_rms_coil_eddy", "improvement", "normalized_residual", "correlation",
            "wall_authority")
    return [{**{k: row.get(k) for k in keys}, "wall_model": wall_model, "window_start": window[0],
             "window_end": window[1], "onset_source": source} for row in metrics["channels"]]


def onset_reference(loaded: Mapping[str, Any]) -> tuple[float, str]:
    """One plasma-free boundary for both sources of a shot: main's, else public's.

    Both sides are put on one clock, so a side whose own timing is off shows
    as a shifted trace instead of being realigned out of sight.
    """
    from vaft.omas.vacuum_magnetics import plasma_free_boundary, plasma_window

    errors = []
    for source in ("main", "public"):
        if source not in loaded:
            continue
        try:
            onset, how = plasma_free_boundary(plasma_window(loaded[source]))
            return float(onset), f"{source}:{how}"
        except Exception as error:
            errors.append(f"{source}: {error!r}")
    raise RuntimeError("no plasma onset on either side: " + "; ".join(errors))


def onset_rows(ods: Any, onset: float, keep_channels: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """Measured against vacuum (coil+eddy) magnetics on the onset-aligned grid.

    Per family (``ip``, ``probe`` and ``loop``, and each EFIT sub-family) and grid instant, the median over
    channels of ``|measured - vacuum| / max|measured|`` before the onset (the
    channel's own vacuum amplitude), with the channel count. ``ip`` has no
    vacuum model: the compensated plasma current is expected to be zero before
    the onset, so its row is the measured current itself [kA]. The second list
    holds every channel's measured and vacuum value at :data:`ONSET_POINT_MS`;
    with ``keep_channels`` the third holds every channel's whole trace.
    """
    from vaft.omas.general import signal_time
    from vaft.omas.vacuum_magnetics import synthetic_vacuum_magnetics

    grid = onset + ONSET_GRID_MS * 1e-3
    pre = ONSET_GRID_MS < 0
    start = float(np.asarray(signal_time(ods, "pf_active.coil.0.current.data"), dtype=float)[0])
    channels = synthetic_vacuum_magnetics(ods, per_family=None, window=(grid[0], grid[-1] + 1e-6),
                                          validity_window=(start, onset))
    families: dict[str, list[np.ndarray]] = {}
    traces, points = [], []
    at_points = np.isin(ONSET_GRID_MS, ONSET_POINT_MS)
    for channel in channels:
        inside = (grid >= channel.time[0]) & (grid <= channel.time[-1])
        measured = np.where(inside, np.interp(grid, channel.time, channel.measured), np.nan)
        vacuum = np.where(inside, np.interp(grid, channel.time, channel.coil_eddy), np.nan)
        usable = np.interp(grid, channel.time, channel.usable.astype(float)) > 0.5
        measured[~usable] = np.nan
        scale = np.nanmax(np.abs(measured[pre])) if np.isfinite(measured[pre]).any() else np.nan
        if np.isfinite(scale) and scale > 0:
            kind = "loop" if "flux_loop" in channel.kind else "probe"
            for family in (channel.family, kind):
                families.setdefault(family, []).append(np.abs(measured - vacuum) / scale)
        points += [{"family": "loop" if "flux_loop" in channel.kind else "probe", "channel": channel.name,
                    "t_ms": float(t), "measured": float(m), "vacuum": float(v)}
                   for t, m, v in zip(ONSET_GRID_MS[at_points], measured[at_points], vacuum[at_points])
                   if np.isfinite(m) and np.isfinite(v)]
        if keep_channels:
            traces += [{"family": channel.family, "channel": channel.name, "t_ms": float(t), "measured": float(m),
                        "vacuum": float(v)} for t, m, v in zip(ONSET_GRID_MS, measured, vacuum)]
    rows = []
    for family, stack in families.items():
        stack = np.vstack(stack)
        counts = np.isfinite(stack).sum(axis=0)
        with np.errstate(all="ignore"):
            median = np.nanmedian(np.where(counts > 0, stack, 0.0), axis=0)
        rows += [{"family": family, "t_ms": float(t), "value": float(v) if n else np.nan, "n_channels": int(n)}
                 for t, v, n in zip(ONSET_GRID_MS, median, counts)]
    ip = np.asarray(ods["magnetics.ip.0.data"], dtype=float)
    ip_time = np.asarray(signal_time(ods, "magnetics.ip.0.data"), dtype=float)
    inside = (grid >= ip_time[0]) & (grid <= ip_time[-1])
    points += [{"family": "ip", "channel": "ip", "t_ms": float(t), "measured": float(v) * 1e-3, "vacuum": 0.0}
               for t, v, ok in zip(ONSET_GRID_MS, np.interp(grid, ip_time, ip), inside)
               if ok and t in ONSET_POINT_MS]
    rows += [{"family": "ip", "t_ms": float(t), "value": float(v) * 1e-3 if ok else np.nan, "n_channels": int(ok)}
             for t, v, ok in zip(ONSET_GRID_MS, np.interp(grid, ip_time, ip), inside)]
    if keep_channels:
        traces += [{"family": "ip", "channel": "ip", "t_ms": float(t), "measured": float(v) * 1e-3 if ok else np.nan,
                    "vacuum": 0.0} for t, v, ok in zip(ONSET_GRID_MS, np.interp(grid, ip_time, ip), inside)]
    return rows, points, traces


def onset_qa_rows(ods: Any, onset: float) -> list[dict[str, Any]]:
    """Every channel at :data:`ONSET_QA_MS`, measured against vacuum, with the input QA verdict.

    The diagnostics magnetics QA (:func:`vaft.validation.magnetics.validate_magnetics_signals`,
    #189) is run afresh on each product, so both sides are judged by one
    detector rather than by whatever validity each happened to store, and no
    channel is dropped for its stored validity: ``qa_code`` is the per-sample
    validity at the instant (-2 invalid, -1 suspect, 0 valid, 1 valid but
    flagged) and ``qa_reason`` the channel's reason. The plasma current has no
    such QA (``qa_code`` empty) and no vacuum value but zero.
    """
    from vaft.omas.general import signal_time
    from vaft.omas.vacuum_magnetics import synthetic_vacuum_magnetics
    from vaft.validation.imas import resolve_signal_time
    from vaft.validation.magnetics import QUANTITY_BY_KIND, validate_magnetics_signals
    from vaft.validation.validity import VALIDITY_INVALID

    instant = onset + ONSET_QA_MS * 1e-3
    report = {(q.kind, q.index): q for q in validate_magnetics_signals(ods)}
    window = (instant - 3e-3, instant + 1e-3)
    channels = synthetic_vacuum_magnetics(ods, per_family=None, window=window, validity_window=window,
                                          min_validity=VALIDITY_INVALID)
    rows = []
    for channel in channels:
        if not channel.time[0] <= instant <= channel.time[-1]:
            continue
        row = {"family": "loop" if "flux_loop" in channel.kind else "probe", "channel": channel.name,
               "t_ms": ONSET_QA_MS, "measured": float(np.interp(instant, channel.time, channel.measured)),
               "vacuum": float(np.interp(instant, channel.time, channel.coil_eddy)),
               "qa_code": np.nan, "qa_scalar": np.nan, "qa_status": "", "qa_reason": ""}
        quality = report.get((channel.kind, channel.index))
        if quality is not None and quality.projectable:
            quantity = QUANTITY_BY_KIND[channel.kind][0]
            time = np.asarray(resolve_signal_time(ods, f"magnetics.{channel.kind}.{channel.index}.{quantity}"),
                              dtype=float)
            codes = np.asarray(quality.validity_timed)
            if time.size == codes.size and time.size:
                row["qa_code"] = int(codes[int(np.argmin(np.abs(time - instant)))])
            row.update(qa_scalar=int(quality.validity), qa_status=str(quality.status.value),
                       qa_reason=str(quality.reason)[:200])
        rows.append(row)
    ip_time = np.asarray(signal_time(ods, "magnetics.ip.0.data"), dtype=float)
    if ip_time[0] <= instant <= ip_time[-1]:
        ip = np.asarray(ods["magnetics.ip.0.data"], dtype=float)
        rows.append({"family": "ip", "channel": "ip", "t_ms": ONSET_QA_MS,
                     "measured": float(np.interp(instant, ip_time, ip)) * 1e-3, "vacuum": 0.0})
    return rows


# --------------------------------------------------------------------------
# Legacy stability
# --------------------------------------------------------------------------

def legacy_stability(shot_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """DCON W_t per n and RDCON Delta' per surface from a legacy ``linear_stability`` tree."""
    from vaft.code.gpec import read_dcon_text, read_pest3_matching_text

    rows, surfaces = [], []
    root = shot_dir / "linear_stability"
    if not root.is_dir():
        return rows, surfaces
    for time_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        time_ms = int(time_dir.name)
        for n_dir in sorted((time_dir / "dcon").glob("nn=*")):
            n = int(n_dir.name.split("=")[1])
            row = {"time_ms": time_ms, "n": n, "solver": "dcon"}
            try:
                text = read_dcon_text(n_dir / "dcon.out")
                energies = text["energies"]
                row.update({"status": "ok" if energies else "no_energies",
                            "w_t": energies[0]["total"].real if energies else None,
                            "mpsi": text["truncation"].get("mpsi"), "q95": text["scalars"].get("q95")})
            except Exception as error:
                row.update({"status": "read_error", "error": repr(error)[:300]})
            rows.append(row)
        for n_dir in sorted((time_dir / "rdcon").glob("nn=*")):
            n = int(n_dir.name.split("=")[1])
            row = {"time_ms": time_ms, "n": n, "solver": "rdcon"}
            try:
                result = read_pest3_matching_text(n_dir, mode=n)
                diagonal = result.delta_prime_diagonal()
                values = [d["delta_prime_real"] for d in diagonal]
                row.update({"status": "ok", "msing": result.msing, "mpsi": result.metadata.get("mpsi"),
                            "delta_prime_max": max(values) if values else None,
                            "n_positive_delta_prime": sum(v > 0 for v in values),
                            "classical_tearing_drive": any(v > 0 for v in values) if values else None})
                for d in diagonal:
                    surfaces.append({"time_ms": time_ms, "solver": "rdcon", **d})
            except FileNotFoundError as error:
                row["status"] = "no_delta_gw" if "delta_gw" in str(error) else "no_dcon_out"
            except Exception as error:
                row.update({"status": "read_error", "error": repr(error)[:300]})
            rows.append(row)
    return rows, surfaces


# --------------------------------------------------------------------------
# Output plumbing
# --------------------------------------------------------------------------

class Table:
    """Append-only CSV whose columns grow as new keys appear (rewritten on growth)."""

    def __init__(self, path: Path):
        self.path = path
        self.columns: list[str] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                self.columns = next(reader, [])

    def extend(self, rows: Iterable[Mapping[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        new = [k for row in rows for k in row if k not in self.columns]
        new = list(dict.fromkeys(new))
        if new:
            old_rows = []
            if self.path.exists():
                with self.path.open(newline="", encoding="utf-8") as handle:
                    old_rows = list(csv.DictReader(handle))
            self.columns += new
            # Rewritten through a temporary file and a rename, so an interrupted
            # rewrite cannot truncate rows whose shots are already marked done.
            scratch = self.path.with_suffix(".csv.tmp")
            with scratch.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(old_rows)
            os.replace(scratch, self.path)
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            for row in rows:
                writer.writerow({k: _cell(v) for k, v in row.items()})

    append_rows = extend


def _cell(value: Any) -> Any:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else repr(float(value))
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value)
    return value


def _done(output: Path, check: str) -> set[int]:
    path = output / f"done_{check}.txt"
    return {int(x) for x in path.read_text().split()} if path.exists() else set()


def _mark(output: Path, check: str, shot: int) -> None:
    with (output / f"done_{check}.txt").open("a") as handle:
        handle.write(f"{shot}\n")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def process_shot(shot: int, args, tables: Mapping[str, Table], points_budget: list[int]) -> None:
    import vaft
    from vaft.validation.pipeline_comparison import pair_slices

    checks = set(args.checks)
    efit_checks = checks & {"fit", "virial", "thomson"}
    if efit_checks and not all(shot in _done(args.output, c) for c in efit_checks):
        main_times = main_equilibrium_times(shot)
        time.sleep(args.hsds_sleep)
        sides = side_slices(shot, legacy_root=args.legacy_root, current_filedb=args.current_filedb,
                            main_times=main_times)
        pairing = pair_slices([s["time_s"] for s in sides["legacy"]], [s["time_s"] for s in sides["current"]])
        partner: dict[tuple[str, int], tuple[int, float]] = {}
        for k, pair in enumerate(pairing.pairs):
            partner[("legacy", pair.legacy_index)] = (k, pair.offset)
            partner[("current", pair.current_index)] = (k, pair.offset)
        if shot not in _done(args.output, "slices"):
            rows = []
            for side, entries in sides.items():
                for i, entry in enumerate(entries):
                    pair_id, offset = partner.get((side, i), (None, None))
                    rows.append({"side": side, "shot": shot, "time_ms": entry["time_ms"], "time_s": entry["time_s"],
                                 "gfile": entry["gfile"], "has_mfile": entry["mfile"] is not None,
                                 "has_afile": entry["afile"] is not None,
                                 "pair_id": None if pair_id is None else f"{shot}:{pair_id}",
                                 "pair_offset_s": offset})
            if not rows:
                rows.append({"side": None, "shot": shot, "status": "no_slices",
                             "main_has_equilibrium": main_times is not None})
            tables["slices"].extend(rows)
            _mark(args.output, "slices", shot)

        diagnostics = None
        if "thomson" in checks and shot not in _done(args.output, "thomson") and shot in args.thomson_shots:
            # Only shots whose HSDS public product holds Thomson are tried, so a
            # failed read here is an error that propagates and is retried.
            diagnostics = vaft.database.load(shot, source="public", paths=["thomson_scattering"], cache="off")
            time.sleep(args.hsds_sleep)
            if not len(diagnostics.get("thomson_scattering.channel", [])):
                raise RuntimeError("HSDS public lists thomson_scattering but it has no channels")

        for check in ("fit", "virial", "thomson"):
            if check not in checks or shot in _done(args.output, check):
                continue
            if check == "thomson" and diagnostics is None:
                tables[check].append_rows([{"side": None, "shot": shot, "status": "no_thomson"}])
                _mark(args.output, check, shot)
                continue
            rows, points, arrays, channels = [], [], {}, []
            for side, entries in sides.items():
                for i, entry in enumerate(entries):
                    pair_id = partner.get((side, i), (None, None))[0]
                    base = {"side": side, "shot": shot, "time_ms": entry["time_ms"],
                            "pair_id": None if pair_id is None else f"{shot}:{pair_id}"}
                    try:
                        if check == "fit":
                            row, slice_points, slice_arrays = fit_row(entry)
                            if pair_id is not None:
                                arrays[(side, pair_id)] = slice_arrays
                                if points_budget[0] > 0:
                                    points += [{**base, **p} for p in slice_points]
                        elif check == "virial":
                            row = virial_row(entry)
                        else:
                            row, slice_channels = thomson_row(entry, diagnostics)
                            channels += [{**base, **c} for c in slice_channels]
                    except Exception as error:
                        row = {"status": "error", "error": repr(error)[:300]}
                        if args.verbose:
                            traceback.print_exc()
                    rows.append({**base, **row})
            if check == "fit":
                for row in rows:
                    if row["pair_id"] is None:
                        continue
                    k = int(row["pair_id"].split(":")[1])
                    if ("legacy", k) in arrays and ("current", k) in arrays:
                        row.update(common_channel_fit(arrays[("legacy", k)], arrays[("current", k)])[row["side"]])
            if points:
                pairs = sorted({p["pair_id"] for p in points})[: max(points_budget[0], 0)]
                points = [p for p in points if p["pair_id"] in pairs]
                points_budget[0] -= len(pairs)
                tables["constraint_points"].extend(points)
            if channels:
                tables["thomson_channels"].extend(channels)
            tables[check].extend(rows)
            _mark(args.output, check, shot)

    if "vacuum" in checks and shot not in _done(args.output, "vacuum"):
        rows, loaded = [], {}
        for source in ("public", "main"):
            # A failed read propagates: the shot is logged and left for a resume
            # rather than marked done with a load-error row (an hsget that ran out
            # of file descriptors once dropped 234 shots that way).
            try:
                loaded[source] = vaft.database.load(
                    shot, source=source, paths=["magnetics", "pf_active", "pf_passive"], cache="off")
            finally:
                time.sleep(args.hsds_sleep)
        for source, ods in loaded.items():
            for wall_model in WALL_MODELS:
                try:
                    if wall_model == "stored":
                        target = ods
                    elif "main" not in loaded:
                        raise RuntimeError("no main product to take the common wall geometry from")
                    else:
                        target = common_wall_ods(ods, loaded["main"])
                    rows += [{"source": source, "shot": shot, **r} for r in vacuum_rows(target, wall_model)]
                except Exception as error:
                    rows.append({"source": source, "shot": shot, "wall_model": wall_model, "status": "error",
                                 "reason": repr(error)[:300]})
        tables["vacuum"].extend(rows)
        _mark(args.output, "vacuum", shot)

    if "onset" in checks and shot not in _done(args.output, "onset"):
        loaded = {}
        for source in ("public", "main"):
            try:
                loaded[source] = vaft.database.load(
                    shot, source=source, paths=["magnetics", "pf_active", "pf_passive"], cache="off")
            finally:
                time.sleep(args.hsds_sleep)
        onset, how = onset_reference(loaded)
        for source, ods in loaded.items():
            try:
                rows, points, traces = onset_rows(ods, onset, keep_channels=shot in args.trace_shots)
            except Exception as error:
                rows, points, traces = [{"family": "error", "t_ms": np.nan, "value": np.nan, "n_channels": 0,
                                         "reason": repr(error)[:300]}], [], []
            base = {"source": source, "shot": shot, "onset": onset, "onset_source": how}
            tables["onset"].extend({**base, **r} for r in rows)
            tables["onset_points"].extend({**base, **r} for r in points)
            if args.onset_qa:
                try:
                    qa = onset_qa_rows(ods, onset)
                except Exception as error:
                    qa = [{"family": "error", "reason": repr(error)[:300]}]
                tables["onset_qa"].extend({**base, **r} for r in qa)
            tables["onset_traces"].extend({**base, **r} for r in traces)
        _mark(args.output, "onset", shot)

    if "stability" in checks and shot not in _done(args.output, "stability"):
        rows, surfaces = legacy_stability(args.legacy_root / str(shot))
        tables["stability_legacy"].extend({"shot": shot, **r} for r in rows)
        tables["stability_legacy_surfaces"].extend({"shot": shot, **s} for s in surfaces)
        _mark(args.output, "stability", shot)


def read_shots(args) -> list[int]:
    shots: set[int] = set(args.shots or [])
    if args.shots_file:
        shots |= {int(x) for x in Path(args.shots_file).read_text().split() if x.strip().isdigit()}
    if not shots:
        shots = {int(p.name) for p in args.legacy_root.iterdir() if p.name.isdigit() and p.is_dir()}
    shots = sorted(shots)
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        shots = shots[index::count]
    return shots


def write_manifest(args, shots: Sequence[int]) -> None:
    suffix = "" if not args.shard else "_" + args.shard.replace("/", "of")
    manifest_path = args.output / f"manifest{suffix}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    manifest.update({
        "schema": 1,
        "legacy_root": str(args.legacy_root),
        "current_filedb": str(args.current_filedb),
        "current_membership": "HSDS main equilibrium.time within %.1e s" % MAIN_TIME_TOLERANCE,
        "pairing": "nearest time, one to one, closest first, |dt| <= 5e-4 s "
                   "(vaft.validation.pipeline_comparison.pair_slices)",
        "atlas_dir": None if args.atlas_dir is None else str(args.atlas_dir),
    })
    manifest["runs"].append({"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "commit": _git_commit(), "checks": list(args.checks), "shots": len(shots),
                             "shard": args.shard})
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def copy_atlas(args) -> None:
    if args.atlas_dir is None:
        return
    for name in ("atlas_n.csv", "atlas_surfaces.csv", "schema.json", "README.md"):
        source = Path(args.atlas_dir).expanduser() / name
        if source.exists():
            shutil.copy2(source, args.output / f"stability_current_{name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--current-filedb", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS))
    parser.add_argument("--shots", nargs="*", type=int)
    parser.add_argument("--shots-file")
    parser.add_argument("--shard", help="i/n: process every n-th shot starting at i")
    parser.add_argument("--hsds-sleep", type=float, default=1.0)
    parser.add_argument("--thomson-shots", help="file of shots whose HSDS public product holds thomson_scattering")
    parser.add_argument("--trace-shots", nargs="*", type=int, default=[],
                        help="shots whose every onset-aligned channel trace is kept")
    parser.add_argument("--onset-qa", action="store_true",
                        help="with the onset check, also judge every channel at -1 ms by the input QA")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.legacy_root = args.legacy_root.expanduser()
    args.current_filedb = args.current_filedb.expanduser()
    args.output = args.output.expanduser()
    args.output.mkdir(parents=True, exist_ok=True)

    args.thomson_shots = (set() if not args.thomson_shots else
                          {int(x) for x in Path(args.thomson_shots).read_text().split() if x.strip().isdigit()})
    shots = read_shots(args)
    write_manifest(args, shots)
    copy_atlas(args)
    args.trace_shots = set(args.trace_shots)
    names = ("slices", "fit", "constraint_points", "virial", "thomson", "thomson_channels", "vacuum",
             "onset", "onset_points", "onset_qa", "onset_traces", "stability_legacy", "stability_legacy_surfaces")
    suffix = "" if not args.shard else "_" + args.shard.replace("/", "of")
    tables = {name: Table(args.output / f"{name}{suffix}.csv") for name in names}
    # The point budget is shared by all shards and survives a resume: each shard
    # takes its share and subtracts the pairs its table already holds.
    shards = int(args.shard.split("/")[1]) if args.shard else 1
    held = 0
    points_path = args.output / f"constraint_points{suffix}.csv"
    if points_path.exists():
        with points_path.open(newline="", encoding="utf-8") as handle:
            held = len({row.get("pair_id") for row in csv.DictReader(handle)})
    budget = [CONSTRAINT_POINT_SLICES // shards - held]
    for k, shot in enumerate(shots):
        start = time.time()
        try:
            process_shot(shot, args, tables, budget)
        except Exception as error:  # one shot never stops the run; it stays undone for a resume
            print(f"{shot}: {error!r}", file=sys.stderr, flush=True)
            with (args.output / f"errors{suffix}.txt").open("a", encoding="utf-8") as handle:
                handle.write(f"{shot}\t{error!r}\n")
            if args.verbose:
                traceback.print_exc()
        print(f"[{k + 1}/{len(shots)}] {shot} {time.time() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
